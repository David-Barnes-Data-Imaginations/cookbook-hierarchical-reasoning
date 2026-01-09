"""
Two-Stage Curriculum SFT Training
=================================
Adapted from NVIDIA Nemotron methodology for QLoRA on RTX 4090.

Key Features:
1. Two-stage curriculum (4K → 8K context)
2. /think and /no_think mode control flags
3. Sample tracking for RL exclusion
4. Dual-mode training (thinking + direct response variants)

Usage:
    python scripts/sft_curriculum_trainer.py --stage 1
    python scripts/sft_curriculum_trainer.py --stage 2
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from datasets import load_dataset, Dataset, concatenate_datasets
from dotenv import load_dotenv
from tqdm import tqdm

# Fix: Load .env from project root, not from scripts/ directory
PROJECT_ROOT = Path(__file__).parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    print(f"✅ Loaded environment from {ENV_PATH}")
else:
    load_dotenv()  # Try default locations
    print("⚠️ No .env found in project root, trying default locations...")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Hardware-adapted context lengths (RTX 4090)
STAGE_1_MAX_SEQ_LENGTH = 16384   # Foundation stage
STAGE_2_MAX_SEQ_LENGTH = 32768   # Extended reasoning stage

# Stage 1 sample sizes (multi-category foundation)
STAGE_1_TOTAL_SAMPLES = 3000
STAGE_1_MATH_RATIO = 0.50       # ~50%
STAGE_1_CODE_RATIO = 0.30       # ~30%
STAGE_1_STEM_RATIO = 0.20       # ~20%

# Stage 2 sample sizes (all reasoning=on, no tool_calling for shorter context compatibility)
STAGE_2_TOTAL_SAMPLES = 3000
STAGE_2_MATH_RATIO = 0.61       # ~61%
STAGE_2_CODE_RATIO = 0.32       # ~32%
STAGE_2_STEM_RATIO = 0.07       # ~7%

# LoRA configuration
LORA_RANK = 128
LORA_ALPHA = 128

# Training parameters
STAGE_1_EPOCHS = 1
STAGE_2_EPOCHS = 1
LEARNING_RATE = 2e-5  # Higher for SFT than RL
BATCH_SIZE = 1        # REDUCED: Lower batch = less sudden VRAM spike
GRADIENT_ACCUMULATION = 8  # INCREASED: Compensate for smaller batch

# ============================================================================
# STABILITY SETTINGS (prevent system crashes/reboots)
# ============================================================================
# These reduce peak power draw from simultaneous CPU+GPU load

import torch
import os

# Limit PyTorch CPU threads (prevents CPU power spike during data loading)
NUM_CPU_THREADS = 4  # Reduce from default (all cores) to limit power draw
torch.set_num_threads(NUM_CPU_THREADS)
os.environ["OMP_NUM_THREADS"] = str(NUM_CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_CPU_THREADS)

# Limit data processing parallelism  
NUM_PROC_WORKERS = 4  # Reduced from 36 to prevent CPU power spike during tokenization

# Disable packing (can cause sudden memory spikes)
USE_PACKING = False

print(f"⚡ Stability mode: {NUM_CPU_THREADS} CPU threads, {NUM_PROC_WORKERS} data workers, packing={USE_PACKING}")

# File paths (absolute paths to ensure they save to project directory)
SAMPLE_TRACKING_FILE = PROJECT_ROOT / "sft_used_samples.json"
STRATEGIC_GRAMS_FILE = PROJECT_ROOT / "strategic_grams_deduplicated.json"

# ============================================================================
# SYSTEM PROMPTS WITH MODE CONTROL
# ============================================================================

# Thinking mode system prompt (detailed reasoning)
THINKING_SYSTEM_PROMPT = """You are an advanced reasoning assistant. When asked to think through a problem, show your complete reasoning process.

When you see /think in the user message, respond in this format:
<think>
[Show your step-by-step reasoning here]
</think>
<answer>
[Your final answer here]
</answer>"""

# Direct mode system prompt (no reasoning, fast response)
DIRECT_SYSTEM_PROMPT = """You are an advanced reasoning assistant. When asked for a direct answer, respond concisely without showing your work.

When you see /no_think in the user message, respond directly with just the answer - no thinking tags or explanation."""


# ============================================================================
# SAMPLE TRACKING
# ============================================================================

def load_used_samples():
    """Load the tracking file of samples used in SFT."""
    tracking_file = Path(SAMPLE_TRACKING_FILE)
    if tracking_file.exists():
        with open(tracking_file, "r") as f:
            return json.load(f)
    return {"stage_1": {}, "stage_2": {}}


def save_used_samples(tracking_data):
    """Save the tracking file to disk.
    
    This saves the question IDs used during SFT so they can be excluded
    from RL training later.
    """
    tracking_file = Path(SAMPLE_TRACKING_FILE)
    with open(tracking_file, "w") as f:
        json.dump(tracking_data, f, indent=2)
    print(f"💾 Saved sample tracking to '{tracking_file}'")
    print(f"   Total Stage 1 samples tracked: {sum(len(v) for v in tracking_data.get('stage_1', {}).values())}")
    print(f"   Total Stage 2 samples tracked: {sum(len(v) for v in tracking_data.get('stage_2', {}).values())}")


def mark_samples_used(stage: str, source: str, sample_ids: list, tracking_data: dict):
    """Mark samples as used for a given stage and source."""
    if stage not in tracking_data:
        tracking_data[stage] = {}
    if source not in tracking_data[stage]:
        tracking_data[stage][source] = []
    tracking_data[stage][source].extend(sample_ids)
    return tracking_data


# ============================================================================
# DATA LOADING AND FORMATTING
# ============================================================================

def extract_thinking_and_answer(content: str) -> tuple[str, str]:
    """Extract thinking and answer portions from assistant response."""
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    
    thinking = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else content.strip()
    
    return thinking, answer


def format_nemotron_for_sft(example, mode="thinking"):
    """
    Format Nemotron example for SFT training.
    
    Args:
        example: Raw Nemotron dataset example
        mode: "thinking" for <think> format, "direct" for no thinking
    
    Returns:
        Formatted example with messages and tracking info
    """
    messages = example.get('messages', [])
    
    user_content = ""
    assistant_content = ""
    
    for msg in messages:
        if msg['role'] == 'user':
            user_content = msg['content']
        elif msg['role'] == 'assistant':
            assistant_content = msg['content']
    
    if not user_content or not assistant_content:
        return None
    
    # Extract thinking and answer from original response
    thinking, answer = extract_thinking_and_answer(assistant_content)
    
    if mode == "thinking":
        # Add /think flag and format with thinking tags
        system_prompt = THINKING_SYSTEM_PROMPT
        user_message = f"/think {user_content}"
        
        if thinking:
            assistant_message = f"<think>\n{thinking}\n</think>\n<answer>\n{answer}\n</answer>"
        else:
            # If no explicit thinking, use the full content as thinking
            assistant_message = f"<think>\n{assistant_content}\n</think>\n<answer>\n{answer}\n</answer>"
    
    else:  # direct mode
        system_prompt = DIRECT_SYSTEM_PROMPT
        user_message = f"/no_think {user_content}"
        assistant_message = answer  # Just the answer, no tags
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        "mode": mode,
        "source": "nemotron_math"
    }


def estimate_token_length(text: str) -> int:
    """Rough token count estimation (avg 4 chars per token for English)."""
    return len(text) // 4


def get_example_text_length(example) -> int:
    """Get the total text length of an example for context length filtering."""
    messages = example.get('messages', [])
    total_chars = 0
    for msg in messages:
        total_chars += len(msg.get('content', ''))
    return total_chars


def filter_by_reasoning(dataset, reasoning_value: str):
    """
    Filter dataset by the 'reasoning' column value.
    
    Args:
        dataset: HuggingFace dataset or iterable
        reasoning_value: 'on' or 'off'
    
    Returns:
        Filtered examples as a generator
    """
    for example in dataset:
        if example.get('reasoning') == reasoning_value:
            yield example


def load_stage_1_data(tracking_data: dict):
    """
    Load Stage 1 training data from Nemotron splits.
    
    Stage 1 focuses on:
    - Multi-category foundation: ~50% math, ~30% code, ~20% stem
    - PARALLEL RESPONSES: Each sample creates two training examples:
      1. With /think flag and full <think>...</think> reasoning
      2. With /no_think flag and just the direct answer
    - Shorter context (16K) - samples exceeding this are skipped
    
    This implements the NVIDIA paper's "parallel responses" strategy where
    the model sees both thinking and non-thinking examples for the same input.
    """
    print("\n📚 Loading Stage 1 Data (Multi-Category Foundation + Parallel Responses)...")
    print("=" * 50)
    print(f"   Max context: {STAGE_1_MAX_SEQ_LENGTH} tokens (samples exceeding this will be skipped)")
    print(f"   🔄 Parallel mode: Each sample → 2 examples (thinking + direct)")
    
    # Approximate max chars (4 chars per token is conservative estimate)
    max_chars = STAGE_1_MAX_SEQ_LENGTH * 4
    
    # Calculate BASE sample counts (will be doubled due to parallel responses)
    # Halve the targets since each sample produces 2 examples
    base_samples = STAGE_1_TOTAL_SAMPLES // 2
    math_samples = int(base_samples * STAGE_1_MATH_RATIO)
    code_samples = int(base_samples * STAGE_1_CODE_RATIO)
    stem_samples = int(base_samples * STAGE_1_STEM_RATIO)
    
    print(f"   Target distribution (base samples, will 2x with parallel):")
    print(f"     - Math: {math_samples} → {math_samples * 2} ({STAGE_1_MATH_RATIO*100:.0f}%)")
    print(f"     - Code: {code_samples} → {code_samples * 2} ({STAGE_1_CODE_RATIO*100:.0f}%)")
    print(f"     - STEM: {stem_samples} → {stem_samples * 2} ({STAGE_1_STEM_RATIO*100:.0f}%)")
    
    all_examples = []
    used_sample_ids = {"math": [], "code": [], "stem": []}
    total_skipped_long = 0
    total_skipped_empty = 0
    
    # Helper function to load from a split with parallel responses
    def load_from_split_parallel(split_name: str, target_count: int, source_key: str):
        """Load samples and create BOTH thinking and direct versions for each."""
        nonlocal all_examples, used_sample_ids, total_skipped_long, total_skipped_empty
        
        print(f"\n🌊 Loading {target_count} samples from {split_name} split (→ {target_count * 2} with parallel)...")
        skipped_long = 0
        skipped_empty = 0
        thinking_count = 0
        direct_count = 0
        
        try:
            stream = load_dataset(
                "nvidia/Nemotron-Post-Training-Dataset-v1",
                split=split_name,
                streaming=True
            )
            
            count = 0
            for idx, example in enumerate(tqdm(stream, total=target_count * 2, desc=split_name)):
                if count >= target_count:
                    break
                
                # Check content length before processing
                text_len = get_example_text_length(example)
                if text_len > max_chars:
                    skipped_long += 1
                    continue
                
                # Create BOTH versions for parallel response training
                formatted_thinking = format_nemotron_for_sft(example, mode="thinking")
                formatted_direct = format_nemotron_for_sft(example, mode="direct")
                
                if formatted_thinking and formatted_direct:
                    # Add thinking version
                    formatted_thinking["source"] = source_key
                    formatted_thinking["parallel_id"] = f"{source_key}_{idx}"  # Link parallel pairs
                    all_examples.append(formatted_thinking)
                    thinking_count += 1
                    
                    # Add direct (no thinking) version
                    formatted_direct["source"] = source_key
                    formatted_direct["parallel_id"] = f"{source_key}_{idx}"  # Same ID = same question
                    all_examples.append(formatted_direct)
                    direct_count += 1
                    
                    used_sample_ids[source_key].append(idx)
                    count += 1
                else:
                    skipped_empty += 1
            
            print(f"   ✅ Loaded {count} base samples → {thinking_count + direct_count} total examples")
            print(f"      - Thinking mode: {thinking_count}")
            print(f"      - Direct mode: {direct_count}")
            if skipped_long > 0:
                print(f"   ⏭️  Skipped {skipped_long} (too long)")
            total_skipped_long += skipped_long
            total_skipped_empty += skipped_empty
            return count
        except Exception as e:
            print(f"   ⚠️ Could not load {split_name} split: {e}")
            return 0
    
    # Load from each split with parallel response generation
    load_from_split_parallel("math", math_samples, "math")
    load_from_split_parallel("code", code_samples, "code")
    load_from_split_parallel("stem", stem_samples, "stem")
    
    # Update tracking
    for source_key in ["math", "code", "stem"]:
        tracking_data = mark_samples_used("stage_1", source_key, used_sample_ids[source_key], tracking_data)
    
    # Count by mode for summary
    thinking_total = sum(1 for ex in all_examples if ex.get("mode") == "thinking")
    direct_total = sum(1 for ex in all_examples if ex.get("mode") == "direct")
    
    print(f"\n   📊 Total Stage 1 examples: {len(all_examples)}")
    print(f"   📊 By mode:")
    print(f"       - Thinking (/think): {thinking_total}")
    print(f"       - Direct (/no_think): {direct_total}")
    print(f"   📊 By category:")
    for source_key in ["math", "code", "stem"]:
        count = len(used_sample_ids[source_key])
        print(f"       - {source_key}: {count} base → {count * 2} parallel")
    print(f"   ⏭️  Total skipped (too long): {total_skipped_long}")
    print(f"   ⏭️  Total skipped (empty): {total_skipped_empty}")
    
    return Dataset.from_list(all_examples), tracking_data


def load_stage_2_data(tracking_data: dict):
    """
    Load Stage 2 training data.
    
    Stage 2 focuses on:
    - Longer reasoning chains (32K context)
    - ALL examples in thinking mode (reasoning=on)
    - Distribution: ~61% math, ~32% code, ~7% stem
    """
    print("\n📚 Loading Stage 2 Data (Extended Reasoning)...")
    print("=" * 50)
    print(f"   Max context: {STAGE_2_MAX_SEQ_LENGTH} tokens (samples exceeding this will be skipped)")
    
    # Approximate max chars (4 chars per token is conservative estimate)
    max_chars = STAGE_2_MAX_SEQ_LENGTH * 4
    
    # Calculate sample counts from ratios
    math_samples = int(STAGE_2_TOTAL_SAMPLES * STAGE_2_MATH_RATIO)
    code_samples = int(STAGE_2_TOTAL_SAMPLES * STAGE_2_CODE_RATIO)
    stem_samples = int(STAGE_2_TOTAL_SAMPLES * STAGE_2_STEM_RATIO)
    
    print(f"   Target distribution:")
    print(f"     - Math: {math_samples} ({STAGE_2_MATH_RATIO*100:.0f}%)")
    print(f"     - Code: {code_samples} ({STAGE_2_CODE_RATIO*100:.0f}%)")
    print(f"     - STEM: {stem_samples} ({STAGE_2_STEM_RATIO*100:.0f}%)")
    
    all_examples = []
    used_sample_ids = {"math": [], "code": [], "stem": []}
    total_skipped_long = 0
    total_skipped_empty = 0
    
    # Helper function to load from a split
    def load_from_split(split_name: str, target_count: int, source_key: str):
        """Load samples from a specific split, all in thinking mode."""
        nonlocal all_examples, used_sample_ids, total_skipped_long, total_skipped_empty
        
        print(f"\n🌊 Loading {target_count} samples from {split_name} split...")
        skipped_long = 0
        skipped_empty = 0
        
        try:
            stream = load_dataset(
                "nvidia/Nemotron-Post-Training-Dataset-v1",
                split=split_name,
                streaming=True
            )
            
            count = 0
            for idx, example in enumerate(tqdm(stream, total=target_count * 2, desc=split_name)):
                if count >= target_count:
                    break
                
                # Check content length before processing
                text_len = get_example_text_length(example)
                if text_len > max_chars:
                    skipped_long += 1
                    continue
                
                # Stage 2: ALL thinking mode
                formatted = format_nemotron_for_sft(example, mode="thinking")
                
                if formatted:
                    formatted["source"] = source_key
                    all_examples.append(formatted)
                    used_sample_ids[source_key].append(idx)
                    count += 1
                else:
                    skipped_empty += 1
            
            print(f"   ✅ Loaded {count} {split_name} examples (thinking mode)")
            if skipped_long > 0:
                print(f"   ⏭️  Skipped {skipped_long} (too long)")
            total_skipped_long += skipped_long
            total_skipped_empty += skipped_empty
            return count
        except Exception as e:
            print(f"   ⚠️ Could not load {split_name} split: {e}")
            return 0
    
    # Load from each split with the specified proportions (no tool_calling)
    load_from_split("math", math_samples, "math")
    load_from_split("code", code_samples, "code")
    load_from_split("stem", stem_samples, "stem")
    
    # Update tracking
    for source_key in ["math", "code", "stem"]:
        tracking_data = mark_samples_used("stage_2", source_key, used_sample_ids[source_key], tracking_data)
    
    print(f"\n   📊 Total Stage 2 examples: {len(all_examples)}")
    print(f"   📊 By category:")
    for source_key in ["math", "code", "stem"]:
        count = len(used_sample_ids[source_key])
        print(f"       - {source_key}: {count}")
    print(f"   ⏭️  Total skipped (too long): {total_skipped_long}")
    print(f"   ⏭️  Total skipped (empty): {total_skipped_empty}")
    
    return Dataset.from_list(all_examples), tracking_data


# ============================================================================
# TRAINING
# ============================================================================

def run_sft_training(dataset, stage: int, max_seq_length: int):
    """
    Run SFT training using Unsloth/TRL.
    """
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from huggingface_hub import login, HfApi
    
    # Robust HuggingFace login with multiple fallbacks
    hf_token = os.getenv('HF_TOKEN')
    logged_in = False
    
    if hf_token:
        # Strip whitespace/newlines that might have crept in from .env
        hf_token = hf_token.strip().strip('"').strip("'")
        
        # Validate token format (HF tokens start with "hf_")
        if hf_token.startswith("hf_"):
            try:
                login(token=hf_token, add_to_git_credential=False)
                print("✅ Logged in with HF_TOKEN")
                logged_in = True
            except Exception as e:
                print(f"⚠️ HF_TOKEN login failed: {e}")
        else:
            print(f"⚠️ HF_TOKEN doesn't look valid (should start with 'hf_')")
            print(f"   Token preview: {hf_token[:10]}...")
    
    if not logged_in:
        # Try to use cached credentials from `huggingface-cli login`
        try:
            api = HfApi()
            user_info = api.whoami()
            print(f"✅ Using cached HuggingFace credentials (user: {user_info.get('name', 'unknown')})")
            logged_in = True
        except Exception:
            print("⚠️ No cached HuggingFace credentials found")
            print("   Run `huggingface-cli login` or set HF_TOKEN in .env")
            print("   Continuing without login (will fail for private models)...")
    
    output_dir = f"qwen3-14b-v2-sft-stage{stage}"
    
    print(f"\n🚀 Starting Stage {stage} SFT Training")
    print("=" * 50)
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   Training examples: {len(dataset)}")
    print(f"   Output directory: {output_dir}")
    
    # Load model - different approach for Stage 1 vs Stage 2
    if stage == 2:
        # Stage 2: Load Stage 1 checkpoint directly and continue training those adapters
        stage_1_checkpoint = "qwen3-14b-v2-sft-stage1"
        if os.path.exists(stage_1_checkpoint):
            print(f"\n⏳ Loading Stage 1 checkpoint from {stage_1_checkpoint}...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=stage_1_checkpoint,  # Load the checkpoint directly
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                fast_inference=False,
            )
            print("   ✅ Loaded Stage 1 adapters (will continue training)")
        else:
            raise FileNotFoundError(f"Stage 1 checkpoint not found at {stage_1_checkpoint}. Run Stage 1 first!")
    else:
        # Stage 1: Load base model and attach new LoRA adapters
        print("\n⏳ Loading Qwen3-14B Base...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-14B-Base-bnb-4bit",
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=False,
        )
        
        # Attach LoRA (only for Stage 1 - Stage 2 already has them)
        print("🔗 Attaching LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=LORA_ALPHA,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    
    # Set Qwen3 chat template on tokenizer (base model doesn't have one)
    # This is the standard ChatML format used by Qwen3
    QWEN3_CHAT_TEMPLATE = """{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system
You are a helpful assistant.<|im_end|>
' }}{% endif %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"""
    
    if tokenizer.chat_template is None:
        tokenizer.chat_template = QWEN3_CHAT_TEMPLATE
        print("   ✅ Set Qwen3 ChatML template on tokenizer")
    
    # Format function for the trainer
    def formatting_func(example):
        """Convert messages to Qwen3 ChatML format string.
        
        Note: Unsloth processes in batched mode, so example["messages"] 
        is a list of message-lists, not a single conversation.
        """
        all_messages = example["messages"]
        
        # Handle both single and batched modes
        if isinstance(all_messages[0], dict):
            # Single mode: messages is [{"role": ..., "content": ...}, ...]
            conversations = [all_messages]
        else:
            # Batched mode: messages is [[{"role": ..., ...}, ...], [...], ...]
            conversations = all_messages
        
        results = []
        for messages in conversations:
            # Build the formatted string
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            results.append(formatted)
        
        return results
    
    # Training configuration
    training_args = SFTConfig(
        output_dir=output_dir,
        
        # Optimization
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="paged_adamw_8bit",
        
        # Batching
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        
        # Duration
        num_train_epochs=STAGE_1_EPOCHS if stage == 1 else STAGE_2_EPOCHS,
        save_steps=200,
        logging_steps=10,
        
        # Memory
        fp16=False,
        bf16=True,
        
        # SFT specific
        max_seq_length=max_seq_length,
        packing=USE_PACKING,  # Disabled for stability (prevents sudden memory spikes)
        
        # Logging
        report_to="tensorboard",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        args=training_args,
    )
    
    # Train!
    print("\n🏋️ Starting training...")
    trainer.train()
    
    # Save
    print(f"\n💾 Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Stage {stage} training complete!")
    return model, tokenizer


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Curriculum SFT Training")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True,
                        help="Training stage (1 = Foundation, 2 = Extended)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only load data, don't train (for testing)")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"  TWO-STAGE CURRICULUM SFT - STAGE {args.stage}")
    print("=" * 60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load tracking data
    tracking_data = load_used_samples()
    
    # Load data for the appropriate stage
    if args.stage == 1:
        dataset, tracking_data = load_stage_1_data(tracking_data)
        max_seq_length = STAGE_1_MAX_SEQ_LENGTH
    else:
        dataset, tracking_data = load_stage_2_data(tracking_data)
        max_seq_length = STAGE_2_MAX_SEQ_LENGTH
    
    # Save updated tracking
    save_used_samples(tracking_data)
    
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=42)
    
    # Show sample
    print("\n📝 Sample formatted example:")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"   Mode: {sample.get('mode', 'unknown')}")
        print(f"   Source: {sample.get('source', 'unknown')}")
        print(f"   Messages: {len(sample['messages'])} parts")
        for msg in sample['messages']:
            preview = msg['content'][:100].replace('\n', ' ')
            print(f"     [{msg['role']}]: {preview}...")
    
    if args.dry_run:
        print("\n⚠️ Dry run mode - skipping training")
        print(f"   Would train on {len(dataset)} examples")
        return
    
    # Run training
    run_sft_training(dataset, args.stage, max_seq_length)
    
    print("\n" + "=" * 60)
    print(f"  STAGE {args.stage} COMPLETE!")
    print("=" * 60)
    print(f"  Next steps:")
    if args.stage == 1:
        print("    1. Review Stage 1 checkpoint")
        print("    2. Run: python scripts/sft_curriculum_trainer.py --stage 2")
    else:
        print("    1. Review Stage 2 checkpoint")
        print("    2. Proceed to RL training (samples tracked in sft_used_samples.json)")
    print()


if __name__ == "__main__":
    main()
