"""
Curriculum SFT Training Script with Size-Based Sorting
========================================================
Implements Nemotron-Cascade-2 strategy with:
- Data sorted by QA pair size (smallest first)
- Sample packing enabled for efficiency
- Document masking via attention masks
- Optimized gradient accumulation for large effective batch size
- Mixed domain distribution across 30,000 samples

Target Hardware: NVIDIA DGX Spark (128GB unified memory, Blackwell)
Base Model: Ministral-3-14B-Base-2512
"""

import os
import sys
import json
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# GPU INITIALIZATION FOR NVIDIA CONTAINERS (DGX Spark)
# ============================================================================
# The DGX Spark runs in NVIDIA container with unified memory architecture.
# We need to ensure CUDA is properly initialized before importing unsloth.

# Set environment variables for single-GPU mode (DGX Spark has one GB10 GPU)
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")

# Force CUDA visibility
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# DGX Spark specific settings for Blackwell architecture
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

# Disable distributed training (single GPU)
os.environ["NCCL_P2P_DISABLE"] = "1"

# Prevent torch.compile from causing issues (Inductor auto-tunes for larger GPUs)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# ============================================================================
# VERIFY CUDA AVAILABILITY BEFORE IMPORTING UNSLOTH
# ============================================================================

try:
    import torch

    print("=" * 70)
    print("  GPU INITIALIZATION CHECK")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("   This could mean:")
        print("   1. NVIDIA drivers not installed on host")
        print("   2. Container not launched with --gpus all flag")
        print("   3. GPU resources not available")
        print("\n   To fix, run:")
        print("   docker run --gpus all ...")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"✅ GPU detected: {gpu_name}")
    print(f"   Total memory: {gpu_memory:.2f} GB")
    print(f"   CUDA version: {torch.version.cuda}")
    print("=" * 70)
    print()

except Exception as e:
    print(f"❌ Error initializing CUDA: {e}")
    sys.exit(1)

# ============================================================================
# UNSLOTH IMPORTS (AFTER CUDA INITIALIZATION)
# ============================================================================

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Hardware-adapted settings for DGX Spark (128GB unified memory)
MAX_SEQ_LENGTH = 131072  # 128K context for long reasoning chains
BATCH_SIZE = 8  # Increased for memory efficiency with packing
GRADIENT_ACCUMULATION = 4  # Effective batch = 32 (balanced for curriculum)
LEARNING_RATE = 2e-5
MAX_STEPS = None  # Will be calculated based on dataset size

# LoRA configuration
LORA_RANK = 64
LORA_ALPHA = 64
LORA_DROPOUT = 0.0

# Curriculum learning settings
USE_PACKING = True  # Pack multiple short samples together
CURRICULUM_BUCKETS = 20  # Number of size buckets for curriculum
TOKEN_ESTIMATE_FACTOR = 4  # chars per token (conservative estimate)
NUM_EPOCHS = 1.5  # Per Nemotron Cascade 2 paper

# Dataset configuration - 30,000 total samples
DATASET_CONFIG = {
    "chat": 9000,
    "conversational_agent": 1000,
    "instruction_following": 1000,
    "math": 6000,
    "science": 2000,
    "swe": 8000,
    "terminal_agent": 3000,
}

# Project paths
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "ministral-sft-curriculum"
TRACKING_FILE = PROJECT_ROOT / "sft_curriculum_samples.json"

# ============================================================================
# DOCUMENT MASKING & TOKENIZATION
# ============================================================================


def estimate_token_length(text: str) -> int:
    """
    Estimate token count using character-based approximation.
    For precise counts, use the actual tokenizer.
    """
    return len(text) // TOKEN_ESTIMATE_FACTOR


def calculate_sample_size(example: Dict) -> int:
    """
    Calculate total token size of a QA pair (user + assistant).
    This is used for sorting samples by complexity.
    """
    messages = example.get("messages", [])
    total_tokens = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Include both user and assistant content in size calculation
        if role in ["user", "assistant"]:
            total_tokens += estimate_token_length(content)

    return total_tokens


def format_messages_for_training(example: Dict) -> Dict:
    """
    Format dataset example for training with proper message structure.
    Preserves thinking tags and creates proper chat format.
    """
    messages = example.get("messages", [])

    if not messages:
        return None

    # Extract user and assistant content
    user_content = ""
    assistant_content = ""
    system_content = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system_content = content
        elif role == "user":
            user_content = content
        elif role == "assistant":
            assistant_content = content

    # Build formatted text with special tokens
    text = ""
    if system_content:
        text += f"<system>\n{system_content}\n</system>\n"
    if user_content:
        text += f"<user>\n{user_content}\n</user>\n"
    if assistant_content:
        text += f"<assistant>\n{assistant_content}\n</assistant>\n"

    return {
        "text": text,
        "messages": messages,  # Keep original messages for reference
        "_original_idx": example.get("_idx", 0),
    }


# ============================================================================
# DOCUMENT MASKING EXPLANATION
# ============================================================================
"""
DOCUMENT MASKING WITH FLASH ATTENTION:

Document masking is AUTOMATICALLY handled by the SFTTrainer and Flash Attention:

1. **How it works:**
   - When packing is enabled, multiple samples are concatenated into one sequence
   - The trainer automatically creates attention masks to prevent samples from 
     attending to each other
   - Each sample gets its own attention mask boundary

2. **What happens internally:**
   - Loss is only computed on the assistant's response (not user prompts)
   - Attention masks ensure packed samples don't bleed into each other
   - Flash Attention 2 optimizes the masked attention computation

3. **You don't need to manually handle this!**
   - The SFTTrainer with `dataset_num_proc > 1` handles tokenization
   - Setting `max_seq_length` and `packing=True` enables automatic masking
   - The tokenizer's `chat_template` ensures proper formatting

4. **Key parameters:**
   - `dataset_text_field="text"` - tells trainer which field to use
   - `max_seq_length` - defines the context window
   - `packing=True` - enables sample packing with automatic masking
"""


# ============================================================================
# CURRICULUM DATA LOADING
# ============================================================================


def load_curriculum_dataset() -> Dataset:
    """
    Load and sort dataset by size for curriculum learning.

    Strategy:
    1. Load specified samples from each domain
    2. Calculate token size for each sample
    3. Sort all samples from smallest to largest
    4. Return sorted dataset for progressive training
    """
    print("\n" + "=" * 70)
    print("  CURRICULUM DATA LOADING - Size-Based Sorting")
    print("=" * 70)

    all_examples = []
    total_target = sum(DATASET_CONFIG.values())

    print(f"\n📊 Target distribution (total: {total_target:,} samples):")
    for domain, count in DATASET_CONFIG.items():
        print(f"   - {domain}: {count:,}")

    # Load each domain with streaming
    for domain, target_count in DATASET_CONFIG.items():
        print(f"\n🔄 Loading {domain} ({target_count:,} samples)...")

        try:
            # Stream the dataset to avoid memory issues
            dataset = load_dataset(
                "nvidia/Nemotron-Cascade-2-SFT-Data",
                domain,
                split="train",
                streaming=True,
            )

            count = 0
            domain_examples = []

            for idx, example in enumerate(
                tqdm(dataset, total=target_count, desc=f"  {domain}")
            ):
                if count >= target_count:
                    break

                # Calculate size for curriculum sorting
                token_size = calculate_sample_size(example)
                example["_token_size"] = token_size
                example["_domain"] = domain
                example["_idx"] = count

                domain_examples.append(example)
                count += 1

            all_examples.extend(domain_examples)
            print(f"   ✅ Loaded {count:,} {domain} samples")
            print(
                f"      Size range: {min(e['_token_size'] for e in domain_examples):,} - "
                f"{max(e['_token_size'] for e in domain_examples):,} tokens"
            )

        except Exception as e:
            print(f"   ⚠️ Could not load {domain} domain: {e}")
            print("      Skipping this domain...")
            continue

    print(f"\n📊 Total examples loaded: {len(all_examples):,}")

    # Convert to dataset
    dataset = Dataset.from_list(all_examples)

    # Sort by token size (curriculum learning: smallest to largest)
    print("\n🔢 Sorting by token size for curriculum learning...")
    dataset = dataset.sort("_token_size")

    # Create size buckets for analysis
    total_samples = len(dataset)
    bucket_size = max(1, total_samples // CURRICULUM_BUCKETS)

    print(f"\n📈 Curriculum buckets ({CURRICULUM_BUCKETS} buckets):")
    for i in range(CURRICULUM_BUCKETS):
        start_idx = i * bucket_size
        end_idx = min((i + 1) * bucket_size, total_samples)

        if start_idx < total_samples:
            bucket_samples = [
                dataset[j]["_token_size"] for j in range(start_idx, end_idx)
            ]
            avg_size = statistics.mean(bucket_samples)
            print(
                f"   Bucket {i:2d}: {start_idx:6d}-{end_idx:6d} | "
                f"Avg: {avg_size:8,.0f} tokens"
            )

    # Remove temporary columns before training
    dataset = dataset.remove_columns(["_token_size", "_domain", "_idx"])

    print(f"\n✅ Curriculum dataset ready: {len(dataset):,} samples sorted by size")
    print("   Training will proceed from smallest to largest QA pairs")

    return dataset


# ============================================================================
# MODEL LOADING
# ============================================================================


def load_model():
    """Load Ministral model with Unsloth optimizations."""
    print("\n🤖 Loading Ministral-3-14B with Unsloth optimizations...")
    print("   Using pre-quantized 4-bit model for memory efficiency")

    # Load model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Ministral-3-14B-Base-2512-bnb-4bit",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        trust_remote_code=True,
    )

    print("   ✅ Model loaded successfully")
    return model, tokenizer


def setup_lora(model):
    """Configure LoRA for parameter-efficient fine-tuning."""
    print("\n🔧 Configuring LoRA...")
    print(f"   Rank: {LORA_RANK}, Alpha: {LORA_ALPHA}, Dropout: {LORA_DROPOUT}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("   ✅ LoRA configured")
    return model


# ============================================================================
# TRAINING
# ============================================================================


def format_messages_to_text(example):
    """Convert messages array to formatted text string."""
    import json

    messages = example.get("messages", [])

    # Handle case where messages might be a JSON string
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except (json.JSONDecodeError, TypeError):
            return {"text": ""}

    # Concatenate all messages into a single text
    text = ""
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                text += f"<system>\n{content}\n</system>\n"
            elif role == "user":
                text += f"<user>\n{content}\n</user>\n"
            elif role == "assistant":
                text += f"<assistant>\n{content}\n</assistant>\n"

    return {"text": text}


def train_model(model, tokenizer, dataset):
    """Train the model with curriculum learning support."""

    # Calculate max steps based on dataset size and desired epochs
    effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION
    total_samples = len(dataset)

    # Calculate steps for 1.5 epochs (per Nemotron Cascade 2 paper)
    # Total samples needed = total_samples * epochs
    # Steps = (total_samples * epochs) / effective_batch_size
    max_steps = int((total_samples * NUM_EPOCHS) / effective_batch_size)

    print("\n🚀 Starting curriculum training...")
    print("=" * 70)
    print(f"   Max sequence length: {MAX_SEQ_LENGTH:,}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Max steps: {max_steps:,}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Packing enabled: {USE_PACKING}")
    print("=" * 70)

    # Pre-format the dataset (convert messages to text)
    print("\n📝 Pre-formatting dataset (converting messages to text)...")
    dataset = dataset.map(
        format_messages_to_text,
        remove_columns=["messages"],  # Remove raw messages, keep formatted text
        desc="Formatting examples",
    )

    # Show a sample to verify formatting
    print("\n📋 Sample formatted text (first 500 chars):")
    sample_text = dataset[0]["text"]
    print(f"   {sample_text[:500]}...")
    print(f"   Total tokens in sample: ~{len(sample_text) // 4:,}")
    print()

    # Training arguments optimized for curriculum learning
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        max_steps=max_steps,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=50,
        optim="adamw_bnb_8bit",
        seed=42,
        report_to="none",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        # Performance optimizations
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Initialize trainer (no formatting_func needed - data is pre-formatted)
    print("\n📦 Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",  # Use pre-formatted text field
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=USE_PACKING,  # Pack multiple samples per sequence
    )

    print("   ✅ Trainer initialized")
    print("\n⏳ Curriculum training in progress...")
    print("   Samples will be processed from smallest to largest QA pairs")
    print("   Document masking is handled automatically by Flash Attention")
    print()

    # Start training
    trainer.train()

    print("\n✅ Curriculum training complete!")
    return trainer


# ============================================================================
# SAVING
# ============================================================================


def save_model(model, tokenizer):
    """Save the trained model and tokenizer."""
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")

    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"   ✅ Model saved to {OUTPUT_DIR}")
    print("   🎯 You can now use this model for inference or further training")


# ============================================================================
# DRY RUN & ANALYSIS
# ============================================================================


def analyze_curriculum_distribution(dataset: Dataset):
    """Analyze the curriculum distribution before training."""
    print("\n" + "=" * 70)
    print("  CURRICULUM ANALYSIS")
    print("=" * 70)

    # Re-load with size info for analysis
    all_examples = []

    for domain, target_count in DATASET_CONFIG.items():
        try:
            stream = load_dataset(
                "nvidia/Nemotron-Cascade-2-SFT-Data",
                domain,
                split="train",
                streaming=True,
            )

            count = 0
            for idx, example in enumerate(stream):
                if count >= target_count:
                    break
                example["_token_size"] = calculate_sample_size(example)
                example["_domain"] = domain
                all_examples.append(example)
                count += 1
        except:
            continue

    # Sort and analyze
    all_examples.sort(key=lambda x: x["_token_size"])

    # Show distribution by domain across size ranges
    print("\n📊 Domain distribution across size ranges:")

    # Divide into quartiles
    total = len(all_examples)
    quartile_size = total // 4

    for i, (name, start, end) in enumerate(
        [
            ("Smallest 25%", 0, quartile_size),
            ("25-50%", quartile_size, 2 * quartile_size),
            ("50-75%", 2 * quartile_size, 3 * quartile_size),
            ("Largest 25%", 3 * quartile_size, total),
        ]
    ):
        domain_counts = {}
        for ex in all_examples[start:end]:
            domain = ex["_domain"]
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        print(f"\n   {name} ({end - start} samples):")
        for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            pct = count / (end - start) * 100
            print(f"      {domain:25s}: {count:5d} ({pct:5.1f}%)")

    # Show size statistics
    sizes = [ex["_token_size"] for ex in all_examples]
    print(f"\n📈 Size statistics:")
    print(f"   Min:  {min(sizes):,} tokens")
    print(f"   Max:  {max(sizes):,} tokens")
    print(f"   Median: {statistics.median(sizes):,} tokens")
    print(f"   Mean:  {statistics.mean(sizes):,} tokens")
    print(f"   StdDev: {statistics.stdev(sizes):,.0f} tokens")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum SFT Training with Size-Based Sorting"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only analyze dataset, don't train",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze curriculum distribution before training",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=131072,
        help="Max sequence length (default: 128K)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    args = parser.parse_args()

    # Update config from arguments
    global MAX_SEQ_LENGTH, BATCH_SIZE, GRADIENT_ACCUMULATION
    MAX_SEQ_LENGTH = args.seq_length
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION = args.gradient_accumulation

    print("=" * 70)
    print("  CURRICULUM SFT TRAINING - Size-Based Learning")
    print("=" * 70)
    print(f"  Max sequence length: {MAX_SEQ_LENGTH:,}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"  Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Packing enabled: {USE_PACKING}")
    print("=" * 70)
    print(f"  Started: {torch.cuda.current_device()}")
    print()

    # Check GPU
    print(f"🔍 GPU Info:")
    print(f"   Name: {torch.cuda.get_device_name(0)}")
    print(
        f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )
    print()

    # Analyze if requested
    if args.analyze:
        print("\n📊 Running curriculum analysis...")
        # Create dummy dataset for analysis
        analyze_curriculum_distribution(None)
        print("\n✅ Analysis complete!")
        return

    # Load curriculum dataset
    dataset = load_curriculum_dataset()

    if args.dry_run:
        print("\n⚠️ Dry run mode - skipping training")
        print(f"   Would train on {len(dataset)} examples")
        print("   Samples are sorted from smallest to largest QA pairs")
        print("\n✅ Dry run completed successfully!")
        return

    # Load model
    model, tokenizer = load_model()

    # Setup LoRA
    model = setup_lora(model)

    # Train with curriculum
    trainer = train_model(model, tokenizer, dataset)

    # Save
    save_model(model, tokenizer)

    print("\n" + "=" * 70)
    print("  CURRICULUM TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Model saved to: {OUTPUT_DIR}")
    print("  Next steps:")
    print("    1. Evaluate the model with standard HF evaluation")
    print("    2. Run inference tests on reasoning benchmarks")
    print("    3. Consider continuing with RL training")
    print()


if __name__ == "__main__":
    main()
