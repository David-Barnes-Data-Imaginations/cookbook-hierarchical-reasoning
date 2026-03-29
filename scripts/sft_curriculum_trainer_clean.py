"""
Two-Stage Curriculum SFT Training (Clean Ministral Version)
============================================================
SFT training for Ministral models on NVIDIA DGX Spark (Blackwell architecture).
Clean version with all Nemotron/Mamba-specific patches removed.

Key Features:
1. Two-stage curriculum (128K context for Ministral-14B)
2. Single high-quality examples with thinking tags (no parallel responses)
3. Sample tracking for RL exclusion
4. Uses Ministral-3-14B-Base (pure Mistral architecture, no Mamba)
5. Memory optimized - halved memory usage by removing parallel responses

Usage:
    python scripts/sft_curriculum_trainer_clean.py --stage 1
    python scripts/sft_curriculum_trainer_clean.py --stage 2
"""

# Ensure single-process mode environment variables are set before
# importing libraries that may initialize distributed training (Accelerate).
# The container may pre-populate RANK/WORLD_SIZE which causes issues
# when loading models with device_map='auto'. Overriding early avoids that.
import os

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")

import sys
import json
import argparse

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

# Hardware-adapted context lengths (DGX Spark — 128GB unified memory)
STAGE_1_MAX_SEQ_LENGTH = (
    131072  # Foundation stage — matches Stage 2 (DGX Spark handles 32K fine)
)
STAGE_2_MAX_SEQ_LENGTH = 131072  # Extended reasoning stage

# Stage 1 sample sizes (multi-category foundation - single examples)
STAGE_1_TOTAL_SAMPLES = 3000
STAGE_1_MATH_RATIO = 0.50  # ~50%
STAGE_1_CODE_RATIO = 0.30  # ~30%
STAGE_1_STEM_RATIO = 0.20  # ~20%

# Stage 2 sample sizes (all reasoning=on, single examples)
STAGE_2_TOTAL_SAMPLES = 3000
STAGE_2_MATH_RATIO = 0.61  # ~61%
STAGE_2_CODE_RATIO = 0.32  # ~32%
STAGE_2_STEM_RATIO = 0.07  # ~7%

# LoRA configuration
LORA_RANK = 64
LORA_ALPHA = 64

# Training parameters
STAGE_1_EPOCHS = 1
STAGE_2_EPOCHS = 1
LEARNING_RATE = 2e-5  # Higher for SFT than RL
BATCH_SIZE = 1  # Reduced to prevent OOM (was 2)
GRADIENT_ACCUMULATION = 4  # Effective batch = 4 (reduced from 8 for stability)

# ============================================================================
# HARDWARE SETTINGS (DGX Spark: Grace CPU 20 ARM cores, 128GB unified memory)
# ============================================================================

# DGX Spark GB10: shared memory per SM is 101,376 bytes, but torch.compile
# (Inductor/Triton) auto-tunes kernels sized for larger GPUs (~120KB+).
# Disable Dynamo to force eager-mode execution and avoid SMEM OOM.
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
# Note: No Mamba-specific env vars needed - Ministral is pure transformer architecture

import torch

# DGX Spark Grace CPU has 20 ARM Neoverse N2 cores — use most of them
NUM_CPU_THREADS = 16
torch.set_num_threads(NUM_CPU_THREADS)
os.environ["OMP_NUM_THREADS"] = str(NUM_CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_CPU_THREADS)

# The NVIDIA PyTorch container sets distributed training env vars by default,
# which causes Accelerate to refuse training when device_map='auto' is set.
# Override them to signal single-process (one GPU) mode.
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# More data workers — enough memory and cores to handle it
NUM_PROC_WORKERS = 8

# Enable packing for training efficiency (memory is no longer the bottleneck)
USE_PACKING = True

# Cleanup handler to prevent segfaults on exit
import atexit


def cleanup_on_exit():
    """Clean up resources before exit to prevent segfaults."""
    try:
        import gc

        gc.collect()
    except Exception:
        pass  # Ignore cleanup errors


atexit.register(cleanup_on_exit)

print(
    f"⚡ DGX Spark mode: {NUM_CPU_THREADS} CPU threads, {NUM_PROC_WORKERS} data workers, packing={USE_PACKING}"
)

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
        try:
            with open(tracking_file, "r") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️  '{tracking_file}' is malformed — starting fresh.")
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
    print(
        f"   Total Stage 1 samples tracked: {sum(len(v) for v in tracking_data.get('stage_1', {}).values())}"
    )
    print(
        f"   Total Stage 2 samples tracked: {sum(len(v) for v in tracking_data.get('stage_2', {}).values())}"
    )


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
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)

    thinking = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else content.strip()

    return thinking, answer


def format_ministral_for_sft(example):
    """
    Format Ministral example for SFT training.

    Uses the original DeepSeek-generated content with thinking tags preserved.
    No parallel responses - just one high-quality example per sample.

    Args:
        example: Raw dataset example from Nemotron-Post-Training-Dataset

    Returns:
        Formatted example with messages and tracking info
    """
    messages = example.get("messages", [])

    user_content = ""
    assistant_content = ""

    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
        elif msg["role"] == "assistant":
            assistant_content = msg["content"]

    if not user_content or not assistant_content:
        return None

    # Extract thinking and answer from original response
    thinking, answer = extract_thinking_and_answer(assistant_content)

    # Use thinking system prompt (we're only doing thinking mode now)
    system_prompt = THINKING_SYSTEM_PROMPT
    user_message = f"/think {user_content}"

    # Preserve the original thinking format from DeepSeek
    if thinking:
        assistant_message = (
            f"<think>\n{thinking}\n</think>\n<answer>\n{answer}\n</answer>"
        )
    else:
        # If no explicit thinking tags, use the full content as thinking
        assistant_message = (
            f"<think>\n{assistant_content}\n</think>\n<answer>\n{answer}\n</answer>"
        )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        "mode": "thinking",  # Always thinking mode
        "source": "ministral_training_data",
    }


def estimate_token_length(text: str) -> int:
    """Rough token count estimation (avg 4 chars per token for English)."""
    return len(text) // 4


def get_example_text_length(example) -> int:
    """Get the total text length of an example for context length filtering."""
    messages = example.get("messages", [])
    total_chars = 0
    for msg in messages:
        total_chars += len(msg.get("content", ""))
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
        if example.get("reasoning") == reasoning_value:
            yield example


def load_stage_1_data(tracking_data: dict):
    """
    Load Stage 1 training data from Nemotron splits.

    Stage 1 focuses on:
    - Multi-category foundation: ~50% math, ~30% code, ~20% stem
    - Single high-quality examples with thinking tags (no parallel responses)
    - Uses DeepSeek-generated reasoning chains from Nemotron dataset
    - Samples exceeding max context are skipped
    """
    print("\n📚 Loading Stage 1 Data (Multi-Category Foundation - Single Examples)...")
    print("=" * 50)
    print(
        f"   Max context: {STAGE_1_MAX_SEQ_LENGTH} tokens (samples exceeding this will be skipped)"
    )
    print(f"   🎯 Single mode: Each sample → 1 example (thinking only)")

    # Approximate max chars (4 chars per token is conservative estimate)
    max_chars = STAGE_1_MAX_SEQ_LENGTH * 4

    # Calculate sample counts directly (no doubling)
    math_samples = int(STAGE_1_TOTAL_SAMPLES * STAGE_1_MATH_RATIO)
    code_samples = int(STAGE_1_TOTAL_SAMPLES * STAGE_1_CODE_RATIO)
    stem_samples = int(STAGE_1_TOTAL_SAMPLES * STAGE_1_STEM_RATIO)

    print(f"   Target distribution:")
    print(f"     - Math: {math_samples} ({STAGE_1_MATH_RATIO * 100:.0f}%)")
    print(f"     - Code: {code_samples} ({STAGE_1_CODE_RATIO * 100:.0f}%)")
    print(f"     - STEM: {stem_samples} ({STAGE_1_STEM_RATIO * 100:.0f}%)")

    all_examples = []
    used_sample_ids = {"math": [], "code": [], "stem": []}
    total_skipped_long = 0
    total_skipped_empty = 0

    # Helper function to load from a split (single examples only)
    def load_from_split(split_name: str, target_count: int, source_key: str):
        """Load samples with thinking format only."""
        nonlocal all_examples, used_sample_ids, total_skipped_long, total_skipped_empty

        print(f"\n🌊 Loading {target_count} samples from {split_name} split...")
        skipped_long = 0
        skipped_empty = 0

        try:
            stream = load_dataset(
                "nvidia/Nemotron-Post-Training-Dataset-v1",
                split=split_name,
                streaming=True,
            )

            count = 0
            for idx, example in enumerate(
                tqdm(stream, total=target_count, desc=split_name)
            ):
                if count >= target_count:
                    break

                # Check content length before processing
                text_len = get_example_text_length(example)
                if text_len > max_chars:
                    skipped_long += 1
                    continue

                # Create single thinking version
                formatted = format_ministral_for_sft(example)

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

    # Load from each split (single examples)
    load_from_split("math", math_samples, "math")
    load_from_split("code", code_samples, "code")
    load_from_split("stem", stem_samples, "stem")

    # Update tracking
    for source_key in ["math", "code", "stem"]:
        tracking_data = mark_samples_used(
            "stage_1", source_key, used_sample_ids[source_key], tracking_data
        )

    print(f"\n   📊 Total Stage 1 examples: {len(all_examples)}")
    print(f"   📊 By category:")
    for source_key in ["math", "code", "stem"]:
        count = len(used_sample_ids[source_key])
        print(f"       - {source_key}: {count}")
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
    print(
        f"   Max context: {STAGE_2_MAX_SEQ_LENGTH} tokens (samples exceeding this will be skipped)"
    )

    # Approximate max chars (4 chars per token is conservative estimate)
    max_chars = STAGE_2_MAX_SEQ_LENGTH * 4

    # Calculate sample counts from ratios
    math_samples = int(STAGE_2_TOTAL_SAMPLES * STAGE_2_MATH_RATIO)
    code_samples = int(STAGE_2_TOTAL_SAMPLES * STAGE_2_CODE_RATIO)
    stem_samples = int(STAGE_2_TOTAL_SAMPLES * STAGE_2_STEM_RATIO)

    print(f"   Target distribution:")
    print(f"     - Math: {math_samples} ({STAGE_2_MATH_RATIO * 100:.0f}%)")
    print(f"     - Code: {code_samples} ({STAGE_2_CODE_RATIO * 100:.0f}%)")
    print(f"     - STEM: {stem_samples} ({STAGE_2_STEM_RATIO * 100:.0f}%)")

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
                streaming=True,
            )

            count = 0
            for idx, example in enumerate(
                tqdm(stream, total=target_count, desc=split_name)
            ):
                if count >= target_count:
                    break

                # Check content length before processing
                text_len = get_example_text_length(example)
                if text_len > max_chars:
                    skipped_long += 1
                    continue

                # Stage 2: ALL thinking mode
                formatted = format_ministral_for_sft(example, mode="thinking")

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
        tracking_data = mark_samples_used(
            "stage_2", source_key, used_sample_ids[source_key], tracking_data
        )

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


def run_sft_training(dataset, stage: int, max_seq_length: int, use_qlora: bool = True):
    """
    Export dataset to JSONL and launch NeMo Automodel for hardware-accelerated training.
    """
    import subprocess
    import shutil
    import yaml

    output_dir = PROJECT_ROOT / f"ministral-sft-stage{stage}"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Export Dataset to JSONL
    data_file = PROJECT_ROOT / f"stage{stage}_data.jsonl"
    print(f"\n💾 Saving {len(dataset)} examples to {data_file} for NeMo...")

    with open(data_file, "w") as f:
        for example in dataset:
            f.write(json.dumps({"messages": example["messages"]}) + "\n")

    # 2. Prepare NeMo Automodel Command
    automodel_dir = PROJECT_ROOT / "Automodel"
    if not automodel_dir.exists():
        raise FileNotFoundError(
            f"NeMo Automodel repository not found at {automodel_dir}. Did you run 'git clone'?"
        )

    # 3. Generate a patched YAML that replaces the squad dataset with JSONL messages dataset.
    #    We do this by loading the base YAML and overriding the dataset sections in
    #    Python — avoids CLI-merge issues where the old squad keys (dataset_name, split)
    #    would be passed as kwargs to ChatDataset.__init__() causing a TypeError.
    # Use Mistral YAML as base (Ministral uses same architecture)
    base_yaml_rel = "examples/llm_finetune/mistral/mistral_nemo_2407_squad_peft.yaml"
    base_yaml_path = automodel_dir / base_yaml_rel
    with open(base_yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Use NeMo's built-in ChatDataset for JSONL (no custom jsonl_messages.py needed)
    # Note: tokenizer will be automatically resolved from the model during instantiation
    # For base models with pre-formatted data, we need to provide a chat template
    # that doesn't modify the existing thinking tags in the assistant content
    # This template preserves the exact content as-is (passthrough)
    # IMPORTANT: Simple template without newlines to avoid YAML parsing issues
    passthrough_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )

    cfg["dataset"] = {
        "_target_": "nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset",
        "path_or_dataset_id": str(data_file),
        "split": "train",
        "chat_template": passthrough_template,
    }
    # val_every_steps > max_steps effectively disables validation (no separate val JSONL)
    cfg["validation_dataset"] = {
        "_target_": "nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset",
        "path_or_dataset_id": str(data_file),
        "split": "train",
        "chat_template": passthrough_template,
    }
    cfg["step_scheduler"]["val_every_steps"] = 999999

    # CRITICAL: Keep tokenizer field empty so NeMo will auto-resolve from model
    # ChatDataset needs a tokenizer with chat template support
    # For base models without chat templates, we provide a custom template
    # Don't delete tokenizer - let NeMo auto-resolve it from the model config

    # CRITICAL FIX: Use DDP instead of FSDP2 for single-GPU DGX Spark (Blackwell)
    # ============================================================================
    # FSDP2 is a multi-GPU tensor-sharding strategy that causes issues on single-GPU
    # setups. On DGX Spark's single unified-memory GPU (128GB), DDP is the correct choice:
    # - No tensor sharding needed (entire model fits in memory)
    # - Simpler and more stable for single-GPU training
    # - Avoids potential issues with model attribute access
    cfg["distributed"] = {"strategy": "ddp"}
    print("   🛡️  Using DDP strategy (optimal for single-GPU Blackwell)")

    # trust_remote_code=True ensures proper model loading
    cfg["model"]["trust_remote_code"] = True

    # Disable Liger kernel to avoid potential issues with model internals
    cfg["model"]["use_liger_kernel"] = False

    # Disable packed_sequence - with 128GB memory and batch_size=2, not needed
    cfg["packed_sequence"]["packed_sequence_size"] = 0

    # Use clean YAML path for Ministral
    patched_yaml_rel = f"examples/llm_finetune/mistral/ministral_sft_stage{stage}.yaml"
    patched_yaml_path = automodel_dir / patched_yaml_rel

    # Custom YAML dumper to prevent multiline string issues with chat_template
    class SingleLineYAMLDumper(yaml.SafeDumper):
        pass

    def str_representer(dumper, data):
        # Force all strings to be single-line to avoid YAML parsing issues
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    SingleLineYAMLDumper.add_representer(str, str_representer)

    with open(patched_yaml_path, "w") as f:
        yaml.dump(
            cfg,
            f,
            Dumper=SingleLineYAMLDumper,
            default_flow_style=False,
            allow_unicode=True,
        )
    print(f"   📝 Wrote patched YAML → {patched_yaml_path}")

    # Debug: Show dataset config to verify no tokenizer key
    print(f"   🔍 Dataset config: {cfg['dataset']}")
    if "tokenizer" in cfg["dataset"]:
        print(f"   ⚠️  WARNING: tokenizer key found in dataset config!")
    else:
        print(f"   ✅ No tokenizer key in dataset config (will be auto-resolved)")

    print(f"\n🚀 Launching Stage {stage} NeMo SFT Training")
    print("=" * 50)
    print(f"   Mode: {'QLoRA' if use_qlora else 'Full LoRA'}")
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   Dataset: {data_file}")

    # Base model or continue from stage 1 checkpoint
    if stage == 1:
        # Use Ministral-3-14B-Base-2512 (pure Mistral architecture, ~14B params)
        # We provide a custom chat template since base models don't have one
        # Ministral is optimized for reasoning and has excellent performance
        # QLoRA will apply 4-bit quantization during training
        model_id = "mistralai/Ministral-3-14B-Base-2512"
        print("   🛡️  Using Ministral-3-14B-Base-2512 with custom chat template")
    else:
        # Load from Stage 1 checkpoint (saved in HuggingFace format, not .nemo)
        model_id = str(PROJECT_ROOT / "ministral-sft-stage1")

    # Calculate steps based on dataset size and batch size
    effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION
    max_steps = (len(dataset) // effective_batch_size) * (
        STAGE_1_EPOCHS if stage == 1 else STAGE_2_EPOCHS
    )

    cmd = [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "/workspace/patched_finetune.py",  # monkey-patches DDPManager then runs finetune.py
        "-c",
        patched_yaml_rel,
        "--model.pretrained_model_name_or_path",
        str(model_id),
        "--packed_sequence.packed_sequence_size",
        str(max_seq_length),
        "--step_scheduler.local_batch_size",
        str(BATCH_SIZE),
        "--step_scheduler.max_steps",
        str(max_steps),
        "--step_scheduler.gradient_accumulation_steps",
        str(GRADIENT_ACCUMULATION),  # CRITICAL: Override to prevent OOM
        "--step_scheduler.num_train_epochs",
        str(STAGE_1_EPOCHS if stage == 1 else STAGE_2_EPOCHS),  # Override epochs
        "--exp_manager.explicit_log_dir",
        str(output_dir),
    ]

    # Add QLoRA (4-bit quantization) overrides if requested
    if use_qlora:
        cmd.extend(
            [
                "--quantization.load_in_4bit",
                "True",
                "--quantization.load_in_8bit",
                "False",
                "--quantization.bnb_4bit_compute_dtype",
                "bfloat16",
                "--quantization.bnb_4bit_use_double_quant",
                "True",
                "--quantization.bnb_4bit_quant_type",
                "nf4",
                "--quantization.bnb_4bit_quant_storage",
                "bfloat16",
            ]
        )

    print("\nExecuting Command:")
    print(" ".join(cmd))
    print("\n" + "-" * 50)

    # 3. Execute
    try:
        # Pass environment variables including our DGX configs
        env = os.environ.copy()

        # CRITICAL: Ensure HF_TOKEN is passed to subprocess for gated models
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            env["HF_TOKEN"] = hf_token
            print("   🔑 HF_TOKEN found and will be passed to subprocess")
        else:
            print("   ⚠️  WARNING: HF_TOKEN not found in environment!")
            print("      Ministral-3-14B-Base is a gated model - you need to either:")
            print(
                "      1. Export HF_TOKEN in your shell: export HF_TOKEN='your_token'"
            )
            print("      2. Or login with: hf auth login")
            print("      3. Or add it to your .env file")

        # Add /workspace to PYTHONPATH so jsonl_messages.py is importable by NeMo's config loader.
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = "/workspace" + (
            ":" + existing_pythonpath if existing_pythonpath else ""
        )

        # Enable detailed error reporting
        env["PYTHONFAULTHANDLER"] = "1"
        env["CUDA_LAUNCH_BLOCKING"] = "1"

        # Run with stderr and stdout captured for better error reporting
        result = subprocess.run(
            cmd,
            cwd=str(automodel_dir),
            env=env,
            check=False,  # Don't raise immediately, we want to capture output
            text=True,
        )

        if result.returncode != 0:
            print(
                f"\n❌ NeMo Automodel training failed with exit code: {result.returncode}"
            )
            if result.stderr:
                print("\n--- STDERR ---")
                print(result.stderr[-3000:])  # Last 3000 chars of error
            if result.stdout:
                print("\n--- STDOUT (last 2000 chars) ---")
                print(result.stdout[-2000:])
            sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ NeMo Automodel training failed with exit code: {e.returncode}")
        sys.exit(1)

    print(f"\n✅ Stage {stage} training complete! Checkpoints saved to {output_dir}")
    return output_dir


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Curriculum SFT Training")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        required=True,
        help="Training stage (1 = Foundation, 2 = Extended)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load data, don't train (for testing)",
    )
    parser.add_argument(
        "--qlora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use QLoRA (4-bit quantized base, default). Use --no-qlora for full LoRA.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"  TWO-STAGE CURRICULUM SFT - STAGE {args.stage}")
    print(
        f"  QLoRA mode: {'ON  (4-bit quantized base + paged optimizer)' if args.qlora else 'OFF (full-precision LoRA)'}"
    )
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
        for msg in sample["messages"]:
            preview = msg["content"][:100].replace("\n", " ")
            print(f"     [{msg['role']}]: {preview}...")

    if args.dry_run:
        print("\n⚠️ Dry run mode - skipping training")
        print(f"   Would train on {len(dataset)} examples")
        print("\n✅ Dry run completed successfully!")
        # CRITICAL: Force immediate exit to avoid streaming dataset thread cleanup issues
        # This prevents the "PyGILState_Release" error during Python shutdown
        import os

        os._exit(0)  # Immediate exit without cleanup (safe for dry-run)

    # Run training
    run_sft_training(dataset, args.stage, max_seq_length, use_qlora=args.qlora)

    print("\n" + "=" * 60)
    print(f"  STAGE {args.stage} COMPLETE!")
    print("=" * 60)
    print(f"  Next steps:")
    if args.stage == 1:
        print("    1. Review Stage 1 checkpoint")
        print("    2. Run: python scripts/sft_curriculum_trainer.py --stage 2")
    else:
        print("    1. Review Stage 2 checkpoint")
        print(
            "    2. Proceed to RL training (samples tracked in sft_used_samples.json)"
        )
    print()


if __name__ == "__main__":
    main()
