"""
Two-Stage Curriculum SFT Training
=================================
Adapted from NVIDIA Nemotron methodology for full LoRA on DGX Spark.

Key Features:
1. Two-stage curriculum (16K → 32K context)
2. /think and /no_think mode control flags
3. Sample tracking for RL exclusion
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

# Hardware-adapted context lengths (DGX Spark — 128GB unified memory)
STAGE_1_MAX_SEQ_LENGTH = (
    131072  # Foundation stage — matches Stage 2 (DGX Spark handles 32K fine)
)
STAGE_2_MAX_SEQ_LENGTH = 131072  # Extended reasoning stage

# Stage 1 sample sizes (multi-category foundation)
STAGE_1_TOTAL_SAMPLES = 3000
STAGE_1_MATH_RATIO = 0.50  # ~50%
STAGE_1_CODE_RATIO = 0.30  # ~30%
STAGE_1_STEM_RATIO = 0.20  # ~20%

# Stage 2 sample sizes (all reasoning=on, no tool_calling for shorter context compatibility)
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
BATCH_SIZE = 2  # DGX Spark has 128GB unified memory — larger batch is fine
GRADIENT_ACCUMULATION = 4  # Effective batch = 8 (same as before)

# ============================================================================
# HARDWARE SETTINGS (DGX Spark: Grace CPU 20 ARM cores, 128GB unified memory)
# ============================================================================

import os

# DGX Spark GB10: shared memory per SM is 101,376 bytes, but torch.compile
# (Inductor/Triton) auto-tunes kernels sized for larger GPUs (~120KB+).
# Disable Dynamo to force eager-mode execution and avoid SMEM OOM.
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
os.environ["MAMBA_SSM_FORCE_TRITON"] = (
    "1"  # Fix: bypass missing selective_scan_cuda CUDA ext (Blackwell)
)

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

# More data workers — enough memory and cores to handle it
NUM_PROC_WORKERS = 8

# Enable packing for training efficiency (memory is no longer the bottleneck)
USE_PACKING = True

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


def format_nemotron_for_sft(example, mode="thinking"):
    """
    Format Nemotron example for SFT training.

    Args:
        example: Raw Nemotron dataset example
        mode: "thinking" for <think> format, "direct" for no thinking

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

    if mode == "thinking":
        # Add /think flag and format with thinking tags
        system_prompt = THINKING_SYSTEM_PROMPT
        user_message = f"/think {user_content}"

        if thinking:
            assistant_message = (
                f"<think>\n{thinking}\n</think>\n<answer>\n{answer}\n</answer>"
            )
        else:
            # If no explicit thinking, use the full content as thinking
            assistant_message = (
                f"<think>\n{assistant_content}\n</think>\n<answer>\n{answer}\n</answer>"
            )

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
        "source": "nemotron_math",
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
    - PARALLEL RESPONSES: Each sample creates two training examples:
      1. With /think flag and full <think>...</think> reasoning
      2. With /no_think flag and just the direct answer
    - Shorter context (16K) - samples exceeding this are skipped

    This implements the NVIDIA paper's "parallel responses" strategy where
    the model sees both thinking and non-thinking examples for the same input.
    """
    print(
        "\n📚 Loading Stage 1 Data (Multi-Category Foundation + Parallel Responses)..."
    )
    print("=" * 50)
    print(
        f"   Max context: {STAGE_1_MAX_SEQ_LENGTH} tokens (samples exceeding this will be skipped)"
    )
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
    print(
        f"     - Math: {math_samples} → {math_samples * 2} ({STAGE_1_MATH_RATIO * 100:.0f}%)"
    )
    print(
        f"     - Code: {code_samples} → {code_samples * 2} ({STAGE_1_CODE_RATIO * 100:.0f}%)"
    )
    print(
        f"     - STEM: {stem_samples} → {stem_samples * 2} ({STAGE_1_STEM_RATIO * 100:.0f}%)"
    )

    all_examples = []
    used_sample_ids = {"math": [], "code": [], "stem": []}
    total_skipped_long = 0
    total_skipped_empty = 0

    # Helper function to load from a split with parallel responses
    def load_from_split_parallel(split_name: str, target_count: int, source_key: str):
        """Load samples and create BOTH thinking and direct versions for each."""
        nonlocal all_examples, used_sample_ids, total_skipped_long, total_skipped_empty

        print(
            f"\n🌊 Loading {target_count} samples from {split_name} split (→ {target_count * 2} with parallel)..."
        )
        skipped_long = 0
        skipped_empty = 0
        thinking_count = 0
        direct_count = 0

        try:
            stream = load_dataset(
                "nvidia/Nemotron-Post-Training-Dataset-v1",
                split=split_name,
                streaming=True,
            )

            count = 0
            for idx, example in enumerate(
                tqdm(stream, total=target_count * 2, desc=split_name)
            ):
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
                    formatted_thinking["parallel_id"] = (
                        f"{source_key}_{idx}"  # Link parallel pairs
                    )
                    all_examples.append(formatted_thinking)
                    thinking_count += 1

                    # Add direct (no thinking) version
                    formatted_direct["source"] = source_key
                    formatted_direct["parallel_id"] = (
                        f"{source_key}_{idx}"  # Same ID = same question
                    )
                    all_examples.append(formatted_direct)
                    direct_count += 1

                    used_sample_ids[source_key].append(idx)
                    count += 1
                else:
                    skipped_empty += 1

            print(
                f"   ✅ Loaded {count} base samples → {thinking_count + direct_count} total examples"
            )
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
        tracking_data = mark_samples_used(
            "stage_1", source_key, used_sample_ids[source_key], tracking_data
        )

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
                tqdm(stream, total=target_count * 2, desc=split_name)
            ):
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

    output_dir = PROJECT_ROOT / f"mistral-nemo-sft-stage{stage}"
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

    # 3. Generate a patched YAML that replaces the squad dataset with ChatDataset.
    #    We do this by loading the base YAML and overriding the dataset sections in
    #    Python — avoids CLI-merge issues where the old squad keys (dataset_name, split)
    #    would be passed as kwargs to ChatDataset.__init__() causing a TypeError.
    # CRITICAL: Use Mistral YAML to avoid loading Mistral3 (vision) model
    base_yaml_rel = "examples/llm_finetune/mistral/mistral_nemo_2407_squad_peft.yaml"
    base_yaml_path = automodel_dir / base_yaml_rel
    with open(base_yaml_path) as f:
        cfg = yaml.safe_load(f)

    chat_ds_cfg = {
        "_target_": "jsonl_messages.make_jsonl_messages_dataset",
        "path_or_dataset_id": str(data_file),
        "split": "train",  # required: train_ft.py reads cfg_ds.split for packed-sequence setup
    }
    cfg["dataset"] = chat_ds_cfg
    # val_every_steps > max_steps effectively disables validation (no separate val JSONL)
    cfg["validation_dataset"] = dict(chat_ds_cfg)
    cfg["step_scheduler"]["val_every_steps"] = 999999
    # CRITICAL FIX: Use DDP instead of FSDP2 for single-GPU DGX Spark (Blackwell)
    # ============================================================================
    # FSDP2 is a multi-GPU tensor-sharding strategy that actively BREAKS NemotronH:
    # - FSDP2 wrapping re-registers and flattens child modules
    # - This clobbers the `model.model` attribute (NemotronHModel sub-object)
    # - Causes `AttributeError: 'NemotronHForCausalLM' object has no attribute 'model'`
    # - Happens at the very first training step regardless of QLoRA/LoRA
    #
    # On DGX Spark's single unified-memory GPU (128GB), DDP is the correct choice:
    # - No tensor sharding needed (entire model fits in memory)
    # - Preserves model structure and attributes
    # - Works correctly with NemotronH's hybrid Mamba2/Attention architecture
    # Replace the ENTIRE distributed block — DDP rejects fsdp2-only keys like
    # sequence_parallel, dp_size, tp_size, cp_size.
    cfg["distributed"] = {"strategy": "ddp"}
    print(
        "   🛡️  Using DDP strategy (FSDP2 disabled for single-GPU Blackwell compatibility)"
    )
    # Fix: trust_remote_code=True is REQUIRED — the installed NemotronHConfig doesn't
    # recognize the '-' character in the model's config.json pattern (KeyError: '-').
    # Keep trust_remote_code=True (the installed NemotronH class is used anyway since
    # it's natively registered in transformers 5.3.0).
    cfg["model"]["trust_remote_code"] = True

    # CRITICAL: Force Automodel's NemotronH implementation (NOT transformers')
    # ============================================================================
    # The checkpoint 'nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base' has architectures:
    #   ["NemotronHForCausalLM"]
    #
    # transformers has a native nemotron_h module that is INCOMPLETE:
    # - Its NemotronHForCausalLM does NOT initialize self.model properly
    # - Causes: AttributeError: 'NemotronHForCausalLM' object has no attribute 'model'
    #
    # Automodel has a COMPLETE implementation in:
    #   nemo_automodel.components.models.nemotron_v3.model
    # - Properly initializes: self.model = NemotronV3Model(config, backend=self.backend)
    # - Supports the hybrid Mamba2/Attention architecture correctly
    #
    # We MUST ensure Automodel's version is used by:
    # 1. Setting force_hf=False (so custom model resolution is attempted)
    # 2. Ensuring the architecture name matches exactly
    cfg["model"]["force_hf"] = False
    print("   🛡️  Disabled force_hf to use Automodel's NemotronH implementation")

    # Additional safeguard: Explicitly set the architecture to ensure matching
    # This helps the custom model resolver identify the correct implementation
    if "model" not in cfg:
        cfg["model"] = {}
    cfg["model"]["_target_"] = "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"
    # Fix: Liger kernel (_apply_liger_kernel_to_instance) patches model internals and
    # has been observed to remove self.model on NemotronH → AttributeError in forward().
    # Disable it for QLoRA; Triton LoRA kernels are still used via peft.use_triton.
    cfg["model"]["use_liger_kernel"] = False

    # Double-check: Ensure FSDP2 is NOT enabled (safety guard)
    if cfg.get("distributed", {}).get("strategy") == "fsdp2":
        raise ValueError(
            "FSDP2 strategy detected! This will cause 'AttributeError: NemotronHForCausalLM "
            "object has no attribute model' on single-GPU Blackwell (DGX Spark). "
            "Please use strategy: ddp instead."
        )

    # CRITICAL: Explicitly specify model type to avoid Mistral3 (vision) being loaded
    # Ministral-3-14B is a text-only Mistral model, NOT Mistral3ForConditionalGeneration
    if "model_type" not in cfg.get("model", {}):
        cfg["model"]["model_type"] = "ministral"
        print("   🛡️  Explicitly set model_type='ministral' to avoid vision model")

    # Debug: Print the model config to verify force_hf is set
    print(f"   📋 Model config before saving: {cfg.get('model', {})}")
    if "force_hf" not in cfg.get("model", {}):
        print("   ⚠️  WARNING: force_hf not found in model config!")
    else:
        print(f"   ✅ force_hf set to: {cfg['model']['force_hf']}")
    if "model_type" not in cfg.get("model", {}):
        print("   ⚠️  WARNING: model_type not set - may load wrong architecture!")
    else:
        print(f"   ✅ model_type set to: {cfg['model']['model_type']}")

    # Fix: packed_sequence forces attn_implementation=flash_attention_2 (kernel_patches.py:182).
    # Flash Attention 2 may further interfere with the model structure on Blackwell.
    # Disable packing — with 128GB unified memory and batch_size=2 this is fine.
    cfg["packed_sequence"]["packed_sequence_size"] = 0

    patched_yaml_rel = f"examples/llm_finetune/nemotron/nemotron_sft_stage{stage}.yaml"
    patched_yaml_path = automodel_dir / patched_yaml_rel
    with open(patched_yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"   📝 Wrote patched YAML → {patched_yaml_path}")

    # Debug: Verify the YAML content
    import subprocess

    result = subprocess.run(
        ["grep", "-A", "5", "model:", str(patched_yaml_path)],
        capture_output=True,
        text=True,
    )
    print(f"   🔍 YAML 'model' section:\n{result.stdout}")

    print(f"\n🚀 Launching Stage {stage} NeMo SFT Training")
    print("=" * 50)
    print(f"   Mode: {'QLoRA' if use_qlora else 'Full LoRA'}")
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   Dataset: {data_file}")

    # Base model or continue from stage 1 checkpoint
    # CRITICAL: Use Mistral-Nemo (text-only) NOT Ministral-3 (Mistral3 architecture with vision tower)
    # Mistral3 architecture always loads vision components → OOM on single GPU
    base_yaml = "examples/llm_finetune/mistral/mistral_nemo_2407_squad_peft.yaml"
    if stage == 1:
        # Use Mistral-Nemo-Base-2407 (pure text, no vision tower, ~12B params)
        # This is the correct choice for text-only SFT training
        # QLoRA will apply 4-bit quantization during training
        model_id = "mistralai/Mistral-Nemo-Base-2407"
        print("   🛡️  Using Mistral-Nemo (text-only, no vision tower)")
    else:
        # Load from Stage 1 checkpoint
        model_id = str(PROJECT_ROOT / "mistral-nemo-sft-stage1.nemo")

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
        return

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
