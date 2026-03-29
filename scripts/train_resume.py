"""
Resume Training Script with Automatic Checkpoint Detection
============================================================
Automatically finds the most recent checkpoint and continues training.
Supports both LoRA and full fine-tuning checkpoints.

Features:
- Auto-detects latest checkpoint from output directory
- Loads optimizer state and training step
- Continues curriculum learning from where it left off
- Handles both Unsloth LoRA and full fine-tuning checkpoints

Target Hardware: NVIDIA DGX Spark (128GB unified memory, Blackwell)
"""

import os
import json
import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# GPU INITIALIZATION FOR NVIDIA CONTAINERS (DGX Spark)
# ============================================================================

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# ============================================================================
# VERIFY CUDA AVAILABILITY
# ============================================================================

try:
    import torch

    print("=" * 70)
    print("  GPU INITIALIZATION CHECK")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("   To fix, run: docker run --gpus all ...")
        import sys

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
    import sys

    sys.exit(1)

# ============================================================================
# IMPORTS
# ============================================================================

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_SEQ_LENGTH = 131072  # 128K context
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1.5

USE_PACKING = True
CURRICULUM_BUCKETS = 20
TOKEN_ESTIMATE_FACTOR = 4

DATASET_CONFIG = {
    "chat": 9000,
    "conversational_agent": 1000,
    "instruction_following": 1000,
    "math": 6000,
    "science": 2000,
    "swe": 8000,
    "terminal_agent": 3000,
}

# Output directory - will be auto-detected if resuming
OUTPUT_DIR = Path(__file__).parent / "ministral-sft-curriculum"
RESUME_FROM_CHECKPOINT = None  # Will be auto-detected

# ============================================================================
# CHECKPOINT DETECTION
# ============================================================================


def find_latest_checkpoint(base_dir: Path) -> Optional[Path]:
    """
    Find the most recent checkpoint in the output directory.

    Searches for:
    - checkpoint-* directories (standard HuggingFace checkpoints)
    - checkpoint-*-* directories (Unsloth-style checkpoints)
    - merlin checkpoints (if using that format)

    Returns:
        Path to latest checkpoint directory, or None if no checkpoint found
    """
    print("\n🔍 Searching for existing checkpoints...")
    print(f"   Base directory: {base_dir}")

    if not base_dir.exists():
        print(f"   ⚠️  Directory does not exist: {base_dir}")
        return None

    # Find all checkpoint directories
    checkpoint_pattern = re.compile(r"checkpoint-\d+")
    checkpoints = []

    for item in base_dir.iterdir():
        if item.is_dir() and checkpoint_pattern.match(item.name):
            # Extract step number from checkpoint name
            try:
                step = int(item.name.split("-")[1])
                checkpoints.append((step, item))
            except (IndexError, ValueError):
                continue

    if not checkpoints:
        print(f"   ℹ️  No checkpoints found in {base_dir}")
        return None

    # Sort by step number (most recent first)
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_checkpoint = checkpoints[0]

    print(f"   ✅ Found {len(checkpoints)} checkpoint(s)")
    print(f"   🎯 Latest checkpoint: {latest_checkpoint.name} (step {latest_step})")

    # Show recent checkpoints
    print(f"   📋 Recent checkpoints:")
    for step, path in checkpoints[:5]:  # Show top 5
        print(f"      - {path.name} (step {step})")

    return latest_checkpoint


def load_training_state(checkpoint_dir: Path) -> Dict:
    """
    Load training state from checkpoint.

    Returns dict with:
    - global_step: Current training step
    - learning_rate: Current learning rate
    - epoch: Current epoch
    """
    state_file = checkpoint_dir / "trainer_state.json"

    if not state_file.exists():
        print(f"   ⚠️  No trainer_state.json found, using defaults")
        return {
            "global_step": 0,
            "learning_rate": LEARNING_RATE,
            "epoch": 0,
        }

    with open(state_file, "r") as f:
        state = json.load(f)

    return {
        "global_step": state.get("global_step", 0),
        "learning_rate": state.get("log_history", [{}])[-1].get(
            "learning_rate", LEARNING_RATE
        )
        if state.get("log_history")
        else LEARNING_RATE,
        "epoch": state.get("epoch", 0),
    }


def calculate_remaining_steps(total_steps: int, completed_steps: int) -> int:
    """Calculate how many steps remain to complete the desired epochs."""
    remaining = total_steps - completed_steps
    return max(0, remaining)


# ============================================================================
# DATA PREPARATION (Same as curriculum script)
# ============================================================================


def estimate_token_length(text: str) -> int:
    return len(text) // TOKEN_ESTIMATE_FACTOR


def calculate_sample_size(example: Dict) -> int:
    messages = example.get("messages", [])
    total_tokens = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["user", "assistant"]:
            total_tokens += estimate_token_length(content)

    return total_tokens


def format_messages_to_text(example):
    messages = example.get("messages", [])

    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except (json.JSONDecodeError, TypeError):
            return {"text": ""}

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


def load_curriculum_dataset() -> Dataset:
    """Load and sort dataset by size for curriculum learning."""
    print("\n" + "=" * 70)
    print("  CURRICULUM DATA LOADING - Size-Based Sorting")
    print("=" * 70)

    all_examples = []
    total_target = sum(DATASET_CONFIG.values())

    print(f"\n📊 Target distribution (total: {total_target:,} samples):")
    for domain, count in DATASET_CONFIG.items():
        print(f"   - {domain}: {count:,}")

    for domain, target_count in DATASET_CONFIG.items():
        print(f"\n🔄 Loading {domain} ({target_count:,} samples)...")

        try:
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

                token_size = calculate_sample_size(example)
                example["_token_size"] = token_size
                example["_domain"] = domain
                example["_idx"] = count

                domain_examples.append(example)
                count += 1

            all_examples.extend(domain_examples)
            print(f"   ✅ Loaded {count:,} {domain} samples")
            if domain_examples:
                print(
                    f"      Size range: {min(e['_token_size'] for e in domain_examples):,} - "
                    f"{max(e['_token_size'] for e in domain_examples):,} tokens"
                )

        except Exception as e:
            print(f"   ⚠️ Could not load {domain} domain: {e}")
            continue

    print(f"\n📊 Total examples loaded: {len(all_examples):,}")

    dataset = Dataset.from_list(all_examples)

    print("\n🔢 Sorting by token size for curriculum learning...")
    dataset = dataset.sort("_token_size")

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
                f"   Bucket {i:2d}: {start_idx:6d}-{end_idx:6d} | Avg: {avg_size:8,.0f} tokens"
            )

    dataset = dataset.remove_columns(["_token_size", "_domain", "_idx"])

    print("\n📝 Pre-formatting dataset...")
    dataset = dataset.map(
        format_messages_to_text,
        remove_columns=["messages"],
        desc="Formatting examples",
        num_proc=8,
    )

    print(f"\n✅ Curriculum dataset ready: {len(dataset):,} samples")

    return dataset


# ============================================================================
# MODEL LOADING
# ============================================================================


def load_model_and_resume(checkpoint_dir: Optional[Path] = None):
    """
    Load model, optionally resuming from checkpoint.

    Args:
        checkpoint_dir: Path to checkpoint directory, or None to start fresh

    Returns:
        Tuple of (model, tokenizer, training_state)
    """
    if checkpoint_dir:
        print(f"\n🤖 Loading model from checkpoint: {checkpoint_dir.name}")
        print("   Continuing from previous training run...")

        # Load from checkpoint (4-bit for LoRA)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(checkpoint_dir),
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            trust_remote_code=True,
        )
    else:
        print("\n🤖 Loading fresh model...")
        print("   Starting new training from scratch")

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

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("   ✅ LoRA configured")
    return model


# ============================================================================
# TRAINING
# ============================================================================


def train_model(
    model,
    tokenizer,
    dataset,
    start_step: int = 0,
    max_steps: int = None,
    checkpoint_dir: Optional[Path] = None,
):
    """Train the model, optionally resuming from a specific step."""

    effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION
    total_samples = len(dataset)

    # Calculate total steps if not provided
    if max_steps is None:
        max_steps = int((total_samples * NUM_EPOCHS) / effective_batch_size)

    # Calculate remaining steps
    remaining_steps = max(0, max_steps - start_step)

    print("\n🚀 Starting training...")
    print("=" * 70)
    print(f"   Max sequence length: {MAX_SEQ_LENGTH:,}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Total steps: {max_steps:,}")
    print(f"   Starting from step: {start_step:,}")
    print(f"   Remaining steps: {remaining_steps:,}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Packing enabled: {USE_PACKING}")
    if checkpoint_dir:
        print(f"   Resuming from checkpoint: {checkpoint_dir.name}")
    print("=" * 70)

    # Determine resume path - use specific checkpoint if provided, otherwise base dir
    resume_path = (
        str(checkpoint_dir)
        if checkpoint_dir
        else (str(OUTPUT_DIR) if OUTPUT_DIR.exists() else None)
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        max_steps=max_steps,  # Continue to final step
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
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Resume-specific settings - point to the specific checkpoint directory
        resume_from_checkpoint=resume_path,
    )

    print("\n📦 Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=USE_PACKING,
    )

    print("   ✅ Trainer initialized")

    if start_step > 0:
        print(f"\n⏳ Resuming training from step {start_step:,}")
    else:
        print("\n⏳ Training in progress...")

    print("   Samples will be processed from smallest to largest QA pairs")
    print("   Document masking is handled automatically by Flash Attention")
    print()

    # Start training (auto-resumes from checkpoint if available)
    # Use the specific checkpoint directory if provided, otherwise the base output directory
    resume_path = (
        str(checkpoint_dir)
        if checkpoint_dir
        else (str(OUTPUT_DIR) if OUTPUT_DIR.exists() else None)
    )
    trainer.train(resume_from_checkpoint=resume_path)

    print("\n✅ Training complete!")
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


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Resume Training with Automatic Checkpoint Detection"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only analyze dataset, don't train",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to specific checkpoint directory (optional, auto-detected if not provided)",
    )
    parser.add_argument(
        "--force-new",
        action="store_true",
        help="Force new training even if checkpoints exist",
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
    global MAX_SEQ_LENGTH, BATCH_SIZE, GRADIENT_ACCUMULATION, OUTPUT_DIR
    MAX_SEQ_LENGTH = args.seq_length
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION = args.gradient_accumulation

    # Auto-detect output directory from checkpoint if provided
    if args.checkpoint_dir:
        OUTPUT_DIR = Path(args.checkpoint_dir).parent
        print(f"\n📁 Using checkpoint directory: {args.checkpoint_dir}")

    print("=" * 70)
    print("  RESUME TRAINING - Automatic Checkpoint Detection")
    print("=" * 70)
    print(f"  Max sequence length: {MAX_SEQ_LENGTH:,}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"  Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Packing enabled: {USE_PACKING}")
    print("=" * 70)
    print()

    # Check GPU
    print(f"🔍 GPU Info:")
    print(f"   Name: {torch.cuda.get_device_name(0)}")
    print(
        f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )
    print()

    # Find latest checkpoint
    checkpoint_dir = None
    if not args.force_new:
        checkpoint_dir = find_latest_checkpoint(OUTPUT_DIR)

    if checkpoint_dir and not args.force_new:
        print(f"\n💡 Found existing checkpoint: {checkpoint_dir.name}")
        print("   Training will resume from this checkpoint")

        # Load training state
        training_state = load_training_state(checkpoint_dir)
        start_step = training_state["global_step"]

        print(f"\n📊 Training state:")
        print(f"   Completed steps: {start_step:,}")
        print(f"   Current epoch: {training_state['epoch']:.2f}")
        print(f"   Learning rate: {training_state['learning_rate']:.2e}")
    else:
        print(f"\n⚠️  No existing checkpoint found or --force-new specified")
        print("   Starting fresh training from step 0")
        start_step = 0

    print()

    # Load dataset
    print("\n📚 Loading curriculum dataset...")
    dataset = load_curriculum_dataset()

    if args.dry_run:
        print("\n⚠️ Dry run mode - skipping training")
        print(f"   Would train on {len(dataset)} examples")
        if checkpoint_dir:
            print(f"   Would resume from: {checkpoint_dir.name}")
        print("\n✅ Dry run completed successfully!")
        return

    # Load model (with checkpoint if available)
    model, tokenizer = load_model_and_resume(checkpoint_dir)

    # Setup LoRA (only if not already present in checkpoint)
    if checkpoint_dir:
        print("\n🔍 Checking if LoRA adapters already exist...")
        # When loading from checkpoint, LoRA is already loaded by FastLanguageModel.from_pretrained
        print("   ✅ LoRA adapters already loaded from checkpoint - skipping setup")
    else:
        print("\n🔧 Setting up LoRA for new training...")
        model = setup_lora(model)

    # Calculate total steps
    effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION
    total_samples = len(dataset)
    max_steps = int((total_samples * NUM_EPOCHS) / effective_batch_size)

    # Train (will auto-resume from checkpoint)
    trainer = train_model(
        model,
        tokenizer,
        dataset,
        start_step=start_step,
        max_steps=max_steps,
        checkpoint_dir=checkpoint_dir,  # Pass checkpoint directory for proper resume
    )

    # Save
    save_model(model, tokenizer)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Model saved to: {OUTPUT_DIR}")
    print("  Next steps:")
    print("    1. Evaluate the model with standard HF evaluation")
    print("    2. Run inference tests on reasoning benchmarks")
    print("    3. Consider continuing with RL training")
    print()


if __name__ == "__main__":
    import statistics  # Import here to avoid circular import

    main()
