"""
Unsloth Training Script for Ministral-3-14B on DGX Spark
=========================================================
Optimized for Blackwell architecture with 128GB unified memory.
Uses pre-quantized Ministral model from Unsloth.

Features:
- 2-3x faster training vs standard methods
- 50% less memory usage
- Built-in optimizations for Blackwell GPUs
- Simple configuration without complex YAML files
- **Automatic sample tracking** for RL exclusion

Sample Tracking:
- Automatically records which JSONL indices were used for training
- Saves to: sft_used_samples.json
- Use `check_used_samples.py` to view what was used
- Generate `rl_exclude_indices.txt` for RL training

Usage:
    python train_unsloth.py                    # Standard training
    python train_unsloth.py --dry-run          # Test without training
    python train_unsloth.py --seq-length 65536 # Custom context length
"""

import os
import json
import argparse
from pathlib import Path

# ============================================================================
# UNSLOTH IMPORTS
# ============================================================================

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# ============================================================================
# CONFIGURATION
# ============================================================================

# Hardware-adapted settings for DGX Spark (128GB unified memory)
MAX_SEQ_LENGTH = 131072  # Start with 32K (can increase to 64K or 128K if needed)
BATCH_SIZE = 2  # Unsloth is more memory-efficient, can use batch 2
GRADIENT_ACCUMULATION = 4  # Effective batch = 8
LEARNING_RATE = 2e-5
MAX_STEPS = 750  # For 3000 samples with batch 8

# LoRA configuration
LORA_RANK = 64
LORA_ALPHA = 64
LORA_DROPOUT = 0.0

# Project paths
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "ministral-sft-unsloth"
DATA_FILE = PROJECT_ROOT / "stage1_data.jsonl"
TRACKING_FILE = PROJECT_ROOT / "sft_used_samples.json"

# ============================================================================
# DATA PREPARATION WITH TRACKING
# ============================================================================


def load_used_samples():
    """Load the tracking file of samples used in SFT."""
    if TRACKING_FILE.exists():
        try:
            with open(TRACKING_FILE, "r") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️  '{TRACKING_FILE}' is malformed — starting fresh.")
    return {"stage_1": {}, "stage_2": {}}


def save_used_samples(tracking_data, stage: str, source: str, sample_indices: list):
    """Save the list of sample indices used for training."""
    if stage not in tracking_data:
        tracking_data[stage] = {}
    if source not in tracking_data[stage]:
        tracking_data[stage][source] = []

    # Add new indices (avoiding duplicates)
    existing = set(tracking_data[stage][source])
    for idx in sample_indices:
        if idx not in existing:
            tracking_data[stage][source].append(idx)

    with open(TRACKING_FILE, "w") as f:
        json.dump(tracking_data, f, indent=2)

    print(f"💾 Saved sample tracking to '{TRACKING_FILE}'")
    print(
        f"   Stage {stage} - {source}: {len(tracking_data[stage][source])} samples tracked"
    )


def format_for_training(example, idx: int):
    """Format dataset example for training."""
    messages = example.get("messages", [])

    # Concatenate all messages into a single text
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<system>\n{content}\n</system>\n"
        elif role == "user":
            text += f"<user>\n{content}\n</user>\n"
        elif role == "assistant":
            text += f"<assistant>\n{content}\n</assistant>\n"

    return {"text": text, "_original_idx": idx}


def load_training_data(stage: str = "stage_1", source: str = "ministral_training_data"):
    """Load and format the training dataset with tracking."""
    print(f"\n📚 Loading training data from {DATA_FILE}...")

    # Load JSONL dataset
    dataset = load_dataset("json", data_files=str(DATA_FILE), split="train")

    # Get current tracking data
    tracking_data = load_used_samples()
    used_indices = set(tracking_data.get(stage, {}).get(source, []))

    print(f"   Previously used samples: {len(used_indices)}")

    # Filter out already used samples if needed
    if used_indices:
        print(f"   🔄 Filtering out {len(used_indices)} previously used samples...")
        indices_to_keep = [i for i in range(len(dataset)) if i not in used_indices]
        dataset = dataset.select(indices_to_keep)
        print(f"   Remaining samples: {len(dataset)}")

    # Format for training and track indices
    print("   🔄 Formatting examples for training...")

    # Track which indices we're using
    used_indices_for_this_run = []

    def format_with_tracking(example):
        idx = example["_idx"] if "_idx" in example else example["__index"]
        used_indices_for_this_run.append(idx)
        return format_for_training(example, idx)

    # Add index to dataset for tracking
    dataset = dataset.map(lambda x, idx: {**x, "_idx": idx}, with_indices=True)

    dataset = dataset.map(format_with_tracking, remove_columns=["_idx"])

    # Save tracking
    if used_indices_for_this_run:
        save_used_samples(tracking_data, stage, source, used_indices_for_this_run)

    print(f"   ✅ Loaded {len(dataset)} examples")
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
        use_gradient_checkpointing="unsloth",  # Memory-efficient checkpointing
        random_state=42,
    )

    print("   ✅ LoRA configured")
    return model


# ============================================================================
# TRAINING
# ============================================================================


def train_model(model, tokenizer, dataset):
    """Train the model with SFT trainer."""
    print("\n🚀 Starting training...")
    print(f"   Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"   Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"   Max steps: {MAX_STEPS}")
    print(f"   Learning rate: {LEARNING_RATE}")

    # Training arguments optimized for DGX Spark
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=50,
        optim="adamw_bnb_8bit",  # Memory-optimized optimizer
        seed=42,
        report_to="none",  # Disable wandb/tensorboard for simplicity
        warmup_steps=100,
        lr_scheduler_type="cosine",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    # Start training
    print("\n⏳ Training in progress...")
    trainer.train()

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
    print("   🎯 You can now use this model for inference or further training")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Unsloth Training for Ministral")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only load data, don't train"
    )
    parser.add_argument(
        "--seq-length", type=int, default=32768, help="Max sequence length"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    args = parser.parse_args()

    # Update config from arguments
    global MAX_SEQ_LENGTH, BATCH_SIZE
    MAX_SEQ_LENGTH = args.seq_length
    BATCH_SIZE = args.batch_size

    print("=" * 70)
    print("  UNSLOTH TRAINING - Ministral-3-14B on DGX Spark")
    print("=" * 70)
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"  Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
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

    # Load data with tracking
    dataset = load_training_data(stage="stage_1", source="ministral_training_data")

    if args.dry_run:
        print("\n⚠️ Dry run mode - skipping training")
        print(f"   Would train on {len(dataset)} examples")
        print("\n✅ Dry run completed successfully!")
        return

    # Load model
    model, tokenizer = load_model()

    # Setup LoRA
    model = setup_lora(model)

    # Train
    trainer = train_model(model, tokenizer, dataset)

    # Save
    save_model(model, tokenizer)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Model saved to: {OUTPUT_DIR}")
    print("  Next steps:")
    print("    1. Evaluate the model with standard HF evaluation")
    print("    2. Run inference tests")
    print("    3. Proceed to Stage 2 training if needed")
    print()


if __name__ == "__main__":
    main()
