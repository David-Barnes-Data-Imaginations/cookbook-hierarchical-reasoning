#!/usr/bin/env python3
"""
Model Evaluation Script with Interactive CLI
=============================================
Evaluate trained models on various benchmarks using lm_eval.

Features:
- Interactive selection of model directories and checkpoints
- Configurable limit for evaluation samples
- Support for both LoRA and full fine-tuned models
- Automatic result saving and formatting
- **Checkpointing**: Saves results periodically to prevent data loss
- **Resume capability**: Can resume from last checkpoint if interrupted

Usage:
    python scripts/evaluate_model.py

Example:
    python scripts/evaluate_model.py
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime  # Fixed: Use datetime instead of torch.cuda
from typing import List, Optional, Dict, Any

# ============================================================================
# DEPENDENCY CHECK
# ============================================================================

try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    import torch
    from transformers import AutoTokenizer

    # Try to import Unsloth first (preferred for your checkpoints)
    try:
        from unsloth import FastLanguageModel

        UNSLOTH_AVAILABLE = True
    except ImportError:
        UNSLOTH_AVAILABLE = False
        from transformers import AutoModelForCausalLM
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("\nPlease install required packages:")
    print("  pip install lm_eval transformers torch accelerate")
    if not UNSLOTH_AVAILABLE:
        print("  pip install unsloth")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default benchmarks to evaluate
DEFAULT_TASKS = [
    ("arc_challenge", "ARC Challenge"),
    ("hellaswag", "HellaSwag"),
    ("winogrande", "Winogrande"),
    ("piqa", "PIQA"),
    ("mmlu", "MMLU"),
    ("gsm8k", "GSM8K"),
    ("truthfulqa_mc2", "TruthfulQA MC2"),
]

# Base directory for checkpoints
CHECKPOINTS_DIR = Path(__file__).parent.parent / "checkpoints"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_section(text: str):
    """Print a formatted section header."""
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}\n")


def get_available_models() -> List[Path]:
    """
    Scan the checkpoints directory for available model directories.

    Returns:
        List of paths to model directories
    """
    if not CHECKPOINTS_DIR.exists():
        print(f"⚠️  Checkpoints directory not found: {CHECKPOINTS_DIR}")
        return []

    models = []
    for item in CHECKPOINTS_DIR.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            models.append(item)

    return sorted(models)


def get_checkpoints(model_dir: Path) -> List[Path]:
    """
    Find all checkpoint directories within a model directory.

    Args:
        model_dir: Path to the model directory

    Returns:
        List of checkpoint paths, sorted by step number (descending)
    """
    checkpoints = []
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")

    for item in model_dir.iterdir():
        if item.is_dir():
            match = checkpoint_pattern.match(item.name)
            if match:
                checkpoints.append((int(match.group(1)), item))

    # Sort by step number (descending - newest first)
    checkpoints.sort(key=lambda x: x[0], reverse=True)

    return [path for _, path in checkpoints]


def display_menu(title: str, options: List[str], show_index: bool = True) -> int:
    """
    Display a numbered menu and get user selection.

    Args:
        title: Menu title
        options: List of options
        show_index: Whether to show index numbers

    Returns:
        Selected index (0-based)
    """
    print_section(title)

    if not options:
        print("⚠️  No options available!")
        return -1

    for i, option in enumerate(options):
        if show_index:
            print(f"  {i + 1}. {option}")
        else:
            print(f"  {option}")

    print()
    return -1


def get_user_selection(prompt: str, min_val: int, max_val: int) -> int:
    """Get a validated integer selection from user."""
    while True:
        try:
            user_input = input(prompt).strip()
            value = int(user_input)

            if min_val <= value <= max_val:
                return value
            else:
                print(f"❌ Please enter a number between {min_val} and {max_val}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n❌ Cancelled!")
            sys.exit(0)


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================


def load_model(checkpoint_path: Path) -> tuple:
    """
    Load a model and tokenizer from checkpoint using Unsloth.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        Tuple of (model, tokenizer)
    """
    print_section("Loading Model")
    print(f"Checkpoint: {checkpoint_path}")

    try:
        # Check if this is a LoRA adapter
        adapter_config_path = checkpoint_path / "adapter_config.json"
        is_lora = adapter_config_path.exists()

        if UNSLOTH_AVAILABLE and is_lora:
            # Use Unsloth for loading (best compatibility with your checkpoints)
            print("🔄 Using Unsloth to load base model, then applying LoRA adapters...")

            # Load adapter config to get base model info
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)

            base_model_path = adapter_config.get("base_model_name_or_path")
            print(f"   Base model: {base_model_path}")

            # Import PEFT for loading adapters
            from peft import PeftModel

            print("🔄 Loading base model with Unsloth (4-bit)...")
            # Load base model only (no adapters yet)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_path,
                max_seq_length=8192,
                load_in_4bit=True,
                trust_remote_code=True,
            )

            print("🔄 Applying LoRA adapters...")
            # Now apply LoRA adapters
            try:
                model = PeftModel.from_pretrained(
                    model,
                    str(checkpoint_path),
                    is_trainable=False,  # We're just evaluating
                )
                model.eval()
                print(
                    "✅ Model with LoRA adapters loaded successfully via Unsloth + PEFT!"
                )
                print(f"   Device: {model.device}")

                # IMPORTANT: Ensure tokenizer is properly configured for lm_eval
                # Unsloth's tokenizer might not have all required attributes
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    print("   ✅ Set tokenizer.pad_token")

                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                    print("   ✅ Set tokenizer.pad_token_id")

                # Ensure model_input_names is set (required by lm_eval)
                if not hasattr(tokenizer, "model_input_names"):
                    tokenizer.model_input_names = ["input_ids", "attention_mask"]
                    print("   ✅ Set tokenizer.model_input_names")

            except Exception as peft_error:
                print(f"⚠️  PEFT loading failed: {peft_error}")
                print("   Trying alternative approach: merging LoRA weights...")

                # Alternative: Try to merge adapters using Unsloth's method
                try:
                    # Merge LoRA weights into base model
                    model = FastLanguageModel.for_inference(model)
                    print("✅ LoRA adapters merged successfully!")

                    # Still need to ensure tokenizer is configured
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id

                except Exception as merge_error:
                    print(f"❌ Both methods failed. Error: {merge_error}")
                    raise

        elif is_lora:
            # Fallback to PEFT if Unsloth not available
            print("⚠️  Unsloth not available, trying PEFT fallback...")
            print("   Note: This may not work with 4-bit Mistral3 models")

            from peft import PeftModel
            from transformers import AutoConfig, AutoTokenizer
            import torch.nn as nn

            # Load adapter config
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)

            base_model_path = adapter_config.get("base_model_name_or_path")
            print(f"   Base model: {base_model_path}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Try to load with PEFT (may fail with 4-bit models)
            print("🔄 Loading base model in 4-bit...")
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            print("🔄 Loading LoRA adapters...")
            model = PeftModel.from_pretrained(
                base_model,
                str(checkpoint_path),
                device_map="auto",
            )

            model.eval()
            print("✅ Model loaded via PEFT!")

        else:
            # Full fine-tuned model (no LoRA)
            print("🔄 Loading full fine-tuned model...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(checkpoint_path),
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if UNSLOTH_AVAILABLE:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=str(checkpoint_path),
                    max_seq_length=8192,
                    load_in_4bit=True,
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    str(checkpoint_path),
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )

            model.eval()
            print("✅ Model loaded successfully!")

        # ============================================================================
        # ENSURE TOKENIZER IS COMPATIBLE WITH lm_eval
        # ============================================================================
        # The lm_eval library requires specific tokenizer attributes
        # This is the key fix for the AssertionError
        print("🔧 Ensuring tokenizer compatibility with lm_eval...")

        # Ensure essential tokenizer attributes are set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("   Set pad_token to eos_token")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("   Set pad_token_id to eos_token_id")

        # Ensure the tokenizer has a model input name attribute
        # This is critical for lm_eval's HFLM wrapper
        if not hasattr(tokenizer, "model_input_names"):
            tokenizer.model_input_names = ["input_ids", "attention_mask"]
            print("   Added model_input_names attribute")

        # For lm_eval compatibility, ensure we have the right attributes
        # Check if tokenizer has the required attributes
        required_attrs = ["pad_token", "pad_token_id", "eos_token", "eos_token_id"]
        for attr in required_attrs:
            if not hasattr(tokenizer, attr) or getattr(tokenizer, attr) is None:
                # Try to set from known values
                if "token" in attr:
                    base_attr = attr.replace("_token", "")
                    if hasattr(tokenizer, base_attr + "_token"):
                        setattr(
                            tokenizer, attr, getattr(tokenizer, base_attr + "_token")
                        )
                        print(f"   Set {attr} from {base_attr}_token")

        print("✅ Tokenizer compatibility checks complete!")

        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback

        traceback.print_exc()
        raise


def ensure_model_compatibility(model, tokenizer):
    """
    Ensure the model and tokenizer are compatible with lm_eval.
    This function adds any missing attributes that lm_eval might need.

    Args:
        model: The model to check
        tokenizer: The tokenizer to check

    Returns:
        Tuple of (modified model, modified tokenizer)
    """
    # Ensure model has essential attributes
    if not hasattr(model, "config"):
        print("⚠️  Model missing 'config' attribute")

    if not hasattr(model, "device"):
        if torch.cuda.is_available():
            model = model.cuda()
        elif hasattr(model, "parameters"):
            model = next(model.parameters()).device

    # Ensure tokenizer has all required attributes for lm_eval
    if not hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "right"

    if not hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "right"

    # Ensure special tokens are properly set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set model-specific attributes if missing
    if not hasattr(model, "get_input_embeddings"):
        print("⚠️  Model missing get_input_embeddings method")

    if not hasattr(model, "get_output_embeddings"):
        print("⚠️  Model missing get_output_embeddings method")

    return model, tokenizer


def run_evaluation(
    model,
    tokenizer,
    tasks: List[str],
    limit: Optional[int] = None,
    batch_size: int = 1,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run lm_eval on the specified tasks with checkpointing support.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        tasks: List of task names
        limit: Number of samples to evaluate (None for no limit)
        batch_size: Batch size for evaluation
        checkpoint_path: Optional path to save intermediate results

    Returns:
        Dictionary with evaluation results
    """
    print_section("Running Evaluation")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Limit: {limit if limit else 'No limit'}")
    print(f"Batch size: {batch_size}")
    print()

    # Warning for multi-modal models
    tokenizer_type = type(tokenizer).__name__
    if "processor" in tokenizer_type.lower():
        print(f"⚠️  WARNING: Using multi-modal processor ({tokenizer_type})")
        print(
            "   lm_eval's HFLM wrapper has limited support for vision-language models."
        )
        print("   Some tasks may fail or produce unexpected results.\n")

    # Validate batch_size
    if batch_size < 1:
        print(f"⚠️  Invalid batch_size {batch_size}, using 1")
        batch_size = 1

    # Wrap model with HFLM
    print("🔄 Setting up lm_eval wrapper...")

    # IMPORTANT: Ensure model and tokenizer are compatible with lm_eval
    model, tokenizer = ensure_model_compatibility(model, tokenizer)

    # Get device info
    if hasattr(model, "device"):
        device = model.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"   Model device: {device}")
    print(f"   Tokenizer type: {type(tokenizer).__name__}")

    # Diagnostic: Check tokenizer attributes
    print("   🔍 Checking tokenizer attributes:")
    tokenizer_attrs = {
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token": getattr(tokenizer, "eos_token", "N/A"),
        "eos_token_id": getattr(tokenizer, "eos_token_id", "N/A"),
        "model_input_names": getattr(tokenizer, "model_input_names", "N/A"),
    }
    for attr, value in tokenizer_attrs.items():
        status = "✅" if value is not None and value != "N/A" else "⚠️"
        print(f"      {status} {attr}: {value}")

    # Create HFLM instance with proper parameters
    # Key fix: Handle multi-modal models (like Pixtral) properly
    try:
        print("   📦 Creating HFLM wrapper with model instance...")

        # Check if we're using a multi-modal processor (like PixtralProcessor)
        tokenizer_type = type(tokenizer).__name__
        is_multimodal = "processor" in tokenizer_type.lower()

        if is_multimodal:
            print(f"   ⚠️  Detected multi-modal processor: {tokenizer_type}")
            print("   🔧 Attempting to extract text tokenizer...")

            # For multi-modal processors, try to get the underlying tokenizer
            if hasattr(tokenizer, "tokenizer"):
                text_tokenizer = tokenizer.tokenizer
                print(
                    f"   ✅ Extracted text tokenizer: {type(text_tokenizer).__name__}"
                )
                tokenizer = text_tokenizer
            elif hasattr(tokenizer, "tokenizer") and hasattr(
                tokenizer, "image_processor"
            ):
                # Some processors have both tokenizer and image_processor
                text_tokenizer = tokenizer.tokenizer
                print(
                    f"   ✅ Extracted text tokenizer: {type(text_tokenizer).__name__}"
                )
                tokenizer = text_tokenizer
            else:
                # Try to use the processor directly but ensure it has tokenizer attributes
                print(
                    "   ⚠️  Could not extract separate tokenizer, using processor as-is"
                )

        # Ensure all required attributes are set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not hasattr(tokenizer, "model_input_names"):
            tokenizer.model_input_names = ["input_ids", "attention_mask"]
        if not hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "right"
        if not hasattr(tokenizer, "truncation_side"):
            tokenizer.truncation_side = "right"

        llm = HFLM(
            pretrained=model,  # Pass the actual model instance
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=2048,  # Set a reasonable max length
            trust_remote_code=True,
            add_special_tokens=False,  # Avoid double tokenization
        )
        print("   ✅ HFLM wrapper created successfully!")

    except AssertionError as ae:
        print(f"   ❌ AssertionError in HFLM: {ae}")
        print("   🔧 This is likely due to multi-modal model incompatibility.")
        print("   🔍 Detailed tokenizer state:")
        print(f"      Type: {type(tokenizer)}")
        print(f"      pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        print(f"      eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

        # Try alternative approach: Create HFLM with just tokenizer
        print("   ⚠️  Attempting alternative HFLM initialization...")

        try:
            # Create HFLM with tokenizer only, then manually set model
            llm = HFLM(
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=2048,
                trust_remote_code=True,
                add_special_tokens=False,
            )
            # Manually set the model
            llm.model = model
            llm.device = device
            if hasattr(model, "config"):
                llm.config = model.config
            if hasattr(model, "get_input_embeddings"):
                llm._model_config = model.config
            print("   ✅ Alternative HFLM setup successful!")

        except Exception as e2:
            print(f"❌ Alternative approach also failed: {e2}")
            print("\n" + "=" * 70)
            print("⚠️  CRITICAL: lm_eval may not support your model type (Pixtral)")
            print("   The HFLM wrapper expects standard text-only models.")
            print("\nPossible solutions:")
            print(
                "  1. Use a text-only evaluation framework that supports multi-modal models"
            )
            print("  2. Evaluate only on tasks that don't require image inputs")
            print("  3. Consider using a different evaluation library like 'eval-hub'")
            print("=" * 70)
            raise
    except Exception as e:
        print(f"⚠️  Initial HFLM setup failed: {e}")
        print("   Trying alternative approach...")

        # Alternative: Create HFLM with just tokenizer and set model manually
        try:
            llm = HFLM(
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=2048,
                trust_remote_code=True,
                add_special_tokens=False,
            )
            # Manually set the model and other attributes
            llm.model = model
            llm.device = device
            if hasattr(model, "config"):
                llm.config = model.config
            print("   ✅ Alternative HFLM setup successful!")
        except Exception as e2:
            print(f"❌ Both HFLM approaches failed: {e2}")
            print("\n" + "=" * 70)
            print("⚠️  Your model may not be compatible with lm_eval's HFLM wrapper.")
            print("   Consider using model-specific evaluation scripts.")
            print("=" * 70)
            raise

    print("✅ Wrapper ready!")
    print(f"🚀 Starting evaluation on {len(tasks)} tasks...")
    print()

    # Run evaluation with checkpointing
    try:
        # If limit is set and large, we'll checkpoint periodically
        should_checkpoint = (
            checkpoint_path is not None and limit is not None and limit > 10
        )

        if should_checkpoint:
            print(f"💾 Checkpointing enabled: will save intermediate results")

        results = lm_eval.simple_evaluate(
            model=llm,
            tasks=tasks,
            num_fewshot=0,
            limit=limit,
            log_samples=True,
        )

        # Save checkpoint if enabled
        if should_checkpoint and checkpoint_path:
            checkpoint_file = (
                checkpoint_path
                / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            save_results(results, checkpoint_file)
            print(f"💾 Intermediate checkpoint saved: {checkpoint_file}")

    except Exception as eval_error:
        print(f"❌ Evaluation failed: {eval_error}")

        # Try to save partial results if we have any
        if checkpoint_path:
            try:
                partial_results = {
                    "error": str(eval_error),
                    "timestamp": datetime.now().isoformat(),
                    "tasks_attempted": tasks,
                    "limit": limit,
                }
                error_file = (
                    checkpoint_path
                    / f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(error_file, "w") as f:
                    json.dump(partial_results, f, indent=2)
                print(f"💾 Error log saved: {error_file}")
            except Exception as save_error:
                print(f"⚠️  Could not save error log: {save_error}")

        import traceback

        traceback.print_exc()
        raise

    print("\n✅ Evaluation complete!")
    return results


def save_results(results: Dict[str, Any], output_path: Path):
    """
    Save evaluation results to JSON file.

    Args:
        results: Evaluation results dictionary
        output_path: Path to save results
    """
    print_section("Saving Results")

    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✅ Results saved to: {output_path}")

        # Also print a summary table
        try:
            from lm_eval.utils import make_table

            print("\n📊 Results Summary:")
            print(make_table(results))
        except Exception as e:
            print(f"⚠️  Could not display table: {e}")

    except Exception as e:
        print(f"❌ Error saving results: {e}")


# ============================================================================
# MAIN CLI FLOW
# ============================================================================


def main():
    """Main CLI flow for model evaluation."""
    print_header("MODEL EVALUATION TOOL")
    print("This tool evaluates trained models on various benchmarks.\n")

    # Step 1: Select model directory
    print_section("Step 1: Select Model Directory")
    models = get_available_models()

    if not models:
        print("❌ No model directories found in", CHECKPOINTS_DIR)
        print("Please ensure you have trained models in the checkpoints directory.")
        sys.exit(1)

    print("Available model directories:")
    for i, model in enumerate(models):
        print(f"  {i + 1}. {model.name}")
    print()

    # Get model selection
    choice = get_user_selection(
        "Select model directory (enter number): ", 1, len(models)
    )
    selected_model = models[choice - 1]
    print(f"✅ Selected: {selected_model.name}")

    # Step 2: Select checkpoint
    print_section("Step 2: Select Checkpoint")
    checkpoints = get_checkpoints(selected_model)

    if not checkpoints:
        print(f"⚠️  No checkpoints found in {selected_model}")
        print("You can use the base model directory directly.")
        use_base = input("Use base model directory? (y/n): ").strip().lower()
        if use_base == "y":
            checkpoint_path = selected_model
            print(f"✅ Using base model: {checkpoint_path.name}")
        else:
            print("❌ Cannot proceed without a checkpoint.")
            sys.exit(1)
    else:
        print("Available checkpoints (newest first):")
        display_count = min(len(checkpoints), 20)
        for i in range(display_count):
            print(f"  {i + 1}. {checkpoints[i].name}")

        if len(checkpoints) > 20:
            print(f"  ... and {len(checkpoints) - 20} more")
        print()

        choice = get_user_selection(
            "Select checkpoint (enter number): ", 1, len(checkpoints)
        )
        checkpoint_path = checkpoints[choice - 1]
        print(f"✅ Selected: {checkpoint_path.name}")

    # Step 3: Select tasks
    print_section("Step 3: Select Evaluation Tasks")
    print("Available tasks:")
    for i, (task_id, task_name) in enumerate(DEFAULT_TASKS):
        print(f"  {i + 1}. {task_name} ({task_id})")
    print()

    print("Enter task numbers separated by commas (e.g., '1,3,5')")
    print("Or press Enter to select all tasks")

    try:
        user_input = input("\nEnter task selection: ").strip()

        if not user_input:
            selected_tasks = [task_id for task_id, _ in DEFAULT_TASKS]
            print(f"✅ Selected all {len(selected_tasks)} tasks")
        else:
            indices = []
            for part in user_input.split(","):
                try:
                    idx = int(part.strip()) - 1
                    if 0 <= idx < len(DEFAULT_TASKS):
                        indices.append(idx)
                    else:
                        print(f"⚠️  Invalid index: {part.strip()}")
                except ValueError:
                    print(f"⚠️  Invalid number: {part.strip()}")

            if not indices:
                print("⚠️  No valid tasks selected, using all tasks")
                selected_tasks = [task_id for task_id, _ in DEFAULT_TASKS]
            else:
                selected_tasks = [DEFAULT_TASKS[i][0] for i in sorted(set(indices))]
                print(
                    f"✅ Selected {len(selected_tasks)} tasks: {', '.join(selected_tasks)}"
                )

    except KeyboardInterrupt:
        print("\n\n❌ Cancelled!")
        sys.exit(0)

    # Step 4: Set limit
    print_section("Step 4: Set Evaluation Limit")
    print("The limit determines how many samples to evaluate per task.")
    print("Higher values give more accurate results but take longer.")
    print()

    while True:
        try:
            user_input = input("Enter a number for 'limit' (0 for no limit): ").strip()
            limit = int(user_input)

            if limit < 0:
                print("❌ Please enter a non-negative number.")
                continue

            if limit == 0:
                print("✅ No limit set - will evaluate all available samples")
            else:
                print(f"✅ Limit set to {limit} samples per task")

            break

        except ValueError:
            print("❌ Please enter a valid number.")

    # Confirmation
    print_section("Confirmation")
    print("Summary of your selection:")
    print(f"  Model:      {selected_model.name}")
    print(f"  Checkpoint: {checkpoint_path.name}")
    print(f"  Tasks:      {', '.join(selected_tasks)}")
    print(f"  Limit:      {limit if limit else 'No limit'}")
    print()

    confirm = input("Proceed with evaluation? (y/n): ").strip().lower()
    if confirm != "y":
        print("❌ Evaluation cancelled.")
        sys.exit(0)

    # Load model
    print("\n⏳ Loading model...")
    model, tokenizer = load_model(checkpoint_path)

    # Set up checkpoint directory for intermediate saves
    checkpoint_dir = Path(__file__).parent.parent / "logs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation with checkpointing support
    results = run_evaluation(
        model, tokenizer, selected_tasks, limit, checkpoint_path=checkpoint_dir
    )

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = (
        f"{selected_model.name}_{checkpoint_path.name}_eval_{timestamp}.json"
    )
    output_path = Path(__file__).parent.parent / "logs" / output_filename

    # Ensure logs directory exists
    output_path.parent.mkdir(exist_ok=True)

    save_results(results, output_path)

    # Final summary
    print_header("EVALUATION COMPLETE")
    print(f"Results saved to: {output_path}")
    print("\nYou can now:")
    print("  1. Compare results across different checkpoints")
    print("  2. Analyze performance on specific tasks")
    print("  3. Use these results to guide further training")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Evaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
