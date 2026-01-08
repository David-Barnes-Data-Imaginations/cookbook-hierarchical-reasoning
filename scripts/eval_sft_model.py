"""
Quick Evaluation Script for SFT-trained Model
==============================================
Merges the Stage 2 adapters and runs lm_eval benchmarks.

Usage:
    python scripts/eval_sft_model.py
    python scripts/eval_sft_model.py --skip-merge  # If already merged
"""

import os
import json
import argparse
from pathlib import Path

# Stability settings (same as training)
import torch
torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"

def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("🧹 GPU memory cleared")


def merge_model():
    """Merge Stage 2 adapters into a single model for evaluation."""
    from unsloth import FastLanguageModel
    
    checkpoint = "qwen3-14b-sft-stage2"
    output_path = "qwen3-14b-sft-merged"
    
    if os.path.exists(output_path):
        print(f"✅ Merged model already exists at {output_path}")
        return output_path
    
    print(f"⏳ Loading checkpoint from {checkpoint}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=8192,
        load_in_4bit=True,
    )
    
    print(f"🔗 Merging adapters and saving to {output_path}...")
    model.save_pretrained_merged(
        output_path,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"✅ Merged model saved!")
    
    # CRITICAL: Unload the model before eval loads another one
    print("🧹 Unloading merge model to free VRAM...")
    del model
    del tokenizer
    clear_gpu_memory()
    
    return output_path


def run_eval(model_path: str, limit: int = 50):
    """Run lm_eval benchmarks on the merged model."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table
    
    print(f"\n⏳ Loading model for evaluation from {model_path}...")
    print("   (Loading in 4-bit to fit in 24GB VRAM)")
    
    # Load in 4-bit quantization - 16-bit 14B model won't fit in 24GB!
    llm = HFLM(
        pretrained=model_path,
        batch_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        # 4-bit quantization to fit in VRAM
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    # Quick eval tasks
    task_list = [
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "piqa",
        "gsm8k",
    ]
    
    print(f"\n🚀 Running evaluation on: {task_list}")
    print(f"   Limit: {limit} samples per task (for quick testing)")
    
    results = lm_eval.simple_evaluate(
        model=llm,
        tasks=task_list,
        num_fewshot=0,
        limit=limit,
        log_samples=True,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(make_table(results))
    
    # Save to JSON
    output_file = "qwen3-14b-sft-eval-results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT-trained model")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merging (if already done)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of samples per task (default: 50)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Custom model path (skip merge)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  QWEN3-14B SFT MODEL EVALUATION")
    print("=" * 60)
    
    if args.model_path:
        model_path = args.model_path
    elif args.skip_merge:
        model_path = "qwen3-14b-sft-merged"
    else:
        model_path = merge_model()
    
    run_eval(model_path, limit=args.limit)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
