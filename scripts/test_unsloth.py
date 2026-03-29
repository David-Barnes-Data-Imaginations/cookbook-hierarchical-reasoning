"""
Quick test to verify Unsloth is working correctly on DGX Spark
"""

import torch
from unsloth import FastLanguageModel

print("=" * 60)
print("  UNSLOTH DGX SPARK VERIFICATION TEST")
print("=" * 60)

# Check GPU
print(f"\n🔍 GPU Information:")
print(f"   GPU Count: {torch.cuda.device_count()}")
print(f"   Current GPU: {torch.cuda.get_device_name(0)}")
print(
    f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
)
print(f"   BF16 Support: {torch.cuda.is_bf16_supported()}")

# Test model loading (small test)
print("\n🧪 Testing model loading...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Ministral-3-14B-Base-2512-bnb-4bit",
        max_seq_length=2048,  # Small for test
        load_in_4bit=True,
    )
    print("   ✅ Model loaded successfully!")

    # Test LoRA setup
    print("\n🧪 Testing LoRA configuration...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )
    print("   ✅ LoRA configured successfully!")

    print("\n" + "=" * 60)
    print("  ✅ ALL TESTS PASSED - Unsloth is ready for training!")
    print("=" * 60)
    print("\nYou can now run: python train_unsloth.py")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTroubleshooting:")
    print("  1. Check if Unsloth is installed: pip list | grep unsloth")
    print("  2. Check GPU availability: nvidia-smi")
    print("  3. Check HF token: export HF_TOKEN='your_token'")
