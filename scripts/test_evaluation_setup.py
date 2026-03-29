#!/usr/bin/env python3
"""
Test script to verify evaluation script dependencies and setup
"""

import sys
from pathlib import Path


def test_imports():
    """Test if all required packages are installed."""
    print("=" * 70)
    print("  TESTING EVALUATION SCRIPT DEPENDENCIES")
    print("=" * 70)
    print()

    tests = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("lm_eval", "LM Evaluation Harness"),
        ("accelerate", "Accelerate"),
        ("peft", "PEFT (LoRA support)"),
    ]

    missing = []

    for package, friendly_name in tests:
        try:
            __import__(package)
            print(f"✅ {friendly_name:30s} - Installed")
        except ImportError:
            print(f"❌ {friendly_name:30s} - NOT INSTALLED")
            missing.append(package)

    print()

    if missing:
        print("⚠️  Missing packages detected!")
        print("\nTo install missing packages, run:")
        print("  pip install lm_eval transformers torch accelerate peft")
        print("\nOr for specific versions:")
        print(
            "  pip install lm_eval==0.4.2 transformers==4.40.0 torch==2.2.0 peft==0.11.0"
        )
        return False
    else:
        print("✅ All dependencies are installed!")
        return True


def test_checkpoint_structure():
    """Test if checkpoint structure is correct."""
    print()
    print("=" * 70)
    print("  TESTING CHECKPOINT STRUCTURE")
    print("=" * 70)
    print()

    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"

    if not checkpoints_dir.exists():
        print(f"❌ Checkpoints directory not found: {checkpoints_dir}")
        return False

    print(f"✅ Checkpoints directory found: {checkpoints_dir}")
    print()

    models = list(checkpoints_dir.iterdir())

    if not models:
        print("⚠️  No model directories found in checkpoints/")
        print("   Make sure you have trained models in this directory.")
        return False

    print(f"Found {len(models)} model directory/ies:")
    for model_dir in models:
        if model_dir.is_dir():
            checkpoints = [
                d for d in model_dir.iterdir() if d.name.startswith("checkpoint-")
            ]
            print(f"  - {model_dir.name}: {len(checkpoints)} checkpoint(s)")

    print()
    print("✅ Checkpoint structure is valid!")
    return True


def test_logs_directory():
    """Test if logs directory exists or can be created."""
    print()
    print("=" * 70)
    print("  TESTING LOGS DIRECTORY")
    print("=" * 70)
    print()

    logs_dir = Path(__file__).parent.parent / "logs"

    if not logs_dir.exists():
        print(f"📁 Creating logs directory: {logs_dir}")
        logs_dir.mkdir(exist_ok=True)

    print(f"✅ Logs directory ready: {logs_dir}")
    return True


def main():
    """Run all tests."""
    print()
    print("🧪 Running evaluation script setup tests...\n")

    results = []

    # Test 1: Imports
    results.append(("Dependencies", test_imports()))

    # Test 2: Checkpoint structure
    results.append(("Checkpoint Structure", test_checkpoint_structure()))

    # Test 3: Logs directory
    results.append(("Logs Directory", test_logs_directory()))

    # Summary
    print()
    print("=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    print()

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("🎉 All tests passed! You're ready to run evaluations.")
        print("\nRun evaluation with:")
        print("  python scripts/evaluate_model.py")
        print()
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print(
            "  - Install missing packages: pip install lm_eval transformers torch accelerate"
        )
        print("  - Add trained models to checkpoints/ directory")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
