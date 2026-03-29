#!/usr/bin/env python3
"""
Utility to check which samples have been used for SFT training.
Helps ensure you don't reuse the same samples in RL training.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
TRACKING_FILE = PROJECT_ROOT / "sft_used_samples.json"


def check_used_samples():
    """Display summary of used samples."""
    print("=" * 70)
    print("  SFT SAMPLE TRACKING SUMMARY")
    print("=" * 70)

    if not TRACKING_FILE.exists():
        print("\n❌ No tracking file found at", TRACKING_FILE)
        print("   This means either:")
        print("   - Training hasn't run yet")
        print("   - Tracking wasn't enabled")
        print("\n   Run training first to generate the tracking file.")
        return

    with open(TRACKING_FILE, "r") as f:
        tracking = json.load(f)

    total_used = 0

    for stage, sources in tracking.items():
        print(f"\n📋 {stage.upper()}")
        print("-" * 70)

        for source, indices in sources.items():
            count = len(indices)
            total_used += count
            print(f"   {source}:")
            print(f"      Total samples used: {count}")

            # Show first and last few indices
            if count > 0:
                first_few = indices[:5]
                last_few = indices[-5:] if count > 5 else []

                print(f"      First 5 indices: {first_few}")
                if last_few:
                    print(f"      Last 5 indices: {last_few}")

                # Show sample distribution
                if count > 10:
                    print(f"      ... and {count - 10} more samples")

    print("\n" + "=" * 70)
    print(f"  TOTAL UNIQUE SAMPLES USED: {total_used}")
    print("=" * 70)

    # Generate list for RL exclusion
    all_indices = []
    for stage, sources in tracking.items():
        for source, indices in sources.items():
            all_indices.extend(indices)

    all_indices = sorted(set(all_indices))  # Remove duplicates

    print(f"\n📝 For RL training, use these indices to EXCLUDE:")
    print(f"   Total to exclude: {len(all_indices)}")

    # Save to a file for easy use in RL training
    exclude_file = PROJECT_ROOT / "rl_exclude_indices.txt"
    with open(exclude_file, "w") as f:
        for idx in all_indices:
            f.write(f"{idx}\n")

    print(f"   Saved to: {exclude_file}")
    print(f"   You can load this in your RL script to filter out used samples.")

    print("\n" + "=" * 70)


def check_specific_stage(stage: str):
    """Check samples used in a specific stage."""
    if not TRACKING_FILE.exists():
        print(f"No tracking file found for {stage}")
        return

    with open(TRACKING_FILE, "r") as f:
        tracking = json.load(f)

    if stage not in tracking:
        print(f"No data found for {stage}")
        return

    print(f"\n📋 {stage.upper()} DETAILS")
    print("-" * 70)

    total = 0
    for source, indices in tracking[stage].items():
        count = len(indices)
        total += count
        print(f"   {source}: {count} samples")
        if count <= 20:
            print(f"      Indices: {indices}")
        else:
            print(f"      First 10: {indices[:10]}")
            print(f"      Last 10: {indices[-10:]}")

    print(f"\n   Total for {stage}: {total}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check used SFT samples")
    parser.add_argument(
        "--stage", type=str, help="Check specific stage (e.g., 'stage_1')"
    )
    args = parser.parse_args()

    if args.stage:
        check_specific_stage(args.stage)
    else:
        check_used_samples()
