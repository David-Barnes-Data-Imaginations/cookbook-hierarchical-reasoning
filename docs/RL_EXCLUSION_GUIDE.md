# RL Training: Excluding SFT Samples

## Overview

To prevent data leakage between SFT and RL training, you need to exclude the samples used in SFT from your RL dataset. This guide shows how to do that.

## Step 1: Check Which Samples Were Used

After SFT training completes, run:

```bash
python check_used_samples.py
```

This will show:
- Which samples were used in Stage 1 and Stage 2
- Total count of used samples
- A file `rl_exclude_indices.txt` with all indices to exclude

Example output:
```
======================================================================
  SFT SAMPLE TRACKING SUMMARY
======================================================================

📋 STAGE_1
----------------------------------------------------------------------
   ministral_training_data:
      Total samples used: 3000
      First 5 indices: [45, 123, 567, 890, 1234]
      Last 5 indices: [98765, 98890, 99001, 99567, 99999]

======================================================================
  TOTAL UNIQUE SAMPLES USED: 3000
======================================================================

📝 For RL training, use these indices to EXCLUDE:
   Total to exclude: 3000
   Saved to: /workspace/rl_exclude_indices.txt
```

## Step 2: Load and Filter in RL Training

### Option A: Using HuggingFace Datasets

```python
from datasets import load_dataset

# Load your RL dataset
rl_dataset = load_dataset("your-rl-dataset", split="train")

# Load excluded indices
with open("rl_exclude_indices.txt", "r") as f:
    exclude_indices = set(int(line.strip()) for line in f if line.strip())

# Filter out excluded samples
# Note: This assumes your dataset has an index column matching the SFT indices
# If indices don't match directly, you may need to use a unique ID field

filtered_dataset = rl_dataset.filter(
    lambda example, idx: idx not in exclude_indices,
    with_indices=True
)

print(f"Original: {len(rl_dataset)}, After filtering: {len(filtered_dataset)}")
```

### Option B: Using a Unique ID Field

If your dataset has unique IDs (like question_id or sample_id):

```python
# Load excluded indices
with open("rl_exclude_indices.txt", "r") as f:
    exclude_ids = set(int(line.strip()) for line in f if line.strip())

# Filter by ID field
filtered_dataset = rl_dataset.filter(
    lambda example: example.get("id") not in exclude_ids
)
```

### Option C: Direct Index Filtering (Fastest)

```python
# Get all indices to keep
all_indices = set(range(len(rl_dataset)))
with open("rl_exclude_indices.txt", "r") as f:
    exclude_indices = set(int(line.strip()) for line in f if line.strip())

indices_to_keep = list(all_indices - exclude_indices)

# Select only the indices to keep
filtered_dataset = rl_dataset.select(indices_to_keep)
```

## Step 3: Verify No Overlap

```python
# Double-check no overlap
used_ids = set()
with open("rl_exclude_indices.txt", "r") as f:
    used_ids = set(int(line.strip()) for line in f if line.strip())

# Sample some RL data to verify
sample = filtered_dataset.shuffle(seed=42).select(range(100))
sample_ids = [example.get("id") for example in sample if "id" in example]

overlap = used_ids.intersection(set(sample_ids))
if overlap:
    print(f"⚠️  WARNING: Found {len(overlap)} overlapping samples!")
else:
    print("✅ No overlap detected - good to go!")
```

## Important Notes

### Index Matching

The tracking system uses **indices from your JSONL file** (`stage1_data.jsonl`). When preparing your RL dataset:

1. **If using the same source dataset** (e.g., Nemotron splits):
   - Make sure indices match the original dataset indices
   - Use the `parallel_id` field if available to match samples

2. **If using a different dataset**:
   - You'll need to match by content or unique IDs
   - Consider adding a unique ID field to both datasets

### Multi-Stage Tracking

The tracking file supports multiple stages:
- `stage_1`: Samples used in first SFT stage
- `stage_2`: Samples used in second SFT stage

Both are automatically combined when generating the exclusion list.

### Manual Override

If you need to manually specify indices to exclude:

```python
# Create custom exclusion file
with open("rl_exclude_indices.txt", "w") as f:
    # Add your indices here
    f.write("123\n")
    f.write("456\n")
    f.write("789\n")
```

## Best Practices

1. **Always check before RL training**: Run `python check_used_samples.py` to see what was used
2. **Keep the tracking file**: `sft_used_samples.json` is your master record
3. **Document your data sources**: Note which dataset splits were used for SFT
4. **Test with a small sample**: Verify no overlap before full RL training

## Example Workflow

```bash
# 1. Run SFT training
python train_unsloth.py

# 2. Check what was used
python check_used_samples.py

# 3. Prepare RL dataset with exclusion
python prepare_rl_dataset.py  # Your script that filters data

# 4. Verify no overlap
python verify_no_overlap.py  # Your verification script

# 5. Start RL training
python train_rl.py
```

## Troubleshooting

### "No tracking file found"

- SFT training hasn't run yet
- The tracking system wasn't enabled in your SFT script
- Check that `sft_used_samples.json` exists in your project root

### "Index mismatch"

- Your RL dataset uses different indexing than SFT
- Solution: Use unique IDs instead of indices, or map indices between datasets

### "Too few samples left after filtering"

- You may have used most of your data in SFT
- Consider:
  - Getting more data for RL
  - Reducing SFT sample size
  - Using data augmentation for RL

---

**Remember**: The goal is to ensure your RL model learns from **new** experiences, not just memorize SFT data! 🎯
