# Sample Tracking System

## How It Works

The Unsloth training script now includes **automatic sample tracking** to prevent reusing the same questions in RL training.

## What Gets Tracked

When you run `train_unsloth.py`:
1. Each sample from your `stage1_data.jsonl` is assigned an **index** (0, 1, 2, 3, ...)
2. These indices are saved to `sft_used_samples.json`
3. The tracking file records:
   - Which stage (stage_1, stage_2)
   - Which source (e.g., "ministral_training_data")
   - **List of indices used**

## Example Tracking File

```json
{
  "stage_1": {
    "ministral_training_data": [45, 123, 567, 890, 1234, ..., 99999]
  },
  "stage_2": {
    "ministral_training_data": [1000, 2000, 3000, ..., 95000]
  }
}
```

## How to Use

### 1. After SFT Training

```bash
# Check what was used
python check_used_samples.py
```

This will show:
- Total samples used per stage
- First and last few indices
- **Auto-generates** `rl_exclude_indices.txt`

### 2. In RL Training

Load the exclusion list:

```python
# Load excluded indices
with open("rl_exclude_indices.txt", "r") as f:
    exclude_indices = set(int(line.strip()) for line in f if line.strip())

# Filter your RL dataset
filtered_dataset = rl_dataset.filter(
    lambda example, idx: idx not in exclude_indices,
    with_indices=True
)
```

## Important Notes

### Index Matching

The tracking uses **indices from your JSONL file**. This works when:

✅ **Same dataset source**: If RL uses the same Nemotron dataset as SFT
- The indices directly match
- Just filter by index

⚠️ **Different dataset source**: If RL uses a different dataset
- You need to match by unique ID or content
- Consider adding `id` fields to both datasets

### Multiple Stages

If you run both Stage 1 and Stage 2:
- Both are tracked separately
- `check_used_samples.py` combines them
- `rl_exclude_indices.txt` contains **all unique indices**

### Re-running Training

If you re-run SFT on the same data:
- The script checks `sft_used_samples.json`
- **Skips previously used samples** (if you enable filtering)
- Updates the tracking file

## Quick Commands

```bash
# Check what was used
python check_used_samples.py

# Check specific stage
python check_used_samples.py --stage stage_1

# View tracking file directly
cat sft_used_samples.json

# View exclusion list
head rl_exclude_indices.txt
```

## File Locations

| File | Purpose |
|------|---------|
| `sft_used_samples.json` | Master tracking file (JSON format) |
| `rl_exclude_indices.txt` | Simple list for RL filtering (one index per line) |
| `check_used_samples.py` | Utility to view tracking |

## Best Practices

1. **Keep the tracking file**: Don't delete `sft_used_samples.json`
2. **Check before RL**: Always run `check_used_samples.py` before RL training
3. **Verify no overlap**: Test a small sample to ensure no duplicates
4. **Document**: Note which dataset splits were used for SFT

## Troubleshooting

### "File not found"

- Training hasn't run yet
- Run `python train_unsloth.py` first

### "Index mismatch"

- Your RL dataset uses different indexing
- Solution: Use unique IDs instead of positional indices

### "Too few samples left"

- You used most data in SFT
- Get more RL data or reduce SFT size

---

**This system ensures your RL training learns from fresh data, not just SFT memorization!** 🎯
