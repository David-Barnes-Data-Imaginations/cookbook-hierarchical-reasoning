# Memory Optimization: Removed Parallel Responses

## Problem

The PC crash during training was likely caused by the **parallel response strategy** which doubled memory usage:
- Each sample was duplicated into 2 examples (thinking + direct)
- This effectively doubled the KV cache requirements during training
- Even with 128GB unified memory on DGX Spark, this could cause OOM crashes

## Solution

Removed the parallel response strategy and simplified to **single high-quality examples**:

### Changes Made

1. **`format_ministral_for_sft()` function**:
   - Removed `mode` parameter (no longer supports "direct" mode)
   - Always generates thinking format with <think>...</think> tags
   - Preserves DeepSeek's original reasoning chains

2. **`load_stage_1_data()` function**:
   - Removed parallel response logic
   - Each sample now creates only **1 example** (not 2)
   - Updated sample counts to use full targets (no halving)
   - Simplified progress tracking and logging

3. **`load_stage_2_data()` function**:
   - Fixed progress bar total (removed `* 2` multiplier)
   - Already was single-example mode, just needed cleanup

4. **Configuration**:
   - Updated comments to reflect single-example approach
   - Removed references to parallel responses throughout

## Benefits

1. **Halved memory usage** during training
2. **Simpler data pipeline** - no complex parallel logic
3. **Higher quality data** - focuses on rich reasoning chains
4. **Faster training** - fewer examples to process
5. **More stable** - less likely to cause OOM crashes

## Trade-offs

- **Lost**: Direct/no_think mode examples
- **Kept**: All high-quality thinking/reasoning examples

The reasoning-focused approach is actually **more aligned** with your goal of building advanced reasoning capabilities. The direct examples were redundant since the model can learn to be concise from the thinking examples anyway.

## New Data Format

Each example now looks like:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an advanced reasoning assistant. When asked to think through a problem..."
    },
    {
      "role": "user",
      "content": "/think [original question]"
    },
    {
      "role": "assistant",
      "content": "<think>\n[DeepSeek's reasoning chain]\n</think>\n<answer>\n[Final answer]\n</answer>"
    }
  ],
  "mode": "thinking",
  "source": "ministral_training_data"
}
```

## Testing

Run with the same commands as before:

```bash
# Dry run to verify
python scripts/sft_curriculum_trainer_clean.py --stage 1 --dry-run

# Actual training
python scripts/sft_curriculum_trainer_clean.py --stage 1
```

The dataset size will be **half** of what it was before (3000 examples instead of 6000), but each example is high-quality reasoning data.

## Future Considerations

If you want direct/no_think examples later, you can:
1. Add them back as a separate dataset
2. Train a separate model on direct responses
3. Use prompt engineering to get concise answers from the thinking model

But for now, focusing on **quality reasoning chains** is the better approach for your advanced reasoning capabilities goal.
