# Bug Fix: Mistral Tokenizer InvalidMessageStructureException

## Problem Summary

When training with Ministral-3-14B-Base on NVIDIA DGX Spark, the training failed with:

```
mistral_common.exceptions.InvalidMessageStructureException: 
Expected last role User or Tool (or Assistant with prefix or continue_final_message set to True) 
for serving but got assistant
```

And later:
```
Expected last role Assistant with prefix False for serving with continue_final_message set to True but got user
```

## Root Cause

The error occurred in `Automodel/nemo_automodel/components/datasets/llm/formatting_utils.py` during the dataset packing phase. The code has a fallback mechanism for tokenizers (like Mistral) that don't support the `return_assistant_tokens_mask` parameter:

1. **First attempt**: Try with `return_assistant_tokens_mask` → fails for Mistral
2. **Fallback #1**: Retry without it, adding `continue_final_message=True` when last message is from assistant
3. **Fallback #2** (for loss mask computation): When computing prompt-only version, the code:
   - Pops the assistant message from the conversation
   - Now the last message is "user"
   - But it was passing `continue_final_message=True` (from the original conversation) → **CRASH**

### Two Bugs in the Fallback Path

**Bug #1**: The second tokenizer call (for prompt-only) was missing `continue_final_message` entirely.

**Bug #2**: Even after adding it, the code was using the **original** `continue_final` value (based on the full conversation with assistant), but after popping the assistant message, the last role is now "user", which requires `continue_final_message=False`.

## The Fixes

### Fix #1: Add `continue_final_message` to the prompt-only call

```python
# Before:
tokenized_prompt = tokenizer.apply_chat_template(
    answer_text,
    tools=tools,
    tokenize=True,
    return_dict=True,
    # Missing: continue_final_message parameter
    padding=False,
    truncation=False,
    max_length=seq_length,
)
```

### Fix #2: Check the ACTUAL last role in the prompt-only version

```python
# After:
# When computing prompt-only, the last message is NOT assistant,
# so continue_final_message should be False regardless of original
prompt_continue_final = answer_text[-1].get("role", "user") == "assistant"
tokenized_prompt = tokenizer.apply_chat_template(
    answer_text,
    tools=tools,
    tokenize=True,
    return_dict=True,
    continue_final_message=prompt_continue_final,  # Check actual last role
    padding=False,
    truncation=False,
    max_length=seq_length,
)
```

## Why This Happened on DGX Spark vs RTX 4090

The error is **not hardware-specific**. It's a bug in the NeMo Automodel code that:
- Only triggers when using Mistral tokenizers (Ministral models)
- Only triggers when `answer_only_loss_mask=True` (which is default for SFT)
- Only triggers during dataset packing (which you enabled with `USE_PACKING = True`)

Your previous RTX 4090 setup likely:
- Used a different model (not Mistral-based)
- Had packing disabled
- Used a different version of NeMo Automodel that didn't have this code path

## Files Modified

- `Automodel/nemo_automodel/components/datasets/llm/formatting_utils.py` (line 424)

## Next Steps

1. **Re-run your training script** - The fix should resolve the error
2. **If you encounter other issues**, they're likely unrelated to this specific bug
3. **Consider reporting this bug** to NVIDIA-NeMo/Automodel on GitHub so it can be fixed upstream

## Testing the Fix

You can test with a dry-run first to verify the dataset loads correctly:

```bash
python scripts/sft_curriculum_trainer_clean.py --stage 1 --dry-run
```

If that succeeds without the `InvalidMessageStructureException`, you're good to proceed with actual training:

```bash
python scripts/sft_curriculum_trainer_clean.py --stage 1
```

## Why This Happened on DGX Spark vs RTX 4090

The error is **not hardware-specific**. It's a bug in the NeMo Automodel code that:
- Only triggers when using Mistral tokenizers (Ministral models)
- Only triggers when `answer_only_loss_mask=True` (which is default for SFT)
- Only triggers during dataset packing (which you enabled with `USE_PACKING = True`)

Your previous RTX 4090 setup likely:
- Used a different model (not Mistral-based)
- Had packing disabled
- Used a different version of NeMo Automodel that didn't have this code path

---

**Note**: This is a patch to the NeMo Automodel code in your local `Automodel/` directory. When you update Automodel in the future, you may need to re-apply this fix unless NVIDIA has fixed it upstream.
