# Summary of Changes: Clean Ministral SFT Setup

## Overview
Removed all Nemotron/Mamba-specific patches and created a clean training setup for Ministral-3-14B on DGX Spark.

## Files Modified

### 1. `scripts/sft_curriculum_trainer_clean.py` (NEW)
**Created as a clean version** of the training script with all Nemotron-specific code removed.

**Changes:**
- ✅ **Removed Mamba environment variable**: Deleted `MAMBA_SSM_FORCE_TRITON=1` (line 98-100)
- ✅ **Renamed function**: `format_nemotron_for_sft` → `format_ministral_for_sft`
- ✅ **Updated source label**: Changed from `"nemotron_math"` to `"ministral_training_data"`
- ✅ **Cleaned YAML generation**:
  - Removed all NemotronH-specific comments and logic
  - Removed `force_hf=False` override (not needed for Ministral)
  - Removed custom model type override (`model_type = "ministral"`)
  - Removed debug grep commands
  - Changed dataset target from `jsonl_messages.make_jsonl_messages_dataset` to `nemo_automodel.components.datasets.llm.chat.make_chat_dataset`
  - Updated YAML output path from `nemotron/nemotron_sft_stage{stage}.yaml` to `mistral/ministral_sft_stage{stage}.yaml`
- ✅ **Updated model IDs**:
  - Stage 1: `mistralai/Mistral-Nemo-Base-2407` → `mistralai/Ministral-3-14B-Base`
  - Stage 2 checkpoint: `mistral-nemo-sft-stage1.nemo` → `ministral-sft-stage1`
- ✅ **Updated output directories**: `mistral-nemo-sft-stage{stage}` → `ministral-sft-stage{stage}`
- ✅ **Updated docstring**: Reflects clean Ministral setup

**Kept intact:**
- ✅ Two-stage curriculum training logic
- ✅ Parallel response generation (/think and /no_think modes)
- ✅ Sample tracking for RL exclusion
- ✅ DDP strategy configuration
- ✅ QLoRA configuration
- ✅ All data loading and formatting logic

### 2. `patched_finetune.py`
**Removed Nemotron-specific Patch 3** while keeping essential BNB/DDP fixes.

**Changes:**
- ❌ **Removed entire Patch 3 section** (lines 141-190):
  - `_patched_resolve_custom_model_cls_for_config` function
  - All NemotronH model resolver logic
  - Comments about forcing Automodel's NemotronH implementation
- ✅ **Kept Patch 1**: DDPManager.parallelize patch for BNB quantization
- ✅ **Kept Patch 2**: infrastructure.py model.module patch for single-GPU
- ✅ **Updated docstring**: Notes that this is a clean version without Nemotron patches

**Why this is safe:**
- Ministral uses standard Mistral architecture (not NemotronH)
- No custom model resolver needed - HF handles it natively
- Only essential patches remain: BNB dtype handling and single-GPU compatibility

### 3. `README.md`
**Completely rewritten** to reflect clean Ministral setup.

**Changes:**
- ✅ **Updated title**: "Clean Ministral Version"
- ✅ **Updated model**: `mistralai/Ministral-3-14B-Base`
- ✅ **Script name**: `sft_curriculum_trainer_clean.py`
- ✅ **Removed Mamba references**: No need for `MAMBA_SSM_FORCE_TRITON`
- ✅ **Updated output paths**: `ministral-sft-stage{stage}/` (HF format, not `.nemo`)
- ✅ **Added "What's Different" section**: Explains the clean-up
- ✅ **Removed custom dataset loader note**: Uses NeMo's built-in ChatDataset
- ✅ **Updated troubleshooting**: Specific to Ministral, not Nemotron

**Kept intact:**
- ✅ Docker setup instructions
- ✅ uv installation steps
- ✅ BitsAndBytes installation
- ✅ Environment variable setup
- ✅ Training commands

## What Was Removed (and Why It's Safe)

### 1. Mamba-Specific Environment Variables
```python
# REMOVED:
os.environ["MAMBA_SSM_FORCE_TRITON"] = "1"
```
**Why safe**: Ministral is pure transformer architecture - no Mamba kernels needed.

### 2. NemotronH Model Resolver Patch
```python
# REMOVED from patched_finetune.py:
def _patched_resolve_custom_model_cls_for_config(config):
    # ... all NemotronH forcing logic
```
**Why safe**: Ministral uses standard Mistral architecture that HF handles natively. No custom resolver needed.

### 3. Custom Dataset Loader Reference
```python
# CHANGED from:
_target_: jsonl_messages.make_jsonl_messages_dataset

# TO:
_target_: nemo_automodel.components.datasets.llm.chat.make_chat_dataset
```
**Why safe**: NeMo's built-in ChatDataset handles JSONL format perfectly. No custom loader needed.

### 4. force_hf Override
```python
# REMOVED:
cfg["model"]["force_hf"] = False
```
**Why safe**: Ministral is a standard HF model - no need to force custom implementation.

### 5. model_type Override
```python
# REMOVED:
cfg["model"]["model_type"] = "ministral"
```
**Why safe**: HF auto-detects Ministral architecture from config.json.

## What Was Preserved

### Critical Training Logic
- ✅ Two-stage curriculum (Stage 1: foundation, Stage 2: reasoning)
- ✅ Parallel response generation (/think and /no_think modes)
- ✅ Sample tracking for RL exclusion (prevents data contamination)
- ✅ Data loading from Nemotron-Post-Training-Dataset-v1
- ✅ Category ratios (50% math, 30% code, 20% STEM for Stage 1)
- ✅ Context length handling (128K)

### Essential Patches
- ✅ DDPManager.parallelize patch (BNB quantization compatibility)
- ✅ infrastructure.py model.module patch (single-GPU compatibility)
- ✅ DDP strategy (not FSDP2) for single-GPU Blackwell

### Hardware Optimizations
- ✅ 128K context length support
- ✅ QLoRA 4-bit quantization
- ✅ Batch size 2 with gradient accumulation 4
- ✅ 16 CPU threads, 8 data workers

## Testing Checklist

Before running full training, verify:

1. **Dry run**: 
   ```bash
   python scripts/sft_curriculum_trainer_clean.py --stage 1 --dry-run
   ```
   Should load data without errors.

2. **Model loading**:
   - Verify Ministral-3-14B-Base downloads successfully
   - Check no NemotronH-related errors in logs

3. **Minimal training**:
   ```bash
   # Run just 2-3 steps to verify setup
   # (Ctrl+C after seeing training start)
   python scripts/sft_curriculum_trainer_clean.py --stage 1
   ```

4. **Full training**:
   - Stage 1 should complete and save to `ministral-sft-stage1/`
   - Stage 2 should load from Stage 1 checkpoint
   - No Mamba/Nemotron-related errors

## Expected Output

### Checkpoint Locations
- Stage 1: `/workspace/ministral-sft-stage1/`
- Stage 2: `/workspace/ministral-sft-stage2/`

### Checkpoint Format
- HuggingFace format (not `.nemo`)
- Contains: `model.safetensors`, `config.json`, `tokenizer.json`, etc.

### Training Logs
Should see:
```
🛡️  Using Ministral-3-14B-Base (pure Mistral architecture)
🛡️  Using DDP strategy (optimal for single-GPU Blackwell)
📝 Wrote patched YAML → examples/llm_finetune/mistral/ministral_sft_stage1.yaml
🚀 Launching Stage 1 NeMo SFT Training
```

Should NOT see:
- Any references to NemotronH
- Mamba-related warnings
- `force_hf` messages
- Custom model resolver logs

## Migration from Old Version

If you were using the old `sft_curriculum_trainer.py`:

1. **Backup your data**:
   ```bash
   cp scripts/sft_curriculum_trainer.py scripts/sft_curriculum_trainer_old.py
   ```

2. **Use the new script**:
   ```bash
   python scripts/sft_curriculum_trainer_clean.py --stage 1
   ```

3. **Update checkpoints path**:
   - Old: `mistral-nemo-sft-stage1.nemo`
   - New: `ministral-sft-stage1/`

4. **No Mamba env vars needed**:
   - Remove `export MAMBA_SSM_FORCE_TRITON=1` from your setup

## Notes

- **jsonl_messages.py**: Can be kept for reference but not used by clean script
- **Old README**: Saved as `old_before_change_README.md` for reference
- **Original script**: `sft_curriculum_trainer.py` kept unchanged for comparison
- **Clean script**: `sft_curriculum_trainer_clean.py` is the new recommended version

## Next Steps

1. Test with `--dry-run` flag
2. Run minimal training (2-3 steps)
3. Verify no Nemotron/Mamba errors
4. Proceed with full Stage 1 training
5. Continue to Stage 2 after Stage 1 completes

---

**Created**: 2026-03-22  
**Purpose**: Clean Ministral SFT setup for DGX Spark (Blackwell)
