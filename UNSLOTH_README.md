# Unsloth Training Guide for DGX Spark

## Quick Start

### 1. Install Unsloth

Inside your Docker container:

```bash
# Install Unsloth (optimized for Blackwell)
pip install unsloth

# Verify installation
python -c "import unsloth; print('Unsloth installed successfully!')"
```

### 2. Prepare Your Data

Ensure your training data is in JSONL format at:
```
/workspace/stage1_data.jsonl
```

Each line should be a JSON object with a "messages" field:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### 3. Run Training

```bash
# Standard training (32K context, batch size 2)
python train_unsloth.py

# Dry run (test without training)
python train_unsloth.py --dry-run

# Custom settings
python train_unsloth.py --seq-length 65536 --batch-size 4
```

## Configuration

### Default Settings (Optimized for DGX Spark)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Sequence Length | 32,768 | Start here, can increase to 64K or 128K |
| Batch Size | 2 | Memory-efficient for 128GB unified memory |
| Gradient Accumulation | 4 | Effective batch = 8 |
| Learning Rate | 2e-5 | Standard for SFT |
| LoRA Rank | 64 | Good balance of quality/memory |
| Max Steps | 750 | For 3000 samples |

### Memory Usage Estimates

- **32K context**: ~40-50GB VRAM
- **64K context**: ~60-70GB VRAM  
- **128K context**: ~90-100GB VRAM

With 128GB unified memory, you can safely use up to 128K context.

## Troubleshooting

### OOM (Out of Memory)

If you see "Killed" or OOM errors:

```bash
# Reduce sequence length
python train_unsloth.py --seq-length 16384

# Reduce batch size
python train_unsloth.py --batch-size 1
```

### Model Loading Issues

If the model fails to load:

```bash
# Check HF token
export HF_TOKEN="your_token_here"

# Or login with HF CLI
hf auth login
```

### Slow Training

Unsloth should be 2-3x faster than standard methods. If it's slow:

```bash
# Ensure you're using the correct GPU
nvidia-smi  # Should show your DGX Spark GPU

# Check if Unsloth is using optimized kernels
python -c "from unsloth import is_bf16_supported; print('BF16 supported:', is_bf16_supported())"
```

## Expected Training Time

For 3000 samples with 32K context:
- **Training time**: ~1-2 hours
- **Memory usage**: ~40-50GB
- **Speed**: ~2-3x faster than NeMo

## Output

After training completes:
- **Model checkpoint**: `/workspace/ministral-sft-unsloth/`
- **Format**: HuggingFace format (ready for use)
- **Files**: `model.safetensors`, `tokenizer.json`, `adapter_config.json`

## Next Steps

1. **Evaluate**: Test the model on validation set
2. **Stage 2**: Train with more data or longer context
3. **Inference**: Use the model for generation

## Why Unsloth Works Better

1. **Memory Optimized**: 50% less memory than standard methods
2. **Blackwell Native**: Built for your DGX Spark architecture
3. **Simpler**: No complex YAML configuration
4. **Faster**: 2-3x training speed
5. **Pre-optimized**: Ministral model already quantized

## Comparison with NeMo

| Feature | NeMo | Unsloth |
|---------|------|---------|
| Memory Usage | High (OOM-prone) | Low (optimized) |
| Setup Complexity | Complex (YAML hacking) | Simple (one script) |
| Training Speed | Standard | 2-3x faster |
| DGX Spark Support | Generic | Native |
| Error Rate | High (config issues) | Low |

## Support

- **Unsloth Docs**: https://docs.unsloth.ai/
- **DGX Spark Forum**: https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10
- **Unsloth GitHub**: https://github.com/unslothai/unsloth

---

**Good luck with your training! 🚀**
