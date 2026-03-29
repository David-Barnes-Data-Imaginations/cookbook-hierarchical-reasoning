# Cookbook: Hierarchical Reasoning on DGX Spark (Unsloth Version)

Curriculum SFT training for **Ministral-3-14B** on NVIDIA DGX Spark (Blackwell architecture, 128GB unified memory) using **Unsloth** for 2-3x faster training and 50% less memory usage.

**New approach** - Switched from NeMo to Unsloth for simpler setup and better performance on Blackwell GPUs.

## Hardware

- **DGX Spark**: Blackwell GB10 GPU with 128GB unified memory
- **Single-GPU setup**: Optimized for Blackwell architecture
- **QLoRA**: 4-bit quantized base model for memory efficiency
- **Model**: Ministral-3-14B-Base (~14B params, pure Mistral architecture)
- **Unsloth**: 2-3x faster training, 50% less memory usage

## Quick Start

### 1. Pull and Run Docker Container

```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --oom-score-adj 1000 --shm-size 64gb \
  -it --rm \
  -v /home/david-barnes/Projects/cookbook-hierarchical-reasoning:/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.11-py3
```

### 2. Setup Unsloth Environment

```bash
# Install Unsloth and dependencies
pip install unsloth trl transformers accelerate bitsandbytes

# Optional: Install with specific versions for stability
pip install unsloth==2025.1.0 trl==0.12.0 transformers==4.48.0
```

### 3. Run Training Scripts

```bash
# Set environment variables for DGX Spark
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export HF_DATASETS_NUM_PROC=8
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export TORCH_CUDA_ARCH_LIST="12.0"
export PYTHONFAULTHANDLER=1
export HF_TOKEN=$HF_TOKEN

# Quick test - dry run to verify setup
python train_unsloth.py --dry-run

# Standard QLoRA training (recommended)
python train_unsloth.py

# Custom configuration
python train_unsloth.py --seq-length 65536 --batch-size 4
```

## Curriculum Learning Scripts

### Size-Based Curriculum Training (Recommended)

Train on 30,000 samples from **Nemotron-Cascade-2-SFT-Data**, sorted by size (smallest to largest):

```bash
# Full curriculum training with 1.5 epochs (per Nemotron paper)
python train_curriculum.py

# Analyze curriculum distribution before training
python train_curriculum.py --analyze

# Dry run (verify data loading without training)
python train_curriculum.py --dry-run

# Custom configuration
python train_curriculum.py --seq-length 131072 --batch-size 8 --gradient-accumulation 4
```

**Dataset Distribution (30,000 samples):**
- `chat`: 9,000 samples (30%)
- `swe`: 8,000 samples (27%)
- `math`: 6,000 samples (20%)
- `terminal_agent`: 3,000 samples (10%)
- `science`: 2,000 samples (7%)
- `conversational_agent`: 1,000 samples (3%)
- `instruction_following`: 1,000 samples (3%)

### Resume Training (Auto-Detect Checkpoints)

Automatically finds and continues from the latest checkpoint:

```bash
# Auto-detect and resume from latest checkpoint
python train_resume.py

# Force fresh training (ignore existing checkpoints)
python train_resume.py --force-new

# Specify a specific checkpoint directory
python train_resume.py --checkpoint-dir ./ministral-sft-unsloth/checkpoint-684

# Dry run (verify resume without training)
python train_resume.py --dry-run
```

### Full Fine-Tuning (No LoRA)

For when you have NVLink connected and need maximum performance:

```bash
# Check memory requirements first
python train_full_finetune.py --memory-check

# Full fine-tuning (updates all model parameters)
python train_full_finetune.py

# Dry run
python train_full_finetune.py --dry-run

# Custom configuration
python train_full_finetune.py --batch-size 2 --gradient-accumulation 8
```

**Note:** Full fine-tuning requires ~100GB VRAM. Use with NVLink for best performance.

### 2. Setup NeMo Automodel

```bash
# 1. Clone the Automodel repository
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel

# 2. Patch pyproject.toml to allow using the container's pre-installed PyTorch
sed -i 's/match-runtime = true/match-runtime = false/g' pyproject.toml

# 3. Install dependencies (using uv for speed and isolation)
pip install uv
uv venv --system-site-packages

export TORCH_CUDA_ARCH_LIST="10.0;12.0"
export CUDA_NVRTC=0
export NVRTC_DISABLE_NVRTC=1

# Install Automodel without Mamba dependencies (Ministral is pure transformer - no Mamba needed)
uv sync --inexact \
  --no-install-package torch \
  --no-install-package torchvision \
  --no-install-package triton \
  --no-install-package nvidia-cublas-cu12 \
  --no-install-package nvidia-cuda-cupti-cu12 \
  --no-install-package nvidia-cuda-nvrtc-cu12 \
  --no-install-package nvidia-cuda-runtime-cu12 \
  --no-install-package nvidia-cudnn-cu12 \
  --no-install-package nvidia-cufft-cu12 \
  --no-install-package nvidia-cufile-cu12 \
  --no-install-package nvidia-curand-cu12 \
  --no-install-package nvidia-cusolver-cu12 \
  --no-install-package nvidia-cusparse-cu12 \
  --no-install-package nvidia-cusparselt-cu12 \
  --no-install-package nvidia-nccl-cu12 \
  --no-install-package transformer-engine \
  --no-install-package nvidia-modelopt \
  --no-install-package nvidia-modelopt-core \
  --no-install-package flash-attn \
  --no-install-package transformer-engine-cu12 \
  --no-install-package transformer-engine-torch \
  --no-install-package nv-grouped-gemm \
  --no-install-package deep-ep \
  --no-install-package causal-conv1d

# 4. Install BitsAndBytes for QLoRA support
uv pip install scikit-build-core cmake python-dotenv datasets==4.3.0 ninja build packaging

CMAKE_ARGS="-DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=80;86;87;89;90;100;120" \
CMAKE_BUILD_PARALLEL_LEVEL=8 \
uv pip install --reinstall --no-cache --no-deps --no-build-isolation \
  git+https://github.com/bitsandbytes-foundation/bitsandbytes.git@50be19c39698e038a1604daf3e1b939c9ac1c342

# 5. Return to project root
cd ..

# 6. Activate the Automodel venv
source Automodel/.venv/bin/activate
```

### 3. Run Training

```bash
# Set environment variables for DGX Spark
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export HF_DATASETS_NUM_PROC=8
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export TORCH_CUDA_ARCH_LIST="10.0;12.0"
export PYTHONFAULTHANDLER=1
export HF_TOKEN=$HF_TOKEN
# Make sure you're in /workspace (project root), NOT /workspace/Automodel
python scripts/sft_curriculum_trainer_clean.py --stage 1 --dry-run 
# Stage 1: Foundation training (3000 samples, 128K context)
# QLoRA is default (recommended for DGX Spark)

python scripts/sft_curriculum_trainer_clean.py --stage 1

# Full LoRA (no quantization — only if you need maximum convergence speed)
python scripts/sft_curriculum_trainer_clean.py --stage 1 --no-qlora

# Stage 2: Extended reasoning training (after Stage 1 completes)
python scripts/sft_curriculum_trainer_clean.py --stage 2
```

## Configuration

### Model
- **Base Model**: `mistralai/Ministral-3-14B-Base-2512` (~14B params, text-only)
- **Architecture**: Pure Mistral (no vision tower, no Mamba)
- **Quantization**: QLoRA (4-bit) with BitsAndBytes

### Training Parameters (Unsloth)
- **Context Length**: 131,072 tokens (128K)
- **Batch Size**: 8 (with packing enabled)
- **Gradient Accumulation**: 4
- **Effective Batch Size**: 32
- **LoRA Rank**: 64
- **LoRA Alpha**: 64
- **Learning Rate**: 2e-5 (cosine decay with warmup)
- **Epochs**: 1.5 (per Nemotron Cascade 2 paper)
- **Max Steps**: ~1,407 (for 30,000 samples with 1.5 epochs)

### Curriculum Learning
- **Sorting**: Samples sorted by token size (smallest to largest)
- **Packing**: Enabled for efficiency (multiple samples per sequence)
- **Document Masking**: Automatic via Flash Attention 2
- **Domains**: Mixed across all size ranges (chat, math, science, SWE, etc.)

### Dataset
- **Source**: NVIDIA Nemotron-Cascade-2-SFT-Data
- **Total Samples**: 30,000 (configurable per domain)
- **Format**: JSONL with messages array (system, user, assistant)
- **Processing**: Pre-formatted with custom chat template

### Stage 1: Foundation
- **Samples**: 3000 total (~50% math, ~30% code, ~20% STEM)
- **Single Examples**: Each sample creates one high-quality training example with thinking tags
- **Format**: Uses DeepSeek-generated reasoning chains with <think>...</think> tags preserved
- **Purpose**: Multi-category foundation training with high-quality reasoning data

### Stage 2: Extended Reasoning
- **Samples**: 3000 total (~61% math, ~32% code, ~7% STEM)
- **Mode**: All examples in thinking mode (reasoning=on)
- **Purpose**: Long-form reasoning chains

## Output

- **LoRA Checkpoint**: `ministral-sft-unsloth/` (HuggingFace format with LoRA adapters)
- **Full Fine-Tune Checkpoint**: `ministral-sft-full-finetune/` (Complete model weights)
- **Sample Tracking**: `sft_used_samples.json` (for RL exclusion)
- **Curriculum Metadata**: Training logs show size distribution across 20 buckets

## Key Features

### Curriculum Learning
- **Size-Based Sorting**: Train on smallest QA pairs first, gradually increasing complexity
- **Automatic Bucketing**: 20 size buckets for monitoring distribution
- **Sample Packing**: Multiple short samples concatenated for efficiency
- **Document Masking**: Automatic attention masking between packed samples

### Unsloth Optimizations
- **2-3x Faster Training**: Optimized kernels for Blackwell architecture
- **50% Less Memory**: Efficient 4-bit quantization with LoRA
- **Automatic Gradient Checkpointing**: Memory-efficient backpropagation
- **Flash Attention 2**: Faster attention computation with automatic masking

### Training Resumption
- **Auto-Detect Checkpoints**: Finds latest checkpoint automatically
- **State Preservation**: Continues optimizer state, learning rate schedule, and step count
- **Flexible Resume**: Can force fresh start or resume from specific checkpoint

## Troubleshooting

### GPU Not Detected (Unsloth Error)
If you see "Unsloth cannot find any torch accelerator":
- Ensure container was started with `--gpus all`
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify environment variables are set correctly

### OOM (Out of Memory)
If you see "Killed" without error messages:
- **For LoRA**: Reduce batch size: `python train_curriculum.py --batch-size 4`
- **For Full FT**: Use gradient checkpointing (enabled by default) or reduce sequence length
- Enable packing: Already enabled by default for memory efficiency

### Loss Shows as Zero
- This is normal for the first few steps during initialization
- If loss remains zero after 50+ steps, check:
  - Data formatting is correct (messages properly converted to text)
  - Model is in training mode: `model.train()`
  - Learning rate is not too high or too low

### Checkpoint Resume Issues
If resume fails:
- Verify checkpoint directory exists: `ls -la ministral-sft-unsloth/`
- Check for `trainer_state.json` in checkpoint directory
- Use `--force-new` to start fresh if checkpoint is corrupted

### Silent Failures
Enable detailed error reporting:
```bash
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1
```

### TMUX/SSH Disconnection
To prevent training from stopping when SSH disconnects:

**Using TMUX (Recommended):**
```bash
# Start a tmux session
tmux new -s training

# Run your training inside tmux
python train_curriculum.py

# Detach safely (training continues): Ctrl+B, then D

# Reattach later: tmux attach -t training
```

**Using NOHUP (Simple):**
```bash
# Run in background
nohup python train_curriculum.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

## Scripts Overview

### `train_unsloth.py`
- **Purpose**: Simple Unsloth training script (legacy, for testing)
- **Data**: Loads from `stage1_data.jsonl`
- **Use Case**: Quick testing and validation

### `train_curriculum.py` ⭐ **RECOMMENDED**
- **Purpose**: Size-based curriculum learning with Nemotron-Cascade-2 data
- **Data**: 30,000 samples from multiple domains, sorted by size
- **Features**: 
  - Automatic size sorting (smallest to largest)
  - Sample packing for efficiency
  - 1.5 epochs (per Nemotron paper)
  - 20 size buckets for monitoring
- **Use Case**: Primary training script for reasoning capabilities

### `train_resume.py`
- **Purpose**: Resume training from last checkpoint
- **Features**:
  - Auto-detects latest checkpoint
  - Preserves optimizer state and learning rate schedule
  - Can force fresh start with `--force-new`
- **Use Case**: Recover from interrupted training

### `train_full_finetune.py`
- **Purpose**: Full parameter fine-tuning (no LoRA)
- **Requirements**: ~100GB VRAM (use with NVLink for best results)
- **Features**:
  - Updates all model parameters
  - Gradient checkpointing for memory efficiency
  - Same curriculum learning as LoRA version
- **Use Case**: Maximum performance when NVLink is connected

### `evaluate_model.py` ⭐ **RECOMMENDED FOR EVALS**
- **Purpose**: Interactive model evaluation on various benchmarks
- **Features**:
  - Interactive CLI for selecting models and checkpoints
  - Support for multiple benchmark tasks (ARC, MMLU, GSM8K, etc.)
  - Configurable limit for evaluation samples
  - Automatic result saving with timestamps
  - Formatted output tables for easy comparison
- **Use Case**: Evaluate trained models to track reasoning improvements
- **Dependencies**: `pip install lm_eval`

**Usage:**
```bash
# Interactive evaluation
python scripts/evaluate_model.py

# The script will guide you through:
# 1. Selecting a model directory (ministral-sft-curriculum or ministral-sft-unsloth)
# 2. Selecting a specific checkpoint (e.g., checkpoint-1000)
# 3. Choosing which tasks to evaluate (or select all)
# 4. Setting a limit on samples per task (0 for no limit)
```

**Example Output:**
```
📊 Results Summary:
|    tasks    |acc|
|-------------|--:|
|   arc_challenge | 0.65|
|    hellaswag | 0.72|
|    winogrande | 0.68|
|       piqa | 0.75|
|       mmlu | 0.48|
|      gsm8k | 0.52|
|truthfulqa_mc2 | 0.41|
```

Results are automatically saved to `logs/` directory with timestamps for easy comparison across checkpoints.

### `scripts/sft_curriculum_trainer_clean.py` (Deprecated)
- **Status**: Legacy script using NeMo Automodel (no longer recommended)
- **Replaced by**: Unsloth-based scripts above

## What's Different from the NeMo Version?

### Switched to Unsloth
1. **Framework**: Changed from NeMo Automodel to Unsloth for simpler setup
2. **Speed**: 2-3x faster training with optimized kernels
3. **Memory**: 50% less memory usage
4. **No YAML Configs**: Direct Python configuration instead of complex YAML files
5. **Native HF Format**: Saves in standard HuggingFace format

### Curriculum Learning Enhancements
1. **Size-Based Sorting**: Train on smallest QA pairs first (smallest to largest)
2. **Automatic Bucketing**: 20 size buckets for monitoring distribution
3. **Sample Packing**: Multiple short samples concatenated for efficiency
4. **Document Masking**: Automatic attention masking via Flash Attention 2

### Training Resumption
1. **Auto-Detect Checkpoints**: Automatically finds latest checkpoint
2. **State Preservation**: Continues optimizer state, learning rate, and step count
3. **Flexible Resume**: Can force fresh start or resume from specific checkpoint

### Dataset Updates
1. **Nemotron-Cascade-2**: Using latest NVIDIA dataset (15.8M samples available)
2. **Curriculum Distribution**: 30,000 samples across 7 domains mixed by size
3. **Pre-Formatting**: Messages automatically converted to text format

### Simplified Workflow
1. **No NeMo Automodel**: Direct Unsloth training without complex setup
2. **Single Script**: One script handles all training (no Stage 1/Stage 2 separation)
3. **Automatic Configuration**: Smart defaults for DGX Spark hardware

## Next Steps

After SFT training completes:
1. **Evaluate the model** with standard HF evaluation tools (e.g., MMLU, GSM8K, HumanEval)
2. **Run inference tests** on reasoning benchmarks to verify curriculum learning effectiveness
3. **Proceed to RL training** (samples tracked in `sft_used_samples.json` for exclusion)
4. **When NVLink arrives**: Consider full fine-tuning with `train_full_finetune.py` for maximum performance
5. **Experiment with different curriculum strategies**: Try domain-specific sorting or multi-stage curriculum

## Performance Tips

### Optimize for DGX Spark
- Use `--batch-size 8` with packing for best throughput
- Enable `bf16` precision (automatic on Blackwell)
- Keep gradient accumulation at 4 for stable gradients

### Monitor Training
- Watch loss values in logs (should be 0.3-0.6 for SFT)
- Check gradient norms (should be 0.05-0.15)
- Verify learning rate follows cosine schedule

### Memory Management
- For 128K context: Use packing to fit more samples
- For full fine-tuning: Reduce batch size to 2-4
- Monitor VRAM with `nvidia-smi` during training

## Citation

If you use this code or the Nemotron-Cascade-2 dataset in your research:

```bibtex
@article{Nemotron_Cascade_2,
  title={Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation},
  author={Yang, Zhuolin and Liu, Zihan and Chen, Yang and Dai, Wenliang and Wang, Boxin and Lin, Sheng-Chieh and Lee, Chankyu and Chen, Yangyi and Jiang, Dongfu and He, Jiafan and Pi, Renjie and Lam, Grace and Lee, Nayeon and Bukharin, Alexander and Shoeybi, Mohammad and Catanzaro, Bryan and Ping, Wei},
  year={2026}
}
```
