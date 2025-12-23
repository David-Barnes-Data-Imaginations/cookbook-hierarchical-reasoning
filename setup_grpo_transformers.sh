#!/bin/bash

# Configuration
SSH_HOST="dataimaginations-heirarchical-reasoning@ssh.hf.space"
IDENTITY_FILE="~/.ssh/id_ed25519"

echo "🚀 Setting up clean GRPO environment with Transformers (no Unsloth)..."

ssh -o ServerAliveInterval=30 -i $IDENTITY_FILE $SSH_HOST 'bash -s' << 'EOF'
set -e

# Initialize Conda
source /home/user/miniconda/etc/profile.d/conda.sh
conda activate unsloth_env

echo "🗑️  Removing Unsloth (to avoid patching conflicts)..."
pip uninstall -y unsloth unsloth_zoo 2>/dev/null || true

echo "⬇️  Installing core libraries for GRPO with Transformers..."
uv pip install --upgrade --no-cache-dir \
    "torch>=2.0.0" \
    "transformers>=4.45.0" \
    "trl>=0.11.0" \
    "peft>=0.7.0" \
    "accelerate>=0.33.0" \
    "bitsandbytes>=0.43.0" \
    "datasets" \
    "tensorboard" \
    "huggingface_hub" \
    "ipywidgets"

echo ""
echo "✅ Setup Complete!"
echo ""
echo "📦 Installed versions:"
pip list | grep -E "torch|transformers|trl|peft|accelerate|bitsandbytes"
echo ""
echo "👉 Restart your Jupyter kernel and you're ready to go!"
EOF
