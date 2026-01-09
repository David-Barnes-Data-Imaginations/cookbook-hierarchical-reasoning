#!/bin/bash

# Configuration
SSH_HOST="dataimaginations-heirarchical-reasoning@ssh.hf.space"
IDENTITY_FILE="~/.ssh/id_ed25519"

echo "🧹 Clearing Unsloth cache and trying compatible versions..."

ssh -o ServerAliveInterval=30 -i $IDENTITY_FILE $SSH_HOST 'bash -s' << 'EOF'
set -e

# Initialize Conda
source /home/user/miniconda/etc/profile.d/conda.sh
conda activate unsloth_env

echo "🗑️  Removing corrupted Unsloth cache..."
rm -rf /tmp/unsloth_compiled_cache
rm -rf ~/.cache/unsloth

echo "⬇️  Installing TRL 0.10.1 (BEFORE GRPOTrainer was added - will train with standard SFT instead)..."
pip uninstall -y trl
pip install "trl==0.10.1" --no-cache-dir

echo "✅ Setup Complete!"
echo ""
echo "📝 IMPORTANT: Your notebook needs to be updated to use SFTTrainer instead of GRPOTrainer."
echo "   GRPOTrainer doesn't exist in TRL 0.10.1, but standard SFT will work fine."
EOF
