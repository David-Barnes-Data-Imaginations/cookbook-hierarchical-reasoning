#!/bin/bash

# Configuration
SSH_HOST="dataimaginations-heirarchical-reasoning@ssh.hf.space"
IDENTITY_FILE="~/.ssh/id_ed25519"

echo "🚀 Connecting to remote Space to set up Python 3.11 environment..."

ssh -o ServerAliveInterval=30 -i $IDENTITY_FILE $SSH_HOST 'bash -s' << 'EOF'
set -e # Exit on error

# 1. Initialize Conda
source /home/user/miniconda/etc/profile.d/conda.sh

# 2. Create Environment (if not exists)
if ! conda info --envs | grep -q "unsloth_env"; then
    echo "📦 Creating 'unsloth_env' with Python 3.11..."
    conda create -n unsloth_env python=3.11 -y
else
    echo "✅ 'unsloth_env' already exists."
fi

# 3. Activate
echo "BS"
conda activate unsloth_env

# 4. Install Jupyter Kernel
echo "🔌 Installing Jupyter Kernel..."
pip install ipykernel &> /dev/null
python -m ipykernel install --user --name=unsloth_env --display-name "Python 3.11 (Unsloth)"

# 5. Install Core Libraries
echo "⬇️  Installing Unsloth, TRL, & Dependencies (this heavily uses the GPU/Network)..."
# Using standard pip here ensures compatibility with the active conda python
pip install unsloth vllm "trl>=0.26.0" peft accelerate bitsandbytes datasets pandas tensorboard huggingface_hub &> pip_log.txt

echo "✅ Setup Complete!"
echo "---------------------------------------------------------"
echo "👉 GO TO YOUR BROWSER NOW:"
echo "1. Refresh the page (F5)"
echo "2. Click 'Kernel' -> 'Change Kernel' -> 'Python 3.11 (Unsloth)'"
echo "---------------------------------------------------------"
EOF
