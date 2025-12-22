# Run the script before connecting:
./setup_remote_env.sh

# first:
ssh -o ServerAliveInterval=30 -i ~/.ssh/id_ed25519 -L 8888:localhost:8888 dataimaginations-heirarchical-reasoning@ssh.hf.space

# then:
/home/user/miniconda/bin/jupyter lab --port 8888 --no-browser --ip=0.0.0.0 --NotebookApp.token='huggingface'

# finally, open in browser:
http://localhost:8888/lab?token=huggingface

# then, click 'Kernel' -> 'Change Kernel' -> 'Python 3.11 (Unsloth)'

# To set your HF_TOKEN permanently on the remote server, Run this in your local terminal:
ssh -i ~/.ssh/id_ed25519 dataimaginations-heirarchical-reasoning@ssh.hf.space "echo 'export HF_TOKEN=hf_' >> ~/.bashrc"