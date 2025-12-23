# Run the script before connecting:
./setup_grpo_transformers.sh

# first:
ssh -o ServerAliveInterval=30 -i ~/.ssh/id_ed25519 -L 8888:localhost:8888 dataimaginations-heirarchical-reasoning@ssh.hf.space

# then:
/home/user/miniconda/bin/jupyter lab --port 8888 --no-browser --ip=0.0.0.0 --NotebookApp.token='huggingface'

# finally, open in browser:
http://localhost:8888/lab?token=huggingface

# then, click 'Kernel' -> 'Change Kernel' -> 'Python 3.11 (Unsloth)'

# To upload the reasoning dataset, Run this in your local terminal:
scp -i ~/.ssh/id_ed25519 reasoning_dataset.json dataimaginations-heirarchical-reasoning@ssh.hf.space:~/app/

# Or to upload both the test and train use:
scp -i ~/.ssh/id_ed25519 reasoning_dataset*.json dataimaginations-heirarchical-reasoning@ssh.hf.space:~/app/



# Text only models

Leading Open Source LLMs Under 16B Parameters 
Model Family 	Developer	Parameters	Release Date	License
Phi-3 Mini/Small/Medium	Microsoft	3.8B, 7B, 14B	April 2024	MIT
Llama 3	Meta AI	8B	April 2024	Llama Community License
Mistral	Mistral AI	7B	Sept 2023	Apache 2.0
Gemma 2	Google DeepMind	9B	2025	Gemma License
Qwen 2.5	Alibaba Cloud	7B, 14B	Sept 2024	Apache 2.0
Falcon 2	TII	11B	May 2024	Apache 2.0 with AUP
StableLM 2	Stability AI	1.6B, 3B, 12B	Jan 2024	Stability AI Community/Enterprise
StarCoder2	BigCode	3B, 7B, 15B	2024	Apache 2.0