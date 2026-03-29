

What is NCCL?
NCCL (pronounced "Nickel") stands for the NVIDIA Collective Communications Library.

When you train a model across multiple GPUs or nodes (like your two DGX Sparks), the machines need to constantly talk to each other to share data. Every time they process a batch of data, they have to synchronize their "gradients" (the math updates) so both Sparks are updating the exact same model weights.

NCCL is the underlying engine that makes this communication happen over your NVLink/QSFP cable at incredibly high speeds. Instead of using standard internet protocols (which are too slow), NCCL talks directly to the networking hardware and GPU memory to bypass CPU bottlenecks.

PyTorch natively uses NCCL under the hood for distributed training (via a module called DistributedDataParallel or FSDP).

`train_full_finetune.py` will not work across two Sparks so you need to change:

1. Remove the Single-Node Hardcoding
```
# Set environment variables for single-GPU mode
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")
```


# Disable distributed training (single GPU)
`os.environ["NCCL_P2P_DISABLE"] = "1"`
You must delete or comment out all of these lines. * WORLD_SIZE="1" tells PyTorch that there is only one machine. You want this to be 2.

`MASTER_ADDR="localhost"` prevents the second Spark from finding the first one.

`NCCL_P2P_DISABLE="1"` turns off the NCCL features you are trying to use when using two Sparks

2. Install and Build NCCL
Because DGX Sparks use the new Blackwell architecture, PyTorch's default NCCL might not be optimized for them yet. You will need to strictly follow the steps in that NVIDIA Stacked Sparks Guide you found. It guides you on how to compile NCCL from source specifically for Blackwell (compute_121).

3. Change How You Launch the Script
For the single Spark you would run:
`python scripts/train_full_finetune.py`

Once you connect the Sparks and install NCCL, you won't launch it with standard Python anymore. You will use torchrun (PyTorch's distributed launcher) or Hugging Face accelerate.

On Spark 1 (the Master node), you will run something like:

`torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="<SPARK_1_IP>" --master_port=12355 scripts/train_full_finetune.py`

On Spark 2, run:
`torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="<SPARK_1_IP>" --master_p`