# Patched ddp.py — original at Automodel/nemo_automodel/components/distributed/ddp.py
# Patch: skip .to(torch.bfloat16) for BitsAndBytes quantized models (world_size=1 path).
# BitsAndBytes models cannot be cast to a new dtype after load; the original code
# unconditionally did model.to("cuda").to(torch.bfloat16) which raises ValueError.
#
# This file shadows the original via PYTHONPATH=/workspace/patches:... so Python
# resolves it first. All other behaviour is identical to the original.

import logging

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from nemo_automodel.components.distributed.config import DDPConfig
from nemo_automodel.components.distributed.parallelizer import _extract_model_layers

logger = logging.getLogger(__name__)


def _is_bnb_quantized(model) -> bool:
    """Return True if *model* is loaded with BitsAndBytes quantization."""
    # transformers sets is_quantized=True and quantization_method="bitsandbytes"
    if getattr(model, "is_quantized", False):
        return True
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "quantization_config", None) is not None:
        return True
    return False


class DDPManager:
    """
    Manager for distributed training using PyTorch's DDP.

    This manager wraps models with DistributedDataParallel for data-parallel
    distributed training.

    Args:
        config (DDPConfig): Configuration for DDP distributed training.
    """

    def __init__(self, config: DDPConfig):
        self.config = config

        # Extract config fields for easy access
        self.activation_checkpointing = config.activation_checkpointing
        self.backend = config.backend

        # Setup distributed environment
        self._setup_distributed()

    def _setup_distributed(self):
        """Initialize device configuration for DDP."""
        if not dist.is_available():
            raise RuntimeError("torch.distributed not available")

        if not dist.is_initialized():
            raise RuntimeError("expected torch.distributed to be initialized")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Pin GPU if using NCCL
        if self.backend == "nccl":
            local_gpu = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(local_gpu)
            self.device = torch.device("cuda", index=local_gpu)
        else:
            self.device = torch.device("cpu")

    def parallelize(self, model):
        """
        Wraps the given model with DistributedDataParallel (DDP).

        For world_size == 1 (single GPU), skips DDP wrapping and just moves
        the model to CUDA.  The dtype cast to bfloat16 is skipped for
        BitsAndBytes quantized models — they cannot be re-cast after loading.
        """
        if dist.get_world_size() == 1:
            logger.info("World size is 1, skipping parallelization.")
            model = model.to("cuda")
            # PATCH: BitsAndBytes quantized models raise ValueError on .to(dtype).
            # Skip the cast; the model was already loaded in the correct dtype.
            if not _is_bnb_quantized(model):
                model = model.to(torch.bfloat16)
            else:
                logger.info(
                    "BitsAndBytes quantized model detected — skipping .to(bfloat16) cast."
                )
            if self.activation_checkpointing:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                else:
                    logger.error("Model does not support gradient checkpointing. Skipping.")
            return model

        if self.activation_checkpointing:
            if hasattr(model, "config") and getattr(model.config, "use_cache", None) is not False:
                try:
                    model.config.use_cache = False
                except Exception:
                    pass

            layers = _extract_model_layers(model)
            for i, layer in enumerate(layers):
                if hasattr(layer, "mlp"):
                    layers[i].mlp = checkpoint_wrapper(layer.mlp)
                if hasattr(layer, "self_attn"):
                    layers[i].self_attn = checkpoint_wrapper(layers[i].self_attn)
                if hasattr(layer, "input_layernorm"):
                    layers[i].input_layernorm = checkpoint_wrapper(layers[i].input_layernorm)
                if hasattr(layer, "post_attention_layernorm"):
                    layers[i].post_attention_layernorm = checkpoint_wrapper(layers[i].post_attention_layernorm)

        return DDP(model.to(self.device), device_ids=[self.device] if self.device.type == "cuda" else None)
