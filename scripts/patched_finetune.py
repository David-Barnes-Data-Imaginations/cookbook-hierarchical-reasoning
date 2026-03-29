"""
patched_finetune.py — drop in /workspace (project root)

Monkey-patches DDPManager.parallelize to skip .to(bfloat16) for BitsAndBytes
quantized models, then runs the real Automodel finetune.py.

Why:  ddp.py world_size==1 path unconditionally does model.to(bfloat16),
      which raises ValueError for BnB quantized models.
How:  We import DDPManager from the real venv package first, replace the
      parallelize method in-place, then exec finetune.py so all subsequent
      imports see the patched class.

Note: This is a clean version with all Nemotron-specific patches removed.
      Only keeps essential BNB/DDP fixes for single-GPU training.
"""

import logging
import runpy
import sys

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _is_bnb_quantized(model) -> bool:
    """Return True if model was loaded with BitsAndBytes quantization."""
    if getattr(model, "is_quantized", False):
        return True
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "quantization_config", None) is not None:
        return True
    return False


# ── Apply patch before finetune.py touches DDPManager ────────────────────────
from nemo_automodel.components.distributed.ddp import DDPManager  # noqa: E402

_orig_parallelize = DDPManager.parallelize


def _patched_parallelize(self, model):
    if dist.get_world_size() == 1:
        logger.info("World size is 1, skipping parallelization.")
        model = model.to("cuda")
        if _is_bnb_quantized(model):
            # BitsAndBytes models cannot be re-cast after load — skip dtype conversion.
            logger.info("BitsAndBytes quantized model: skipping .to(bfloat16) cast.")
        else:
            model = model.to(torch.bfloat16)
        if self.activation_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            else:
                logger.error("Model does not support gradient checkpointing. Skipping.")
        return model
    # Multi-GPU path: fall through to original (DDP wrapping).
    return _orig_parallelize(self, model)


DDPManager.parallelize = _patched_parallelize
logger.info("DDPManager.parallelize patched for BitsAndBytes QLoRA compatibility.")

# ── Patch 2: fix infrastructure.py apply_model_infrastructure ────────────────
# CRITICAL: Multiple lines assume model.module exists (DDP-wrapped), but for
# world_size=1, our patched DDPManager returns the model unwrapped → no .module.
# Fix: Replace ALL occurrences of model.module with getattr(model, 'module', model)
import inspect
import re
import nemo_automodel._transformers.infrastructure as _infra

_infra_src = inspect.getsource(_infra.apply_model_infrastructure)

# Replace ALL occurrences of model.module access patterns
patterns_to_fix = [
    # Pattern 1: setattr(model.module, ...)
    (r"setattr\(model\.module,", 'setattr(getattr(model, "module", model),'),
    # Pattern 2: model.module.something
    (r"model\.module\.([a-zA-Z_][a-zA-Z0-9_]*)", 'getattr(model, "module", model).\\1'),
    # Pattern 3: model.module[...]]
    (r"model\.module\[", 'getattr(model, "module", model)['),
]

_infra_fixed = _infra_src
replacements_made = 0
for pattern, replacement in patterns_to_fix:
    new_src = re.sub(pattern, replacement, _infra_fixed)
    if new_src != _infra_fixed:
        replacements_made += 1
        _infra_fixed = new_src

# Also guard the model.to(device) call for BnB
_infra_fixed = re.sub(
    r"model\.to\(device, non_blocking=True\)",
    "(model.to(device, non_blocking=True) if not _is_bnb_quantized(model) else model)",
    _infra_fixed,
)

if replacements_made == 0:
    logger.warning(
        "Patch 2: No model.module patterns found - checking if already patched or different structure"
    )
    # Show what we're looking for
    logger.warning("Searching for 'model.module' in infrastructure.py...")
    if "model.module" in _infra_src:
        logger.warning("Found 'model.module' in source - pattern may need adjustment")
    else:
        logger.warning("No 'model.module' found - might already be patched")
else:
    logger.info(
        f"Patch 2: Made {replacements_made} replacements for model.module patterns"
    )
    exec(
        compile(_infra_fixed, _infra.__file__ or "<string>", "exec"),
        dict(vars(_infra)),
    )
    # Also make _is_bnb_quantized available inside the infrastructure module namespace
    _infra._is_bnb_quantized = _is_bnb_quantized
    logger.info(
        "apply_model_infrastructure patched for QLoRA single-GPU compatibility."
    )

# ── Patch 2b: update auto_model's local binding ──────────────────────────────
# auto_model.py imported apply_model_infrastructure with 'from ... import' at module
# load time, creating its own reference to the ORIGINAL function object.
# Our exec() updated _infra.apply_model_infrastructure but NOT auto_model's copy.
# We must explicitly rebind it in auto_model's namespace too.
import nemo_automodel._transformers.auto_model as _auto_model

_auto_model.apply_model_infrastructure = _infra.apply_model_infrastructure
logger.info("auto_model.apply_model_infrastructure rebound to patched version.")

# Also patch any other modules that might have imported it
try:
    import nemo_automodel.recipes.llm.train_ft as _train_ft

    if hasattr(_train_ft, "apply_model_infrastructure"):
        _train_ft.apply_model_infrastructure = _infra.apply_model_infrastructure
        logger.info("train_ft.apply_model_infrastructure rebound to patched version.")
except Exception as _e:
    logger.warning("Could not patch train_ft.apply_model_infrastructure: %s", _e)

# ── Run finetune.py as __main__ (sys.argv is already set correctly by uv) ────
runpy.run_path("examples/llm_finetune/finetune.py", run_name="__main__")
