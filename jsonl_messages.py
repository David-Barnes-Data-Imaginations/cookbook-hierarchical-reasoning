"""
jsonl_messages.py  —  drop into /workspace (project root = /workspace in container)
and set PYTHONPATH=/workspace so NeMo Automodel's config loader can resolve:
  _target_: jsonl_messages.make_jsonl_messages_dataset

Handles OpenAI-format JSONL {"messages": [...]} without requiring the tokenizer
to have a chat_template attribute — works with base models (nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base).
"""

import json
import logging
from typing import Optional, Union

from datasets import Dataset

from nemo_automodel.components.datasets.llm.formatting_utils import (
    _add_pad_token,
    _has_chat_template,
    format_chat_template,
    format_prompt_completion,
)

logger = logging.getLogger(__name__)


def _load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_jsonl_messages_dataset(
    tokenizer,
    path_or_dataset_id: Union[str, list],
    seq_length: Optional[int] = None,
    limit_dataset_samples: Optional[int] = None,
    padding: Union[str, bool] = "do_not_pad",
    truncation: Union[str, bool] = "do_not_truncate",
    split: Optional[str] = None,   # accepted but ignored for local files; required by train_ft.py cfg_ds.split
):
    """
    Load a JSONL file with OpenAI-format messages and tokenize for SFT.

    Uses format_chat_template when the tokenizer has a chat_template (instruct
    models), and falls back to simple prompt-completion concatenation for base
    models that don't expose chat_template through the NeMo tokenizer wrapper.

    Args:
        tokenizer:              NeMo/HF tokenizer injected by Automodel.
        path_or_dataset_id:     Local path to a .jsonl file (or list of paths).
        seq_length:             Max sequence length; None = no truncation/padding.
        limit_dataset_samples:  Cap the number of examples loaded.
        padding:                Padding strategy for the formatting utils.
        truncation:             Truncation strategy for the formatting utils.

    Returns:
        datasets.Dataset with keys: input_ids, labels, attention_mask.
    """
    # ─── load ────────────────────────────────────────────────────────────────
    if isinstance(path_or_dataset_id, (list, tuple)):
        rows = []
        for p in path_or_dataset_id:
            rows.extend(_load_jsonl(p))
    else:
        rows = _load_jsonl(path_or_dataset_id)

    if limit_dataset_samples is not None:
        rows = rows[:limit_dataset_samples]

    logger.info("Loaded %d examples from %s", len(rows), path_or_dataset_id)

    # ─── tokenize ────────────────────────────────────────────────────────────
    eos_token_id = getattr(tokenizer, "eos_token_id", 0)
    pad_token_id = _add_pad_token(tokenizer) or eos_token_id
    use_chat_template = _has_chat_template(tokenizer)

    if use_chat_template:
        logger.info("Tokenizer has chat_template → format_chat_template")
    else:
        logger.info("Tokenizer has no chat_template (base model) → prompt-completion fallback")

    processed, skipped = [], 0

    for row in rows:
        messages = row.get("messages", [])

        if use_chat_template:
            try:
                example = format_chat_template(
                    tokenizer=tokenizer,
                    formatted_text=list(messages),   # copy — mutated internally
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    seq_length=seq_length,
                    padding=padding,
                    truncation=truncation,
                )
            except Exception as exc:
                logger.warning("Skipping example (chat_template error): %s", exc)
                skipped += 1
                continue
        else:
            # Concatenate system + user as prompt, assistant as answer.
            system_parts = [m["content"] for m in messages if m.get("role") == "system"]
            user_parts   = [m["content"] for m in messages if m.get("role") == "user"]
            asst_parts   = [m["content"] for m in messages if m.get("role") == "assistant"]

            if not user_parts or not asst_parts:
                skipped += 1
                continue

            prompt_pieces = []
            if system_parts:
                prompt_pieces.append(system_parts[0])
            prompt_pieces.append(user_parts[0])
            prompt = "\n\n".join(prompt_pieces) + "\n\n"
            answer = asst_parts[0]

            try:
                example = format_prompt_completion(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    answer=answer,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    seq_length=seq_length,
                    padding=padding,
                    truncation=truncation,
                )
            except Exception as exc:
                logger.warning("Skipping example (tokenizer error): %s", exc)
                skipped += 1
                continue

        # Only keep examples with at least one supervised label token.
        if any(label != -100 for label in example.get("labels", [])):
            processed.append(example)
        else:
            skipped += 1

    logger.info("Dataset ready: %d kept, %d skipped.", len(processed), skipped)

    if not processed:
        raise RuntimeError(
            f"No valid training examples produced from {path_or_dataset_id}. "
            "Check that the JSONL has 'messages' with user + assistant roles."
        )

    return Dataset.from_list(processed)
