# Cell 6: HICRA Manager Class
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import gc

class HICRA_Manager:
    """Hierarchical Credit Assignment for Reasoning."""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.strategic_grams = [
            "first i need to", "let's look at", "alternatively", "wait",
            "but i'm not sure", "let's see if", "notice that",
            "the final answer is", "let's assume", "we can conclude",
            "implies that", "to solve this", "break it down",
            "suppose that", "checking the", "recall that"
        ]

    def identify_planning_mask(self, input_ids, tokenizer):
        """Creates a mask (True/False) for tokens that are part of planning phrases."""
        batch_size, seq_len = input_ids.shape
        planning_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        for b_idx, text in enumerate(texts):
            text_lower = text.lower()
            for gram in self.strategic_grams:
                if gram in text_lower:
                    # Mark the whole sequence if it contains planning words
                    planning_mask[b_idx] = True
        return planning_mask

    def compute_hicra_advantages(self, advantages, planning_mask):
        """Modify advantages based on planning mask (HICRA core logic)."""
        # Broadcast scalar advantage to token level [Batch, Seq]
        if advantages.dim() == 1:
            advantages = advantages.view(-1, 1).expand_as(planning_mask)

        hicra_adjustment = self.alpha * advantages.abs()
        hicra_adjustment = hicra_adjustment * planning_mask.float()
        return advantages + hicra_adjustment

hicra = HICRA_Manager(alpha=0.2)
print("✅ HICRA Manager initialized")
