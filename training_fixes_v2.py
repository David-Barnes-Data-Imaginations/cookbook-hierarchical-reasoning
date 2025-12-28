# ============================================================================
# TRAINING FIXES V2 - With NaN/Inf Detection and Numerical Stability
# ============================================================================
# 
# Key fixes in this version:
# 1. Clamp advantages to prevent extreme values
# 2. Check for NaN/Inf in loss and skip bad batches
# 3. Lower learning rate (2e-6 instead of 5e-6)
# 4. Clamp log probabilities to prevent -inf
# 5. Add loss scaling to prevent gradient explosion
#
# ============================================================================

import torch
import gc
import traceback
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn.functional as F


def print_gpu_memory(label=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU {label}] {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def check_tensor_health(tensor, name):
    """Check if tensor contains NaN or Inf values"""
    if tensor is None:
        return True
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        print(f"   ⚠️  {name} contains NaN={has_nan}, Inf={has_inf}")
        return False
    return True


def run_stable_hicra_training(model, tokenizer, dataset_train, hicra, 
                               epochs=2, grad_accum_steps=4, 
                               max_new_tokens=100, num_sequences=2,
                               learning_rate=2e-6,  # Lower LR for stability
                               max_advantage=5.0,   # Clamp extreme advantages
                               log_every=4, memory_log_every=20):
    """
    Numerically stable HICRA training loop.
    
    Key stability features:
    - Advantage clamping to [-max_advantage, max_advantage]
    - NaN/Inf detection with batch skipping
    - Lower learning rate
    - Log probability clamping
    - Gradient clipping
    """
    
    print("🚀 Starting STABLE HICRA Training Loop (V2)...")
    print(f"   Config: epochs={epochs}, grad_accum={grad_accum_steps}, lr={learning_rate}")
    print(f"   Generation: max_tokens={max_new_tokens}, sequences={num_sequences}")
    print(f"   Stability: max_advantage={max_advantage}")
    
    # Disable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
        print("   ⚠️  Disabled gradient checkpointing for stability")
    
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = 0
    skipped_steps = 0
    error_count = 0
    max_consecutive_errors = 5
    
    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()  # Reset at start of epoch
        
        for step, batch in enumerate(tqdm(dataset_train, desc=f"Epoch {epoch+1}")):
            # Initialize variables for safe cleanup
            outputs = None
            logits = None
            loss = None
            inputs = None
            planning_mask = None
            
            try:
                # A. Format Input
                prompt = batch['prompt']
                answer = str(batch['answer'])
                
                # B. Tokenize
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                inputs = inputs.to(model.device)
                
                # C. Generate Rollouts (with no_grad to save memory)
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.9,
                            top_p=0.95,  # Added nucleus sampling for stability
                            num_return_sequences=num_sequences,
                            pad_token_id=tokenizer.pad_token_id
                        )
                
                # D. Score the Outputs
                rewards = []
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for text in generated_texts:
                    r = 1.0 if answer in text else -1.0
                    rewards.append(r)
                rewards = torch.tensor(rewards, device=model.device, dtype=torch.float32)
                
                # E. Compute Advantages (with clamping!)
                mean_r = rewards.mean()
                std_r = rewards.std() + 1e-8
                advantages = (rewards - mean_r) / std_r
                
                # STABILITY FIX: Clamp advantages to prevent extreme values
                advantages = torch.clamp(advantages, -max_advantage, max_advantage)
                
                # Check for NaN in advantages
                if not check_tensor_health(advantages, "advantages"):
                    print(f"   Skipping step {step} due to bad advantages")
                    skipped_steps += 1
                    continue
                
                # F. HICRA Planning Mask
                planning_mask = hicra.identify_planning_mask(outputs, tokenizer).to(model.device)
                
                # G. Expand advantages to token level
                token_advantages = advantages.view(-1, 1).expand_as(planning_mask).clone().float()
                final_advantages = hicra.compute_hicra_advantages(token_advantages, planning_mask)
                
                # STABILITY FIX: Clamp final advantages too
                final_advantages = torch.clamp(final_advantages, -max_advantage, max_advantage)
                
                # H. Compute Policy Gradient Loss
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(outputs).logits
                    
                    # Check logits health
                    if not check_tensor_health(logits, "logits"):
                        print(f"   Skipping step {step} due to bad logits")
                        skipped_steps += 1
                        continue
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = outputs[..., 1:].contiguous()
                    
                    # Compute log probs with numerical stability
                    log_probs = F.log_softmax(shift_logits, dim=-1)
                    
                    # STABILITY FIX: Clamp log probs to avoid -inf
                    log_probs = torch.clamp(log_probs, min=-100.0)
                    
                    token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(2)).squeeze(2)
                    
                    # Check token_log_probs health
                    if not check_tensor_health(token_log_probs, "token_log_probs"):
                        print(f"   Skipping step {step} due to bad log probs")
                        skipped_steps += 1
                        continue
                    
                    prompt_len = inputs.input_ids.shape[1]
                    loss_mask = torch.ones_like(token_log_probs)
                    loss_mask[:, :prompt_len] = 0
                    
                    # Align dimensions
                    current_advantages = final_advantages[:, 1:]
                    min_len = min(current_advantages.shape[1], token_log_probs.shape[1])
                    current_advantages = current_advantages[:, :min_len]
                    token_log_probs = token_log_probs[:, :min_len]
                    loss_mask = loss_mask[:, :min_len]
                    
                    # Compute loss
                    masked_sum = (current_advantages * token_log_probs * loss_mask).sum()
                    mask_count = loss_mask.sum() + 1e-8
                    loss = -masked_sum / mask_count
                
                # STABILITY FIX: Check loss health before backward
                if not check_tensor_health(loss, "loss"):
                    print(f"   Skipping step {step} due to bad loss")
                    skipped_steps += 1
                    continue
                
                # STABILITY FIX: Skip if loss is too extreme
                if abs(loss.item()) > 50.0:
                    print(f"   Skipping step {step}: loss too extreme ({loss.item():.2f})")
                    skipped_steps += 1
                    continue
                
                # I. Backward pass
                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()
                
                # J. Optimizer step (with gradient clipping)
                if (step + 1) % grad_accum_steps == 0:
                    # STABILITY FIX: Aggressive gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    
                    # Check if gradients are healthy
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"   ⚠️  Bad gradients at step {step}, skipping optimizer step")
                        optimizer.zero_grad()
                        skipped_steps += 1
                        continue
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Log progress
                if (step + 1) % log_every == 0:
                    print(f"\n   Step {step+1}: Loss={loss.item():.4f}, Mean Reward={mean_r:.2f}")
                
                # Log memory
                if (step + 1) % memory_log_every == 0:
                    print_gpu_memory(f"Step {step+1}")
                
                error_count = 0  # Reset on success
                total_steps += 1
                
            except RuntimeError as e:
                error_str = str(e)
                if "CUDA" in error_str or "device-side assert" in error_str:
                    print(f"\n❌ CUDA Error at step {step}: {e}")
                    print("   This usually means the model weights have become unstable.")
                    print("   Try restarting the kernel and lowering the learning rate further.")
                    return False
                else:
                    error_count += 1
                    print(f"\n⚠️  Error at step {step}: {e}")
                    if error_count >= max_consecutive_errors:
                        print(f"❌ Too many errors. Stopping.")
                        return False
                    
            except Exception as e:
                error_count += 1
                print(f"\n⚠️  Error at step {step}: {e}")
                if error_count >= max_consecutive_errors:
                    print(f"❌ Too many errors. Stopping.")
                    return False
            
            finally:
                # K. SAFE VRAM Cleanup
                for var in [outputs, logits, loss, inputs, planning_mask]:
                    try:
                        if var is not None:
                            del var
                    except:
                        pass
                gc.collect()
                torch.cuda.empty_cache()
    
    print(f"\n✅ Training completed!")
    print(f"   Total steps: {total_steps}")
    print(f"   Skipped steps: {skipped_steps}")
    return True


# ============================================================================
# USAGE - Copy this into your notebook cell
# ============================================================================
"""
# Cell 1: Import and run

exec(open('training_fixes_v2.py').read())

success = run_stable_hicra_training(
    model=model,
    tokenizer=tokenizer, 
    dataset_train=dataset_train,
    hicra=hicra,
    epochs=2,
    grad_accum_steps=4,
    max_new_tokens=100,
    num_sequences=2,
    learning_rate=2e-6,    # Lower than before
    max_advantage=5.0,     # Clamp extreme advantages
    log_every=4,
    memory_log_every=20
)
"""
