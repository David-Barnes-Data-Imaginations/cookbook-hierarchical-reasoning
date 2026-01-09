# ============================================================================
# DIAGNOSTIC CELL - Run this BEFORE your training loop to test for issues
# ============================================================================

import torch
import gc
import traceback

def print_gpu_memory(label=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU {label}] {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def test_single_sample(model, tokenizer, dataset, hicra):
    """
    Test a single sample through the entire pipeline to identify crash points.
    Run this before your full training loop.
    """
    print("="*60)
    print("🔬 DIAGNOSTIC: Testing Single Sample Through Pipeline")
    print("="*60)
    
    batch = dataset[0]
    prompt = batch['prompt']
    answer = batch['answer']
    
    print(f"\n📝 Prompt length: {len(prompt)} chars")
    print(f"📝 Answer: '{answer}' (type: {type(answer).__name__})")
    print_gpu_memory("Initial")
    
    # Step 1: Tokenize
    print("\n[Step 1/5] Tokenizing prompt...")
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print(f"   ✅ Input shape: {inputs.input_ids.shape}")
        print_gpu_memory("After tokenize")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        traceback.print_exc()
        return False
    
    # Step 2: Generate (with reduced params for testing)
    print("\n[Step 2/5] Generating completions (reduced: 2 seqs, 50 tokens)...")
    try:
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,  # Reduced for testing
                    do_sample=True,
                    temperature=0.9,
                    num_return_sequences=2,  # Reduced for testing
                    pad_token_id=tokenizer.pad_token_id
                )
        print(f"   ✅ Output shape: {outputs.shape}")
        print_gpu_memory("After generate")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        traceback.print_exc()
        return False
    
    # Step 3: Decode and check rewards
    print("\n[Step 3/5] Decoding and computing rewards...")
    try:
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        rewards = []
        for text in generated_texts:
            r = 1.0 if str(answer) in text else -1.0
            rewards.append(r)
        rewards_tensor = torch.tensor(rewards, device=model.device)
        print(f"   ✅ Rewards: {rewards}")
        print(f"   ✅ Sample decoded text (first 100 chars): {generated_texts[0][:100]}...")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        traceback.print_exc()
        return False
    
    # Step 4: HICRA mask computation
    print("\n[Step 4/5] Computing HICRA planning mask...")
    try:
        planning_mask = hicra.identify_planning_mask(outputs, tokenizer)
        planning_mask = planning_mask.to(model.device)
        num_planning_tokens = planning_mask.sum().item()
        print(f"   ✅ Mask shape: {planning_mask.shape}")
        print(f"   ✅ Planning tokens found: {num_planning_tokens}")
        print_gpu_memory("After HICRA mask")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        traceback.print_exc()
        return False
    
    # Step 5: Forward pass for logits
    print("\n[Step 5/5] Forward pass for logits...")
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(outputs).logits
        print(f"   ✅ Logits shape: {logits.shape}")
        print_gpu_memory("After forward")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        traceback.print_exc()
        return False
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    del outputs, logits, planning_mask, inputs, rewards_tensor
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("✅ DIAGNOSTIC PASSED - Single sample completed successfully!")
    print("="*60)
    print_gpu_memory("After cleanup")
    return True


# ============================================================================
# IMPROVED TRAINING LOOP - Replace your existing training cell with this
# ============================================================================

def run_improved_hicra_training(model, tokenizer, dataset_train, hicra, 
                                 epochs=2, grad_accum_steps=4, 
                                 max_new_tokens=100, num_sequences=2,
                                 log_every=5, memory_log_every=10):
    """
    Improved HICRA training loop with better memory management and error handling.
    
    Key improvements:
    - Reduced generation parameters for stability
    - Proper exception handling that doesn't lose variables
    - Memory monitoring
    - Graceful error recovery
    """
    import gc
    from tqdm import tqdm
    from torch.optim import AdamW
    import torch.nn.functional as F
    
    print("🚀 Starting Improved HICRA Training Loop...")
    print(f"   Config: epochs={epochs}, grad_accum={grad_accum_steps}")
    print(f"   Generation: max_tokens={max_new_tokens}, sequences={num_sequences}")
    
    # Disable gradient checkpointing (can cause issues with manual loops)
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
        print("   ⚠️  Disabled gradient checkpointing for stability")
    
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-6)
    
    total_steps = 0
    error_count = 0
    max_errors = 10  # Stop if too many consecutive errors
    
    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch + 1}/{epochs}")
        
        for step, batch in enumerate(tqdm(dataset_train, desc=f"Epoch {epoch+1}")):
            # Initialize variables to None for safe cleanup
            outputs = None
            logits = None
            loss = None
            inputs = None
            planning_mask = None
            
            try:
                # A. Format Input
                prompt = batch['prompt']
                answer = str(batch['answer'])  # Ensure string
                
                # B. Tokenize
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                # C. Generate Rollouts
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.9,
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
                
                # E. Compute Advantages
                mean_r = rewards.mean()
                std_r = rewards.std() + 1e-8
                advantages = (rewards - mean_r) / std_r
                
                # F. HICRA Planning Mask
                planning_mask = hicra.identify_planning_mask(outputs, tokenizer).to(model.device)
                
                # G. Expand advantages to token level
                token_advantages = advantages.view(-1, 1).expand_as(planning_mask).clone().float()
                final_advantages = hicra.compute_hicra_advantages(token_advantages, planning_mask)
                
                # H. Compute Policy Gradient Loss
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(outputs).logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = outputs[..., 1:].contiguous()
                    log_probs = F.log_softmax(shift_logits, dim=-1)
                    token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(2)).squeeze(2)
                    
                    prompt_len = inputs.input_ids.shape[1]
                    loss_mask = torch.ones_like(token_log_probs)
                    loss_mask[:, :prompt_len] = 0
                    
                    # Align dimensions
                    current_advantages = final_advantages[:, 1:]
                    min_len = min(current_advantages.shape[1], token_log_probs.shape[1])
                    current_advantages = current_advantages[:, :min_len]
                    token_log_probs = token_log_probs[:, :min_len]
                    loss_mask = loss_mask[:, :min_len]
                    
                    loss = -(current_advantages * token_log_probs * loss_mask).sum() / (loss_mask.sum() + 1e-8)
                
                # I. Backward pass
                loss = loss / grad_accum_steps
                loss.backward()
                
                # J. Optimizer step
                if (step + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Log progress
                if (step + 1) % log_every == 0:
                    print(f"\n   Step {step+1}: Loss={loss.item()*grad_accum_steps:.4f}, Mean Reward={mean_r:.2f}")
                
                # Log memory
                if (step + 1) % memory_log_every == 0:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    print(f"   [Memory] {allocated:.2f}GB allocated")
                
                error_count = 0  # Reset error count on success
                total_steps += 1
                
            except Exception as e:
                error_count += 1
                print(f"\n⚠️  Error at step {step}: {e}")
                
                if error_count >= max_errors:
                    print(f"❌ Too many consecutive errors ({max_errors}). Stopping.")
                    return False
                
                # Continue to cleanup
            
            finally:
                # K. SAFE VRAM Cleanup (always runs)
                for var in [outputs, logits, loss, inputs, planning_mask]:
                    if var is not None:
                        del var
                gc.collect()
                torch.cuda.empty_cache()
    
    print(f"\n✅ Training completed! Total steps: {total_steps}")
    return True


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
Copy these cells into your notebook:

CELL 1 - Diagnostic (run first):
---------------------------------
# Paste the print_gpu_memory and test_single_sample functions above
# Then run:
test_single_sample(model, tokenizer, dataset_train, hicra)


CELL 2 - Training (if diagnostic passes):
------------------------------------------
# Paste run_improved_hicra_training function above
# Then run:
success = run_improved_hicra_training(
    model=model,
    tokenizer=tokenizer, 
    dataset_train=dataset_train,
    hicra=hicra,
    epochs=2,
    grad_accum_steps=4,
    max_new_tokens=100,   # Reduced from 300
    num_sequences=2,       # Reduced from 4
    log_every=5,
    memory_log_every=10
)
"""
