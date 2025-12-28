# Cell 8: HICRA Manual Training Loop
print("🚀 Starting HICRA Manual Training...")

# Create a simple iterator
pbar = tqdm(enumerate(dataset_train), total=min(MAX_STEPS, len(dataset_train)))

for step, batch in pbar:
    if step >= MAX_STEPS:
        break
    
    try:
        # Unsloth/GRPO typically handles formatting, here we do it manually
        prompt_text = batch['prompt'][-1]['content']  # Get the user query
        target_answer = batch['answer']

        # 1. GENERATE (Rollout)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.8,
                num_return_sequences=NUM_GENERATIONS,  # Group Size (G)
                pad_token_id=tokenizer.pad_token_id
            )

        # 2. SCORE (Reward Calculation)
        completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        rewards = []
        for comp in completions:
            # Simple correctness + format check
            r = 1.0 if str(target_answer) in comp else -1.0
            rewards.append(r)

        rewards = torch.tensor(rewards, device="cuda", dtype=torch.float32)

        # 3. COMPUTE ADVANTAGE (GRPO Formula)
        # A = (r - mean) / std
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        # 4. APPLY HICRA (This is where the magic happens)
        # Identify which outputs used planning
        planning_mask = hicra.identify_planning_mask(outputs, tokenizer)

        # Modify the advantages based on the mask
        # If they planned but failed (Adv < 0), this brings Adv closer to 0 (Dampens penalty)
        # If they planned and succeeded (Adv > 0), this increases Adv (Amplifies reward)
        final_advantages = hicra.compute_hicra_advantages(advantages, planning_mask)

        # 5. COMPUTE LOSS & BACKPROP
        # We re-run the forward pass to get gradients
        logits = model(outputs).logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = outputs[..., 1:].contiguous()

        # Cross Entropy (Reduction=None to keep token-level loss)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_loss = token_loss.view(shift_labels.size())

        # Weight the loss by our HICRA Advantages
        weighted_loss = token_loss * final_advantages[:, :-1]  # Align shapes
        loss = weighted_loss.mean()

        # Optimization
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"Loss: {loss.item():.4f} | R: {mean_r:.2f}")

        # Clean Memory
        del inputs, outputs, logits, loss
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"⚠️ Step {step} error: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        continue

print("✅ HICRA Training Complete!")
