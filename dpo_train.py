import os
import time
import torch
from torch.nn import functional as F

from tokenizer import BPETokenizer
from dpo_dataset import DPODataset
from model import GPT

# =============================================================================
# 1. Hyperparameters & Configuration
# =============================================================================
device = 'cpu'
torch.set_num_threads(4)
torch.manual_seed(42)

# DPO config
batch_size = 4
seq_len = 256
max_iters = 100
eval_interval = 20
save_interval = 50
beta = 0.1 # KL penalty strength. 0.1 allows some drift but stays close to ref model

# Learning rate config for DPO (Very low!)
learning_rate = 2e-6
weight_decay = 0.0
grad_clip = 1.0

# =============================================================================
# 2. Helper Functions (Moved up for safe importing)
# =============================================================================
def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    """
    Computes the sum of per-token log probabilities for the non-masked tokens.
    logits: (B, T, V)
    labels: (B, T) with -100 for masked tokens
    """
    loss_mask = (labels != -100)
    
    # Temporarily replace -100 with 0 so gather doesn't crash
    labels_cloned = labels.clone()
    labels_cloned[labels_cloned == -100] = 0
    
    # Get log probabilities of all tokens
    per_token_logps = F.log_softmax(logits, dim=-1)
    
    # Gather the logprobs of the actual target tokens
    per_token_logps = torch.gather(per_token_logps, dim=2, index=labels_cloned.unsqueeze(2)).squeeze(2)
    
    # Zero out masked tokens
    per_token_logps = per_token_logps * loss_mask
    
    # SUM over the sequence (as requested)
    return per_token_logps.sum(dim=-1)


def main():
    # =============================================================================
    # 3. Setup Data & Models
    # =============================================================================
    print("Loading Tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load("tokenizer.json")

    print("\nInitializing DPO Dataset...")
    train_data = DPODataset("dpo_dataset.jsonl", tokenizer, seq_len=seq_len)
    dataset_size = len(train_data.data)

    print(f"\nInitializing Models from SFT checkpoint on {device}...")
    checkpoint_path = "model_sft_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # POLICY MODEL (The one we are training)
    policy_model = GPT(**config)
    policy_model.load_state_dict(checkpoint['model'], strict=True)
    policy_model.to(device)

    # REFERENCE MODEL (The frozen baseline)
    ref_model = GPT(**config)
    ref_model.load_state_dict(checkpoint['model'], strict=True)
    ref_model.to(device)

    # Freeze reference model
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval() # Ensure dropout is disabled so logprobs are deterministic

    # =============================================================================
    # 4. Setup Optimizer
    # =============================================================================
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def save_checkpoint(step, filepath):
        save_dict = {
            'model': policy_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'config': config,
        }
        torch.save(save_dict, filepath)
        print(f"--> Saved checkpoint to {filepath}")

    # =============================================================================
    # 5. Automated Verification / Sanity Checks
    # =============================================================================
    print("\n--- Running Sanity Checks ---")
    # Check 1: get_batch_logps math
    dummy_logits = torch.tensor([[[2.0, 1.0, 0.1], [0.5, 2.5, 0.1], [1.0, 1.0, 1.0]]]) # (1, 3, 3)
    dummy_labels = torch.tensor([[0, 1, -100]]) # (1, 3) - masked last token
    logps = get_batch_logps(dummy_logits, dummy_labels)
    expected = -0.6211
    assert abs(logps.item() - expected) < 1e-3, f"Math error in get_batch_logps: {logps.item()} != {expected}"
    print("✅ get_batch_logps computation matches hand-calculated expected values.")

    # Check 2: Reference model determinism
    test_x, _, _, _ = train_data.get_batch(1)
    with torch.no_grad():
        logits1, _ = ref_model(test_x)
        logits2, _ = ref_model(test_x)
    assert torch.allclose(logits1, logits2), "Reference model is not deterministic! Check if .eval() is set properly."
    print("✅ Reference model is fully deterministic (dropout disabled).")

    # =============================================================================
    # 6. Main DPO Loop
    # =============================================================================
    print("\n🚀 Starting Direct Preference Optimization (DPO) Loop...\n")

    t0 = time.time()

    try:
        for step in range(max_iters + 1):
            
            # --- A. Save Checkpoint ---
            if step > 0 and step % save_interval == 0:
                save_checkpoint(step, f"model_rlhf_step_{step}.pt")
                
            if step == max_iters:
                break
                
            # --- B. Fetch Batch ---
            x_chosen, y_chosen, x_rejected, y_rejected = train_data.get_batch(batch_size)
            x_chosen, y_chosen = x_chosen.to(device), y_chosen.to(device)
            x_rejected, y_rejected = x_rejected.to(device), y_rejected.to(device)
            
            # --- C. Reference Model Forward Pass ---
            with torch.no_grad():
                ref_chosen_logits, _ = ref_model(x_chosen)
                ref_rejected_logits, _ = ref_model(x_rejected)
                
                ref_chosen_logps = get_batch_logps(ref_chosen_logits, y_chosen)
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, y_rejected)
                
            # --- D. Policy Model Forward Pass ---
            policy_chosen_logits, _ = policy_model(x_chosen)
            policy_rejected_logits, _ = policy_model(x_rejected)
            
            policy_chosen_logps = get_batch_logps(policy_chosen_logits, y_chosen)
            policy_rejected_logps = get_batch_logps(policy_rejected_logits, y_rejected)
            
            # --- E. DPO Loss Calculation ---
            policy_logratio = policy_chosen_logps - policy_rejected_logps
            ref_logratio = ref_chosen_logps - ref_rejected_logps
            
            # DPO Loss: -log(sigmoid(beta * (policy_logratio - ref_logratio)))
            losses = -F.logsigmoid(beta * (policy_logratio - ref_logratio))
            loss = losses.mean()
            
            # Backward pass
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # --- F. Logging ---
            if step % 10 == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                time_per_step = (dt / 10.0) * 1000 if step > 0 else 0
                
                # Log mean logratios to see if policy is favoring chosen
                p_ratio = policy_logratio.mean().item()
                r_ratio = ref_logratio.mean().item()
                
                print(f"iter {step:4d} | loss {loss.item():.4f} | p_ratio {p_ratio:7.2f} | r_ratio {r_ratio:7.2f} | time {time_per_step:.2f}ms")

        save_checkpoint(max_iters, "model_rlhf_final.pt")
        print("\n🎉 DPO RLHF Complete!")
    except KeyboardInterrupt:
        print(f"\n[!] Training interrupted at step {step} by user (Ctrl+C).")
        print("Saving current progress before exiting...")
        save_checkpoint(step, "model_interrupted.pt")
        print("✅ Progress saved. Safe to exit now.")


if __name__ == "__main__":
    main()
