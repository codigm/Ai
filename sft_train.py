import os
import time
import math
import torch
from torch.nn import functional as F

from tokenizer import BPETokenizer
from sft_dataset import ChatDataset
from model import GPT

# =============================================================================
# 1. Hyperparameters & Configuration
# =============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4) # Explicitly set threads for stable CPU performance
torch.manual_seed(42)    # For reproducibility

# SFT config
batch_size = 16
accumulation_steps = 8 # Effective batch size = 128
seq_len = 256
max_iters = 20000      # Enough iterations to loop through the 52k dataset a few times
eval_interval = 500    # How often to check loss
save_interval = 5000   # Save every 5000 steps to avoid spamming Kaggle disk

# Learning rate config for SFT
learning_rate = 5e-5   # Much smaller than pre-training!
min_lr = 5e-6
warmup_iters = 10      # Short warmup
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0

# =============================================================================
# 2. Setup Data & Model
# =============================================================================
print("Loading Tokenizer...")
tokenizer = BPETokenizer()
tokenizer.load("tokenizer.json")

print("\nInitializing Chat Dataset...")
train_data = ChatDataset("chat_dataset.jsonl", tokenizer, seq_len=seq_len)
dataset_size = len(train_data.data)

print(f"\nInitializing Model from checkpoint on {device}...")
checkpoint_path = "model_chat_base.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

config = checkpoint['config']
# IMPORTANT: Reconstruct model exactly using the saved config
model = GPT(**config)

# Load the weights from the checkpoint
# Using strict=True to ensure we aren't silently missing the chat embeddings
model.load_state_dict(checkpoint['model'], strict=True)
model.to(device)

# =============================================================================
# 3. Setup Optimizer with Weight Decay Splitting
# =============================================================================
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2))

# =============================================================================
# 4. Learning Rate Scheduler (Cosine with Warmup)
# =============================================================================
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# =============================================================================
# 5. Helper Functions
# =============================================================================
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    losses = torch.zeros(10)
    for k in range(10):
        X, Y = train_data.get_batch(batch_size)
        X, Y = X.to(device), Y.to(device)
        
        # We compute loss manually to explicitly use ignore_index=-100
        logits, _ = model(X)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), Y.view(B*T), ignore_index=-100)
        
        losses[k] = loss.item()
    out['train'] = losses.mean().item()
    model.train()
    return out

def save_checkpoint(step, filepath):
    # Maintain exact same format as pre-training
    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'config': config,
    }
    torch.save(save_dict, filepath)
    print(f"--> Saved checkpoint to {filepath}")

# =============================================================================
# 6. Main SFT Loop
# =============================================================================
print("\n🚀 Starting Supervised Fine-Tuning Loop...\n")

t0 = time.time()

try:
    for step in range(max_iters + 1):
        
        # --- A. Validation Check ---
        if step % eval_interval == 0 or step == max_iters:
            losses = estimate_loss()
            # Calculate epoch (how many times we've seen the whole dataset)
            epoch = (step * batch_size) / dataset_size
            print(f"Step {step:4d} | Epoch {epoch:.2f} | train loss {losses['train']:.4f}")
            
        # --- B. Save Checkpoint ---
        if step > 0 and step % save_interval == 0:
            save_checkpoint(step, f"model_sft_step_{step}.pt")
            
        if step == max_iters:
            break
            
        # --- C. Training Step ---
        X, Y = train_data.get_batch(batch_size)
        X, Y = X.to(device), Y.to(device)
        
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Forward pass
        logits, _ = model(X)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), Y.view(B*T), ignore_index=-100)
        
        raw_loss = loss.item()
        
        # Scale for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Only step the optimizer every `accumulation_steps` mini-batches
        if (step + 1) % accumulation_steps == 0 or (step + 1) == max_iters:
            # Gradient clipping
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            # Optimizer step
            optimizer.step()
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
        
        # --- D. Logging ---
        if step % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            time_per_step = (dt / 10.0) * 1000 if step > 0 else 0
            norm_val = norm.item() if 'norm' in locals() else 0.0
            print(f"iter {step:4d} | loss {raw_loss:.4f} | lr {lr:.4e} | norm {norm_val:.4f} | time/step {time_per_step:.2f}ms")

    save_checkpoint(max_iters, "model_sft_final.pt")
    print("\n🎉 Supervised Fine-Tuning Complete!")
except KeyboardInterrupt:
    print(f"\n[!] Training interrupted at step {step} by user (Ctrl+C).")
    print("Saving current progress before exiting...")
    save_checkpoint(step, "model_interrupted.pt")
    print("✅ Progress saved. Safe to exit now.")
