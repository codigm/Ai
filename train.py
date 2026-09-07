import os
import time
import math
import torch
from torch.nn import functional as F

from tokenizer import BPETokenizer
from dataset import PolyglotDataset
from model import GPT

# =============================================================================
# 1. Hyperparameters & Configuration
# =============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4) # Explicitly set threads for stable CPU performance
torch.manual_seed(42)    # For reproducibility

# Training config
batch_size = 16 # Micro-batch size for GPU
accumulation_steps = 8 # Effective batch size = 16 * 8 = 128
seq_len = 256
max_iters = 5000 # Increased steps for full training
eval_interval = 250
eval_iters = 20
save_interval = 500

# Model config (Nano GPT)
config = {
    "vocab_size": 8192,
    "max_seq_len": seq_len,
    "d_model": 384,
    "n_heads": 6,
    "n_layers": 6,
    "dropout": 0.1
}

# Learning rate config
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 100
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

print("\nInitializing Datasets (This might take a moment if files aren't cached)...")
# dataset.py is currently hardcoded to limit files for speed, which is perfect for this test.
train_data = PolyglotDataset("corpus", tokenizer, split='train', val_ratio=0.1, seq_len=seq_len)
val_data = PolyglotDataset("corpus", tokenizer, split='val', val_ratio=0.1, seq_len=seq_len)

print(f"\nInitializing Model on {device}...")
model = GPT(**config)
model.to(device)

# =============================================================================
# 3. Setup Optimizer with Weight Decay Splitting
# =============================================================================
# Standard practice: Do not apply weight decay to LayerNorms and Biases
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
    # 1) Linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    # 2) If beyond max iterations, return min lr
    if it > max_iters:
        return min_lr
    # 3) Cosine decay down to min learning rate
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
    model.eval() # Switch to evaluation mode (turns off dropout)
    for split, dataset in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataset.get_batch(batch_size)
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train() # Switch back to training mode
    return out

def save_checkpoint(step, filepath):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'config': config,
    }
    torch.save(checkpoint, filepath)
    print(f"--> Saved checkpoint to {filepath}")

# =============================================================================
# 6. Main Training Loop
# =============================================================================
print("\n🚀 Starting Training Loop...\n")

# To track time per step
t0 = time.time()

import itertools

try:
    for step in itertools.count():
        
        # --- A. Validation Check ---
        if step % eval_interval == 0:
            losses = estimate_loss()
            train_ppl = math.exp(losses['train'])
            val_ppl = math.exp(losses['val'])
            print(f"Step {step:4d} | train loss {losses['train']:.4f} (ppl: {train_ppl:.2f}) | val loss {losses['val']:.4f} (ppl: {val_ppl:.2f})")
            
        # --- B. Save Checkpoint ---
        if step > 0 and step % save_interval == 0:
            save_checkpoint(step, "model_latest.pt")
            
        # --- C. Training Step ---
        # Fetch batch
        X, Y = train_data.get_batch(batch_size)
        X, Y = X.to(device), Y.to(device)
        
        # Set learning rate for this step
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Forward pass
        logits, loss = model(X, targets=Y)
        
        # We need the unscaled loss for logging
        raw_loss = loss.item()
        
        # Scale the loss to account for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass (accumulates gradients)
        loss.backward()
        
        # Only step the optimizer every `accumulation_steps` mini-batches
        if (step + 1) % accumulation_steps == 0 or (step + 1) == max_iters:
            # Gradient clipping (prevents exploding gradients)
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
            # If we print every 10 steps, dt is the time for 10 steps.
            # Let's print time per step.
            time_per_step = (dt / 10.0) * 1000 if step > 0 else 0
            
            # Note: norm might be unbound if step 0 doesn't hit the optimizer step,
            # but usually we log at step 0 which is just a random starting point. 
            # We'll safely get norm if it exists.
            norm_val = norm.item() if 'norm' in locals() else 0.0
            print(f"iter {step:4d} | loss {raw_loss:.4f} | lr {lr:.4e} | norm {norm_val:.4f} | time/step {time_per_step:.2f}ms")

    save_checkpoint(max_iters, "model_final.pt")

except KeyboardInterrupt:
    print(f"\n[!] Training interrupted at step {step} by user (Ctrl+C).")
    print("Saving current progress before exiting...")
    save_checkpoint(step, "model_interrupted.pt")
    print("✅ Progress saved. Safe to exit now.")

# =============================================================================
# 7. Qualitative Check (Generation)
# =============================================================================
print("\n--- Qualitative Check ---")
print("Generating 100 tokens from the model (expect gibberish syntax at 500 steps, but it should look somewhat code-like).")

model.eval()
context = "pub fn main() !void {\n"
idx = torch.tensor([tokenizer.encode(context)], dtype=torch.long, device=device)

# Simple autoregressive generation loop
with torch.no_grad():
    for _ in range(100):
        # crop context to max seq len
        idx_cond = idx[:, -seq_len:]
        
        # forward pass
        logits, _ = model(idx_cond)
        
        # focus only on the last time step
        logits = logits[:, -1, :]
        
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

generated_text = tokenizer.decode(idx[0].tolist())
print("\n[GENERATED OUTPUT]:\n")
print(generated_text)
print("\n--------------------------")
