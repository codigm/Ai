import torch
from torch.nn import functional as F
import sys

from tokenizer import BPETokenizer
from model import GPT

def generate(model, idx, max_new_tokens, max_seq_len, temperature=1.0, top_k=None, stop_tokens=None):
    """
    Takes a conditioning sequence of indices idx (LongTensor of shape (b,t)) and completes
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Uses KV Cache for O(n) generation speed.
    """
    if stop_tokens is None:
        stop_tokens = []
        
    past_key_values = None

    for _ in range(max_new_tokens):
        # 1. Context Selection
        if past_key_values is None:
            # First pass: process the entire context prompt
            idx_cond = idx[:, -max_seq_len:]
        else:
            # Subsequent passes: only process the single newest token!
            idx_cond = idx[:, -1:]
            
        # 2. Forward pass with KV caching
        logits, _, past_key_values = model(idx_cond, use_cache=True, past_key_values=past_key_values)
        
        # 3. Focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        
        # 4. Temperature Scaling
        logits = logits / temperature
        
        # 5. Top-K Masking
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # 6. Softmax
        probs = F.softmax(logits, dim=-1)
        
        # 7. Multinomial Sampling
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 8. Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
        # 9. Stop token checking
        if idx_next.item() in stop_tokens:
            break
        
    return idx

def generate_stream(model, idx, max_new_tokens, max_seq_len, temperature=1.0, top_k=None, stop_tokens=None):
    """
    Generator that yields the next token at each step for streaming.
    Uses KV Cache for lightning-fast SSE streaming.
    """
    if stop_tokens is None:
        stop_tokens = []
        
    past_key_values = None
        
    for _ in range(max_new_tokens):
        if past_key_values is None:
            idx_cond = idx[:, -max_seq_len:]
        else:
            idx_cond = idx[:, -1:]
            
        logits, _, past_key_values = model(idx_cond, use_cache=True, past_key_values=past_key_values)
        
        logits = logits[:, -1, :] 
        logits = logits / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        idx = torch.cat((idx, idx_next), dim=1)
        
        yield idx_next.item()
        
        if idx_next.item() in stop_tokens:
            break

if __name__ == "__main__":
    device = 'cpu'
    checkpoint_path = "model_final.pt"
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        print("Error: Could not find model_final.pt. Did Phase 3 training complete?")
        sys.exit(1)
        
    # Reconstruct the exact model architecture using the saved config
    config = checkpoint['config']
    model = GPT(**config)
    
    # Load the weights (Use strict=False to bypass RoPE architecture changes)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    
    # CRITICAL: Switch to evaluation mode to disable dropout.
    # Leaving dropout on during inference injects random noise into the outputs!
    model.eval()
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load("tokenizer.json")
    
    # Prepare the prompt
    prompt = "def calculate_fibonacci("
    print(f"\nPrompt: '{prompt}'\n")
    
    # Encode prompt to tensor
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    max_new_tokens = 150
    
    print("==================================================")
    print("Test 1: Low Temperature (T=0.1) - Safe/Deterministic")
    print("*(Note: Expect to see repetition loops here due to the tiny model size)*")
    print("==================================================")
    
    # CRITICAL: Wrap generation in torch.no_grad() to save memory and compute
    with torch.no_grad():
        out_idx = generate(
            model, 
            idx, 
            max_new_tokens=max_new_tokens, 
            max_seq_len=config['max_seq_len'],
            temperature=0.1, 
            top_k=50
        )
    print(tokenizer.decode(out_idx[0].tolist()))
    
    print("\n==================================================")
    print("Test 2: High Temperature (T=1.5) - Creative/Chaotic")
    print("==================================================")
    
    with torch.no_grad():
        out_idx = generate(
            model, 
            idx, 
            max_new_tokens=max_new_tokens, 
            max_seq_len=config['max_seq_len'],
            temperature=1.5, 
            top_k=50
        )
    print(tokenizer.decode(out_idx[0].tolist()))
    print("\n==================================================")
