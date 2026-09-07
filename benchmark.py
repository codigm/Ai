import os
import sys
import math
import torch
import torch.nn.functional as F
from tokenizer import BPETokenizer
from model import GPT
from sft_dataset import ChatDataset
from torch.utils.data import DataLoader

def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            logits, _ = model(x)
            
            # Reshape for cross_entropy: (B*T, V)
            logits = logits.view(-1, logits.size(-1))
            targets = y.view(-1)
            
            # Calculate sum of losses for the batch
            # ignore_index=-100 ignores the padded tokens and the user prompt,
            # which is exactly what ChatDataset sets them to!
            loss = F.cross_entropy(logits, targets, reduction='sum', ignore_index=-100)
            
            # Count how many valid tokens were predicted
            valid_tokens = (targets != -100).sum().item()
            
            total_loss += loss.item()
            total_tokens += valid_tokens
            
    # Perplexity = exp(average NLL)
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

def main():
    # Use GPU if available to speed up evaluation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading tokenizer and dataset...")
    tokenizer = BPETokenizer()
    tokenizer.load("tokenizer.json")
    
    # We evaluate perplexity on the chat dataset format.
    # This measures how well the model predicts assistant responses!
    chat_dataset = ChatDataset("chat_dataset.jsonl", tokenizer)
    
    # Wrap the custom ChatDataset list so PyTorch DataLoader can use it
    class TorchDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
            
    dataset = TorchDataset(chat_dataset.data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    phases = [
        ("Base (Pre-trained)", "model_final.pt"),
        ("After SFT", "model_sft_final.pt"),
        ("After DPO", "model_rlhf_final.pt")
    ]
    
    base_ppl = None
    
    print(f"\nBenchmarking on {device.upper()}...")
    print("-" * 65)
    print(f"{'Phase':<25} | {'Perplexity':<15} | {'Δ from Base'}")
    print("-" * 65)
    
    for phase_name, path in phases:
        if not os.path.exists(path):
            print(f"{phase_name:<25} | {'N/A (Missing)':<15} | -")
            continue
            
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = GPT(**config)
        # strict=False in case older checkpoints used absolute positional embeddings
        model.load_state_dict(checkpoint['model'], strict=False)
        
        # Handle vocab size mismatch: the base model was trained with 8192 tokens,
        # but the chat dataset uses 3 extra special tokens (<|user|>, <|assistant|>, <|end|>).
        # We resize the embedding/output layers so the forward pass doesn't crash.
        tokenizer_vocab_size = len(tokenizer.vocab) + len(tokenizer.special_tokens)
        model_vocab_size = config['vocab_size']
        
        if model_vocab_size < tokenizer_vocab_size:
            import torch.nn as nn
            new_size = tokenizer_vocab_size
            
            # Resize token embedding
            old_emb = model.token_embedding
            new_emb = nn.Embedding(new_size, config['d_model'])
            new_emb.weight.data[:model_vocab_size] = old_emb.weight.data
            # Initialize new token embeddings with small random values
            nn.init.normal_(new_emb.weight.data[model_vocab_size:], mean=0.0, std=0.02)
            model.token_embedding = new_emb
            
            # Resize lm_head (output projection)
            old_head = model.lm_head
            new_head = nn.Linear(config['d_model'], new_size, bias=False)
            new_head.weight.data[:model_vocab_size] = old_head.weight.data
            nn.init.normal_(new_head.weight.data[model_vocab_size:], mean=0.0, std=0.02)
            model.lm_head = new_head
        
        model.to(device)
        
        ppl = compute_perplexity(model, dataloader, device)
        
        if base_ppl is None:
            base_ppl = ppl
            delta = "—"
        else:
            pct_change = ((ppl - base_ppl) / base_ppl) * 100
            sign = "+" if pct_change > 0 else ""
            delta = f"{sign}{pct_change:.1f}%"
            
        print(f"{phase_name:<25} | {ppl:<15.2f} | {delta}")
        
    print("-" * 65)

if __name__ == "__main__":
    main()
