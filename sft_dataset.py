import json
import torch
import random
from tokenizer import BPETokenizer

class ChatDataset:
    def __init__(self, jsonl_path, tokenizer, seq_len=256):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data = []
        
        # Load dataset
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                
                # Format with chat tokens
                prompt_text = f"<|user|>\n{item['prompt']}<|end|>\n<|assistant|>\n"
                response_text = f"{item['response']}<|end|>"
                
                prompt_ids = self.tokenizer.encode(prompt_text, add_special_token=False)
                response_ids = self.tokenizer.encode(response_text, add_special_token=False)
                
                full_ids = prompt_ids + response_ids
                
                # Truncate if too long
                if len(full_ids) > self.seq_len + 1:
                    full_ids = full_ids[:self.seq_len + 1]
                    
                x = full_ids[:-1]
                y = full_ids[1:]
                
                pad_id = self.tokenizer.special_tokens["<|endoftext|>"]
                x_pad_len = self.seq_len - len(x)
                
                if x_pad_len > 0:
                    x.extend([pad_id] * x_pad_len)
                    y.extend([pad_id] * x_pad_len)
                    
                x_tensor = torch.tensor(x, dtype=torch.long)
                y_tensor = torch.tensor(y, dtype=torch.long)
                
                # CRITICAL: Loss masking. 
                # We do not want to penalize the model for failing to predict the user's prompt.
                # full_ids = [P0, P1, P2, R0, R1] (where P is prompt, R is response)
                # x = [P0, P1, P2, R0]
                # y = [P1, P2, R0, R1]
                # y[2] is R0 (the first token of the assistant's response). 
                # We want gradients to flow for y[2] (R0) and y[3] (R1). 
                # We mask y[0] (P1) and y[1] (P2). 
                # The number of tokens to mask is len(prompt_ids) - 1.
                num_prompt_tokens_to_mask = len(prompt_ids) - 1
                y_tensor[:num_prompt_tokens_to_mask] = -100
                
                # Also mask the padding tokens
                if x_pad_len > 0:
                    y_tensor[-x_pad_len:] = -100
                    
                self.data.append((x_tensor, y_tensor))
                
        print(f"Loaded {len(self.data)} chat examples.")
        
    def get_batch(self, batch_size):
        # Sample random indices
        ix = torch.randint(0, len(self.data), (batch_size,))
        x_batch = torch.stack([self.data[i][0] for i in ix])
        y_batch = torch.stack([self.data[i][1] for i in ix])
        return x_batch, y_batch

if __name__ == "__main__":
    print("--- Testing Chat Data Pipeline ---")
    tokenizer = BPETokenizer()
    tokenizer.load("tokenizer.json")
    
    dataset = ChatDataset("chat_dataset.jsonl", tokenizer, seq_len=256)
    X, Y = dataset.get_batch(2)
    
    print("\nBatch shapes:")
    print(f"X: {list(X.shape)}")
    print(f"Y: {list(Y.shape)}")
    
    print("\nAnalyzing the first sequence in batch:")
    print("X[0]:")
    print(tokenizer.decode(X[0].tolist()))
    
    # Analyze where the masking ends
    y_list = Y[0].tolist()
    first_unmasked = next(i for i, val in enumerate(y_list) if val != -100)
    print(f"\nFirst unmasked target token in Y is at index {first_unmasked} (ID: {y_list[first_unmasked]})")
    print(f"This should correspond to the first token of the assistant's response.")
    print("Decoded unmasked portion of Y (What the model is actually trained to predict):")
    
    unmasked_y = [v for v in y_list if v != -100]
    print(tokenizer.decode(unmasked_y))
    print("\n✅ Verification complete.")
