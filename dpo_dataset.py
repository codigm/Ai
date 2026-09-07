import json
import torch
import random
from tokenizer import BPETokenizer

class DPODataset:
    def __init__(self, jsonl_path, tokenizer, seq_len=256):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                
                # Format with chat tokens
                prompt_text = f"<|user|>\n{item['prompt']}<|end|>\n<|assistant|>\n"
                chosen_text = f"{item['chosen']}<|end|>"
                rejected_text = f"{item['rejected']}<|end|>"
                
                prompt_ids = self.tokenizer.encode(prompt_text, add_special_token=False)
                chosen_ids = self.tokenizer.encode(chosen_text, add_special_token=False)
                rejected_ids = self.tokenizer.encode(rejected_text, add_special_token=False)
                
                x_chosen, y_chosen = self._pad_and_mask(prompt_ids, chosen_ids)
                x_rejected, y_rejected = self._pad_and_mask(prompt_ids, rejected_ids)
                
                self.data.append((x_chosen, y_chosen, x_rejected, y_rejected))
                
        print(f"Loaded {len(self.data)} DPO examples.")
        
    def _pad_and_mask(self, prompt_ids, response_ids):
        full_ids = prompt_ids + response_ids
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
        
        num_prompt_tokens_to_mask = len(prompt_ids) - 1
        y_tensor[:num_prompt_tokens_to_mask] = -100
        
        if x_pad_len > 0:
            y_tensor[-x_pad_len:] = -100
            
        return x_tensor, y_tensor
        
    def get_batch(self, batch_size):
        ix = torch.randint(0, len(self.data), (batch_size,))
        x_chosen = torch.stack([self.data[i][0] for i in ix])
        y_chosen = torch.stack([self.data[i][1] for i in ix])
        x_rejected = torch.stack([self.data[i][2] for i in ix])
        y_rejected = torch.stack([self.data[i][3] for i in ix])
        return x_chosen, y_chosen, x_rejected, y_rejected

if __name__ == "__main__":
    print("--- Testing DPO Data Pipeline ---")
    tokenizer = BPETokenizer()
    tokenizer.load("tokenizer.json")
    
    dataset = DPODataset("dpo_dataset.jsonl", tokenizer, seq_len=256)
    x_c, y_c, x_r, y_r = dataset.get_batch(2)
    
    print("\nBatch shapes:")
    print(f"x_chosen: {list(x_c.shape)}")
    print(f"y_chosen: {list(y_c.shape)}")
    print(f"x_rejected: {list(x_r.shape)}")
    print(f"y_rejected: {list(y_r.shape)}")
    
    y_list = y_c[0].tolist()
    first_unmasked = next(i for i, val in enumerate(y_list) if val != -100)
    print(f"\nFirst unmasked target token in y_chosen is at index {first_unmasked}")
    
    unmasked_y = [v for v in y_list if v != -100]
    print("\nDecoded unmasked portion of y_chosen:")
    print(tokenizer.decode(unmasked_y))
    print("\n✅ Verification complete.")
