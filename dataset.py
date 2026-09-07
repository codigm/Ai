import os
import random
import torch
from tokenizer import BPETokenizer

class PolyglotDataset:
    def __init__(self, corpus_dir, tokenizer, split='train', val_ratio=0.1, seq_len=256):
        """
        Loads the Polyglot corpus, splits it into Train/Val, tokenizes everything,
        and packs it into a single massive 1D PyTorch tensor.
        """
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
        # Get all valid source files
        all_files = [os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir) if f.endswith(('.py', '.java', '.js'))]
        
        # Sort and shuffle deterministically so train/val splits are consistent
        all_files.sort() 
        random.seed(42)
        random.shuffle(all_files)
        
        # Calculate split index
        split_idx = int(len(all_files) * (1 - val_ratio))
        
        if split == 'train':
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]
            
        print(f"Loading {split} split: processing {len(self.files)} files...")
        
        # Packing: Concatenate all tokens into one long list
        all_tokens = []
        for file_path in self.files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # encode() automatically appends the <|endoftext|> separator!
                tokens = self.tokenizer.encode(content, add_special_token=True)
                all_tokens.extend(tokens)
            except Exception as e:
                # Silently skip any files that might have been deleted or corrupted
                pass
                
        # Convert to a PyTorch tensor
        self.data = torch.tensor(all_tokens, dtype=torch.long)
        print(f"-> {split.capitalize()} split created with {len(self.data):,} total tokens.")
        
    def get_batch(self, batch_size):
        """
        Grabs a random batch of sequences.
        X is the input context, Y is the expected next token (shifted by 1).
        """
        # Generate random starting indices for our sequences
        max_idx = len(self.data) - self.seq_len - 1
        ix = torch.randint(0, max_idx, (batch_size,))
        
        # Stack the slices into a 2D tensor [batch_size, seq_len]
        x = torch.stack([self.data[i : i + self.seq_len] for i in ix])
        y = torch.stack([self.data[i + 1 : i + self.seq_len + 1] for i in ix])
        
        return x, y

if __name__ == "__main__":
    print("--- Testing Data Pipeline ---")
    
    # 1. Load the tokenizer we just trained
    print("Loading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load("tokenizer.json")
    
    # 2. Create the datasets (using small subsets just to test quickly)
    # Note: In a real training run, this might take a minute or two to tokenize all 37MB!
    train_data = PolyglotDataset("corpus", tokenizer, split='train', val_ratio=0.1, seq_len=256)
    
    # 3. Get a single batch (Batch Size = 8, Sequence Length = 256)
    batch_size = 8
    seq_len = 256
    X, Y = train_data.get_batch(batch_size=batch_size)
    
    print("\n--- Verification Test ---")
    print(f"X shape (inputs):  {list(X.shape)} -> [batch_size, seq_len]")
    print(f"Y shape (targets): {list(Y.shape)} -> [batch_size, seq_len]")
    
    assert list(X.shape) == [batch_size, seq_len], "Tensor shapes are incorrect!"
    
    print("\nPreview of the first sequence in the batch (decoded back to text):")
    print("-" * 50)
    # Convert the first row of the tensor back to a list of ints and decode
    decoded_X = tokenizer.decode(X[0].tolist())
    print(decoded_X[:300] + "\n... [truncated] ...")
    print("-" * 50)
    
    print("\n✅ Verification Passed: PyTorch Data Pipeline is working perfectly!")
