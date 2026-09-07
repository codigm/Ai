import os
import json

def get_stats(ids):
    """Counts the frequency of adjacent pairs of integers."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """Replaces all occurrences of `pair` in `ids` with the new token `idx`."""
    newids = []
    i = 0
    while i < len(ids):
        # If we find the pair, replace it with the new token
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class BPETokenizer:
    def __init__(self):
        self.merges = {} # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # Special tokens
        self.special_tokens = {"<|endoftext|>": 256}
        self.vocab[256] = b"<|endoftext|>"
        
    def train(self, corpus_dir, vocab_size, max_train_bytes=3 * 1024 * 1024):
        """
        Trains the BPE tokenizer on raw bytes.
        To avoid taking hours in pure Python, we sample a chunk of the corpus (e.g., 3MB)
        which is more than enough to learn common Zig idioms.
        """
        print(f"Reading data from {corpus_dir}...")
        raw_bytes = bytearray()
        for file_name in os.listdir(corpus_dir):
            if file_name.endswith('.zig'):
                file_path = os.path.join(corpus_dir, file_name)
                with open(file_path, 'rb') as f:
                    content = f.read()
                    raw_bytes.extend(content)
                    # Stop reading if we hit our training limit
                    if len(raw_bytes) >= max_train_bytes:
                        break
                        
        actual_bytes = len(raw_bytes)
        print(f"Training tokenizer on {actual_bytes / (1024*1024):.2f} MB of data...")
        print("This might take a few minutes in pure Python. Grab a coffee! ☕")
        
        ids = list(raw_bytes)
        
        # We start adding new tokens at index 257 (0-255 are raw bytes, 256 is special)
        num_merges = vocab_size - 257 
        
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
                
            # Find the pair with the highest frequency
            best = max(stats, key=stats.get)
            idx = 257 + i
            
            if (i+1) % 100 == 0:
                # Try to get a string representation for debugging
                try:
                    p0 = self.vocab[best[0]].decode('utf-8', errors='replace')
                    p1 = self.vocab[best[1]].decode('utf-8', errors='replace')
                    readable = repr(p0 + p1)
                except:
                    readable = str(best)
                print(f"Merge {i+1}/{num_merges}: {best} -> {idx} {readable} (count: {stats[best]})")
                
            self.merges[best] = idx
            self.vocab[idx] = self.vocab[best[0]] + self.vocab[best[1]]
            ids = merge(ids, best, idx)
            
        print("Training complete.")
        
    def encode(self, text, add_special_token=False):
        """Converts a string of text into a list of token IDs, handling special tokens."""
        import re
        # Find all special tokens
        special_strs = list(self.special_tokens.keys())
        # Sort them by length descending to match longest first
        special_strs.sort(key=len, reverse=True)
        # Create a regex pattern to split the text
        pattern = '(' + '|'.join(map(re.escape, special_strs)) + ')'
        parts = re.split(pattern, text)
        
        ids = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                b = part.encode('utf-8')
                part_ids = list(b)
                
                while len(part_ids) >= 2:
                    stats = get_stats(part_ids)
                    # We must apply merges in the exact order they were learned!
                    pair = min(stats.keys(), key=lambda p: self.merges.get(p, float('inf')))
                    
                    if pair not in self.merges:
                        break
                    
                    idx = self.merges[pair]
                    part_ids = merge(part_ids, pair, idx)
                ids.extend(part_ids)
                
        if add_special_token:
            ids.append(self.special_tokens["<|endoftext|>"])
            
        return ids
        
    def decode(self, ids):
        """Converts a list of token IDs back into a string."""
        b = bytearray()
        special_idx_to_str = {v: k for k, v in self.special_tokens.items()}
        decoded_parts = []
        
        for idx in ids:
            if idx in special_idx_to_str:
                if len(b) > 0:
                    decoded_parts.append(b.decode('utf-8', errors='replace'))
                    b = bytearray()
                decoded_parts.append(special_idx_to_str[idx])
            elif idx in self.vocab:
                b.extend(self.vocab[idx])
                
        if len(b) > 0:
            decoded_parts.append(b.decode('utf-8', errors='replace'))
            
        return "".join(decoded_parts)
        
    def save(self, filepath):
        """Saves the learned merges and vocabulary to a JSON file."""
        # JSON dictionary keys must be strings
        merges_str = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        data = {
            "merges": merges_str,
            "special_tokens": self.special_tokens
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f)
            
    def load(self, filepath):
        """Loads a previously saved tokenizer from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.merges = {}
        for k, v in data["merges"].items():
            p1, p2 = map(int, k.split(','))
            self.merges[(p1, p2)] = v
            
        self.special_tokens = data["special_tokens"]
        
        # Rebuild vocabulary
        self.vocab = {i: bytes([i]) for i in range(256)}
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode('utf-8')
            
        # Merges must be processed in order of their token ID to build the vocab correctly
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for (p1, p2), idx in sorted_merges:
            self.vocab[idx] = self.vocab[p1] + self.vocab[p2]

if __name__ == "__main__":
    # Target vocabulary size: 8192
    tokenizer = BPETokenizer()
    
    # We train on a 3MB subset of the corpus so it runs in a reasonable time in pure Python
    tokenizer.train("corpus", vocab_size=8192, max_train_bytes=3 * 1024 * 1024)
    tokenizer.save("tokenizer.json")
    print("\nTokenizer saved to tokenizer.json")
    
    # Run the automated verification test
    print("\n--- Verifying Tokenizer ---")
    test_str = "pub fn main() !void {\n    std.debug.print(\"Hello world!\", .{});\n}"
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    
    print("Original text:\n" + test_str)
    print(f"\nEncoded tokens (length {len(encoded)}):\n", encoded)
    print("\nDecoded text:\n" + decoded)
    
    assert test_str == decoded, "ERROR: Lossless reconstruction failed!"
    print("\n✅ Verification Passed: decode(encode(text)) == text")
