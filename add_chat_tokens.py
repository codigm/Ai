import json

def main():
    print("Adding chat special tokens to tokenizer.json...")
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    special = data.get("special_tokens", {})
    # 256 is already <|endoftext|>
    # The normal vocabulary goes up to 8191.
    # We add our 3 new tokens starting at 8192.
    special["<|user|>"] = 8192
    special["<|assistant|>"] = 8193
    special["<|end|>"] = 8194
    
    data["special_tokens"] = special
    
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)
        
    print("Success! Special tokens added:")
    print(special)

if __name__ == "__main__":
    main()
