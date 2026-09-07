import os
import json
import random
import urllib.request

# Configuration
# Set MAX_EXAMPLES to None to use the full 52,000 dataset (for Kaggle/Colab)
# Set it to 1000 for a fast local test on your PC.
MAX_EXAMPLES = None 
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
OUTPUT_FILE = "chat_dataset.jsonl"

def main():
    print("Downloading Stanford Alpaca dataset...")
    # Download the JSON file
    req = urllib.request.Request(ALPACA_URL, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return

    print(f"Successfully downloaded {len(data)} examples.")
    
    # Shuffle the data so we get a good random mix if we truncate it
    random.seed(42)
    random.shuffle(data)
    
    if MAX_EXAMPLES is not None:
        data = data[:MAX_EXAMPLES]
        print(f"Truncated to {MAX_EXAMPLES} examples for fast training.")
        
    print(f"Formatting and saving to {OUTPUT_FILE}...")
    
    # Format to our chat dataset structure
    formatted_data = []
    for item in data:
        # Alpaca has 'instruction', 'input', and 'output'
        # We combine instruction and input for the prompt
        prompt = item['instruction']
        if item.get('input', '').strip():
            prompt += "\n\n" + item['input']
            
        formatted_item = {
            "prompt": prompt.strip(),
            "response": item['output'].strip()
        }
        formatted_data.append(formatted_item)
        
    # Save as JSONL (JSON Lines)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + '\n')
            
    print(f"Done! Saved {len(formatted_data)} conversations to {OUTPUT_FILE}.")
    print("You can now upload this project to Kaggle and run `uv run python sft_train.py`")

if __name__ == "__main__":
    main()
