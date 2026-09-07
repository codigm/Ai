import os
import sys
import json
import ast
import torch

from tokenizer import BPETokenizer
from model import GPT
from generate import generate

PYTHON_KEYWORDS = ["def", "class", "import", "from", "return", "if", "elif", "else", "for", "while", "try", "except", "with", "as", "pass", "yield", "raise"]

PROMPTS = [
    "def calculate_fibonacci(n):",
    "class User:",
    "def binary_search(arr, target):",
    "import os\nimport sys\n",
    "def quicksort(arr):",
    "def parse_json(input_str):",
    "class ConfigParser:",
    "def __init__(self):",
    "for i in range(10):",
    "if __name__ == '__main__':"
]

def main():
    device = 'cpu'
    
    # Try to load the best model first (fallback to older phases if needed)
    checkpoint_paths = ["model_rlhf_final.pt", "model_sft_final.pt", "model_final.pt"]
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
            
    if not checkpoint_path:
        print("Error: Could not find any model checkpoint. Did training complete?")
        sys.exit(1)
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
        
    # Reconstruct the model
    config = checkpoint['config']
    model = GPT(**config)
    # Use strict=False because older checkpoints might have absolute position embeddings
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    
    tokenizer = BPETokenizer()
    tokenizer.load("tokenizer.json")
    
    results = []
    syntax_passes = 0
    keyword_passes = 0
    
    print(f"Evaluating {len(PROMPTS)} prompts...\n")
    
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            # Generate text
            idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            out_idx = generate(
                model, 
                idx, 
                max_new_tokens=100, 
                max_seq_len=config['max_seq_len'],
                temperature=0.8, 
                top_k=50,
                stop_tokens=[8194]
            )
            
            # Post-process to remove the <|end|> token if it was generated
            generated_idx = out_idx[0].tolist()
            if generated_idx and generated_idx[-1] == 8194:
                generated_idx = generated_idx[:-1]
                
            generated_code = tokenizer.decode(generated_idx)
            
            # Soft Metric Check
            has_keyword = any(kw in generated_code for kw in PYTHON_KEYWORDS)
            if has_keyword:
                keyword_passes += 1
                
            # Hard Metric Check: Python AST parse
            try:
                ast.parse(generated_code)
                is_valid_syntax = True
                stderr_output = ""
                syntax_passes += 1
            except SyntaxError as e:
                is_valid_syntax = False
                stderr_output = str(e)
                
            print(f"[{i+1}/{len(PROMPTS)}] Prompt: {repr(prompt)}")
            print(f"   Has Keywords: {'Yes' if has_keyword else 'No'}")
            print(f"   AST Check:    {'PASS' if is_valid_syntax else 'FAIL'}")
            if not is_valid_syntax and stderr_output:
                print(f"   Error:        {stderr_output}")
            print("-" * 40)
            
            results.append({
                "prompt": prompt,
                "generated_code": generated_code,
                "has_keyword": has_keyword,
                "is_valid_syntax": is_valid_syntax,
                "stderr": stderr_output
            })
            
    # Save to JSONL
    out_file = "eval_results.jsonl"
    with open(out_file, "w", encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"\nEvaluation Complete! Results saved to {out_file}")
    print("=== Final Scores ===")
    print(f"Soft Metric (Keyword Accuracy): {keyword_passes}/{len(PROMPTS)} ({(keyword_passes/len(PROMPTS))*100:.1f}%)")
    print(f"Hard Metric (Syntax Accuracy):  {syntax_passes}/{len(PROMPTS)} ({(syntax_passes/len(PROMPTS))*100:.1f}%)")

if __name__ == "__main__":
    main()
