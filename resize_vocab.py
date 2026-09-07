import torch
import torch.nn as nn
from model import GPT

def main():
    device = 'cpu'
    checkpoint_path = 'model_final.pt'
    output_path = 'model_chat_base.pt'
    
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    old_vocab_size = config['vocab_size']
    new_vocab_size = 8195
    print(f"Current vocab size: {old_vocab_size}, New vocab size: {new_vocab_size}")
    
    # Update config
    config['vocab_size'] = new_vocab_size
    
    # We instantiate the model with the old config first so we can load the state dict cleanly
    old_config = config.copy()
    old_config['vocab_size'] = old_vocab_size
    
    model = GPT(**old_config)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    # 1. Get the original weights
    old_weight = model.token_embedding.weight.data # Shape: (8192, d_model)
    d_model = old_weight.shape[1]
    
    # 2. Create the new weights
    new_weight = torch.empty((new_vocab_size, d_model))
    
    # 3. Copy the old weights
    new_weight[:old_vocab_size, :] = old_weight
    
    # 4. Initialize the new embeddings (for tokens 8192, 8193, 8194)
    # Using normal distribution (mean=0.0, std=0.02) to match _init_weights in model.py
    nn.init.normal_(new_weight[old_vocab_size:, :], mean=0.0, std=0.02)
    
    # 5. Create new layers with the new size
    new_embedding = nn.Embedding(new_vocab_size, d_model)
    new_embedding.weight.data = new_weight
    
    # Because out_features changes, we need a new Linear layer too
    new_lm_head = nn.Linear(d_model, new_vocab_size, bias=False)
    
    # **CRITICAL: Weight Tying**
    # We tie the lm_head weights to be the EXACT same tensor as the token_embedding weights.
    new_lm_head.weight = new_embedding.weight
    
    # Replace the old layers in the model
    model.token_embedding = new_embedding
    model.lm_head = new_lm_head
    
    # 6. Verify weight tying survived
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr(), "Weight tying failed! Tensors are completely independent."
    print("✅ Weight tying assertion passed.")
    print(f"New token_embedding weight shape: {model.token_embedding.weight.shape}")
    
    # Save the new checkpoint
    print(f"Saving resized model to {output_path}...")
    torch.save({
        'model': model.state_dict(),
        'config': config
    }, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
