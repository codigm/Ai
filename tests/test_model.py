import pytest
import torch
from model import GPT

@pytest.fixture
def tiny_config():
    return {
        "vocab_size": 256,
        "max_seq_len": 64,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.0
    }

def test_gpt_initialization(tiny_config):
    model = GPT(**tiny_config)
    assert model is not None
    assert model.get_num_params() > 0

def test_gpt_forward_pass_shape(tiny_config):
    model = GPT(**tiny_config)
    model.eval()
    
    batch_size = 2
    seq_len = 16
    
    # Create dummy input tokens
    dummy_x = torch.randint(0, tiny_config["vocab_size"], (batch_size, seq_len))
    
    with torch.no_grad():
        logits, loss = model(dummy_x)
        
    # Verify shape: (Batch Size, Sequence Length, Vocab Size)
    assert logits.shape == (batch_size, seq_len, tiny_config["vocab_size"]), "Logits shape is incorrect"
    
    # Loss should be None if targets are not provided
    assert loss is None, "Loss should be None when targets are omitted"

def test_gpt_forward_pass_with_targets(tiny_config):
    model = GPT(**tiny_config)
    model.eval()
    
    batch_size = 2
    seq_len = 16
    
    dummy_x = torch.randint(0, tiny_config["vocab_size"], (batch_size, seq_len))
    dummy_y = torch.randint(0, tiny_config["vocab_size"], (batch_size, seq_len))
    
    with torch.no_grad():
        logits, loss = model(dummy_x, targets=dummy_y)
        
    assert loss is not None, "Loss must be calculated when targets are provided"
    assert loss.item() > 0, "Cross entropy loss should be positive"

def test_gpt_max_seq_len_assertion(tiny_config):
    model = GPT(**tiny_config)
    
    # Exceed the max_seq_len deliberately
    too_long_seq = tiny_config["max_seq_len"] + 1
    dummy_x = torch.randint(0, tiny_config["vocab_size"], (1, too_long_seq))
    
    with pytest.raises(AssertionError, match="Cannot forward sequence of length"):
        model(dummy_x)
