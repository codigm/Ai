import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# =============================================================================
# RoPE (Rotary Positional Embeddings) Helpers
# =============================================================================
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the complex exponential frequencies for RoPE.
    dim: dimension of each attention head
    end: maximum sequence length
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)  # [0, 1, ..., end-1]
    freqs = torch.outer(t, freqs)  # Outer product: [end, dim // 2]
    # Create polar complex numbers: cos(freqs) + i*sin(freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape freqs_cis from (T, hs//2) to (1, 1, T, hs//2) to match (B, n_heads, T, hs//2)"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[2], x.shape[-1])
    shape = [d if i == 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Applies the rotary positional embeddings to the query and key tensors.
    xq, xk shapes: (B, n_heads, T, hs)
    """
    # 1. Reshape xq and xk to pairs of floats and cast to complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 2. Reshape freqs_cis for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # 3. Multiply by complex frequencies (this rotates them in 2D space!)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


# =============================================================================
# Transformer Modules
# =============================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, max_seq_len=256):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.d_model = d_model
        
        # Key, query, value projections for all heads, packed in a single linear layer
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask to ensure that attention is only applied to the left in the sequence.
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len))
                                     .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x, freqs_cis, use_cache=False, layer_cache=None):
        B, T, C = x.size()
        
        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Split heads and move the head dimension forward: (B, n_heads, T, head_size)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        # --- ROPE: APPLY ROTARY POSITIONAL EMBEDDINGS ---
        q, k = apply_rotary_emb(q, k, freqs_cis)
        # ------------------------------------------------
        
        if use_cache:
            if layer_cache is not None:
                k_cache, v_cache = layer_cache
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)
            present_cache = (k, v)
        else:
            present_cache = None
            
        # Sequence length of keys/values (includes past tokens if cached)
        seq_len_k = k.size(2)
        
        # Causal Self-Attention: (B, nh, T, hs) @ (B, nh, hs, T_k) -> (B, nh, T, T_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply the causal mask (tokens cannot look into the future)
        # Only needed if processing multiple tokens at once (e.g., initial prompt)
        if T > 1:
            att = att.masked_fill(self.bias[:, :, :T, :seq_len_k] == 0, float('-inf'))
        
        # Softmax to get probabilities
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Multiply probabilities by the actual values
        y = att @ v # (B, nh, T, T_k) @ (B, nh, T_k, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final output projection
        y = self.resid_dropout(self.c_proj(y))
        
        if use_cache:
            return y, present_cache
        return y

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, max_seq_len=256):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, dropout)

    def forward(self, x, freqs_cis, use_cache=False, layer_cache=None):
        if use_cache:
            attn_out, present_cache = self.attn(self.ln_1(x), freqs_cis, use_cache=True, layer_cache=layer_cache)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, present_cache
        else:
            x = x + self.attn(self.ln_1(x), freqs_cis)
            x = x + self.mlp(self.ln_2(x))
            return x

class GPT(nn.Module):
    def __init__(self, vocab_size=8192, max_seq_len=256, d_model=384, n_heads=6, n_layers=6, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # Embedding table (NOTICE: self.position_embedding is completely deleted!)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        
        # Precompute RoPE frequencies for the maximum sequence length
        head_dim = d_model // n_heads
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len * 2) # give it some buffer
        self.register_buffer("freqs_cis", freqs_cis)
        
        # Transformer blocks (Use nn.ModuleList because we are iterating over them to pass freqs_cis)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, dropout, max_seq_len) for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection (LM Head)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, use_cache=False, past_key_values=None):
        B, T = idx.size()
        
        # Infer the absolute starting position for RoPE
        past_length = 0
        if past_key_values is not None:
            # past_key_values is a list of tuples (k_cache, v_cache) for each layer
            # k_cache has shape (B, n_heads, T_past, head_size)
            past_length = past_key_values[0][0].size(2)
            
        assert past_length + T <= self.max_seq_len, f"Cannot forward sequence of length {past_length + T}, max_seq_len is only {self.max_seq_len}"
        
        # Extract embeddings
        x = self.token_embedding(idx) # (B, T, d_model)
        x = self.drop(x)
        
        # Get RoPE frequencies for the current sequence positions
        freqs_cis = self.freqs_cis[past_length : past_length + T]
        
        next_key_values = [] if use_cache else None
        
        # Pass through the transformer blocks
        for i, block in enumerate(self.blocks):
            layer_cache = past_key_values[i] if past_key_values is not None else None
            
            if use_cache:
                x, present_cache = block(x, freqs_cis, use_cache=True, layer_cache=layer_cache)
                next_key_values.append(present_cache)
            else:
                x = block(x, freqs_cis)
        
        # Final LayerNorm
        x = self.ln_f(x)
        
        # Get raw logits (predictions for the next word)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B*T, C)
            targets_reshaped = targets.view(B*T)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
            
        if use_cache:
            return logits, loss, next_key_values
        return logits, loss

    def get_num_params(self):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

if __name__ == "__main__":
    print("--- Testing RoPE Transformer Architecture ---")
    
    # 1. Instantiate the model with our Nano config
    config = {
        "vocab_size": 8192,
        "max_seq_len": 256,
        "d_model": 384,
        "n_heads": 6,
        "n_layers": 6,
        "dropout": 0.1
    }
    
    model = GPT(**config)
    print(f"Model created. Total parameters: {model.get_num_params():,}")
    
    # 2. Create a dummy batch of data
    batch_size = 8
    seq_len = 256
    dummy_x = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    dummy_y = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    print(f"\nForwarding dummy input of shape: {list(dummy_x.shape)}...")
    
    # 3. Forward pass
    logits, loss = model(dummy_x, targets=dummy_y)
    
    print("\n--- Verification Test ---")
    print(f"Logits shape: {list(logits.shape)} -> Expected: [{batch_size}, {seq_len}, {config['vocab_size']}]")
    assert list(logits.shape) == [batch_size, seq_len, config["vocab_size"]], "Logits shape is incorrect!"
    
    print(f"Calculated Initial Loss: {loss.item():.4f}")
    assert loss is not None, "Loss calculation failed!"
    
    print("\n✅ Verification Passed: RoPE Architecture is properly configured!")
