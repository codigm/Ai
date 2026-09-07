# Polyglot AI: A From-Scratch LLM Pipeline

This repository contains a complete, end-to-end pipeline for training a Generative Pre-trained Transformer (GPT) language model from absolute scratch. 

> [!IMPORTANT]
> **Educational Scope**
> This model is intentionally **tiny** (~10M parameters, 384-dimensional embeddings, 6 layers). It is designed as an educational sandbox for learning the mathematics and engineering behind modern AI pipelines. It is *not* intended to compete with or replicate the performance of massive production models like GPT-4 or Llama 3. The goal of this project is full transparency and understanding of the architecture, from tokenization to reinforcement learning.

## Features & Architecture

This repository covers the complete lifecycle of a modern Large Language Model:

1. **Custom Tokenizer (`tokenizer.py`)**: A Byte-Pair Encoding (BPE) tokenizer trained purely on the target dataset without relying on external libraries like `tiktoken`.
2. **Transformer Architecture (`model.py`)**: A clean, readable implementation of a NanoGPT-style Transformer with Rotary Position Embeddings (RoPE), Multi-Head Attention, and SwiGLU activations.
3. **Pre-Training (`train.py`)**: The initial self-supervised training phase where the model learns syntax and structure using next-token prediction.
4. **Supervised Fine-Tuning / SFT (`sft_train.py`)**: Formatting the model into a conversational assistant using explicit `<|user|>` and `<|assistant|>` chat formatting.
5. **Direct Preference Optimization / DPO (`dpo_train.py`)**: Aligning the model using Reinforcement Learning from Human Feedback without complex reward models, relying solely on chosen/rejected prompt pairs.

## Evaluation

Training progress is evaluated using **Perplexity** (the standard exponentiated Cross-Entropy Loss) on a held-out validation set. Perplexity acts as an intuitive measure of how "confused" the model is during token prediction. 

## Dependency Management

This project strictly pins dependencies to ensure reproducible training environments. 
All dependencies are locked in `requirements.txt` and `pyproject.toml`.

```bash
uv pip install -r requirements.txt
```

## Known "Footguns"

> [!WARNING]
> **Checkpoint Loading & Strictness**
> In `server.py`, the model checkpoint is loaded using `strict=False`. This is an explicit workaround for an architectural shift made during training (moving from absolute learned position embeddings in Phase 3 to RoPE in Phase 5). While this prevents crashes on missing `pos_emb` weights, it will also *silently ignore* other architectural mismatches. Proceed with caution if you alter `config` definitions between checkpoints.
