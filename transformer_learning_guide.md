# Transformer From Scratch: Learning & Implementation Guide

This guide breaks down the ambitious project of building a GPT-style language model from scratch. We will build a model specialized in writing Zig code. 

For every phase, we break it down into three parts:
1. **What we are doing**: The goal of the phase.
2. **What you need to learn**: The fundamental concepts you need to grasp, assuming you are starting from zero.
3. **How we will solve it**: The practical steps and code we will write to accomplish it.

---

## Phase 1: Tokenizer & Data Pipeline (Weeks 1-2)

### What we are doing
Computers cannot read text; they can only process numbers. Before we can train our AI, we need to convert thousands of Zig code files into sequences of integer numbers (tokens). We also need a pipeline to feed these numbers to our model efficiently in "batches".

### What you need to learn
- **Python Fundamentals**: File I/O (reading/writing files), strings, lists, loops, and dictionaries.
- **Data Structures**: Hash maps (dictionaries in Python) for mapping words to numbers.
- **Byte-Pair Encoding (BPE)**: An algorithm used by OpenAI. It starts by treating every single character as a token. Then, it finds the most frequently adjacent pair of tokens (e.g., `t` and `h` in `th`) and merges them into a new single token. It repeats this until we have a target vocabulary size.
- **Tensors**: Think of a tensor as an Excel spreadsheet of numbers, but it can have more than two dimensions (rows/columns). A 1D tensor is a list, a 2D tensor is a grid, a 3D tensor is a cube of numbers.
- **Batching & Padding**: AI models learn best when they process multiple examples at once (a batch). Since Zig code snippets have different lengths, we have to add "padding" (dummy empty tokens) to the shorter snippets so they form a perfect rectangular tensor.

### How we will solve it
1. We will write a Python script to download open-source Zig code from GitHub.
2. We will implement the BPE algorithm in pure Python to build our vocabulary, resulting in an `encode(text)` function that turns Zig code into numbers, and a `decode(numbers)` function that turns it back.
3. We will use PyTorch (`torch.Tensor`) to organize these numbers into perfectly sized batches ready for training.

---

## Phase 2: Transformer Architecture (Weeks 3-5)

### What we are doing
This is where we build the "brain" of the AI. We are implementing the decoder-only architecture from the famous "Attention Is All You Need" paper. This architecture is what powers ChatGPT. We will write the complex math that allows the model to understand the context of the code.

### What you need to learn
- **Linear Algebra Basics**: You need to understand vectors (lists of numbers), matrices (grids of numbers), and matrix multiplication (how to multiply grids together). This is the fundamental math of neural networks.
- **Embeddings**: How to map our integer tokens into a high-dimensional space. Think of it like giving each word GPS coordinates in a "meaning" space. Words with similar meanings will have similar coordinates.
- **Positional Encoding**: Transformers process all words at exactly the same time. We have to mathematically inject a "timestamp" into each word's embedding so the model knows the order of the words.
- **Self-Attention**: The heart of the model. This is the math that allows the word "bank" to know if it's sitting next to "river" or "money". 
  - *Queries (Q), Keys (K), Values (V)*: You'll learn how every word acts as a query (asking for context), a key (providing context), and a value (the actual content).
  - *Softmax*: A math function that turns raw numbers into probabilities (percentages that add up to 100%).
- **Multi-Head Attention**: Running multiple self-attention mechanisms in parallel so the model can look for different types of relationships simultaneously (e.g., one head looks at syntax, another looks at variables).
- **Feed-Forward Networks & Layer Norm**: Simple neural network layers that process the attention output, and normalization techniques to keep the math from spiraling out of control.

### How we will solve it
We will create Python classes using PyTorch (`nn.Module`). However, we will strictly enforce a rule: we cannot use PyTorch's built-in transformer tools (like `nn.MultiheadAttention`). We will write out the exact matrix multiplications (`torch.matmul`), additions, and softmax functions ourselves.

---

## Phase 3: Training Loop (Weeks 6-7)

### What we are doing
We are teaching the brain to predict the future. We will feed our model a sequence of Zig code and ask it to guess the very next token. We will grade its guess, figure out where it made a mistake, and tweak its internal "knobs" (weights) so it guesses better next time.

### What you need to learn
- **High-Level Calculus (Gradients)**: You don't need to do the math by hand, but you need the concept. A gradient is a compass that points in the direction of *higher* error. We want to take a step in the *opposite* direction to reduce the error.
- **Loss Functions (Cross-Entropy)**: The mathematical formula we use to "grade" the model's guess. It penalizes the model heavily if it confidently guessed the wrong word.
- **Backpropagation**: PyTorch does this for us automatically (Autograd), but it's the process of the error flowing backward through all the matrices to calculate how much each individual number contributed to the mistake.
- **Optimizers (Adam)**: The algorithm that actually turns the knobs. "Adam" is the industry standard—it smartly adjusts the size of the tweaks based on past history.
- **Learning Rate Schedules**: You want to make big adjustments to the model when it's just starting out (warmup), and very tiny, careful adjustments when it's almost fully trained (decay).

### How we will solve it
We will write a `train.py` loop. It will load a batch of data, pass it through the model (forward pass) to get predictions, calculate the Cross-Entropy loss, ask PyTorch to calculate gradients (`loss.backward()`), and finally ask the Adam optimizer to adjust the weights (`optimizer.step()`). We will log the loss to ensure it is going down over time.

---

## Phase 4: Inference & Sampling (Week 8)

### What we are doing
Our model is trained! Now we want to actually use it to write code. We give it a prompt (e.g., `fn reverseLinkedList(`) and ask it to generate the rest of the function.

### What you need to learn
- **Autoregressive Generation**: The process of taking the model's prediction, adding it to our prompt, and feeding the whole thing *back* into the model to predict the next word, over and over in a loop.
- **Sampling Strategies**: The model doesn't just output one word; it outputs a probability for *every* word in the vocabulary.
  - *Temperature*: A math trick to make the probabilities more uniform (high temp = crazy/creative code) or more spiked (low temp = predictable/safe code).
  - *Top-K*: A rule that says "only pick from the top 50 most likely words, ignore the rest."
  - *Top-P (Nucleus)*: A dynamic rule that picks words until their combined probabilities reach a certain threshold (e.g., 90%).
- **KV Cache (Stretch Goal)**: Because we loop continuously, recalculating the attention for words we've already processed is a waste of time. KV Caching saves the previous calculations to make text generation significantly faster.

### How we will solve it
We will write a `generate()` function that runs our autoregressive loop. It will apply temperature and Top-K/Top-P logic to the model's output probabilities, randomly sample a word based on those probabilities, and append it to the sequence.

---

## Phase 5: Research Extension (Weeks 9-10)

### What we are doing
To prove deep understanding, we will read an actual AI research paper and implement a modern enhancement that didn't exist in the original 2017 Transformer paper.

### What you need to learn
*(Assuming we choose RoPE - Rotary Positional Embeddings, used in LLaMA)*
- **Complex Numbers / Trigonometry**: RoPE uses sine and cosine rotations in a 2D plane to encode position, rather than simply adding a number to the embedding. It allows the model to better understand the *relative* distance between words.

### How we will solve it
We will replace our basic Positional Encoding with a RoPE implementation. We will write the math to rotate our Query and Key vectors before they are multiplied together in the attention mechanism.

---

## Phase 6: Evaluation & Serving Layer (Weeks 11-12)

### What we are doing
We need to prove our model actually works, and we need to wrap it in an API so a user interface (like a chat box) can communicate with it.

### What you need to learn
- **Software Evaluation**: How to systematically test an AI model. For code generation, the best test is: "Does this code actually compile?"
- **Zig Basics**: We need to know enough about Zig to invoke its compiler from the command line.
- **Web APIs & HTTP**: How software communicates over a network using requests and JSON data.
- **FastAPI**: A fast, modern Python framework for building web servers.

### How we will solve it
1. **Evaluation**: We will write an evaluation script that feeds 100 hidden prompts to the model, saves the outputs to `.zig` files, and automatically runs `zig build-exe` to see what percentage of them successfully compile.
2. **Serving**: We will write a `server.py` file using FastAPI. It will expose a `/generate` endpoint. When you send a JSON request with a prompt to this endpoint, the server will run the model's inference loop and return the generated Zig code over the network.
