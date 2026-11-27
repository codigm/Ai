import os
import torch
import torch.nn.functional as F
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Literal, List

from tokenizer import BPETokenizer
from model import GPT
from generate import generate, generate_stream

# Global objects to be initialized in the lifespan
model = None
tokenizer = None
config = None
device = 'cpu'

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, config
    
    candidate_paths = [
        p for p in [
            os.environ.get("MODEL_PATH"),
            "model_sft_final.pt",
            "model_step_9500.pt",
        ]
        if p
    ]
    checkpoint_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if checkpoint_path is None:
        raise RuntimeError(
            "Error: Could not find a model checkpoint (tried model_sft_final.pt, model_step_9500.pt)."
        )

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    model = GPT(**config)
    
    # [FOOTGUN WARNING]
    # We load with strict=False to gracefully ignore missing absolute position embedding keys.
    # Why? Phase 3 used learned Absolute Position Embeddings (`pos_emb`), but our Phase 5
    # refactor switched to Rotary Position Embeddings (RoPE), removing `pos_emb` from the GPT class.
    # Setting strict=False suppresses errors for these missing keys, allowing us to load older checkpoints.
    # WARNING: This will also silently ignore *other* missing or mismatched layers! Ensure architectures match.
    import logging
    logging.warning("Loading checkpoint with strict=False. Ignoring missing keys (expected for pos_emb).")
    
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)

    tokenizer = BPETokenizer()
    tokenizer.load("tokenizer.json")

    # If loading a pre-trained checkpoint (e.g. vocab_size=8192) with the chat tokenizer (8195),
    # resize embeddings + tied lm_head so special chat tokens (<|user|>, <|assistant|>, <|end|>) don't crash.
    target_vocab_size = max(
        [config['vocab_size'], 8195]
        + [tid + 1 for tid in getattr(tokenizer, 'special_tokens', {}).values()]
    )
    if config['vocab_size'] < target_vocab_size:
        old_vocab_size = config['vocab_size']
        old_weight = model.token_embedding.weight.data
        d_model = old_weight.shape[1]
        new_weight = torch.empty((target_vocab_size, d_model), device=device)
        new_weight[:old_vocab_size, :] = old_weight
        torch.nn.init.normal_(new_weight[old_vocab_size:, :], mean=0.0, std=0.02)

        new_embedding = torch.nn.Embedding(target_vocab_size, d_model).to(device)
        new_embedding.weight.data = new_weight
        new_lm_head = torch.nn.Linear(d_model, target_vocab_size, bias=False).to(device)
        new_lm_head.weight = new_embedding.weight

        model.token_embedding = new_embedding
        model.lm_head = new_lm_head
        config['vocab_size'] = target_vocab_size
        print(f"Resized model vocab from {old_vocab_size} to {target_vocab_size} to match tokenizer.")
    
    # CRITICAL: Set to eval mode exactly once at startup
    model.eval()
    
    print("Model and Tokenizer loaded successfully. Ready for inference!")
    yield # Server is running
    # Cleanup logic here (if any)
    print("Shutting down...")

app = FastAPI(title="NanoGPT Chat API", lifespan=lifespan)

# Add CORS Middleware for external UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static chat UI
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The starting text to condition the model.")
    temperature: float = Field(0.8, ge=0.01, le=2.0, description="Creativity setting.")
    max_tokens: int = Field(50, ge=1, le=256, description="Maximum number of tokens to generate.")
    top_k: int = Field(50, ge=1, description="Top-K sampling constraint.")

class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str

@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is currently loading or unavailable.")
    
    try:
        idx = torch.tensor([tokenizer.encode(req.prompt)], dtype=torch.long, device=device)
        
        # Wrapped in no_grad for safe, cheap inference
        with torch.no_grad():
            out_idx = generate(
                model, 
                idx, 
                max_new_tokens=req.max_tokens, 
                max_seq_len=config['max_seq_len'],
                temperature=req.temperature, 
                top_k=req.top_k
            )
            
        generated_code = tokenizer.decode(out_idx[0].tolist())
        return GenerateResponse(prompt=req.prompt, generated_text=generated_code)
        
    except Exception as e:
        # Prevent 500 stack traces from leaking directly to the client
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="A list of messages comprising the conversation so far")
    temperature: float = Field(0.8, ge=0.01, le=2.0, description="Creativity setting.")
    max_tokens: int = Field(50, ge=1, le=256, description="Maximum number of tokens to generate.")
    top_k: int = Field(50, ge=1, description="Top-K sampling constraint.")

class ChatResponse(BaseModel):
    message: ChatMessage
    generated_text: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is currently loading or unavailable.")
    
    # Simple history formatting. Note: max_seq_len is only 256, so a long history will fail!
    prompt_str = ""
    for msg in req.messages:
        prompt_str += f"<|{msg.role}|>\n{msg.content}<|end|>\n"
    prompt_str += "<|assistant|>\n"
    
    try:
        idx = torch.tensor([tokenizer.encode(prompt_str)], dtype=torch.long, device=device)
        
        input_len = idx.shape[1]
        allowed_new = config['max_seq_len'] - input_len
        if allowed_new <= 0:
            raise HTTPException(status_code=400, detail="Prompt is too long for the maximum sequence length of 256.")
            
        actual_max_tokens = min(req.max_tokens, allowed_new)
        
        with torch.no_grad():
            out_idx = generate(
                model, 
                idx, 
                max_new_tokens=actual_max_tokens, 
                max_seq_len=config['max_seq_len'],
                temperature=req.temperature, 
                top_k=req.top_k,
                stop_tokens=[8194] # <|end|>
            )
            
        generated_idx = out_idx[0][input_len:].tolist()
        
        if generated_idx and generated_idx[-1] == 8194:
            generated_idx = generated_idx[:-1]
            
        generated_code = tokenizer.decode(generated_idx)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=generated_code),
            generated_text=generated_code
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# =============================================================================
# SSE Streaming Endpoint
# =============================================================================
@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """Stream tokens one-by-one via Server-Sent Events."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is currently loading or unavailable.")

    # Build the prompt string from conversation history (same format as /chat)
    prompt_str = ""
    for msg in req.messages:
        prompt_str += f"<|{msg.role}|>\n{msg.content}<|end|>\n"
    prompt_str += "<|assistant|>\n"

    try:
        idx = torch.tensor([tokenizer.encode(prompt_str)], dtype=torch.long, device=device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tokenization failed: {str(e)}")

    input_len = idx.shape[1]
    allowed_new = config['max_seq_len'] - input_len
    if allowed_new <= 0:
        raise HTTPException(status_code=400, detail="Prompt is too long for the maximum sequence length of 256.")

    actual_max_tokens = min(req.max_tokens, allowed_new)

    async def token_generator():
        """Inline generation loop that yields each token as an SSE event."""
        nonlocal idx
        stop_token = 8194  # <|end|>
        past_key_values = None

        with torch.no_grad():
            for _ in range(actual_max_tokens):
                # Context Selection
                if past_key_values is None:
                    idx_cond = idx[:, -config['max_seq_len']:]
                else:
                    idx_cond = idx[:, -1:]

                # Forward pass with KV caching
                logits, _, past_key_values = model(idx_cond, use_cache=True, past_key_values=past_key_values)
                logits = logits[:, -1, :]

                # Temperature scaling
                logits = logits / req.temperature

                # Top-K masking
                if req.top_k is not None:
                    v, _ = torch.topk(logits, min(req.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Softmax + sample
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                idx = torch.cat((idx, idx_next), dim=1)

                token_id = idx_next.item()

                # Stop on end token
                if token_id == stop_token:
                    break

                # Decode and yield this single token
                token_text = tokenizer.decode([token_id])
                yield f"data: {token_text}\n\n"

                # Yield control to the event loop so the response can flush
                await asyncio.sleep(0)

        # Signal completion
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering if behind a proxy
        },
    )

if __name__ == "__main__":
    import uvicorn
    # Note: Running with a single worker because PyTorch CPU inference 
    # isn't entirely safe/efficient for true concurrency on a small machine.
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("RELOAD", "false").lower() == "true"
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=reload)
