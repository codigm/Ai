import os
import pytest
from tokenizer import BPETokenizer

@pytest.fixture
def tokenizer():
    tok = BPETokenizer()
    # We assume tokenizer.json exists in the root directory since we are testing an existing project
    tok.load("tokenizer.json")
    return tok

def test_tokenizer_roundtrip_basic(tokenizer):
    text = "pub fn main() !void {\n    std.debug.print(\"Hello, world!\", .{});\n}"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert text == decoded, "Lossless reconstruction failed for basic Zig code"

def test_tokenizer_roundtrip_empty(tokenizer):
    text = ""
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert text == decoded, "Lossless reconstruction failed for empty string"

def test_tokenizer_roundtrip_unicode_and_emojis(tokenizer):
    text = "const greeting = \"こんにちは 🌍\";"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert text == decoded, "Lossless reconstruction failed for Unicode/Emojis"

def test_tokenizer_special_tokens(tokenizer):
    text = "<|endoftext|> hello <|endoftext|>"
    encoded = tokenizer.encode(text)
    
    # <|endoftext|> should map to the special token ID (256)
    assert 256 in encoded, "Special token was not encoded to its mapped ID"
    
    decoded = tokenizer.decode(encoded)
    assert text == decoded, "Special token roundtrip failed"

def test_tokenizer_add_special_token_flag(tokenizer):
    text = "hello"
    encoded_with_special = tokenizer.encode(text, add_special_token=True)
    assert encoded_with_special[-1] == 256, "add_special_token flag did not append special token"
