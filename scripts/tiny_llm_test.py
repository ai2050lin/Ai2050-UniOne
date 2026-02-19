"""Tiny model test - distilgpt2"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import json
import time

print("Loading tiny GPT-2 (distilgpt2)...")

from transformers import GPT2LMHeadModel, GPT2Tokenizer

try:
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    print("Tokenizer OK")
except Exception as e:
    print(f"Tokenizer error: {e}")
    exit(1)

try:
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model.eval()
    print(f"Model OK: {model.config.n_layer} layers")
except Exception as e:
    print(f"Model error: {e}")
    exit(1)

print("\nTesting generation...")
prompt = "Hello"
inputs = tokenizer(prompt, return_tensors='pt')

try:
    with torch.no_grad():
        out = model(**inputs)
    print(f"Forward pass OK, logits shape: {out.logits.shape}")
except Exception as e:
    print(f"Forward error: {e}")
    exit(1)

print("\nAll tests passed!")
