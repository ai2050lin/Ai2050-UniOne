import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
import json
import time

print('=' * 50)
print('LLM Validation Test (transformers)')
print('=' * 50)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

print('Loading GPT-2 on CPU...')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
print(f'Model: {model.config.n_layer} layers, {model.config.n_embd} dim')

results = {"device": "cpu", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

# Test generation
print('\nTest generation:')
prompt = 'The capital of France is'
inputs = tokenizer(prompt, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    next_token = outputs.logits[0, -1].argmax(dim=-1)
    next_word = tokenizer.decode(next_token)

print(f"'{prompt}' -> '{next_word}'")
results["generation"] = {"prompt": prompt, "next_token": next_word}

# Extract activation
print('\nExtracting activations...')
activations = {}

def get_hook(name):
    def hook(module, input, output):
        activations[name] = output[0].detach()
    return hook

hooks = []
layer_indices = [0, 3, 6, 9, 11]
for i in layer_indices:
    h = model.transformer.h[i].register_forward_hook(get_hook(f'layer_{i}'))
    hooks.append(h)

with torch.no_grad():
    _ = model(**inputs)

for h in hooks:
    h.remove()

print(f'Extracted: {len(activations)} layers')
results["activations"] = {k: list(v.shape) for k, v in activations.items()}

# Curvature
print('\nCurvature calculation:')
from sklearn.decomposition import PCA

curvatures = {}
for name, act in activations.items():
    flat = act.reshape(-1, act.size(-1)).numpy()
    n_samples, n_features = flat.shape
    n_comp = min(3, n_samples - 1, n_features - 1)
    if n_comp < 1:
        continue
    pca = PCA(n_components=n_comp)
    pca.fit(flat)
    curv = 1 - np.sum(pca.explained_variance_ratio_[:n_comp])
    curvatures[name] = float(curv)
    print(f'  {name}: curvature={curv:.3f}')

results["curvatures"] = curvatures

# Intervention test
print('\nIntervention test:')
def intervention_hook(module, input, output):
    noise = torch.randn_like(output[0]) * 0.1
    return (output[0] + noise,)

h = model.transformer.h[6].register_forward_hook(intervention_hook)
with torch.no_grad():
    outputs_int = model(**inputs)
    next_token_int = outputs_int.logits[0, -1].argmax(dim=-1)
    next_word_int = tokenizer.decode(next_token_int)
h.remove()

print(f"  Original: '{next_word}' -> Intervened: '{next_word_int}'")
changed = next_word != next_word_int
print(f"  Changed: {changed}")
results["intervention"] = {"changed": changed}

# Save
os.makedirs("tempdata", exist_ok=True)
with open("tempdata/real_llm_validation.json", "w") as f:
    json.dump(results, f, indent=2)

print(f'\nSaved: tempdata/real_llm_validation.json')
print('\nDone!')
