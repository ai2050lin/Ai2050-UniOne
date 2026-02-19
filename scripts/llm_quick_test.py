"""快速 LLM 测试 - CPU 版本"""
import torch
torch.set_default_device('cpu')
print('Device: CPU')

from transformer_lens import HookedTransformer

print('Loading GPT-2...')
model = HookedTransformer.from_pretrained('gpt2-small', device='cpu')
print(f'Model: {model.cfg.n_layers} layers, {model.cfg.d_model} dim')

print('\\nTest generation:')
tokens = model.to_tokens('Paris is')
with torch.no_grad():
    out = model.generate(tokens, max_new_tokens=3)
print(f'Output: {model.to_string(out[0])}')

print('\\nDone!')
