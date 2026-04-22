import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = 'D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c'
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True).to('cuda')
model.eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)

layer = model.model.layers[0]
activations = {}

def hook_fn(module, inp, out):
    if isinstance(out, tuple):
        activations['h'] = out[0][0, -1, :].detach().float().clone()
        activations['shape'] = str(out[0].shape)
    else:
        activations['h'] = out[0, -1, :].detach().float().clone()
        activations['shape'] = str(out.shape)

h = layer.register_forward_hook(hook_fn)
ids = tokenizer('Hello world', return_tensors='pt')['input_ids'].to('cuda')
with torch.no_grad():
    _ = model(ids)
h.remove()

if 'h' in activations:
    norm = torch.norm(activations['h']).item()
    print(f"OK shape={activations['shape']} norm={norm:.2f}")
else:
    print("FAILED - no activation captured")
