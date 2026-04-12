"""Test hook on Qwen3 to debug P428"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

mdl = AutoModelForCausalLM.from_pretrained(
    r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
    dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True, device_map="cpu",
)
if torch.cuda.is_available():
    mdl = mdl.to("cuda")
mdl.eval()
tok = AutoTokenizer.from_pretrained(
    r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
    trust_remote_code=True, local_files_only=True, use_fast=False,
)

layer = mdl.model.layers[5]

collected = {}

def attn_hook(m, i, o):
    if isinstance(o, tuple):
        collected["attn"] = o[0].detach().cpu().float()
    else:
        collected["attn"] = o.detach().cpu().float()

def mlp_hook(m, i, o):
    if isinstance(o, tuple):
        collected["mlp"] = o[0].detach().cpu().float()
    else:
        collected["mlp"] = o.detach().cpu().float()

h1 = layer.self_attn.register_forward_hook(attn_hook)
h2 = layer.mlp.register_forward_hook(mlp_hook)

prompt = "The apple is"
toks = tok(prompt, return_tensors="pt").to(mdl.device)
with torch.no_grad():
    _ = mdl(toks.input_ids)

h1.remove()
h2.remove()

print("attn collected:", "attn" in collected)
print("mlp collected:", "mlp" in collected)
if "attn" in collected:
    print("attn shape:", collected["attn"].shape, "norm:", collected["attn"][0, -1, :].norm().item())
if "mlp" in collected:
    print("mlp shape:", collected["mlp"].shape, "norm:", collected["mlp"][0, -1, :].norm().item())

del mdl
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Done")
