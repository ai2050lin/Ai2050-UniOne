"""Quick test: load model, run 1 patching pair, verify everything works."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading DS7B 8bit...")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    trust_remote_code=True,
    device_map="auto",
    load_in_8bit=True,
)
model.eval()
print("Model loaded!")

# Test 1 pair
sent_clean = "The cat sleeps on the mat."
sent_source = "The cat slept on the mat."
ids_clean = tokenizer(sent_clean, return_tensors="pt")["input_ids"]
ids_source = tokenizer(sent_source, return_tensors="pt")["input_ids"]

device = next(model.parameters()).device
ids_clean = ids_clean.to(device)
ids_source = ids_source.to(device)

# Get clean hidden at L0
captured = {}
def hook_fn(module, input, output):
    captured["h"] = output[0][0, -1, :].detach()

layer = model.model.layers[0]
handle = layer.register_forward_hook(hook_fn)
with torch.no_grad():
    _ = model(ids_clean)
handle.remove()
clean_h = captured["h"].cpu().float()
print(f"Clean hidden shape: {clean_h.shape}, norm: {clean_h.norm():.1f}")

# Get source hidden
captured2 = {}
def hook_fn2(module, input, output):
    captured2["h"] = output[0][0, -1, :].detach()

handle2 = layer.register_forward_hook(hook_fn2)
with torch.no_grad():
    _ = model(ids_source)
handle2.remove()
source_h = captured2["h"].cpu().float()
print(f"Source hidden shape: {source_h.shape}, norm: {source_h.norm():.1f}")

# Patch
captured3 = {}
def patch_hook(module, input, output):
    h = output[0].clone()
    h[0, -1, :] = source_h.to(h.device)
    captured3["h"] = h[0, -1, :].detach()
    return (h,) + output[1:]

handle3 = layer.register_forward_hook(patch_hook)
with torch.no_grad():
    _ = model(ids_clean)
handle3.remove()
patched_h = captured3["h"].cpu().float()

l2 = torch.norm(patched_h - clean_h, p=2).item()
print(f"Patched L2: {l2:.1f}")
print("SUCCESS: Model loading and patching work!")
