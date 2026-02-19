"""快速 GPT-2 测试"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Loading GPT-2...")

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Model loaded!")

# 测试生成
prompt = "Hello world"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"Input: {prompt}")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Output: {result}")

# 测试激活提取
print("\nTesting activation extraction...")

activations = {}

def hook_fn(module, input, output):
    activations["layer_0"] = output[0].detach()

hook = model.transformer.h[0].register_forward_hook(hook_fn)

with torch.no_grad():
    _ = model(**inputs)

hook.remove()

print(f"Captured activation shape: {activations['layer_0'].shape}")

# 测试几何干预
print("\nTesting geometric intervention...")

activation = activations["layer_0"]
reference = torch.randn(10, activation.size(-1))  # 随机参考点

# 热核扩散
distances = torch.cdist(activation.reshape(-1, activation.size(-1)), reference)
weights = torch.exp(-distances**2 / 4)
weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
diffused = torch.matmul(weights, reference).reshape_as(activation)

intervention_diff = torch.norm(diffused - activation).item()
print(f"Intervention difference: {intervention_diff:.4f}")

print("\nTest completed successfully!")
