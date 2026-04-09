"""简单测试DeepSeek7B模型是否能正常加载和推理"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"

print(f"[1/4] 加载tokenizer: {model_path}")
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"  tokenizer OK, vocab_size={tok.vocab_size}")

print(f"[2/4] 加载模型 (bfloat16, device_map=auto)...")
mdl = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
mdl.eval()
print(f"  模型加载OK, n_layers={len(mdl.model.layers)}, d_model={mdl.config.hidden_size}")

print(f"[3/4] GPU内存检查...")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB used")

print(f"[4/4] 简单推理测试...")
prompt = "The capital of France is"
inputs = tok(prompt, return_tensors="pt").to(mdl.device)
with torch.no_grad():
    out = mdl.generate(**inputs, max_new_tokens=20, do_sample=False)
result = tok.decode(out[0], skip_special_tokens=True)
print(f"  输入: '{prompt}'")
print(f"  输出: '{result}'")

# 提取hidden states测试
print(f"\n[额外] Hidden states测试...")
inputs2 = tok("The cat sat on the mat", return_tensors="pt").to(mdl.device)
with torch.no_grad():
    out2 = mdl(**inputs2, output_hidden_states=True)
h0 = out2.hidden_states[0][0, -1].float()
hf = out2.hidden_states[-1][0, -1].float()
cos_sim = torch.nn.functional.cosine_similarity(h0.unsqueeze(0), hf.unsqueeze(0)).item()
print(f"  n_hidden_states: {len(out2.hidden_states)}")
print(f"  ||h_0|| = {h0.norm().item():.4f}")
print(f"  ||h_L|| = {hf.norm().item():.4f}")
print(f"  cos(h_0, h_L) = {cos_sim:.4f}")

print("\n=== DeepSeek7B测试通过! ===")
