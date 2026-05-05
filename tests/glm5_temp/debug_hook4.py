"""Verify hook is actually injecting perturbation correctly"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, numpy as np
sys.path.insert(0, '../glm5')
from model_utils import load_model, get_layers, release_model

model, tokenizer, device = load_model('deepseek7b')

text = 'The scientist discovered a new element'
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Test 1: No hook vs identity hook
with torch.no_grad():
    out1 = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    logits1_full = out1.logits[0, -1].cpu().float().numpy()
    top_id = int(np.argmax(logits1_full))
    hs_L26_1 = out1.hidden_states[26][0, -1, :5].cpu().float().numpy()
    hs_L27_1 = out1.hidden_states[27][0, -1, :5].cpu().float().numpy()
    print(f"No hook: top_id={top_id}, top_logit={logits1_full[top_id]:.4f}, L26[:5]={hs_L26_1}, L27[:5]={hs_L27_1}")

# Test 2: Hook that returns the SAME hidden state (identity)
hs_to_inject = out1.hidden_states[26][0].detach().clone()

def identity_hook(module, input, output):
    if isinstance(output, tuple):
        return (hs_to_inject.unsqueeze(0).to(output[0].dtype),) + output[1:]
    return hs_to_inject.unsqueeze(0).to(output.dtype)

handle = get_layers(model)[26].register_forward_hook(identity_hook)
with torch.no_grad():
    out2 = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
handle.remove()
logits2 = out2.logits[0, -1].cpu().float().numpy()
hs_L27_2 = out2.hidden_states[27][0, -1, :5].cpu().float().numpy()
print(f"Identity hook: top_logit={logits2[top_id]:.4f}, L27[:5]={hs_L27_2}")
print(f"  Logits match: {np.allclose(logits1_full, logits2, atol=1e-3)}")

# Test 3: Hook with LARGE perturbation (eps=10)
W_U = model.lm_head.weight.detach().cpu().float().numpy()
wu_top = W_U[top_id]; wu_top = wu_top / np.linalg.norm(wu_top)

hs_pert = hs_to_inject.clone()
hs_pert[-1] = hs_pert[-1] + 10.0 * torch.tensor(wu_top, dtype=hs_pert.dtype, device=device)

def pert_hook(module, input, output):
    if isinstance(output, tuple):
        return (hs_pert.unsqueeze(0).to(output[0].dtype),) + output[1:]
    return hs_pert.unsqueeze(0).to(output.dtype)

handle2 = get_layers(model)[26].register_forward_hook(pert_hook)
with torch.no_grad():
    out3 = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
handle2.remove()
logits3 = out3.logits[0, -1].cpu().float().numpy()
hs_L27_3 = out3.hidden_states[27][0, -1, :5].cpu().float().numpy()
print(f"Perturbed (eps=10, W_U[top]): top_logit={logits3[top_id]:.4f}")
print(f"  L27[:5]={hs_L27_3}")
print(f"  Top logit change: {logits3[top_id]-logits1_full[top_id]:.4f}")

# Test 4: Large random perturbation
np.random.seed(42)
rand_d = np.random.randn(W_U.shape[1]); rand_d = rand_d / np.linalg.norm(rand_d)
hs_pert2 = hs_to_inject.clone()
hs_pert2[-1] = hs_pert2[-1] + 10.0 * torch.tensor(rand_d, dtype=hs_pert2.dtype, device=device)

def pert_hook2(module, input, output):
    if isinstance(output, tuple):
        return (hs_pert2.unsqueeze(0).to(output[0].dtype),) + output[1:]
    return hs_pert2.unsqueeze(0).to(output.dtype)

handle3 = get_layers(model)[26].register_forward_hook(pert_hook2)
with torch.no_grad():
    out4 = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
handle3.remove()
logits4 = out4.logits[0, -1].cpu().float().numpy()
print(f"Perturbed (eps=10, random): top_logit={logits4[top_id]:.4f}")
print(f"  Top logit change: {logits4[top_id]-logits1_full[top_id]:.4f}")

# Key comparison
print(f"\n===== COMPARISON =====")
print(f"W_U[top] direction: top_Δ = {logits3[top_id]-logits1_full[top_id]:.4f}")
print(f"Random direction:   top_Δ = {logits4[top_id]-logits1_full[top_id]:.4f}")
print(f"Ratio: {abs(logits3[top_id]-logits1_full[top_id]) / (abs(logits4[top_id]-logits1_full[top_id])+1e-10):.2f}x")

release_model(model)
