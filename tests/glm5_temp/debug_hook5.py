"""Correct approach: modify hidden state BEFORE the layer processes it,
not AFTER. Use register_forward_pre_hook instead of register_forward_hook.

Actually, the correct approach for injecting perturbation at layer L is:
1. Get the input to layer L (which is the output of layer L-1 after layernorm)
2. Add perturbation to that input
3. Let layer L process normally

This means we need to modify the INPUT to layer L+1 (which is output of layer L).

But actually the cleanest approach: use the model's forward with a modified 
hidden_states cache. Since we can't do that easily, let's use a different 
approach: compute the perturbation effect analytically.

For the LAST layer (L27 = n_layers-1):
  output_logits = W_U @ LayerNorm(h_L27)
  
If we add perturbation δ to h_L27:
  Δlogits ≈ W_U @ J_LN @ δ  (where J_LN is Jacobian of LayerNorm)
  
This is the ONLY case where we can directly compute the effect without 
going through nonlinear layers.

For earlier layers, we need to propagate through nonlinear layers,
which requires the hook approach but correctly.
"""

import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, numpy as np
sys.path.insert(0, '../glm5')
from model_utils import load_model, get_layers, release_model

model, tokenizer, device = load_model('deepseek7b')

text = 'The scientist discovered a new element'
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

with torch.no_grad():
    out_base = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
logits_base = out_base.logits[0, -1].cpu().float().numpy()
top_id = int(np.argmax(logits_base))
second_id = int(np.argsort(logits_base)[-2])

print(f"Top: id={top_id}, logit={logits_base[top_id]:.4f}")
print(f"Second: id={second_id}, logit={logits_base[second_id]:.4f}")

# Get final hidden state (before layernorm and W_U projection)
hs_final = out_base.hidden_states[-1][0, -1].detach().clone()  # [d_model]
print(f"Final hidden state norm: {hs_final.float().norm():.4f}, dtype: {hs_final.dtype}")

# Compute logits analytically: logits = W_U @ LayerNorm(h_final) + bias
# For the LAST layer, this is the cleanest test

W_U = model.lm_head.weight.detach().cpu().float().numpy()  # [vocab, d_model]

# Apply layer norm manually
# Find the final layernorm
n_layers = len(get_layers(model))
final_ln = model.model.norm  # The final layer norm
print(f"Final layernorm: {type(final_ln).__name__}")

# Compute LayerNorm(h_final)
hs_final_f = hs_final.float().unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
with torch.no_grad():
    hs_ln = final_ln(hs_final_f.to(device))  # [1, 1, d_model]
hs_ln_np = hs_ln[0, 0].cpu().float().numpy()

# Verify: W_U @ hs_ln should ≈ logits (maybe with bias)
logits_computed = W_U @ hs_ln_np
print(f"Computed logits[top_id] = {logits_computed[top_id]:.4f}")
print(f"Actual logits[top_id] = {logits_base[top_id]:.4f}")
print(f"Match: {np.abs(logits_computed[top_id] - logits_base[top_id]) < 0.1}")

# Now: perturbation test at the FINAL layer (no nonlinear layers between perturbation and logits)
# Δlogits ≈ W_U @ J_LN @ δ
# where J_LN is the Jacobian of LayerNorm at h_final

# For LayerNorm: J_LN ≈ (1/σ) * (I - μ_∇ - h̄·h̄^T/σ²) ≈ (1/σ) * I (approximately)
# For simplicity, let's compute numerically

eps_jac = 0.01
hs_plus = (hs_final_f + eps_jac * torch.randn_like(hs_final_f)).to(device)
hs_minus = (hs_final_f - eps_jac * torch.randn_like(hs_final_f)).to(device)

# Actually, let me just directly test: add perturbation to h_final, apply LN, compute logits
print("\n===== Direct perturbation at FINAL hidden state (before LayerNorm) =====")

d_model = W_U.shape[1]
directions = {
    'W_U[top]': W_U[top_id] / np.linalg.norm(W_U[top_id]),
    'margin': (W_U[top_id] - W_U[second_id]) / np.linalg.norm(W_U[top_id] - W_U[second_id]),
    '-margin': -(W_U[top_id] - W_U[second_id]) / np.linalg.norm(W_U[top_id] - W_U[second_id]),
}

np.random.seed(42)
for i in range(3):
    rd = np.random.randn(d_model)
    directions[f'rand{i}'] = rd / np.linalg.norm(rd)

for eps_val in [0.01, 0.1, 1.0, 5.0]:
    print(f"\neps={eps_val}:")
    for name, dir_np in directions.items():
        d_t = torch.tensor(dir_np, dtype=torch.float32, device=device)
        hs_pert = hs_final_f.clone().to(device) + eps_val * d_t.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            hs_pert_ln = final_ln(hs_pert.to(hs_final_f.dtype))
        
        logits_pert = W_U @ hs_pert_ln[0, 0].cpu().float().numpy()
        top_change = logits_pert[top_id] - logits_base[top_id]
        margin_change = (logits_pert[top_id] - logits_pert[second_id]) - (logits_base[top_id] - logits_base[second_id])
        new_top = int(np.argmax(logits_pert))
        
        print(f"  {name:10s}: top_Δ={top_change:+.4f}, margin_Δ={margin_change:+.4f}, flipped={new_top != top_id}")

# Also test: perturbation AFTER LayerNorm (directly on the LN output)
print("\n===== Direct perturbation AFTER LayerNorm (linear regime) =====")
hs_ln_base = hs_ln[0, 0].detach().clone()  # [d_model]

for eps_val in [0.01, 0.1, 1.0]:
    print(f"\neps={eps_val}:")
    for name, dir_np in directions.items():
        d_t = torch.tensor(dir_np, dtype=torch.float32, device=device)
        hs_ln_pert = hs_ln_base + eps_val * d_t.to(hs_ln_base.dtype)
        
        logits_pert = W_U @ hs_ln_pert.cpu().float().numpy()
        top_change = logits_pert[top_id] - logits_base[top_id]
        margin_change = (logits_pert[top_id] - logits_pert[second_id]) - (logits_base[top_id] - logits_base[second_id])
        
        print(f"  {name:10s}: top_Δ={top_change:+.4f}, margin_Δ={margin_change:+.4f}")

release_model(model)
