"""Debug hook injection — verify perturbation is correctly applied"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, numpy as np
sys.path.insert(0, '../glm5')
from model_utils import load_model, get_layers, release_model

model, tokenizer, device = load_model('deepseek7b')

text = 'The scientist discovered a new element'
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Get base
with torch.no_grad():
    out_base = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
logits_base = out_base.logits[0, -1].cpu().float().numpy()
top_id = int(np.argmax(logits_base))
print(f"Base top logit: {logits_base[top_id]:.4f}")

# Get hidden state at L20
hs_base = out_base.hidden_states[20][0, -1].detach().clone()
print(f"hs_base shape: {hs_base.shape}, dtype: {hs_base.dtype}, norm: {hs_base.norm():.4f}")

del out_base
torch.cuda.empty_cache()

# Test hook with perturbation
li = 20
eps = 0.01
d = torch.randn(hs_base.shape, dtype=torch.float32, device=device)
d = d / d.norm()

hs_pert = (hs_base + eps * d).to(hs_base.dtype)

captured_input = []
captured_output = []

def debug_hook(module, input, output):
    # Capture what's being passed
    if isinstance(output, tuple):
        orig_hs = output[0]
        captured_output.append(orig_hs.detach().clone())
        # Replace with perturbed
        new_hs = hs_pert.unsqueeze(0).unsqueeze(0).expand_as(orig_hs)
        return (new_hs,) + output[1:]
    return output

handle = get_layers(model)[li].register_forward_hook(debug_hook)
with torch.no_grad():
    out_pert = model(input_ids=input_ids, attention_mask=attention_mask)
handle.remove()

logits_pert = out_pert.logits[0, -1].cpu().float().numpy()

print(f"Perturbed top logit: {logits_pert[top_id]:.4f}")
print(f"Logit change: {logits_pert[top_id] - logits_base[top_id]:.6f}")

if captured_output:
    orig = captured_output[0]
    print(f"Original output shape: {orig.shape}")
    print(f"Original vs perturbed hidden state diff (first token): {(orig[0,-1] - hs_pert).norm():.6f}")

# Now test with DIFFERENT directions — should give DIFFERENT logit changes
print("\n--- Testing direction sensitivity ---")
for eps_val in [0.001, 0.01, 0.1]:
    changes = []
    for seed in range(5):
        torch.manual_seed(seed)
        d_i = torch.randn(hs_base.shape, dtype=torch.float32, device=device)
        d_i = d_i / d_i.norm()
        hs_i = (hs_base + eps_val * d_i).to(hs_base.dtype)
        
        def make_hook_i(hs):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (hs.unsqueeze(0).unsqueeze(0).expand_as(output[0]),) + output[1:]
                return hs.unsqueeze(0).unsqueeze(0).expand_as(output)
            return hook_fn
        
        handle_i = get_layers(model)[li].register_forward_hook(make_hook_i(hs_i))
        with torch.no_grad():
            out_i = model(input_ids=input_ids, attention_mask=attention_mask)
        handle_i.remove()
        
        logit_change = out_i.logits[0, -1, top_id].cpu().float().item() - logits_base[top_id]
        changes.append(logit_change)
        del out_i
    
    changes = np.array(changes)
    print(f"eps={eps_val}: mean_change={changes.mean():.4f}, std={changes.std():.6f}, range=[{changes.min():.4f}, {changes.max():.4f}]")
    torch.cuda.empty_cache()

release_model(model)
