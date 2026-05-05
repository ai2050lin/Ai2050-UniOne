"""Fixed hook injection — only perturb last token position"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, numpy as np
sys.path.insert(0, '../glm5')
from model_utils import load_model, get_layers, release_model

model, tokenizer, device = load_model('deepseek7b')

text = 'The scientist discovered a new element'
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
n_tokens = input_ids.shape[1]

with torch.no_grad():
    out_base = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
logits_base = out_base.logits[0, -1].cpu().float().numpy()
top_id = int(np.argmax(logits_base))
print(f"Base top logit: {logits_base[top_id]:.4f}, n_tokens={n_tokens}")

# Get hidden state at L20, last token
hs_base = out_base.hidden_states[20][0, -1].detach().clone()  # [d_model]
hs_base_all = out_base.hidden_states[20][0].detach().clone()  # [n_tokens, d_model]
print(f"hs_base norm: {hs_base.norm():.4f}, dtype: {hs_base.dtype}")

del out_base
torch.cuda.empty_cache()

li = 20

# Test with properly constructed perturbation (only last token position)
print("\n--- Testing with correct last-token-only perturbation ---")
for eps_val in [0.01, 0.1, 0.5, 1.0]:
    changes = []
    W_U = model.lm_head.weight.detach().cpu().float().numpy()
    d_model = W_U.shape[1]
    second_id = int(np.argsort(logits_base)[-2])
    
    # W_U[top] direction
    wu_top = W_U[top_id]; wu_top = wu_top / np.linalg.norm(wu_top)
    # margin direction  
    margin = W_U[top_id] - W_U[second_id]; margin = margin / np.linalg.norm(margin)
    
    for name, direction_np in [('W_U[top]', wu_top), ('margin', margin), ('-margin', -margin)]:
        d_t = torch.tensor(direction_np, dtype=torch.float32, device=device)
        hs_pert_all = hs_base_all.clone()
        hs_pert_all[-1] = hs_pert_all[-1] + eps_val * d_t.to(hs_pert_all.dtype)
        
        def make_hook(hs_full):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (hs_full.unsqueeze(0).to(output[0].dtype),) + output[1:]
                return hs_full.unsqueeze(0).to(output.dtype)
            return hook_fn
        
        handle = get_layers(model)[li].register_forward_hook(make_hook(hs_pert_all))
        with torch.no_grad():
            out_p = model(input_ids=input_ids, attention_mask=attention_mask)
        handle.remove()
        
        logit_change = out_p.logits[0, -1, top_id].cpu().float().item() - logits_base[top_id]
        margin_change = ((out_p.logits[0, -1, top_id] - out_p.logits[0, -1, second_id]) - 
                        (logits_base[top_id] - logits_base[second_id])).cpu().float().item()
        changes.append((name, logit_change, margin_change))
        del out_p
    
    # Random directions
    rand_changes = []
    for seed in range(3):
        np.random.seed(seed)
        rand_d = np.random.randn(d_model); rand_d = rand_d / np.linalg.norm(rand_d)
        d_t = torch.tensor(rand_d, dtype=torch.float32, device=device)
        hs_pert_all = hs_base_all.clone()
        hs_pert_all[-1] = hs_pert_all[-1] + eps_val * d_t.to(hs_pert_all.dtype)
        
        def make_hook2(hs_full):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (hs_full.unsqueeze(0).to(output[0].dtype),) + output[1:]
                return hs_full.unsqueeze(0).to(output.dtype)
            return hook_fn
        
        handle2 = get_layers(model)[li].register_forward_hook(make_hook2(hs_pert_all))
        with torch.no_grad():
            out_r = model(input_ids=input_ids, attention_mask=attention_mask)
        handle2.remove()
        
        rc = out_r.logits[0, -1, top_id].cpu().float().item() - logits_base[top_id]
        rand_changes.append(rc)
        del out_r
    
    print(f"\neps={eps_val}:")
    for name, tc, mc in changes:
        print(f"  {name:10s}: top_change={tc:+.4f}, margin_change={mc:+.4f}")
    print(f"  {'random':10s}: mean={np.mean(rand_changes):+.4f} ± {np.std(rand_changes):.4f}")
    
    torch.cuda.empty_cache()

release_model(model)
