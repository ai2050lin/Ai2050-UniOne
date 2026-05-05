"""Test direction sensitivity at late layers (L25, L26) where effect should be strongest"""
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
print(f"Top: {tokenizer.decode([top_id]).strip()} ({logits_base[top_id]:.2f})")
print(f"Second: {tokenizer.decode([second_id]).strip()} ({logits_base[second_id]:.2f})")
print(f"Margin: {logits_base[top_id]-logits_base[second_id]:.2f}")

W_U = model.lm_head.weight.detach().cpu().float().numpy()
d_model = W_U.shape[1]

# Directions
wu_top = W_U[top_id]; wu_top = wu_top / np.linalg.norm(wu_top)
margin = W_U[top_id] - W_U[second_id]; margin = margin / np.linalg.norm(margin)
# Most adversarial: -margin (should decrease top logit most)
neg_margin = -margin

np.random.seed(42)
rand_dirs = [np.random.randn(d_model) for _ in range(5)]
rand_dirs = [d / np.linalg.norm(d) for d in rand_dirs]

hs_all_layers = {}
for li in [20, 25, 26]:
    hs_all_layers[li] = out_base.hidden_states[li][0].detach().clone()

del out_base
torch.cuda.empty_cache()

# Test each layer
for li in [20, 25, 26]:
    print(f"\n===== Layer {li} =====")
    hs_base_all = hs_all_layers[li]
    
    for eps_val in [0.01, 0.1, 1.0]:
        direction_results = {}
        
        for name, direction_np in [('W_U[top]', wu_top), ('margin', margin), 
                                    ('-margin', neg_margin)]:
            d_t = torch.tensor(direction_np, dtype=torch.float32, device=device)
            hs_pert = hs_base_all.clone()
            hs_pert[-1] = hs_pert[-1] + eps_val * d_t.to(hs_pert.dtype)
            
            def make_hook(hs_full):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        return (hs_full.unsqueeze(0).to(output[0].dtype),) + output[1:]
                    return hs_full.unsqueeze(0).to(output.dtype)
                return hook_fn
            
            handle = get_layers(model)[li].register_forward_hook(make_hook(hs_pert))
            with torch.no_grad():
                out_p = model(input_ids=input_ids, attention_mask=attention_mask)
            handle.remove()
            
            new_logits = out_p.logits[0, -1].cpu().float().numpy()
            new_top = int(np.argmax(new_logits))
            top_change = new_logits[top_id] - logits_base[top_id]
            margin_change = (new_logits[top_id] - new_logits[second_id]) - (logits_base[top_id] - logits_base[second_id])
            direction_results[name] = (top_change, margin_change, new_top != top_id)
            del out_p
        
        # Random
        rand_results = []
        for rd in rand_dirs:
            d_t = torch.tensor(rd, dtype=torch.float32, device=device)
            hs_pert = hs_base_all.clone()
            hs_pert[-1] = hs_pert[-1] + eps_val * d_t.to(hs_pert.dtype)
            
            def make_hook2(hs_full):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        return (hs_full.unsqueeze(0).to(output[0].dtype),) + output[1:]
                    return hs_full.unsqueeze(0).to(output.dtype)
                return hook_fn
            
            handle2 = get_layers(model)[li].register_forward_hook(make_hook2(hs_pert))
            with torch.no_grad():
                out_r = model(input_ids=input_ids, attention_mask=attention_mask)
            handle2.remove()
            
            new_logits = out_r.logits[0, -1].cpu().float().numpy()
            rand_results.append(new_logits[top_id] - logits_base[top_id])
            del out_r
        
        print(f"\n  eps={eps_val}:")
        for name, (tc, mc, flipped) in direction_results.items():
            print(f"    {name:10s}: top_Δ={tc:+.4f}, margin_Δ={mc:+.4f}, flipped={flipped}")
        print(f"    {'random':10s}: top_Δ={np.mean(rand_results):+.4f} ± {np.std(rand_results):.4f}")
        
        torch.cuda.empty_cache()

release_model(model)
