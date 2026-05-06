"""
Phase 59 Exp4: Patch Specificity Control (精简版)
==================================================
验证: Subject patch的特异性 — 随机位置patch是否也有大影响?

关键对比:
  Subject patch Δ vs Object patch Δ vs Random patch Δ
  如果 Subject >> Random → 语法信息的特异性
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np, gc, argparse
import warnings
warnings.filterwarnings('ignore')

from model_utils import load_model, get_model_info, release_model


def run_patching_experiment(model, tokenizer, device, info):
    print("="*70)
    print("★★★ Patch Specificity Control ★★★")
    print("="*70)
    
    # Singular: "The cat chases the mouse"
    # Plural:   "The cats chase the mouse"
    # Patch: Put plural subject into singular sentence
    # Measure: Does verb agreement change?
    
    sing = "The cat chases the mouse"
    plur = "The cats chase the mouse"
    
    sing_ids = tokenizer(sing, return_tensors="pt").to(device)
    plur_ids = tokenizer(plur, return_tensors="pt").to(device)
    
    # Find token positions
    sing_tokens = tokenizer.encode(sing, add_special_tokens=False)
    plur_tokens = tokenizer.encode(plur, add_special_tokens=False)
    
    print(f"  Sing tokens: {sing_tokens}")
    print(f"  Plur tokens: {plur_tokens}")
    print(f"  Sing decoded: {[tokenizer.decode([t]) for t in sing_tokens]}")
    print(f"  Plur decoded: {[tokenizer.decode([t]) for t in plur_tokens]}")
    
    # Get verb token IDs
    chases_id = tokenizer.encode("chases", add_special_tokens=False)[0]
    chase_id = tokenizer.encode("chase", add_special_tokens=False)[0]
    
    # Find positions (with BOS offset)
    # "The cat chases the mouse" → BOS The cat chases the mouse
    # Positions: 0=BOS, 1=The, 2=cat, 3=chases, 4=the, 5=mouse
    subj_pos = 2   # "cat" / "cats"
    verb_pos = 3   # "chases" / "chase"
    obj_pos = 5    # "mouse"
    det_pos = 4    # "the" (before object)
    
    model_layers = model.model.layers
    test_layers = [0, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25]
    
    # Baseline
    with torch.no_grad():
        sing_logits_base = model(**sing_ids).logits.detach().cpu()
        plur_logits_base = model(**plur_ids).logits.detach().cpu()
    
    # Measure agreement at verb position
    # For singular: P(chases) - P(chase) should be positive
    base_sing_agr = (sing_logits_base[0, verb_pos, chases_id] - 
                    sing_logits_base[0, verb_pos, chase_id]).item()
    base_plur_agr = (plur_logits_base[0, verb_pos, chase_id] - 
                    plur_logits_base[0, verb_pos, chases_id]).item()
    
    print(f"\n  Baseline sing agreement: {base_sing_agr:.4f} (chases - chase at verb pos)")
    print(f"  Baseline plur agreement: {base_plur_agr:.4f} (chase - chases at verb pos)")
    
    results = {}
    
    for layer_idx in test_layers:
        if layer_idx >= info.n_layers:
            continue
        
        # Get plural sentence activations at this layer
        plur_act = {}
        def cap_hook(d, li):
            def fn(m, i, o):
                d[li] = (o[0] if isinstance(o, tuple) else o).detach().clone()
            return fn
        
        h = model_layers[layer_idx].register_forward_hook(cap_hook(plur_act, layer_idx))
        with torch.no_grad():
            model(**plur_ids)
        h.remove()
        
        if layer_idx not in plur_act:
            continue
        
        plur_hidden = plur_act[layer_idx]  # [1, seq_len, d_model]
        
        # ===== Patch 1: Subject position (cat → cats) =====
        plur_subj_repr = plur_hidden[0, subj_pos, :]
        
        def make_patch(value, pos):
            applied = [False]
            def fn(m, i, o):
                if not applied[0]:
                    if isinstance(o, tuple):
                        p = o[0].clone()
                        p[:, pos, :] = value.to(p.device)
                        applied[0] = True
                        return (p,) + o[1:]
                    else:
                        p = o.clone()
                        p[:, pos, :] = value.to(p.device)
                        applied[0] = True
                        return p
                return o
            return fn
        
        h1 = model_layers[layer_idx].register_forward_hook(
            make_patch(plur_subj_repr, subj_pos))
        with torch.no_grad():
            subj_patched_logits = model(**sing_ids).logits.detach().cpu()
        h1.remove()
        
        subj_agr = (subj_patched_logits[0, verb_pos, chase_id] - 
                   subj_patched_logits[0, verb_pos, chases_id]).item()
        subj_delta = subj_agr - (-base_sing_agr)  # shift toward plural
        
        # ===== Patch 2: Object position (mouse from plur) =====
        plur_obj_repr = plur_hidden[0, obj_pos, :]
        
        h2 = model_layers[layer_idx].register_forward_hook(
            make_patch(plur_obj_repr, obj_pos))
        with torch.no_grad():
            obj_patched_logits = model(**sing_ids).logits.detach().cpu()
        h2.remove()
        
        obj_agr = (obj_patched_logits[0, verb_pos, chases_id] - 
                  obj_patched_logits[0, verb_pos, chase_id]).item()
        obj_delta = obj_agr - base_sing_agr  # should be small
        
        # ===== Patch 3: Determiner position ("the" from plur) =====
        plur_det_repr = plur_hidden[0, det_pos, :]
        
        h3 = model_layers[layer_idx].register_forward_hook(
            make_patch(plur_det_repr, det_pos))
        with torch.no_grad():
            det_patched_logits = model(**sing_ids).logits.detach().cpu()
        h3.remove()
        
        det_agr = (det_patched_logits[0, verb_pos, chases_id] - 
                  det_patched_logits[0, verb_pos, chase_id]).item()
        det_delta = det_agr - base_sing_agr
        
        # ===== Patch 4: Random noise patch (negative control) =====
        noise = torch.randn_like(plur_subj_repr) * plur_subj_repr.std()
        
        h4 = model_layers[layer_idx].register_forward_hook(
            make_patch(noise, subj_pos))
        with torch.no_grad():
            noise_patched_logits = model(**sing_ids).logits.detach().cpu()
        h4.remove()
        
        noise_agr = (noise_patched_logits[0, verb_pos, chases_id] - 
                    noise_patched_logits[0, verb_pos, chase_id]).item()
        noise_delta = noise_agr - base_sing_agr
        
        results[layer_idx] = {
            'subj_delta': subj_delta,
            'obj_delta': obj_delta,
            'det_delta': det_delta,
            'noise_delta': noise_delta,
        }
        
        spec = abs(subj_delta) / (abs(noise_delta) + 0.001)
        
        print(f"  L{layer_idx:>3}: subj={subj_delta:+.4f}  obj={obj_delta:+.4f}  "
              f"det={det_delta:+.4f}  noise={noise_delta:+.4f}  "
              f"specificity={spec:.1f}x")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Patch Specificity Analysis")
    print(f"{'='*70}")
    print(f"  {'Layer':>6} | {'Subj→Verb':>10} | {'Obj→Verb':>10} | {'Det→Verb':>10} | {'Noise→Verb':>11} | {'Spec':>6}")
    print(f"  {'-'*65}")
    for li in sorted(results.keys()):
        r = results[li]
        spec = abs(r['subj_delta']) / (abs(r['noise_delta']) + 0.001)
        print(f"  L{li:>4} | {r['subj_delta']:>+10.4f} | {r['obj_delta']:>+10.4f} | "
              f"{r['det_delta']:>+10.4f} | {r['noise_delta']:>+11.4f} | {spec:>5.1f}x")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"Model: {info.name}, Layers={info.n_layers}, d_model={info.d_model}")
    
    try:
        results = run_patching_experiment(model, tokenizer, device, info)
    finally:
        release_model(model)
        print("\nDone.")
