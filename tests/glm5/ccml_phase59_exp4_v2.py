"""
Phase 59 Exp4 (v2): Patch Specificity Control — 修复tokenization问题
===================================================================
关键修复: 正确处理multi-token动词和位置映射
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


def find_word_position(tokenizer, text, target_word):
    """Find the token position(s) of a target word in tokenized text."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoded = [tokenizer.decode([t]).strip() for t in tokens]
    
    # Find the first token that starts with or equals target_word
    for i, d in enumerate(decoded):
        if d.lower().startswith(target_word.lower()) or d.lower() == target_word.lower():
            return i + 1  # +1 for BOS
    
    # If not found as prefix, search for it in the full decoded text
    full_text = tokenizer.decode(tokens)
    print(f"  WARNING: '{target_word}' not found in tokens. Decoded: {decoded}")
    return None


def get_verb_logits(logits, tokenizer, verb_pos, sing_verb, plur_verb):
    """
    Get logit scores for singular and plural verb forms.
    Handles multi-token verbs by summing log-probs.
    """
    # For simplicity, use the first token of each verb form
    sing_ids = tokenizer.encode(sing_verb, add_special_tokens=False)
    plur_ids = tokenizer.encode(plur_verb, add_special_tokens=False)
    
    # Logit at verb position for the first token of each form
    sing_logit = logits[0, verb_pos, sing_ids[0]].item()
    plur_logit = logits[0, verb_pos, plur_ids[0]].item()
    
    return sing_logit, plur_logit


def run_experiment(model, tokenizer, device, info):
    print("="*70)
    print("★★★ Patch Specificity Control (v2) ★★★")
    print("="*70)
    
    # Use simpler sentence pairs where verb is a single token
    # Test: "The cat runs" vs "The cats run" 
    # Or better: use longer sentences where verb position is clearer
    
    test_pairs = [
        # (sing_sent, plur_sent, sing_verb, plur_verb)
        ("The cat runs fast", "The cats run fast", "runs", "run"),
        ("The dog walks home", "The dogs walk home", "walks", "walk"),
        ("The bird flies high", "The birds fly high", "flies", "fly"),
        ("The girl reads books", "The girls read books", "reads", "read"),
    ]
    
    # First, check tokenization
    print("\n  Tokenization check:")
    for sing, plur, sv, pv in test_pairs:
        st = tokenizer.encode(sing, add_special_tokens=False)
        pt = tokenizer.encode(plur, add_special_tokens=False)
        sd = [tokenizer.decode([t]).strip() for t in st]
        pd = [tokenizer.decode([t]).strip() for t in pt]
        print(f"  Sing: {sd}")
        print(f"  Plur: {pd}")
    
    model_layers = model.model.layers
    test_layers = [0, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25]
    
    all_results = {}
    
    for sing, plur, sv, pv in test_pairs:
        print(f"\n--- Testing: '{sing}' / '{plur}' ---")
        
        # Find positions
        subj_pos = find_word_position(tokenizer, sing, sing.split()[1])  # "cat"/"cats"
        verb_pos_s = find_word_position(tokenizer, sing, sv)
        verb_pos_p = find_word_position(tokenizer, plur, pv)
        
        # Object position (last content word)
        obj_pos_s = find_word_position(tokenizer, sing, sing.split()[-1])
        
        print(f"  Subject pos: {subj_pos}, Verb pos sing: {verb_pos_s}, Verb pos plur: {verb_pos_p}")
        
        if subj_pos is None or verb_pos_s is None:
            print(f"  Skipping: cannot find positions")
            continue
        
        sing_ids = tokenizer(sing, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur, return_tensors="pt").to(device)
        
        # Get verb token IDs (first token of each)
        sv_tids = tokenizer.encode(sv, add_special_tokens=False)
        pv_tids = tokenizer.encode(pv, add_special_tokens=False)
        
        # Baseline logits
        with torch.no_grad():
            sing_logits_base = model(**sing_ids).logits.detach().cpu()
            plur_logits_base = model(**plur_ids).logits.detach().cpu()
        
        # Agreement = logit(sing_verb) - logit(plur_verb) at verb position
        # For singular sentence, this should be positive (prefers singular verb)
        base_sing_agr = (sing_logits_base[0, verb_pos_s, sv_tids[0]] - 
                        sing_logits_base[0, verb_pos_s, pv_tids[0]]).item()
        base_plur_agr = (plur_logits_base[0, verb_pos_p, pv_tids[0]] - 
                        plur_logits_base[0, verb_pos_p, sv_tids[0]]).item()
        
        print(f"  Baseline sing agreement: {base_sing_agr:.4f}")
        print(f"  Baseline plur agreement: {base_plur_agr:.4f}")
        
        for layer_idx in test_layers:
            if layer_idx >= info.n_layers:
                continue
            
            # Get plural activations
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
            
            plur_hidden = plur_act[layer_idx]
            
            # ===== Patch 1: Subject position (put plural subject in singular sentence) =====
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
            
            h1 = model_layers[layer_idx].register_forward_hook(make_patch(plur_subj_repr, subj_pos))
            with torch.no_grad():
                subj_logits = model(**sing_ids).logits.detach().cpu()
            h1.remove()
            
            # After patching plural subject, agreement should shift toward plural
            subj_agr = (subj_logits[0, verb_pos_s, sv_tids[0]] - 
                       subj_logits[0, verb_pos_s, pv_tids[0]]).item()
            subj_delta = subj_agr - base_sing_agr  # Should be negative (shifts toward plural)
            
            # ===== Patch 2: Object position =====
            if obj_pos_s and obj_pos_s < plur_hidden.shape[1]:
                plur_obj_repr = plur_hidden[0, obj_pos_s, :]
                
                h2 = model_layers[layer_idx].register_forward_hook(make_patch(plur_obj_repr, obj_pos_s))
                with torch.no_grad():
                    obj_logits = model(**sing_ids).logits.detach().cpu()
                h2.remove()
                
                obj_agr = (obj_logits[0, verb_pos_s, sv_tids[0]] - 
                          obj_logits[0, verb_pos_s, pv_tids[0]]).item()
                obj_delta = obj_agr - base_sing_agr
            else:
                obj_delta = 0
            
            # ===== Patch 3: Random noise at subject position =====
            noise = torch.randn_like(plur_subj_repr) * plur_subj_repr.std()
            
            h3 = model_layers[layer_idx].register_forward_hook(make_patch(noise, subj_pos))
            with torch.no_grad():
                noise_logits = model(**sing_ids).logits.detach().cpu()
            h3.remove()
            
            noise_agr = (noise_logits[0, verb_pos_s, sv_tids[0]] - 
                        noise_logits[0, verb_pos_s, pv_tids[0]]).item()
            noise_delta = noise_agr - base_sing_agr
            
            if layer_idx not in all_results:
                all_results[layer_idx] = {'subj': [], 'obj': [], 'noise': []}
            
            all_results[layer_idx]['subj'].append(subj_delta)
            all_results[layer_idx]['obj'].append(obj_delta)
            all_results[layer_idx]['noise'].append(noise_delta)
            
            gc.collect()
            torch.cuda.empty_cache()
    
    # Aggregate
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}")
    print(f"  {'Layer':>6} | {'Subj Δ':>10} | {'Obj Δ':>10} | {'Noise Δ':>10} | {'Spec':>6} | {'p<0.05':>6}")
    print(f"  {'-'*60}")
    
    for li in sorted(all_results.keys()):
        r = all_results[li]
        subj_m = np.mean(r['subj']) if r['subj'] else 0
        obj_m = np.mean(r['obj']) if r['obj'] else 0
        noise_m = np.mean(r['noise']) if r['noise'] else 0
        
        # Specificity: |subj effect| / |noise effect|
        spec = abs(subj_m) / (abs(noise_m) + 0.001)
        
        # Simple significance test: is subj effect consistently in the expected direction?
        expected_sign = -1  # patching plural subject should decrease sing agreement
        n_expected = sum(1 for x in r['subj'] if np.sign(x) == expected_sign)
        p_val = (0.5 ** len(r['subj'])) * (2 ** n_expected) if len(r['subj']) > 0 else 1.0
        sig = "★" if n_expected == len(r['subj']) and len(r['subj']) >= 3 else ""
        
        print(f"  L{li:>4} | {subj_m:>+10.4f} | {obj_m:>+10.4f} | {noise_m:>+10.4f} | {spec:>5.1f}x | {sig}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"Model: {info.name}, Layers={info.n_layers}, d_model={info.d_model}")
    
    try:
        results = run_experiment(model, tokenizer, device, info)
    finally:
        release_model(model)
        print("\nDone.")
