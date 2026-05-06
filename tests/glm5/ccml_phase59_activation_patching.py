"""
Phase 59: Activation Patching — 语法信息的因果验证
====================================================

这是整个项目最关键的实验: 从"观察"走向"因果控制"

核心问题: Residual stream中的语法信息是否真实影响模型输出?

实验设计:
  Exp1: Subject Patching (核心)
    - 把句子A的subject表示patch到句子B
    - 看verb位置的logit是否变化
    - 如果变化 → 语法信息被模型使用

  Exp2: Nonlinear Probe vs Linear Probe
    - 用MLP probe代替linear probe
    - 如果nonlinear显著高于linear → 语法是非线性编码的

  Exp3: 位置保持 vs 语法角色变化
    - 同位置不同角色 → 测试角色信息是否独立于位置

  Exp4: Value向量分析
    - attention × value ≠ attention mass
    - 分析value向量中是否携带语法信息

方法论原理:
  存在 ≠ 被用
  probe可读 ≠ 因果有效
  patching测的是: 是否被用
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np, gc, argparse, random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model_utils import load_model, get_model_info, release_model, safe_decode


# ============================================================
# 工具函数
# ============================================================

def get_residual_stream(model, tokenizer, device, info, text, layers=None):
    """
    Hook residual stream at specified layers for all positions.
    Returns: dict[layer_idx] -> tensor[seq_len, d_model]
    """
    if layers is None:
        layers = list(range(info.n_layers))
    
    # Get model layers based on architecture
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model_layers = model.model.layers
    else:
        raise ValueError("Cannot find model layers")
    
    activations = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is the hidden state after this layer
            # For residual stream: we want the output of the layer
            if isinstance(output, tuple):
                activations[layer_idx] = output[0].detach().clone()
            else:
                activations[layer_idx] = output.detach().clone()
        return hook_fn
    
    # Register hooks
    for li in layers:
        if li < len(model_layers):
            h = model_layers[li].register_forward_hook(make_hook(li))
            hooks.append(h)
    
    # Forward pass
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(**inputs)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Squeeze batch dimension
    result = {}
    for li in layers:
        if li in activations:
            result[li] = activations[li].squeeze(0).cpu()  # [seq_len, d_model]
    
    return result, inputs


def get_logits_with_patch(model, tokenizer, device, info, text, 
                          patch_layer, patch_pos, patch_value,
                          layers_to_monitor=None):
    """
    Run forward pass with a single position patched at a specific layer.
    Returns logits and optionally intermediate activations.
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model_layers = model.model.layers
    else:
        raise ValueError("Cannot find model layers")
    
    activations = {}
    hooks = []
    
    def make_patch_hook(target_layer, target_pos, value):
        applied = [False]
        def hook_fn(module, input, output):
            if not applied[0]:
                if isinstance(output, tuple):
                    patched = output[0].clone()
                    patched[:, target_pos, :] = value.to(patched.device)
                    applied[0] = True
                    return (patched,) + output[1:]
                else:
                    patched = output.clone()
                    patched[:, target_pos, :] = value.to(patched.device)
                    applied[0] = True
                    return patched
            return output
        return hook_fn
    
    def make_monitor_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations[layer_idx] = output[0].detach().clone()
            else:
                activations[layer_idx] = output.detach().clone()
        return hook_fn
    
    # Register patch hook
    if patch_layer < len(model_layers):
        h = model_layers[patch_layer].register_forward_hook(
            make_patch_hook(patch_layer, patch_pos, patch_value)
        )
        hooks.append(h)
    
    # Register monitor hooks
    if layers_to_monitor:
        for li in layers_to_monitor:
            if li < len(model_layers) and li != patch_layer:
                h = model_layers[li].register_forward_hook(make_monitor_hook(li))
                hooks.append(h)
    
    # Forward pass
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return logits, activations, inputs


def compute_logit_diff(logits, token_ids_a, token_ids_b):
    """Compute logit difference between sets of tokens."""
    last_logits = logits[0, -2, :]  # Position before last (verb position typically)
    score_a = torch.max(last_logits[token_ids_a]).item()
    score_b = torch.max(last_logits[token_ids_b]).item()
    return score_a - score_b


def get_token_id(tokenizer, token_str):
    """Get token ID, handling different tokenizer behaviors."""
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    return ids[0] if ids else None


# ============================================================
# Exp1: Subject Activation Patching (核心因果实验)
# ============================================================
def exp1_subject_patching(model, tokenizer, device, info):
    """
    最关键实验: Subject Patching
    
    设计:
      S1: "The cat chased the dog"
      S2: "The dog chased the cat"
    
    Patch: 把S1的"cat"表示塞进S2的"dog"位置
    
    预测:
      如果语法信息被使用 → verb行为的logit会变化
      具体地: S2中"dog"是nsubj, "cat"是dobj
      patch后: 如果"cat"的nsubj信息被传递 → verb后的token分布会偏向S1模式
    
    更精细的设计: 用subject-verb agreement来测试
      "The cat chase" (wrong) vs "The cat chases" (right)
      "The dogs chase" (right) vs "The dogs chases" (wrong)
      
      Patch subject → 看verb agreement是否改变
    """
    print("\n" + "="*70)
    print("★★★ Exp1: Subject Activation Patching — 因果验证 ★★★")
    print("="*70)
    
    # ===== 句子对设计 =====
    # 核心思路: 用subject-verb agreement测试
    
    # Set A: Singular subjects → "chases/bites/sees"
    # Set B: Plural subjects → "chase/bite/see"
    
    sentence_pairs = [
        # (singular_sent, plural_sent, singular_verb, plural_verb, subj_pos)
        ("The cat chases the mouse", "The cats chase the mouse", "chases", "chase", 2),
        ("The dog bites the man", "The dogs bite the man", "bites", "bite", 2),
        ("The girl sees the boy", "The girls see the boy", "sees", "see", 2),
        ("The bird eats the fish", "The birds eat the fish", "eats", "eat", 2),
        ("The wolf kills the deer", "The wolves kill the deer", "kills", "kill", 2),
        ("The fox catches the rabbit", "The foxes catch the rabbit", "catches", "catch", 2),
        ("The bear attacks the deer", "The bears attack the deer", "attacks", "attack", 2),
        ("The snake bites the frog", "The snakes bite the frog", "bites", "bite", 2),
    ]
    
    # Also test with adv-front (position shift)
    adv_sentence_pairs = [
        ("Quickly the cat chases the mouse", "Quickly the cats chase the mouse", "chases", "chase", 3),
        ("Slowly the dog bites the man", "Slowly the dogs bite the man", "bites", "bite", 3),
        ("Quietly the girl sees the boy", "Quietly the girls see the boy", "sees", "see", 3),
        ("Gently the bird eats the fish", "Gently the birds eat the fish", "eats", "eat", 3),
    ]
    
    # PP-front (bigger position shift)
    pp_sentence_pairs = [
        ("In the garden the cat chases the mouse", "In the garden the cats chase the mouse", "chases", "chase", 5),
        ("In the forest the dog bites the man", "In the forest the dogs bite the man", "bites", "bite", 5),
        ("On the hill the girl sees the boy", "On the hill the girls see the boy", "sees", "see", 5),
        ("By the river the bird eats the fish", "By the river the birds eat the fish", "eats", "eat", 5),
    ]
    
    test_sets = [
        ("SVO", sentence_pairs),
        ("Adv-front", adv_sentence_pairs),
        ("PP-front", pp_sentence_pairs),
    ]
    
    # Layers to test
    test_layers = [0, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25]
    
    all_results = {}
    
    for set_name, pairs in test_sets:
        print(f"\n--- {set_name} ---")
        
        layer_results = defaultdict(lambda: {
            'sing2plur_logit_diff': [],
            'plur2sing_logit_diff': [],
            'kl_div': [],
            'sing2plur_verb_change': [],
            'plur2sing_verb_change': [],
        })
        
        for sing_sent, plur_sent, sing_verb, plur_verb, subj_pos in pairs:
            # Tokenize
            sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
            plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
            
            # Find verb position (position after subject+verb_token)
            sing_tokens = tokenizer.encode(sing_sent, add_special_tokens=False)
            plur_tokens = tokenizer.encode(plur_sent, add_special_tokens=False)
            
            # Get verb token IDs
            sing_verb_ids = tokenizer.encode(sing_verb, add_special_tokens=False)
            plur_verb_ids = tokenizer.encode(plur_verb, add_special_tokens=False)
            
            if not sing_verb_ids or not plur_verb_ids:
                print(f"  Skipping: cannot encode verbs for '{sing_sent}'")
                continue
            
            # ===== Baseline: get logits without patching =====
            with torch.no_grad():
                sing_logits_base = model(**sing_ids).logits.detach().cpu()
                plur_logits_base = model(**plur_ids).logits.detach().cpu()
            
            # Baseline logit diff at verb position
            # Find verb position in token sequence
            verb_pos_sing = None
            verb_pos_plur = None
            
            for i, tid in enumerate(sing_tokens):
                decoded = tokenizer.decode([tid]).strip().lower()
                if decoded == sing_verb.lower():
                    verb_pos_sing = i + 1  # +1 for BOS
                    break
            
            for i, tid in enumerate(plur_tokens):
                decoded = tokenizer.decode([tid]).strip().lower()
                if decoded == plur_verb.lower():
                    verb_pos_plur = i + 1
                    break
            
            if verb_pos_sing is None or verb_pos_plur is None:
                # Fallback: estimate from subj_pos
                verb_pos_sing = subj_pos + 1 + 1  # subj_pos (0-indexed in words) +1 for BOS +1
                verb_pos_plur = verb_pos_sing
            
            # Adjust subj_pos for BOS token
            subj_pos_with_bos = subj_pos + 1  # account for BOS
            
            # Compute baseline agreement scores
            # For singular sentence: P(sing_verb) vs P(plur_verb)
            sing_base_verb_logits = sing_logits_base[0, verb_pos_sing, :]
            plur_base_verb_logits = plur_logits_base[0, verb_pos_plur, :]
            
            # Agreement score: logit(sing_verb) - logit(plur_verb) for singular sentence
            base_sing_agreement = (sing_base_verb_logits[sing_verb_ids[0]] - 
                                   sing_base_verb_logits[plur_verb_ids[0]]).item()
            base_plur_agreement = (plur_base_verb_logits[plur_verb_ids[0]] - 
                                   plur_base_verb_logits[sing_verb_ids[0]]).item()
            
            # ===== Patching: for each layer =====
            for layer_idx in test_layers:
                if layer_idx >= info.n_layers:
                    continue
                
                # Get residual stream activations for both sentences
                # Step 1: Get source activation (singular subject representation)
                sing_resids = {}
                plur_resids = {}
                
                def make_capture_hook(act_dict, li):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            act_dict[li] = output[0].detach().clone()
                        else:
                            act_dict[li] = output.detach().clone()
                    return hook_fn
                
                model_layers = model.model.layers
                hooks = []
                
                # Capture singular sentence activations
                h = model_layers[layer_idx].register_forward_hook(
                    make_capture_hook(sing_resids, layer_idx))
                hooks.append(h)
                
                with torch.no_grad():
                    model(**sing_ids)
                
                # Remove and re-register for plural
                hooks[0].remove()
                hooks = []
                
                h = model_layers[layer_idx].register_forward_hook(
                    make_capture_hook(plur_resids, layer_idx))
                hooks.append(h)
                
                with torch.no_grad():
                    model(**plur_ids)
                
                hooks[0].remove()
                
                if layer_idx not in sing_resids or layer_idx not in plur_resids:
                    continue
                
                # Get subject representations
                sing_subj_repr = sing_resids[layer_idx][0, subj_pos_with_bos, :]  # [d_model]
                plur_subj_repr = plur_resids[layer_idx][0, subj_pos_with_bos, :]  # [d_model]
                
                # ===== Patch 1: Singular → Plural (patch sing_subj into plur_sent) =====
                # This should make plural sentence's verb agreement shift toward singular
                patched_logits_sp = {}
                
                def make_patch_fn(value, pos, applied_flag):
                    def hook_fn(module, input, output):
                        if not applied_flag[0]:
                            if isinstance(output, tuple):
                                patched = output[0].clone()
                                patched[:, pos, :] = value.to(patched.device)
                                applied_flag[0] = True
                                return (patched,) + output[1:]
                            else:
                                patched = output.clone()
                                patched[:, pos, :] = value.to(patched.device)
                                applied_flag[0] = True
                                return patched
                        return output
                    return hook_fn
                
                flag1 = [False]
                h1 = model_layers[layer_idx].register_forward_hook(
                    make_patch_fn(sing_subj_repr, subj_pos_with_bos, flag1))
                
                with torch.no_grad():
                    patched_logits_sp = model(**plur_ids).logits.detach().cpu()
                
                h1.remove()
                
                # Patched agreement for sing→plur: should shift toward singular
                patched_sp_verb_logits = patched_logits_sp[0, verb_pos_plur, :]
                patched_sp_agreement = (patched_sp_verb_logits[sing_verb_ids[0]] - 
                                        patched_sp_verb_logits[plur_verb_ids[0]]).item()
                
                # Change in agreement (negative means shifted toward singular)
                delta_sp = patched_sp_agreement - (-base_plur_agreement)  # compare to baseline plural
                
                # ===== Patch 2: Plural → Singular (patch plur_subj into sing_sent) =====
                flag2 = [False]
                h2 = model_layers[layer_idx].register_forward_hook(
                    make_patch_fn(plur_subj_repr, subj_pos_with_bos, flag2))
                
                with torch.no_grad():
                    patched_logits_ps = model(**sing_ids).logits.detach().cpu()
                
                h2.remove()
                
                # Patched agreement for plur→sing: should shift toward plural
                patched_ps_verb_logits = patched_logits_ps[0, verb_pos_sing, :]
                patched_ps_agreement = (patched_ps_verb_logits[plur_verb_ids[0]] - 
                                        patched_ps_verb_logits[sing_verb_ids[0]]).item()
                
                # Change in agreement (negative means shifted toward plural)
                delta_ps = patched_ps_agreement - (-base_sing_agreement)
                
                # ===== KL divergence =====
                from torch.nn.functional import kl_div, log_softmax, softmax
                
                p_base = softmax(plur_logits_base[0, verb_pos_plur, :], dim=-1)
                p_patched = softmax(patched_logits_sp[0, verb_pos_plur, :], dim=-1)
                kl = kl_div(log_softmax(patched_logits_sp[0, verb_pos_plur, :], dim=-1),
                           log_softmax(plur_logits_base[0, verb_pos_plur, :], dim=-1),
                           reduction='sum').item()
                
                # ===== Check verb change =====
                # Does top-1 verb change after patch?
                top1_base = plur_logits_base[0, verb_pos_plur, :].argmax().item()
                top1_patched = patched_logits_sp[0, verb_pos_plur, :].argmax().item()
                verb_changed_sp = 1 if top1_base != top1_patched else 0
                
                top1_base_s = sing_logits_base[0, verb_pos_sing, :].argmax().item()
                top1_patched_s = patched_logits_ps[0, verb_pos_sing, :].argmax().item()
                verb_changed_ps = 1 if top1_base_s != top1_patched_s else 0
                
                # Store results
                layer_results[layer_idx]['sing2plur_logit_diff'].append(delta_sp)
                layer_results[layer_idx]['plur2sing_logit_diff'].append(delta_ps)
                layer_results[layer_idx]['kl_div'].append(kl)
                layer_results[layer_idx]['sing2plur_verb_change'].append(verb_changed_sp)
                layer_results[layer_idx]['plur2sing_verb_change'].append(verb_changed_ps)
            
            # Progress
            print(f"  '{sing_sent[:30]}...' done", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        
        # Aggregate results
        print(f"\n  {'Layer':>6} | {'S→P logitΔ':>12} | {'P→S logitΔ':>12} | {'KL div':>10} | {'S→P verb%':>10} | {'P→S verb%':>10}")
        print(f"  {'-'*70}")
        
        for li in sorted(layer_results.keys()):
            r = layer_results[li]
            sp_ld = np.mean(r['sing2plur_logit_diff']) if r['sing2plur_logit_diff'] else 0
            ps_ld = np.mean(r['plur2sing_logit_diff']) if r['plur2sing_logit_diff'] else 0
            kl = np.mean(r['kl_div']) if r['kl_div'] else 0
            sp_vc = np.mean(r['sing2plur_verb_change']) * 100 if r['sing2plur_verb_change'] else 0
            ps_vc = np.mean(r['plur2sing_verb_change']) * 100 if r['plur2sing_verb_change'] else 0
            
            print(f"  L{li:>4} | {sp_ld:>12.4f} | {ps_ld:>12.4f} | {kl:>10.4f} | {sp_vc:>9.1f}% | {ps_vc:>9.1f}%")
        
        all_results[set_name] = dict(layer_results)
    
    # ===== Cross-position patching =====
    print("\n" + "="*70)
    print("★★★ Exp1b: Cross-Position Subject Patching ★★★")
    print("="*70)
    print("Test: Patch SVO subject into Adv-front subject (different position)")
    print("If syntax info is position-invariant → should still affect verb")
    
    cross_pairs = [
        # (svo_sing, adv_plur, verb, svo_subj_pos, adv_subj_pos)
        ("The cat chases the mouse", "Quickly the cats chase the mouse", "chases", "chase", 2, 3),
        ("The dog bites the man", "Slowly the dogs bite the man", "bites", "bite", 2, 3),
        ("The girl sees the boy", "Quietly the girls see the boy", "sees", "see", 2, 3),
        ("The bird eats the fish", "Gently the birds eat the fish", "eats", "eat", 2, 3),
    ]
    
    cross_layer_results = defaultdict(lambda: {
        'logit_diff': [],
        'verb_change': [],
    })
    
    for svo_sing, adv_plur, sing_verb, plur_verb, svo_subj_pos, adv_subj_pos in cross_pairs:
        svo_ids = tokenizer(svo_sing, return_tensors="pt").to(device)
        adv_ids = tokenizer(adv_plur, return_tensors="pt").to(device)
        
        sing_verb_ids = tokenizer.encode(sing_verb, add_special_tokens=False)
        plur_verb_ids = tokenizer.encode(plur_verb, add_special_tokens=False)
        
        if not sing_verb_ids or not plur_verb_ids:
            continue
        
        # Find verb position in adv sentence
        adv_tokens = tokenizer.encode(adv_plur, add_special_tokens=False)
        verb_pos = None
        for i, tid in enumerate(adv_tokens):
            decoded = tokenizer.decode([tid]).strip().lower()
            if decoded in [plur_verb.lower(), sing_verb.lower()]:
                verb_pos = i + 1
                break
        if verb_pos is None:
            verb_pos = adv_subj_pos + 2  # estimate
        
        svo_subj_bos = svo_subj_pos + 1
        adv_subj_bos = adv_subj_pos + 1
        
        # Baseline
        with torch.no_grad():
            adv_logits_base = model(**adv_ids).logits.detach().cpu()
        
        for layer_idx in test_layers:
            if layer_idx >= info.n_layers:
                continue
            
            # Get SVO subject repr
            svo_resids = {}
            def make_cap(d, li):
                def h(m, i, o):
                    d[li] = (o[0] if isinstance(o, tuple) else o).detach().clone()
                return h
            
            model_layers = model.model.layers
            h1 = model_layers[layer_idx].register_forward_hook(make_cap(svo_resids, layer_idx))
            with torch.no_grad():
                model(**svo_ids)
            h1.remove()
            
            if layer_idx not in svo_resids:
                continue
            
            svo_subj_repr = svo_resids[layer_idx][0, svo_subj_bos, :]
            
            # Patch into adv sentence at adv subject position
            flag = [False]
            def make_patch(value, pos, fl):
                def h(m, i, o):
                    if not fl[0]:
                        if isinstance(o, tuple):
                            p = o[0].clone()
                            p[:, pos, :] = value.to(p.device)
                            fl[0] = True
                            return (p,) + o[1:]
                        else:
                            p = o.clone()
                            p[:, pos, :] = value.to(p.device)
                            fl[0] = True
                            return p
                    return o
                return h
            
            h2 = model_layers[layer_idx].register_forward_hook(
                make_patch(svo_subj_repr, adv_subj_bos, flag))
            with torch.no_grad():
                adv_logits_patched = model(**adv_ids).logits.detach().cpu()
            h2.remove()
            
            # Measure change
            base_agr = (adv_logits_base[0, verb_pos, plur_verb_ids[0]] - 
                       adv_logits_base[0, verb_pos, sing_verb_ids[0]]).item()
            patched_agr = (adv_logits_patched[0, verb_pos, plur_verb_ids[0]] - 
                          adv_logits_patched[0, verb_pos, sing_verb_ids[0]]).item()
            
            delta = patched_agr - base_agr
            
            top1_base = adv_logits_base[0, verb_pos, :].argmax().item()
            top1_patch = adv_logits_patched[0, verb_pos, :].argmax().item()
            vc = 1 if top1_base != top1_patch else 0
            
            cross_layer_results[layer_idx]['logit_diff'].append(delta)
            cross_layer_results[layer_idx]['verb_change'].append(vc)
        
        print(f"  Cross-patch '{svo_sing[:20]}...' → '{adv_plur[:20]}...' done", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"\n  Cross-Position Patching Results:")
    print(f"  {'Layer':>6} | {'LogitΔ':>10} | {'Verb change%':>14}")
    print(f"  {'-'*40}")
    for li in sorted(cross_layer_results.keys()):
        r = cross_layer_results[li]
        ld = np.mean(r['logit_diff']) if r['logit_diff'] else 0
        vc = np.mean(r['verb_change']) * 100 if r['verb_change'] else 0
        print(f"  L{li:>4} | {ld:>10.4f} | {vc:>13.1f}%")
    
    return all_results, cross_layer_results


# ============================================================
# Exp2: Nonlinear Probe vs Linear Probe
# ============================================================
def exp2_nonlinear_probe(model, tokenizer, device, info):
    """
    关键测试: 语法信息是否是非线性编码的?
    
    逻辑:
      linear probe = 50% (随机) 
      nonlinear probe = ? 
      
      如果 nonlinear >> linear → 语法信息存在但是非线性编码
      如果 nonlinear ≈ linear → 语法信息确实弱
    """
    print("\n" + "="*70)
    print("★★★ Exp2: Nonlinear vs Linear Probe ★★★")
    print("="*70)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    
    # Data: multiple sentence types with nsubj at different positions
    sentences = {
        "SVO": [
            ("The cat chased the mouse", {2: "nsubj", 3: "verb", 5: "dobj"}),
            ("The dog bit the man", {2: "nsubj", 3: "verb", 5: "dobj"}),
            ("The girl saw the boy", {2: "nsubj", 3: "verb", 5: "dobj"}),
            ("The bird ate the fish", {2: "nsubj", 3: "verb", 5: "dobj"}),
            ("The wolf killed the deer", {2: "nsubj", 3: "verb", 5: "dobj"}),
            ("The fox caught the rabbit", {2: "nsubj", 3: "verb", 5: "dobj"}),
            ("The bear attacked the deer", {2: "nsubj", 3: "verb", 5: "dobj"}),
            ("The snake bit the frog", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ],
        "Adv": [
            ("Quickly the cat chased the mouse", {3: "nsubj", 4: "verb", 6: "dobj"}),
            ("Slowly the dog bit the man", {3: "nsubj", 4: "verb", 6: "dobj"}),
            ("Quietly the girl saw the boy", {3: "nsubj", 4: "verb", 6: "dobj"}),
            ("Gently the bird ate the fish", {3: "nsubj", 4: "verb", 6: "dobj"}),
            ("Fiercely the wolf killed the deer", {3: "nsubj", 4: "verb", 6: "dobj"}),
            ("Swiftly the fox caught the rabbit", {3: "nsubj", 4: "verb", 6: "dobj"}),
            ("Boldly the bear attacked the deer", {3: "nsubj", 4: "verb", 6: "dobj"}),
            ("Softly the snake bit the frog", {3: "nsubj", 4: "verb", 6: "dobj"}),
        ],
        "PP": [
            ("In the garden the cat chased the mouse", {5: "nsubj", 7: "verb", 9: "dobj"}),
            ("In the forest the dog bit the man", {5: "nsubj", 7: "verb", 9: "dobj"}),
            ("On the hill the girl saw the boy", {5: "nsubj", 7: "verb", 9: "dobj"}),
            ("By the river the bird ate the fish", {5: "nsubj", 7: "verb", 9: "dobj"}),
            ("In the field the wolf killed the deer", {5: "nsubj", 7: "verb", 9: "dobj"}),
            ("At the park the fox caught the rabbit", {5: "nsubj", 7: "verb", 9: "dobj"}),
            ("In the cave the bear attacked the deer", {5: "nsubj", 7: "verb", 9: "dobj"}),
            ("Near the lake the snake bit the frog", {5: "nsubj", 7: "verb", 9: "dobj"}),
        ],
        "CE": [
            ("The cat that the dog chased ran", {2: "nsubj", 5: "dobj", 6: "verb"}),
            ("The bird that the cat caught flew", {2: "nsubj", 5: "dobj", 6: "verb"}),
            ("The man that the bear attacked ran", {2: "nsubj", 5: "dobj", 6: "verb"}),
            ("The fish that the bird ate swam", {2: "nsubj", 5: "dobj", 6: "verb"}),
            ("The deer that the wolf killed fell", {2: "nsubj", 5: "dobj", 6: "verb"}),
            ("The rabbit that the fox caught ran", {2: "nsubj", 5: "dobj", 6: "verb"}),
            ("The frog that the snake bit jumped", {2: "nsubj", 5: "dobj", 6: "verb"}),
            ("The mouse that the cat chased hid", {2: "nsubj", 5: "dobj", 6: "verb"}),
        ],
    }
    
    role_to_id = {"nsubj": 0, "dobj": 1, "verb": 2, "other": 3}
    test_layers = [0, 2, 5, 10, 15, 20, 25]
    
    results = {}
    
    for layer_idx in test_layers:
        if layer_idx >= info.n_layers:
            continue
        
        print(f"\n  --- Layer {layer_idx} ---")
        
        # Collect representations
        X_data = []  # features
        y_data = []  # labels (nsubj/dobj/verb/other)
        pos_data = []  # position (for control)
        
        for stype, sents in sentences.items():
            for sent, role_map in sents:
                # Get residual stream
                resids = {}
                def make_cap(d, li):
                    def h(m, i, o):
                        d[li] = (o[0] if isinstance(o, tuple) else o).detach().clone()
                    return h
                
                model_layers = model.model.layers
                h = model_layers[layer_idx].register_forward_hook(make_cap(resids, layer_idx))
                
                with torch.no_grad():
                    inputs = tokenizer(sent, return_tensors="pt").to(device)
                    model(**inputs)
                
                h.remove()
                
                if layer_idx not in resids:
                    continue
                
                hidden = resids[layer_idx].squeeze(0).cpu().float().numpy()  # [seq_len, d_model]
                
                # Map positions (accounting for BOS)
                bos_offset = 1
                for pos, role in role_map.items():
                    token_pos = pos + bos_offset
                    if token_pos < hidden.shape[0]:
                        X_data.append(hidden[token_pos])
                        y_data.append(role_to_id[role])
                        pos_data.append(pos)
                
                # Add "other" tokens
                all_labeled = set(pos + bos_offset for pos in role_map.keys())
                for i in range(hidden.shape[0]):
                    if i not in all_labeled and i > 0:  # skip BOS
                        X_data.append(hidden[i])
                        y_data.append(role_to_id["other"])
                        pos_data.append(i - bos_offset)
                
                gc.collect()
                torch.cuda.empty_cache()
        
        if len(X_data) < 20:
            print(f"  Not enough data ({len(X_data)} samples), skipping")
            continue
        
        X = np.array(X_data)
        y = np.array(y_data)
        pos = np.array(pos_data)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # ===== Test 1: Linear Probe (same-position) =====
        # Train on all data, test on nsubj/dobj only
        nsubj_mask = y == 0
        dobj_mask = y == 1
        
        # For nsubj vs dobj binary classification
        nd_mask = (y == 0) | (y == 1)
        if nd_mask.sum() < 10:
            print(f"  Not enough nsubj/dobj samples")
            continue
        
        X_nd = X_scaled[nd_mask]
        y_nd = y[nd_mask]
        pos_nd = pos[nd_mask]
        
        # Linear probe
        try:
            linear_scores = cross_val_score(
                LogisticRegression(max_iter=1000, C=1.0),
                X_nd, y_nd, cv=min(3, len(set(y_nd))), scoring='accuracy'
            )
            linear_acc = linear_scores.mean()
        except:
            linear_acc = 0.5
        
        # Nonlinear probe (small MLP)
        try:
            mlp_scores = cross_val_score(
                MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, 
                             early_stopping=True, validation_fraction=0.2),
                X_nd, y_nd, cv=min(3, len(set(y_nd))), scoring='accuracy'
            )
            mlp_acc = mlp_scores.mean()
        except:
            mlp_acc = 0.5
        
        # ===== Test 2: Cross-position probe =====
        # Train on SVO (pos2=nsubj), test on Adv (pos3=nsubj)
        svo_nsubj_mask = (y == 0) & (pos == 2)
        adv_nsubj_mask = (y == 0) & (pos == 3)
        svo_dobj_mask = (y == 1) & (pos == 5)
        adv_dobj_mask = (y == 1) & (pos == 6)
        
        # SVO train: nsubj@pos2 vs dobj@pos5
        svo_train_mask = svo_nsubj_mask | svo_dobj_mask
        # Adv test: nsubj@pos3 vs dobj@pos6
        adv_test_mask = adv_nsubj_mask | adv_dobj_mask
        
        cross_linear_acc = 0.5
        cross_mlp_acc = 0.5
        
        if svo_train_mask.sum() >= 4 and adv_test_mask.sum() >= 4:
            X_svo_train = X_scaled[svo_train_mask]
            y_svo_train = y[svo_train_mask]
            X_adv_test = X_scaled[adv_test_mask]
            y_adv_test = y[adv_test_mask]
            
            # Linear
            try:
                lr = LogisticRegression(max_iter=1000, C=1.0)
                lr.fit(X_svo_train, y_svo_train)
                cross_linear_acc = lr.score(X_adv_test, y_adv_test)
            except:
                pass
            
            # MLP
            try:
                mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                   early_stopping=True)
                mlp.fit(X_svo_train, y_svo_train)
                cross_mlp_acc = mlp.score(X_adv_test, y_adv_test)
            except:
                pass
        
        # ===== Test 3: Position-controlled (remove position signal) =====
        # Idea: project out the position direction, then probe
        # Use PCA: remove top-3 components (position-dominated), then probe
        
        # Fit PCA on all data
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=min(50, X_scaled.shape[1], X_scaled.shape[0]))
        X_pca = pca.fit_transform(X_scaled)
        
        # Remove top-3 components (presumably position-dominated)
        if X_pca.shape[1] > 5:
            # Reconstruct without top-3 PCs
            X_residual = pca.inverse_transform(X_pca) - pca.inverse_transform(
                np.column_stack([X_pca[:, :3], np.zeros((X_pca.shape[0], X_pca.shape[1]-3))])
            )
            # Actually, let's just use components 3 onwards
            X_pc_removed = X_pca[:, 3:]
            
            if nd_mask.sum() >= 10:
                X_pc_nd = X_pc_removed[nd_mask]
                
                # Linear on PC-removed
                try:
                    lr_pc = LogisticRegression(max_iter=1000, C=1.0)
                    pc_scores = cross_val_score(lr_pc, X_pc_nd, y_nd, 
                                               cv=min(3, len(set(y_nd))), scoring='accuracy')
                    pc_removed_linear_acc = pc_scores.mean()
                except:
                    pc_removed_linear_acc = 0.5
                
                # MLP on PC-removed
                try:
                    mlp_pc = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                          early_stopping=True)
                    pc_mlp_scores = cross_val_score(mlp_pc, X_pc_nd, y_nd,
                                                    cv=min(3, len(set(y_nd))), scoring='accuracy')
                    pc_removed_mlp_acc = pc_mlp_scores.mean()
                except:
                    pc_removed_mlp_acc = 0.5
            else:
                pc_removed_linear_acc = 0.5
                pc_removed_mlp_acc = 0.5
        else:
            pc_removed_linear_acc = 0.5
            pc_removed_mlp_acc = 0.5
        
        results[layer_idx] = {
            'linear_same': linear_acc,
            'mlp_same': mlp_acc,
            'linear_cross': cross_linear_acc,
            'mlp_cross': cross_mlp_acc,
            'linear_pc_removed': pc_removed_linear_acc,
            'mlp_pc_removed': pc_removed_mlp_acc,
        }
        
        print(f"  Same-position:  Linear={linear_acc:.3f}  MLP={mlp_acc:.3f}  Δ={mlp_acc-linear_acc:+.3f}")
        print(f"  Cross-position: Linear={cross_linear_acc:.3f}  MLP={cross_mlp_acc:.3f}  Δ={cross_mlp_acc-cross_linear_acc:+.3f}")
        print(f"  PC-removed:     Linear={pc_removed_linear_acc:.3f}  MLP={pc_removed_mlp_acc:.3f}  Δ={pc_removed_mlp_acc-pc_removed_linear_acc:+.3f}")
    
    return results


# ============================================================
# Exp3: Value Vector Analysis (attention × value)
# ============================================================
def exp3_value_vector_analysis(model, tokenizer, device, info):
    """
    关键纠正: attention mass ≠ 信息贡献
    
    attention权重只决定"从哪个token取信息"
    value向量决定"取什么信息"
    
    本实验: 分析value向量中是否包含语法信息
    """
    print("\n" + "="*70)
    print("★★★ Exp3: Value Vector Analysis ★★★")
    print("="*70)
    print("Testing: Does value = f(residual) carry syntax information?")
    print("If yes: low attention mass ≠ low information contribution")
    
    sentences = [
        ("The cat chased the mouse", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ("The dog bit the man", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ("The girl saw the boy", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ("The bird ate the fish", {2: "nsubj", 3: "verb", 5: "dobj"}),
    ]
    
    test_layers = [0, 5, 10, 15, 20, 25]
    model_layers = model.model.layers
    
    results = {}
    
    for layer_idx in test_layers:
        if layer_idx >= info.n_layers:
            continue
        
        print(f"\n  --- Layer {layer_idx} ---")
        
        nsubj_values = []
        dobj_values = []
        other_values = []
        verb_values = []
        
        for sent, role_map in sentences:
            # Hook attention output and value projection
            attn_weights = {}
            attn_outputs = {}
            mlp_outputs = {}
            
            def make_attn_hook(li):
                def hook_fn(module, input, output):
                    # output of attention: (hidden_states, attn_weights, past_key_value)
                    if isinstance(output, tuple) and len(output) >= 2:
                        attn_weights[li] = output[1].detach().clone() if output[1] is not None else None
                        attn_outputs[li] = output[0].detach().clone()
                    else:
                        attn_outputs[li] = output.detach().clone() if not isinstance(output, tuple) else output[0].detach().clone()
                return hook_fn
            
            def make_mlp_hook(li):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        mlp_outputs[li] = output[0].detach().clone()
                    else:
                        mlp_outputs[li] = output.detach().clone()
                return hook_fn
            
            # Register hooks on self_attn and mlp
            layer = model_layers[layer_idx]
            h1 = layer.self_attn.register_forward_hook(make_attn_hook(layer_idx))
            h2 = layer.mlp.register_forward_hook(make_mlp_hook(layer_idx))
            
            with torch.no_grad():
                inputs = tokenizer(sent, return_tensors="pt").to(device)
                model(**inputs)
            
            h1.remove()
            h2.remove()
            
            if layer_idx not in attn_outputs:
                continue
            
            # Attention output = weighted sum of value vectors
            attn_out = attn_outputs[layer_idx].squeeze(0).cpu().numpy()  # [seq_len, d_model]
            
            bos_offset = 1
            for pos, role in role_map.items():
                token_pos = pos + bos_offset
                if token_pos < attn_out.shape[0]:
                    if role == "nsubj":
                        nsubj_values.append(attn_out[token_pos])
                    elif role == "dobj":
                        dobj_values.append(attn_out[token_pos])
                    elif role == "verb":
                        verb_values.append(attn_out[token_pos])
            
            # Other tokens
            all_labeled = set(pos + bos_offset for pos in role_map.keys())
            for i in range(attn_out.shape[0]):
                if i not in all_labeled and i > 0:
                    other_values.append(attn_out[i])
            
            gc.collect()
            torch.cuda.empty_cache()
        
        if not nsubj_values or not dobj_values:
            print(f"  Insufficient data")
            continue
        
        # Compute cosine distances
        nsubj_arr = np.array(nsubj_values)
        dobj_arr = np.array(dobj_values)
        other_arr = np.array(other_values) if other_values else np.array(verb_values)
        
        from scipy.spatial.distance import cosine
        
        # nsubj vs dobj
        nsubj_dobj_dists = []
        for ns in nsubj_arr:
            for db in dobj_arr:
                nsubj_dobj_dists.append(1 - cosine(ns, db))
        
        # nsubj vs other
        nsubj_other_dists = []
        for ns in nsubj_arr:
            for ot in other_arr[:len(nsubj_arr)*len(dobj_arr)]:
                nsubj_other_dists.append(1 - cosine(ns, ot))
        
        # Random pairs
        all_vecs = np.vstack([nsubj_arr, dobj_arr, other_arr])
        random_dists = []
        for _ in range(100):
            i, j = random.sample(range(len(all_vecs)), 2)
            random_dists.append(1 - cosine(all_vecs[i], all_vecs[j]))
        
        ns_db_mean = np.mean(nsubj_dobj_dists) if nsubj_dobj_dists else 0
        ns_ot_mean = np.mean(nsubj_other_dists) if nsubj_other_dists else 0
        rand_mean = np.mean(random_dists) if random_dists else 0
        
        # Can we classify nsubj vs dobj from attention output?
        if len(nsubj_arr) >= 2 and len(dobj_arr) >= 2:
            X_attn = np.vstack([nsubj_arr, dobj_arr])
            y_attn = np.array([0]*len(nsubj_arr) + [1]*len(dobj_arr))
            
            try:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(max_iter=1000, C=1.0)
                scores = cross_val_score(lr, X_attn, y_attn, cv=min(2, len(set(y_attn))), scoring='accuracy')
                attn_cls_acc = scores.mean()
            except:
                attn_cls_acc = 0.5
        else:
            attn_cls_acc = 0.5
        
        results[layer_idx] = {
            'nsubj_dobj_cosine': ns_db_mean,
            'nsubj_other_cosine': ns_ot_mean,
            'random_cosine': rand_mean,
            'attn_output_cls': attn_cls_acc,
        }
        
        print(f"  Attn output cosine: nsubj-dobj={ns_db_mean:.3f}  nsubj-other={ns_ot_mean:.3f}  random={rand_mean:.3f}")
        print(f"  Attn output classification: {attn_cls_acc:.3f}")
    
    return results


# ============================================================
# Exp4: Random Patch Control (Negative Control)
# ============================================================
def exp4_random_patch_control(model, tokenizer, device, info):
    """
    负对照: 随机位置的patch是否也能改变输出?
    
    如果随机patch也有大影响 → patch本身就在破坏
    如果只有subject patch有影响 → 语法信息的特异性
    """
    print("\n" + "="*70)
    print("★★★ Exp4: Random Patch Control (Negative Control) ★★★")
    print("="*70)
    
    sentence = "The cat chases the mouse"
    # Token positions: The(1) cat(2) chases(3) the(4) mouse(5)
    
    test_layers = [0, 5, 10, 15, 20, 25]
    
    # Get verb tokens
    sing_verb_ids = tokenizer.encode("chases", add_special_tokens=False)
    plur_verb_ids = tokenizer.encode("chase", add_special_tokens=False)
    if not sing_verb_ids or not plur_verb_ids:
        print("  Cannot encode verbs, skipping")
        return {}
    
    # Baseline
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        base_logits = model(**inputs).logits.detach().cpu()
    
    # Verb position (after subject + verb token)
    verb_pos = 3 + 1  # chases at word pos 3 → token pos 4 (with BOS)
    subj_pos = 2 + 1  # cat at word pos 2 → token pos 3
    
    base_agreement = (base_logits[0, verb_pos, sing_verb_ids[0]] - 
                     base_logits[0, verb_pos, plur_verb_ids[0]]).item()
    
    results = {}
    
    # Get plural subject sentence
    plur_sentence = "The cats chase the mouse"
    plur_inputs = tokenizer(plur_sentence, return_tensors="pt").to(device)
    
    plur_verb_pos = verb_pos  # same relative position
    
    with torch.no_grad():
        plur_logits = model(**plur_inputs).logits.detach().cpu()
    
    plur_agreement = (plur_logits[0, plur_verb_pos, plur_verb_ids[0]] - 
                     plur_logits[0, plur_verb_pos, sing_verb_ids[0]]).item()
    
    print(f"  Baseline sing agreement: {base_agreement:.4f}")
    print(f"  Baseline plur agreement: {plur_agreement:.4f}")
    
    for layer_idx in test_layers:
        if layer_idx >= info.n_layers:
            continue
        
        # Get plural subject representation
        plur_resids = {}
        def make_cap(d, li):
            def h(m, i, o):
                d[li] = (o[0] if isinstance(o, tuple) else o).detach().clone()
            return h
        
        model_layers = model.model.layers
        h1 = model_layers[layer_idx].register_forward_hook(make_cap(plur_resids, layer_idx))
        with torch.no_grad():
            model(**plur_inputs)
        h1.remove()
        
        if layer_idx not in plur_resids:
            continue
        
        # Patch 1: Subject position (cat → cats repr)
        plur_subj_repr = plur_resids[layer_idx][0, subj_pos, :]
        
        flag = [False]
        def make_patch(value, pos, fl):
            def h(m, i, o):
                if not fl[0]:
                    if isinstance(o, tuple):
                        p = o[0].clone()
                        p[:, pos, :] = value.to(p.device)
                        fl[0] = True
                        return (p,) + o[1:]
                    else:
                        p = o.clone()
                        p[:, pos, :] = value.to(p.device)
                        fl[0] = True
                        return p
                return o
            return h
        
        h2 = model_layers[layer_idx].register_forward_hook(
            make_patch(plur_subj_repr, subj_pos, flag))
        with torch.no_grad():
            subj_patched_logits = model(**inputs).logits.detach().cpu()
        h2.remove()
        
        subj_patched_agreement = (subj_patched_logits[0, verb_pos, sing_verb_ids[0]] - 
                                  subj_patched_logits[0, verb_pos, plur_verb_ids[0]]).item()
        subj_delta = subj_patched_agreement - base_agreement
        
        # Patch 2: Random position (verb position, NOT subject)
        # Use same plural representation but at different position
        # This controls for "patching itself disrupts"
        random_pos = verb_pos  # verb position
        plur_verb_repr = plur_resids[layer_idx][0, random_pos, :]
        
        flag3 = [False]
        h3 = model_layers[layer_idx].register_forward_hook(
            make_patch(plur_verb_repr, random_pos, flag3))
        with torch.no_grad():
            rand_patched_logits = model(**inputs).logits.detach().cpu()
        h3.remove()
        
        rand_patched_agreement = (rand_patched_logits[0, verb_pos, sing_verb_ids[0]] - 
                                  rand_patched_logits[0, verb_pos, plur_verb_ids[0]]).item()
        rand_delta = rand_patched_agreement - base_agreement
        
        # Patch 3: Object position
        obj_pos = 5 + 1  # mouse at word pos 5 → token pos 6
        plur_obj_repr = plur_resids[layer_idx][0, obj_pos, :] if obj_pos < plur_resids[layer_idx].shape[1] else None
        
        obj_delta = 0
        if plur_obj_repr is not None:
            flag4 = [False]
            h4 = model_layers[layer_idx].register_forward_hook(
                make_patch(plur_obj_repr, obj_pos, flag4))
            with torch.no_grad():
                obj_patched_logits = model(**inputs).logits.detach().cpu()
            h4.remove()
            
            obj_patched_agreement = (obj_patched_logits[0, verb_pos, sing_verb_ids[0]] - 
                                    obj_patched_logits[0, verb_pos, plur_verb_ids[0]]).item()
            obj_delta = obj_patched_agreement - base_agreement
        
        results[layer_idx] = {
            'subj_delta': subj_delta,
            'obj_delta': obj_delta,
            'random_delta': rand_delta,
        }
        
        print(f"  L{layer_idx}: subj_patch Δ={subj_delta:+.4f}  obj_patch Δ={obj_delta:+.4f}  verb_patch Δ={rand_delta:+.4f}")
        print(f"         specificity = {subj_delta / (abs(rand_delta) + 0.001):.2f}x (subj vs random)")
    
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 59: Activation Patching")
    parser.add_argument("--model", type=str, default="deepseek7b",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, required=True,
                       choices=["1", "2", "3", "4", "all"],
                       help="Experiment to run")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"Model: {info.name}, Layers={info.n_layers}, d_model={info.d_model}")
    
    try:
        if args.exp in ["1", "all"]:
            results, cross_results = exp1_subject_patching(model, tokenizer, device, info)
        
        if args.exp in ["2", "all"]:
            nl_results = exp2_nonlinear_probe(model, tokenizer, device, info)
        
        if args.exp in ["3", "all"]:
            val_results = exp3_value_vector_analysis(model, tokenizer, device, info)
        
        if args.exp in ["4", "all"]:
            ctrl_results = exp4_random_patch_control(model, tokenizer, device, info)
    
    finally:
        release_model(model)
        print("\nModel released.")
