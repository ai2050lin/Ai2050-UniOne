"""
Phase 60: Syntax Direction Extraction + Precise Patching
=========================================================
核心问题: 语法信息是否是位置不变的隐变量?

方法突破 (vs Phase 59):
  Phase 59: 替换整个向量 → 导致distribution shift (位置编码/上下文破坏)
  Phase 60: 只加语法方向 → 保留位置/上下文, 避免distribution shift

实验设计:
  1. 提取"number方向": 同位置sing/plur主语的均值差
  2. 提取"position方向": 不同位置的均值差
  3. 正交化: 去掉position分量的number方向
  4. 同位置方向patching (验证方向有效)
  5. 跨位置方向patching (核心测试: 语法是否位置不变?)
  6. 控制实验: 错误位置/position方向/全向量替换

数据量: 30 noun-verb对 × 2结构 × 2数量 = 120句
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np, gc, argparse, time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model_utils import load_model, get_model_info, release_model, get_layers

# ===== 句子生成 =====
NVA_PAIRS = [
    # (sing_noun, plur_noun, sing_verb, plur_verb, adverb)
    ("cat", "cats", "runs", "run", "fast"),
    ("dog", "dogs", "walks", "walk", "home"),
    ("bird", "birds", "flies", "fly", "high"),
    ("girl", "girls", "reads", "read", "well"),
    ("boy", "boys", "sings", "sing", "loud"),
    ("man", "men", "works", "work", "hard"),
    ("horse", "horses", "jumps", "jump", "far"),
    ("bear", "bears", "sleeps", "sleep", "long"),
    ("snake", "snakes", "crawls", "crawl", "slow"),
    ("frog", "frogs", "swims", "swim", "deep"),
    ("fox", "foxes", "hunts", "hunt", "alone"),
    ("king", "kings", "rules", "rule", "well"),
    ("student", "students", "studies", "study", "hard"),
    ("teacher", "teachers", "speaks", "speak", "clear"),
    ("doctor", "doctors", "helps", "help", "often"),
    ("tree", "trees", "grows", "grow", "tall"),
    ("car", "cars", "moves", "move", "fast"),
    ("queen", "queens", "leads", "lead", "now"),
    ("child", "children", "plays", "play", "here"),
    ("wolf", "wolves", "howls", "howl", "night"),
    ("driver", "drivers", "drives", "drive", "slow"),
    ("worker", "workers", "builds", "build", "fast"),
    ("player", "players", "wins", "win", "often"),
    ("writer", "writers", "writes", "write", "daily"),
    ("rabbit", "rabbits", "hops", "hop", "fast"),
    ("eagle", "eagles", "soars", "soar", "high"),
    ("tiger", "tigers", "stalks", "stalk", "quiet"),
    ("monkey", "monkeys", "climbs", "climb", "up"),
    ("lion", "lions", "roars", "roar", "loud"),
    ("farmer", "farmers", "plants", "plant", "early"),
]

def generate_sentences():
    """生成所有句子对, 返回结构化数据"""
    svo_data = []  # [(sing_sent, plur_sent, subj_pos, verb_pos, sv, pv, noun_s, noun_p)]
    adv_data = []
    
    for sn, pn, sv, pv, adv in NVA_PAIRS:
        # SVO: "The {noun} {verb} {adverb}"
        svo_sing = f"The {sn} {sv} {adv}"
        svo_plur = f"The {pn} {pv} {adv}"
        svo_data.append((svo_sing, svo_plur, sv, pv, sn, pn))
        
        # Adv-front: "Today the {noun} {verb} {adverb}"
        adv_sing = f"Today the {sn} {sv} {adv}"
        adv_plur = f"Today the {pn} {pv} {adv}"
        adv_data.append((adv_sing, adv_plur, sv, pv, sn, pn))
    
    return svo_data, adv_data


def find_token_position(tokenizer, tokens, target_word):
    """在已tokenize的tokens中找到target_word的位置"""
    decoded = [tokenizer.decode([t]).strip() for t in tokens]
    for i, d in enumerate(decoded):
        if d.lower() == target_word.lower() or d.lower().startswith(target_word.lower()):
            return i + 1  # +1 for BOS
    return None


def collect_activations(model, tokenizer, device, sentences, target_pos_fn, 
                        target_layers, batch_info=""):
    """
    收集句子在目标位置和目标层的activations
    
    Args:
        sentences: list of sentence strings
        target_pos_fn: function(tokenizer, tokens) -> (subj_pos, verb_pos)
        target_layers: list of layer indices
    
    Returns:
        dict: {layer_idx: {"subj": [vectors], "verb": [vectors], "full": [vectors]}}
    """
    layers = get_layers(model)
    results = defaultdict(lambda: {"subj": [], "verb": [], "full": []})
    valid_count = 0
    
    for sent_idx, sent in enumerate(sentences):
        if sent_idx % 10 == 0 and sent_idx > 0:
            print(f"  {batch_info} {sent_idx}/{len(sentences)} done")
        
        toks = tokenizer.encode(sent, add_special_tokens=False)
        subj_pos, verb_pos = target_pos_fn(tokenizer, toks)
        
        if subj_pos is None or verb_pos is None:
            continue
        
        # Hook to capture layer outputs
        captured = {}
        def make_hook(li):
            def fn(m, inp, out):
                if isinstance(out, tuple):
                    captured[li] = out[0].detach().cpu()
                else:
                    captured[li] = out.detach().cpu()
            return fn
        
        hooks = []
        for li in target_layers:
            if li < len(layers):
                hooks.append(layers[li].register_forward_hook(make_hook(li)))
        
        input_ids = tokenizer(sent, return_tensors="pt").to(device)
        with torch.no_grad():
            try:
                model(**input_ids)
            except Exception as e:
                print(f"  ERROR on '{sent[:40]}...': {e}")
                for h in hooks:
                    h.remove()
                continue
        
        for h in hooks:
            h.remove()
        
        for li in target_layers:
            if li in captured:
                h = captured[li]  # [1, seq_len, d_model]
                if subj_pos < h.shape[1]:
                    results[li]["subj"].append(h[0, subj_pos, :].float().numpy())
                if verb_pos < h.shape[1]:
                    results[li]["verb"].append(h[0, verb_pos, :].float().numpy())
        
        valid_count += 1
        del captured
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"  {batch_info} {valid_count}/{len(sentences)} valid sentences")
    return results, valid_count


def extract_directions(svo_sing_act, svo_plur_act, adv_sing_act, adv_plur_act, layer):
    """
    提取number方向和position方向
    
    Returns:
        d_num_pos2: number方向 (从SVO pos2提取)
        d_num_pos3: number方向 (从Adv pos3提取)
        d_pos: position方向 (pos2 vs pos3)
        d_num_clean_pos2: 去掉position分量的number方向
        d_num_clean_pos3: 同上
        diagnostics: 诊断信息
    """
    # SVO: subject at pos2
    sing_pos2 = np.array(svo_sing_act[layer]["subj"])  # [N, d_model]
    plur_pos2 = np.array(svo_plur_act[layer]["subj"])   # [N, d_model]
    
    # Adv-front: subject at pos3
    sing_pos3 = np.array(adv_sing_act[layer]["subj"])   # [N, d_model]
    plur_pos3 = np.array(adv_plur_act[layer]["subj"])   # [N, d_model]
    
    # Number direction at pos2
    mean_sing_pos2 = sing_pos2.mean(axis=0)
    mean_plur_pos2 = plur_pos2.mean(axis=0)
    d_num_pos2 = mean_plur_pos2 - mean_sing_pos2
    
    # Number direction at pos3
    mean_sing_pos3 = sing_pos3.mean(axis=0)
    mean_plur_pos3 = plur_pos3.mean(axis=0)
    d_num_pos3 = mean_plur_pos3 - mean_sing_pos3
    
    # Position direction: mean(all@pos2) - mean(all@pos3)
    all_pos2 = np.vstack([sing_pos2, plur_pos2])
    all_pos3 = np.vstack([sing_pos3, plur_pos3])
    d_pos = all_pos2.mean(axis=0) - all_pos3.mean(axis=0)
    
    # Orthogonalize: remove position component from number direction
    def orthogonalize(v, u):
        """Remove component of v along u"""
        proj = np.dot(v, u) / (np.dot(u, u) + 1e-10) * u
        return v - proj
    
    d_num_clean_pos2 = orthogonalize(d_num_pos2, d_pos)
    d_num_clean_pos3 = orthogonalize(d_num_pos3, d_pos)
    
    # Diagnostics
    def cosine(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    
    diagnostics = {
        "num_pos2_norm": float(np.linalg.norm(d_num_pos2)),
        "num_pos3_norm": float(np.linalg.norm(d_num_pos3)),
        "pos_norm": float(np.linalg.norm(d_pos)),
        "cos_num_pos2_vs_pos3": cosine(d_num_pos2, d_num_pos3),
        "cos_num_pos2_vs_dpos": cosine(d_num_pos2, d_pos),
        "cos_num_pos3_vs_dpos": cosine(d_num_pos3, d_pos),
        "clean_num_pos2_norm": float(np.linalg.norm(d_num_clean_pos2)),
        "clean_num_pos3_norm": float(np.linalg.norm(d_num_clean_pos3)),
        "clean_fraction_pos2": float(np.linalg.norm(d_num_clean_pos2) / (np.linalg.norm(d_num_pos2) + 1e-10)),
        "clean_fraction_pos3": float(np.linalg.norm(d_num_clean_pos3) / (np.linalg.norm(d_num_pos3) + 1e-10)),
        "cos_clean_vs_raw_pos2": cosine(d_num_clean_pos2, d_num_pos2),
    }
    
    return d_num_pos2, d_num_pos3, d_pos, d_num_clean_pos2, d_num_clean_pos3, diagnostics


def direction_patching(model, tokenizer, device, layers, test_sents, 
                       subj_pos_fn, verb_pos_fn, direction, alpha, 
                       patch_pos_fn, layer_idx):
    """
    在指定位置添加方向, 测试动词协议变化
    
    Args:
        test_sents: list of (sing_sent, plur_sent, sv, pv, sn, pn)
        direction: 方向向量 [d_model]
        alpha: 缩放因子
        patch_pos_fn: function that returns the position to patch (may differ from subj_pos)
        layer_idx: patch at which layer
    
    Returns:
        dict with effects per sentence
    """
    direction_t = torch.tensor(direction, dtype=torch.float16, device=device)
    
    effects = []
    
    for sing_sent, plur_sent, sv, pv, sn, pn in test_sents:
        sing_toks = tokenizer.encode(sing_sent, add_special_tokens=False)
        plur_toks = tokenizer.encode(plur_sent, add_special_tokens=False)
        
        subj_pos = subj_pos_fn(tokenizer, sing_toks)
        verb_pos = verb_pos_fn(tokenizer, sing_toks)
        patch_pos = patch_pos_fn(tokenizer, sing_toks)
        
        if subj_pos is None or verb_pos is None or patch_pos is None:
            continue
        
        # Get verb token IDs
        sv_ids = tokenizer.encode(sv, add_special_tokens=False)
        pv_ids = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_ids or not pv_ids:
            continue
        
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        
        # Baseline forward
        with torch.no_grad():
            base_logits = model(**sing_ids).logits.detach().cpu()
        
        if verb_pos >= base_logits.shape[1]:
            continue
        
        base_agr = (base_logits[0, verb_pos, sv_ids[0]] - 
                   base_logits[0, verb_pos, pv_ids[0]]).item()
        
        # Patched forward: add direction at patch_pos
        applied = [False]
        def make_additive_hook(pos, direction_tensor, a):
            def fn(m, inp, out):
                if not applied[0]:
                    if isinstance(out, tuple):
                        p = out[0].clone()
                        p[:, pos, :] += (a * direction_tensor).to(p.dtype)
                        applied[0] = True
                        return (p,) + out[1:]
                    else:
                        p = out.clone()
                        p[:, pos, :] += (a * direction_tensor).to(p.dtype)
                        applied[0] = True
                        return p
                return out
            return fn
        
        hook = layers[layer_idx].register_forward_hook(
            make_additive_hook(patch_pos, direction_t, alpha)
        )
        
        with torch.no_grad():
            patched_logits = model(**sing_ids).logits.detach().cpu()
        
        hook.remove()
        
        patched_agr = (patched_logits[0, verb_pos, sv_ids[0]] - 
                      patched_logits[0, verb_pos, pv_ids[0]]).item()
        
        delta = patched_agr - base_agr  # Negative = shifted toward plural
        
        # Also check top verb token
        base_top = base_logits[0, verb_pos].argmax().item()
        patched_top = patched_logits[0, verb_pos].argmax().item()
        verb_changed = base_top != patched_top
        
        effects.append({
            "delta": delta,
            "base_agr": base_agr,
            "patched_agr": patched_agr,
            "verb_changed": verb_changed,
            "sent": sing_sent,
        })
    
    return effects


def full_vector_patching(model, tokenizer, device, layers, source_sent, target_sent,
                         source_pos_fn, target_pos_fn, verb_pos_fn, layer_idx):
    """Phase 59 style全向量替换 (对比用)"""
    source_toks = tokenizer.encode(source_sent, add_special_tokens=False)
    target_toks = tokenizer.encode(target_sent, add_special_tokens=False)
    
    src_subj_pos = source_pos_fn(tokenizer, source_toks)
    tgt_subj_pos = target_pos_fn(tokenizer, target_toks)
    tgt_verb_pos = verb_pos_fn(tokenizer, target_toks)
    
    if src_subj_pos is None or tgt_subj_pos is None or tgt_verb_pos is None:
        return None
    
    # Get source activation
    source_ids = tokenizer(source_sent, return_tensors="pt").to(device)
    captured = {}
    def cap_hook(li):
        def fn(m, inp, out):
            if isinstance(out, tuple):
                captured[li] = out[0].detach().clone()
            else:
                captured[li] = out.detach().clone()
        return fn
    
    hook = layers[layer_idx].register_forward_hook(cap_hook(layer_idx))
    with torch.no_grad():
        model(**source_ids)
    hook.remove()
    
    if layer_idx not in captured:
        return None
    
    source_repr = captured[layer_idx][0, src_subj_pos, :]
    
    # Apply to target
    target_ids = tokenizer(target_sent, return_tensors="pt").to(device)
    
    # Get target verb info
    # Parse target sentence for verb tokens
    # Assuming target is singular, source is plural
    tgt_words = target_sent.split()
    src_words = source_sent.split()
    
    # Baseline
    with torch.no_grad():
        base_logits = model(**target_ids).logits.detach().cpu()
    
    if tgt_verb_pos >= base_logits.shape[1]:
        return None
    
    # Get sing/plur verb IDs from the sentence pair
    # We need to figure out sv/pv from the sentences
    # Simple heuristic: the 3rd word in SVO is verb
    sv = tgt_words[2] if len(tgt_words) > 2 else None
    pv = src_words[2] if len(src_words) > 2 else None
    if sv is None or pv is None:
        return None
    
    sv_ids = tokenizer.encode(sv, add_special_tokens=False)
    pv_ids = tokenizer.encode(pv, add_special_tokens=False)
    if not sv_ids or not pv_ids:
        return None
    
    base_agr = (base_logits[0, tgt_verb_pos, sv_ids[0]] - 
               base_logits[0, tgt_verb_pos, pv_ids[0]]).item()
    
    # Patched
    applied = [False]
    def make_replace_hook(pos, value):
        def fn(m, inp, out):
            if not applied[0]:
                if isinstance(out, tuple):
                    p = out[0].clone()
                    p[:, pos, :] = value.to(p.device).to(p.dtype)
                    applied[0] = True
                    return (p,) + out[1:]
                else:
                    p = out.clone()
                    p[:, pos, :] = value.to(p.device).to(p.dtype)
                    applied[0] = True
                    return p
            return out
        return fn
    
    hook = layers[layer_idx].register_forward_hook(
        make_replace_hook(tgt_subj_pos, source_repr)
    )
    with torch.no_grad():
        patched_logits = model(**target_ids).logits.detach().cpu()
    hook.remove()
    
    patched_agr = (patched_logits[0, tgt_verb_pos, sv_ids[0]] - 
                  patched_logits[0, tgt_verb_pos, pv_ids[0]]).item()
    
    delta = patched_agr - base_agr
    
    return {"delta": delta, "base_agr": base_agr, "patched_agr": patched_agr}


def run_phase60(model, tokenizer, device, info):
    print("=" * 70)
    print("★★★ Phase 60: Syntax Direction Extraction + Precise Patching ★★★")
    print("=" * 70)
    
    svo_data, adv_data = generate_sentences()
    print(f"\nGenerated: {len(svo_data)} SVO pairs, {len(adv_data)} Adv-front pairs")
    
    layers = get_layers(model)
    target_layers = [0, 5, 10, 15, 18, 20, 25]
    target_layers = [l for l in target_layers if l < info.n_layers]
    
    # ===== Step 1: Collect activations =====
    print("\n" + "=" * 70)
    print("Step 1: Collecting activations...")
    print("=" * 70)
    
    # SVO sentences
    svo_sing_sents = [d[0] for d in svo_data]  # singular
    svo_plur_sents = [d[1] for d in svo_data]  # plural
    adv_sing_sents = [d[0] for d in adv_data]
    adv_plur_sents = [d[1] for d in adv_data]
    
    def svo_pos_fn(tok, toks):
        # SVO: subject at index 1 (after BOS + "The"), verb at index 2
        decoded = [tok.decode([t]).strip() for t in toks]
        subj_pos = None
        verb_pos = None
        for i, d in enumerate(decoded):
            if d.lower() in ["cat","cats","dog","dogs","bird","birds","girl","girls",
                             "boy","boys","man","men","horse","horses","bear","bears",
                             "snake","snakes","frog","frogs","fox","foxes","king","kings",
                             "student","students","teacher","teachers","doctor","doctors",
                             "tree","trees","car","cars","queen","queens","child","children",
                             "wolf","wolves","driver","drivers","worker","workers","player","players",
                             "writer","writers","rabbit","rabbits","eagle","eagles","tiger","tigers",
                             "monkey","monkeys","lion","lions","farmer","farmers"]:
                subj_pos = i + 1  # +1 for BOS
                break
        # Verb is right after subject
        if subj_pos is not None:
            verb_pos = subj_pos + 1
        return subj_pos, verb_pos
    
    def adv_pos_fn(tok, toks):
        # Adv-front: "Today the {noun} {verb} {adv}"
        # [BOS, Today, the, noun, verb, adv]
        decoded = [tok.decode([t]).strip() for t in toks]
        subj_pos = None
        verb_pos = None
        for i, d in enumerate(decoded):
            if d.lower() in ["cat","cats","dog","dogs","bird","birds","girl","girls",
                             "boy","boys","man","men","horse","horses","bear","bears",
                             "snake","snakes","frog","frogs","fox","foxes","king","kings",
                             "student","students","teacher","teachers","doctor","doctors",
                             "tree","trees","car","cars","queen","queens","child","children",
                             "wolf","wolves","driver","drivers","worker","workers","player","players",
                             "writer","writers","rabbit","rabbits","eagle","eagles","tiger","tigers",
                             "monkey","monkeys","lion","lions","farmer","farmers"]:
                subj_pos = i + 1  # +1 for BOS
                break
        if subj_pos is not None:
            verb_pos = subj_pos + 1
        return subj_pos, verb_pos
    
    t0 = time.time()
    
    svo_sing_act, n1 = collect_activations(model, tokenizer, device, svo_sing_sents,
                                            svo_pos_fn, target_layers, "SVO-sing")
    svo_plur_act, n2 = collect_activations(model, tokenizer, device, svo_plur_sents,
                                            svo_pos_fn, target_layers, "SVO-plur")
    adv_sing_act, n3 = collect_activations(model, tokenizer, device, adv_sing_sents,
                                            adv_pos_fn, target_layers, "Adv-sing")
    adv_plur_act, n4 = collect_activations(model, tokenizer, device, adv_plur_sents,
                                            adv_pos_fn, target_layers, "Adv-plur")
    
    print(f"\n  Activation collection took {time.time()-t0:.1f}s")
    print(f"  Valid: SVO-sing={n1}, SVO-plur={n2}, Adv-sing={n3}, Adv-plur={n4}")
    
    # ===== Step 2: Extract directions =====
    print("\n" + "=" * 70)
    print("Step 2: Direction Extraction + Diagnostics")
    print("=" * 70)
    
    all_directions = {}
    all_diagnostics = {}
    
    for li in target_layers:
        if li not in svo_sing_act or not svo_sing_act[li]["subj"]:
            continue
        if li not in svo_plur_act or not svo_plur_act[li]["subj"]:
            continue
        if li not in adv_sing_act or not adv_sing_act[li]["subj"]:
            continue
        if li not in adv_plur_act or not adv_plur_act[li]["subj"]:
            continue
        
        d_num2, d_num3, d_pos, d_clean2, d_clean3, diag = extract_directions(
            svo_sing_act, svo_plur_act, adv_sing_act, adv_plur_act, li
        )
        
        all_directions[li] = {
            "d_num_pos2": d_num2,
            "d_num_pos3": d_num3,
            "d_pos": d_pos,
            "d_num_clean_pos2": d_clean2,
            "d_num_clean_pos3": d_clean3,
        }
        all_diagnostics[li] = diag
        
        print(f"\n  L{li}: ||d_num_pos2||={diag['num_pos2_norm']:.2f}, "
              f"||d_num_pos3||={diag['num_pos3_norm']:.2f}, "
              f"||d_pos||={diag['pos_norm']:.2f}")
        print(f"       cos(num_pos2, num_pos3) = {diag['cos_num_pos2_vs_pos3']:.4f}  "
              f"{'★★★ HIGH → NUMBER POSITION-INVARIANT!' if abs(diag['cos_num_pos2_vs_pos3']) > 0.5 else ''}")
        print(f"       cos(num_pos2, d_pos) = {diag['cos_num_pos2_vs_dpos']:.4f}  "
              f"{'★ LOW → NUMBER ⊥ POSITION' if abs(diag['cos_num_pos2_vs_dpos']) < 0.3 else ''}")
        print(f"       cos(num_pos3, d_pos) = {diag['cos_num_pos3_vs_dpos']:.4f}")
        print(f"       clean_fraction_pos2 = {diag['clean_fraction_pos2']:.4f}")
        print(f"       clean_fraction_pos3 = {diag['clean_fraction_pos3']:.4f}")
    
    # ===== Step 3: Direction Patching Experiments =====
    print("\n" + "=" * 70)
    print("Step 3: Direction Patching Experiments")
    print("=" * 70)
    
    alphas = [0.5, 1.0, 2.0]
    
    # Use first 20 SVO pairs for testing (to keep runtime manageable)
    test_svo = svo_data[:20]
    test_adv = adv_data[:20]
    
    # --- Exp A: Same-position direction patching (SVO) ---
    print("\n--- Exp A: Same-position direction patching (SVO) ---")
    print("  Adding d_number (from SVO pos2) to SVO sing subject @ pos2")
    
    exp_a_results = {}
    for li in target_layers:
        if li not in all_directions:
            continue
        d = all_directions[li]["d_num_pos2"]
        
        for alpha in alphas:
            effects = direction_patching(
                model, tokenizer, device, layers, test_svo,
                lambda tok, toks: svo_pos_fn(tok, toks)[0],  # subj_pos
                lambda tok, toks: svo_pos_fn(tok, toks)[1],  # verb_pos
                d, alpha,
                lambda tok, toks: svo_pos_fn(tok, toks)[0],  # patch at subj pos
                li
            )
            
            if effects:
                mean_delta = np.mean([e["delta"] for e in effects])
                verb_change_pct = np.mean([e["verb_changed"] for e in effects]) * 100
                key = (li, alpha)
                exp_a_results[key] = {"mean_delta": mean_delta, "verb_pct": verb_change_pct, "n": len(effects)}
                
                if alpha == 1.0:
                    sign_match = sum(1 for e in effects if e["delta"] < 0)
                    print(f"  L{li} α={alpha}: Δ={mean_delta:+.4f}, verb%={verb_change_pct:.0f}%, "
                          f"n={len(effects)}, neg={sign_match}/{len(effects)}")
    
    # --- Exp B: Cross-position direction patching (CORE TEST) ---
    print("\n--- Exp B: Cross-position direction patching (CORE TEST) ---")
    print("  Adding d_number (from SVO pos2) to Adv-front sing subject @ pos3")
    
    exp_b_results = {}
    for li in target_layers:
        if li not in all_directions:
            continue
        d = all_directions[li]["d_num_pos2"]  # direction from pos2!
        
        for alpha in alphas:
            effects = direction_patching(
                model, tokenizer, device, layers, test_adv,
                lambda tok, toks: adv_pos_fn(tok, toks)[0],  # subj_pos
                lambda tok, toks: adv_pos_fn(tok, toks)[1],  # verb_pos
                d, alpha,
                lambda tok, toks: adv_pos_fn(tok, toks)[0],  # patch at subj pos (pos3 in adv)
                li
            )
            
            if effects:
                mean_delta = np.mean([e["delta"] for e in effects])
                verb_change_pct = np.mean([e["verb_changed"] for e in effects]) * 100
                key = (li, alpha)
                exp_b_results[key] = {"mean_delta": mean_delta, "verb_pct": verb_change_pct, "n": len(effects)}
                
                if alpha == 1.0:
                    sign_match = sum(1 for e in effects if e["delta"] < 0)
                    print(f"  L{li} α={alpha}: Δ={mean_delta:+.4f}, verb%={verb_change_pct:.0f}%, "
                          f"n={len(effects)}, neg={sign_match}/{len(effects)}")
    
    # --- Exp C: Clean direction (position-removed) patching ---
    print("\n--- Exp C: Clean direction (position-removed) cross-position patching ---")
    print("  Adding d_number_clean (pos-removed, from SVO) to Adv-front @ pos3")
    
    exp_c_results = {}
    for li in target_layers:
        if li not in all_directions:
            continue
        d = all_directions[li]["d_num_clean_pos2"]  # clean direction from pos2!
        
        for alpha in alphas:
            effects = direction_patching(
                model, tokenizer, device, layers, test_adv,
                lambda tok, toks: adv_pos_fn(tok, toks)[0],
                lambda tok, toks: adv_pos_fn(tok, toks)[1],
                d, alpha,
                lambda tok, toks: adv_pos_fn(tok, toks)[0],
                li
            )
            
            if effects:
                mean_delta = np.mean([e["delta"] for e in effects])
                verb_change_pct = np.mean([e["verb_changed"] for e in effects]) * 100
                key = (li, alpha)
                exp_c_results[key] = {"mean_delta": mean_delta, "verb_pct": verb_change_pct, "n": len(effects)}
                
                if alpha == 1.0:
                    sign_match = sum(1 for e in effects if e["delta"] < 0)
                    print(f"  L{li} α={alpha}: Δ={mean_delta:+.4f}, verb%={verb_change_pct:.0f}%, "
                          f"neg={sign_match}/{len(effects)}")
    
    # --- Exp D: Position direction control ---
    print("\n--- Exp D: Position direction control ---")
    print("  Adding d_pos to SVO subject @ pos2 (should NOT change agreement)")
    
    exp_d_results = {}
    for li in target_layers:
        if li not in all_directions:
            continue
        d = all_directions[li]["d_pos"]
        
        effects = direction_patching(
            model, tokenizer, device, layers, test_svo,
            lambda tok, toks: svo_pos_fn(tok, toks)[0],
            lambda tok, toks: svo_pos_fn(tok, toks)[1],
            d, 1.0,
            lambda tok, toks: svo_pos_fn(tok, toks)[0],
            li
        )
        
        if effects:
            mean_delta = np.mean([e["delta"] for e in effects])
            verb_change_pct = np.mean([e["verb_changed"] for e in effects]) * 100
            exp_d_results[li] = {"mean_delta": mean_delta, "verb_pct": verb_change_pct, "n": len(effects)}
            print(f"  L{li}: Δ={mean_delta:+.4f}, verb%={verb_change_pct:.0f}%")
    
    # --- Exp E: Wrong position control ---
    print("\n--- Exp E: Wrong position control ---")
    print("  Adding d_number to VERB position (should NOT change agreement direction)")
    
    exp_e_results = {}
    for li in target_layers:
        if li not in all_directions:
            continue
        d = all_directions[li]["d_num_pos2"]
        
        effects = direction_patching(
            model, tokenizer, device, layers, test_svo,
            lambda tok, toks: svo_pos_fn(tok, toks)[0],
            lambda tok, toks: svo_pos_fn(tok, toks)[1],
            d, 1.0,
            lambda tok, toks: svo_pos_fn(tok, toks)[1],  # patch at VERB position!
            li
        )
        
        if effects:
            mean_delta = np.mean([e["delta"] for e in effects])
            verb_change_pct = np.mean([e["verb_changed"] for e in effects]) * 100
            exp_e_results[li] = {"mean_delta": mean_delta, "verb_pct": verb_change_pct, "n": len(effects)}
            print(f"  L{li}: Δ={mean_delta:+.4f}, verb%={verb_change_pct:.0f}%")
    
    # ===== Step 4: Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY: Direction Patching Results")
    print("=" * 70)
    
    print(f"\n{'Layer':>6} | {'A:same':>12} | {'B:cross':>12} | {'C:clean':>12} | {'D:dpos':>10} | {'E:verb':>10} | {'cos(num)':>8}")
    print("-" * 85)
    
    for li in target_layers:
        if li not in all_directions:
            continue
        
        a = exp_a_results.get((li, 1.0), {})
        b = exp_b_results.get((li, 1.0), {})
        c = exp_c_results.get((li, 1.0), {})
        d_res = exp_d_results.get(li, {})
        e_res = exp_e_results.get(li, {})
        diag = all_diagnostics.get(li, {})
        
        a_str = f"{a.get('mean_delta', 0):+.4f}" if a else "N/A"
        b_str = f"{b.get('mean_delta', 0):+.4f}" if b else "N/A"
        c_str = f"{c.get('mean_delta', 0):+.4f}" if c else "N/A"
        d_str = f"{d_res.get('mean_delta', 0):+.4f}" if d_res else "N/A"
        e_str = f"{e_res.get('mean_delta', 0):+.4f}" if e_res else "N/A"
        cos_str = f"{diag.get('cos_num_pos2_vs_pos3', 0):.3f}" if diag else "N/A"
        
        print(f"  L{li:>3} | {a_str:>12} | {b_str:>12} | {c_str:>12} | {d_str:>10} | {e_str:>10} | {cos_str:>8}")
    
    # Key diagnostic
    print("\n" + "=" * 70)
    print("KEY DIAGNOSTICS")
    print("=" * 70)
    
    for li in target_layers:
        if li not in all_diagnostics:
            continue
        diag = all_diagnostics[li]
        cos_num = diag['cos_num_pos2_vs_pos3']
        cos_pos = diag['cos_num_pos2_vs_dpos']
        clean_frac = diag['clean_fraction_pos2']
        
        # Check if cross-position works
        b = exp_b_results.get((li, 1.0), {})
        c = exp_c_results.get((li, 1.0), {})
        
        b_effective = b.get('mean_delta', 0) < -0.05 if b else False
        c_effective = c.get('mean_delta', 0) < -0.05 if c else False
        
        verdict = ""
        if cos_num > 0.5 and (b_effective or c_effective):
            verdict = "★★★ NUMBER IS POSITION-INVARIANT!"
        elif cos_num > 0.5 and not b_effective and not c_effective:
            verdict = "★ Number direction similar across positions, but patching doesn't transfer (complex readout)"
        elif cos_num < 0.3:
            verdict = "Number direction differs across positions (position-dependent)"
        else:
            verdict = "Partial overlap (need more analysis)"
        
        print(f"\n  L{li}: cos(num_pos2,num_pos3)={cos_num:.3f}, "
              f"cos(num,d_pos)={cos_pos:.3f}, clean_frac={clean_frac:.3f}")
        print(f"       Cross-pos raw Δ={b.get('mean_delta', 'N/A')}, "
              f"clean Δ={c.get('mean_delta', 'N/A')}")
        print(f"       → {verdict}")
    
    # Return all results for memo
    return {
        "diagnostics": all_diagnostics,
        "exp_a": exp_a_results,
        "exp_b": exp_b_results,
        "exp_c": exp_c_results,
        "exp_d": exp_d_results,
        "exp_e": exp_e_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"Model: {info.name}, Layers={info.n_layers}, d_model={info.d_model}")
    
    try:
        results = run_phase60(model, tokenizer, device, info)
    finally:
        release_model(model)
        print("\nDone.")
