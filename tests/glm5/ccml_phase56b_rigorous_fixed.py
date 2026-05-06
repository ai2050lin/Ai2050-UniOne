"""
Phase 56B: 修正版严格验证
========================

修复:
1. Exp4 tokenization问题 → 只用单token伪词
2. Exp2 distance confound → 距离控制的角色区分
3. 重新评估Phase 55B结论, 结合permutation test发现

核心发现需要验证:
- Permutation test: per-layer Real k1% < Perm k1% → 随机position反而更多top-1
- SVO dobj: best_rank=4 → attention根本不追踪dobj
- 只有nsubj被稳定追踪
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


def get_attention_weights(model, tokenizer, sentence, device):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model(input_ids=inputs["input_ids"], output_attentions=True)
            all_attn = outputs.attentions
        except:
            return None, None, None
    attn_dict = {}
    for li, attn in enumerate(all_attn):
        if attn is not None:
            attn_dict[li] = attn.detach().float().cpu().numpy()[0]
    tokens = [safe_decode(tokenizer, tid) for tid in inputs["input_ids"][0].cpu().numpy()]
    return attn_dict, inputs, tokens


def verify_tokenization(tokenizer, sentence, expected_words):
    """验证句子是否1:1 tokenization"""
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    decoded = [safe_decode(tokenizer, t) for t in tokens]
    return len(decoded) == len(expected_words), decoded


def compute_rank_for_dep(attn_dict, head_pos, dep_pos, n_layers):
    """计算per-head的rank"""
    results = []
    for li in range(n_layers):
        if li not in attn_dict:
            continue
        A = attn_dict[li]
        n_heads = A.shape[0]

        if dep_pos < head_pos:
            row = A[:, head_pos, 1:head_pos]
            target = dep_pos - 1
        elif dep_pos > head_pos:
            row = A[:, head_pos, 1:]
            target = dep_pos - 1
        else:
            continue

        if target < 0 or target >= row.shape[1]:
            continue

        for h in range(n_heads):
            score = float(row[h, target])
            sorted_idx = np.argsort(row[h])[::-1]
            rank = int(np.where(sorted_idx == target)[0][0]) + 1 if target in sorted_idx else 999
            results.append({"layer": li, "head": h, "rank": rank, "score": score})

    return results


# ============================================================
# ExpA: 修正版Permutation Test — 分nsubj/dobj分别统计
# ============================================================
def expA_permutation_by_role(model, tokenizer, device, info):
    """
    关键修正: 分别统计nsubj和dobj的permutation结果
    因为nsubj和dobj在句子中的位置特性不同
    """
    print("\n" + "="*70)
    print("★★★ ExpA: Permutation Test by Role ★★★")
    print("="*70)

    # SVO句子 (BOS + The + N1 + V + the + N2)
    svo_sentences = [
        "The cat chased the mouse",
        "The dog bit the man",
        "The girl saw the boy",
        "The king ruled the land",
        "The bird ate the fish",
        "The wolf killed the deer",
    ]

    # Center-embed句子 (BOS + The + N1 + that + the + N2 + V1 + V2)
    center_sentences = [
        "The cat that the dog chased ran",
        "The man that the girl saw left",
        "The bird that the cat watched flew",
        "The king that the man served died",
    ]

    nsubj_real_ranks = []
    dobj_real_ranks = []
    nsubj_perm_ranks = []
    dobj_perm_ranks = []

    N_PERM = 200

    all_sentences = [(s, "svo") for s in svo_sentences] + [(s, "center") for s in center_sentences]

    for sentence, stype in all_sentences:
        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue

        seq_len = len(tokens)
        print(f"\n  [{sentence}] tokens={seq_len}: {tokens}")

        if stype == "svo":
            # BOS + The + N1 + V + the + N2
            deps = [(3, 2, "nsubj"), (3, 5, "dobj")]
        else:
            # BOS + The + N1 + that + the + N2 + V1 + V2
            deps = [(7, 2, "nsubj_long"), (6, 5, "nsubj_short"), (6, 2, "dobj")]

        for head_pos, dep_pos, rel_type in deps:
            if head_pos >= seq_len or dep_pos >= seq_len:
                continue

            # 真实dep
            real_results = compute_rank_for_dep(attn_dict, head_pos, dep_pos, info.n_layers)
            if not real_results:
                continue
            real_best = min(r["rank"] for r in real_results)

            is_nsubj = "nsubj" in rel_type
            if is_nsubj:
                nsubj_real_ranks.append(real_best)
            else:
                dobj_real_ranks.append(real_best)

            # Permutation
            valid_positions = [p for p in range(1, seq_len) if p != head_pos and p != dep_pos]
            perm_ranks = []
            for _ in range(N_PERM):
                fake_dep = random.choice(valid_positions)
                perm_results = compute_rank_for_dep(attn_dict, head_pos, fake_dep, info.n_layers)
                if perm_results:
                    perm_ranks.append(min(r["rank"] for r in perm_results))

            perm_avg = np.mean(perm_ranks) if perm_ranks else 999
            if is_nsubj:
                nsubj_perm_ranks.append(perm_avg)
            else:
                dobj_perm_ranks.append(perm_avg)

            perm_k1 = sum(1 for r in perm_ranks if r == 1) / len(perm_ranks) if perm_ranks else 0
            marker = "★★★" if real_best == 1 and perm_k1 < 0.3 else ("★" if real_best == 1 else "✗")
            print(f"    {tokens[dep_pos]}({dep_pos})→{tokens[head_pos]}({head_pos}) [{rel_type}] "
                  f"real_best={real_best} perm_avg={perm_avg:.1f} perm_k1={perm_k1:.3f} {marker}")

        del attn_dict, inputs
        gc.collect()

    # ===== 分角色统计 =====
    print("\n\n" + "="*70)
    print("★★★ PERMUTATION RESULTS BY ROLE ★★★")
    print("="*70)

    from scipy import stats

    for role_name, real_ranks, perm_ranks in [
        ("NSUBJ (all)", nsubj_real_ranks, nsubj_perm_ranks),
        ("DOBJ (all)", dobj_real_ranks, dobj_perm_ranks),
    ]:
        if not real_ranks:
            continue

        real_k1 = sum(1 for r in real_ranks if r == 1) / len(real_ranks)
        real_avg = np.mean(real_ranks)
        perm_avg = np.mean(perm_ranks) if perm_ranks else 999

        print(f"\n  {role_name}:")
        print(f"    Real best_rank=1 rate: {real_k1:.3f} ({sum(1 for r in real_ranks if r == 1)}/{len(real_ranks)})")
        print(f"    Real avg best_rank: {real_avg:.2f}")
        print(f"    Perm avg best_rank: {perm_avg:.2f}")
        print(f"    Effect size (real-avg vs perm-avg): {perm_avg - real_avg:.2f}")

        if len(real_ranks) >= 3 and perm_ranks and len(perm_ranks) >= 3:
            try:
                stat, pval = stats.mannwhitneyu(real_ranks, perm_ranks, alternative='less')
                print(f"    Mann-Whitney p = {pval:.6f}")
                if pval < 0.05:
                    print(f"    ★ p < 0.05 → 真实dep显著优于随机")
                else:
                    print(f"    ✗ p >= 0.05 → 无法排除随机假阳性")
            except:
                pass

    return nsubj_real_ranks, dobj_real_ranks


# ============================================================
# ExpB: 距离控制的角色区分
# ============================================================
def expB_distance_controlled_role(model, tokenizer, device, info):
    """
    核心问题: Exp2的Jaccard=0.235可能是distance confound

    修正: 比较同距离的不同角色
    - "What did the cat chase" → What(d=3) = dobj
    - "What did chase the cat" → 不合法...
    
    更好的方案: 使用同结构不同wh-word
    - "What did the cat eat" → What = dobj (d=3 from eat)
    - "Where did the cat eat" → Where = adjunct (d=3 from eat)
    - "When did the cat eat" → When = adjunct (d=3 from eat)
    
    → 同距离, 不同角色!
    以及:
    - "Who ate the fish" → Who = nsubj (d=1 from ate)
    - "What ate the fish" → What = nsubj (d=1 from ate)
    → 同距离, 同角色(对照)
    """
    print("\n" + "="*70)
    print("★★★ ExpB: Distance-Controlled Role Discrimination ★★★")
    print("="*70)

    # 同结构、同距离、不同角色的Wh-question
    # 所有句子: What/Where/When/How did the cat VERB
    # → wh-word到verb距离相同(都是d=3), 但语法角色不同

    test_sets = [
        {
            "name": "Same structure (did + S + V), distance=3",
            "sentences": [
                {"sentence": "What did the cat eat", "wh": "What", "role": "dobj", "wh_pos": 1, "verb_pos": 5},
                {"sentence": "Where did the cat eat", "wh": "Where", "role": "adjunct", "wh_pos": 1, "verb_pos": 5},
                {"sentence": "When did the cat eat", "wh": "When", "role": "adjunct", "wh_pos": 1, "verb_pos": 5},
                {"sentence": "How did the cat eat", "wh": "How", "role": "adjunct", "wh_pos": 1, "verb_pos": 5},
            ]
        },
        {
            "name": "Same structure (did + S + V), different verb",
            "sentences": [
                {"sentence": "What did the dog see", "wh": "What", "role": "dobj", "wh_pos": 1, "verb_pos": 5},
                {"sentence": "Where did the dog sleep", "wh": "Where", "role": "adjunct", "wh_pos": 1, "verb_pos": 5},
            ]
        },
        {
            "name": "Wh-as-subject (d=1 from verb)",
            "sentences": [
                {"sentence": "Who ate the fish", "wh": "Who", "role": "nsubj", "wh_pos": 1, "verb_pos": 2},
                {"sentence": "What ate the fish", "wh": "What", "role": "nsubj", "wh_pos": 1, "verb_pos": 2},
            ]
        },
    ]

    all_profiles = {}  # (wh_word, role) → {layer: [wh_attn_scores]}

    for test_set in test_sets:
        print(f"\n--- {test_set['name']} ---")

        for item in test_set["sentences"]:
            sentence = item["sentence"]
            wh = item["wh"]
            role = item["role"]
            wh_pos = item["wh_pos"]
            verb_pos = item["verb_pos"]

            attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
            if attn_dict is None:
                continue

            seq_len = len(tokens)
            print(f"  [{sentence}] {wh}(pos{wh_pos})→verb(pos{verb_pos}) role={role}")
            print(f"  Tokens: {[(i,t) for i,t in enumerate(tokens)]}")

            key = (wh, role, sentence)
            all_profiles[key] = {}

            for li in range(min(info.n_layers, 28)):
                if li not in attn_dict:
                    continue
                A = attn_dict[li]
                n_heads = A.shape[0]

                # verb行: 各head attend to各位置
                verb_row = A[:, verb_pos, :]  # [n_heads, seq_len]
                wh_attn = verb_row[:, wh_pos]  # [n_heads]

                # 统计: 多少head把wh作为top-1
                for h in range(n_heads):
                    sorted_pos = np.argsort(verb_row[h])[::-1]
                    top1_pos = int(sorted_pos[0])

                    if li not in all_profiles[key]:
                        all_profiles[key][li] = []
                    all_profiles[key][li].append({
                        "head": h,
                        "wh_score": float(wh_attn[h]),
                        "wh_is_top1": top1_pos == wh_pos,
                        "top1_pos": top1_pos,
                    })

            del attn_dict, inputs
            gc.collect()

    # ===== 分析: 同距离不同角色的区分 =====
    print("\n\n" + "="*70)
    print("★★★ DISTANCE-CONTROLLED ROLE ANALYSIS ★★★")
    print("="*70)

    # 核心对比: What(dobj) vs Where/When/How(adjunct), 同距离d=3
    for li in [0, 1, 3, 6, 10, 15, 20]:
        dobj_top1_counts = []
        adjunct_top1_counts = []
        nsubj_top1_counts = []

        for key, layers in all_profiles.items():
            wh, role, sentence = key
            if li not in layers:
                continue
            heads = layers[li]
            top1_count = sum(1 for h in heads if h["wh_is_top1"])
            total = len(heads)

            if role == "dobj":
                dobj_top1_counts.append((top1_count, total, sentence))
            elif role == "adjunct":
                adjunct_top1_counts.append((top1_count, total, sentence))
            elif role == "nsubj":
                nsubj_top1_counts.append((top1_count, total, sentence))

        dobj_rate = np.mean([c/t for c,t,_ in dobj_top1_counts]) * 100 if dobj_top1_counts else 0
        adjunct_rate = np.mean([c/t for c,t,_ in adjunct_top1_counts]) * 100 if adjunct_top1_counts else 0
        nsubj_rate = np.mean([c/t for c,t,_ in nsubj_top1_counts]) * 100 if nsubj_top1_counts else 0

        print(f"  L{li:>2}: dobj={dobj_rate:5.1f}%  adjunct={adjunct_rate:5.1f}%  nsubj={nsubj_rate:5.1f}%  "
              f"|dobj-adjunct|={abs(dobj_rate-adjunct_rate):5.1f}%")

    # ===== 最关键测试: What(dobj) vs Where(adjunct) 在同结构中 =====
    print("\n\n--- ★★★ Critical: What(dobj) vs Where/When/How(adjunct) at SAME distance ★★★ ---")

    for li in [0, 1, 3, 10, 15]:
        what_scores = []
        where_scores = []

        for key, layers in all_profiles.items():
            wh, role, sentence = key
            if li not in layers:
                continue
            heads = layers[li]

            if wh == "What" and role == "dobj":
                what_scores.extend([h["wh_score"] for h in heads])
            elif wh in ["Where", "When", "How"] and role == "adjunct":
                where_scores.extend([h["wh_score"] for h in heads])

        if what_scores and where_scores:
            what_mean = np.mean(what_scores)
            where_mean = np.mean(where_scores)
            diff = what_mean - where_mean

            # t-test
            try:
                from scipy import stats
                t, p = stats.ttest_ind(what_scores, where_scores)
                sig = "★" if p < 0.05 else ""
                print(f"  L{li:>2}: What(dobj)={what_mean:.4f} vs Where/When/How(adj)={where_mean:.4f} "
                      f"Δ={diff:+.4f} p={p:.4f} {sig}")
            except:
                print(f"  L{li:>2}: What(dobj)={what_mean:.4f} vs Where/When/How(adj)={where_mean:.4f} "
                      f"Δ={diff:+.4f}")

    print("\n\n  ★ If What(dobj) >> Where/When/How(adjunct) at same distance → genuine role tracking")
    print("  ✗ If What ≈ Where/When/How → only distance/position, not role")


# ============================================================
# ExpC: 修正版Template Control (单token伪词)
# ============================================================
def expC_template_control_fixed(model, tokenizer, device, info):
    """
    修正: 使用tokenizer验证1:1映射
    避免伪词被拆成多个token
    """
    print("\n" + "="*70)
    print("★★★ ExpC: Template Control (Fixed Tokenization) ★★★")
    print("="*70)

    # 先验证哪些伪词是单token
    test_words = ["cat", "dog", "run", "walk", "big", "red", "fast", "slow",
                  "the", "a", "is", "was", "can", "will", "it", "he"]

    single_token_words = []
    for w in test_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            single_token_words.append(w)
            print(f"  '{w}' → single token ✓")
        else:
            print(f"  '{w}' → {len(ids)} tokens ✗")

    # 使用真实词但打乱结构, 确保tokenization正确
    test_cases = [
        {"name": "Normal SVO", "sentence": "The cat chased the mouse",
         "deps": [(3, 2, "nsubj"), (3, 5, "dobj")]},
        {"name": "SVO with different words", "sentence": "The dog saw the bird",
         "deps": [(3, 2, "nsubj"), (3, 5, "dobj")]},
        # 关键测试: 同词不同序
        {"name": "OSV (conflict!)", "sentence": "The mouse the cat chased",
         "deps": [(4, 3, "nsubj"), (4, 2, "dobj")]},  # chased←cat(nsubj), chased←mouse(dobj)
        # 打乱结构
        {"name": "Random order 1", "sentence": "Cat mouse the chased the",
         "deps": [(3, 0, "nsubj")]},  # 这个标注已经无意义
        {"name": "Random order 2", "sentence": "The chased mouse the cat",
         "deps": [(1, 4, "nsubj")]},
        # 没有function words
        {"name": "No function words", "sentence": "Cat chased mouse",
         "deps": [(1, 0, "nsubj"), (1, 2, "dobj")]},
        # VSO (少见但合法)
        # 难以用简单句测试
    ]

    # 先验证所有句子的tokenization
    print("\n--- Tokenization Verification ---")
    valid_cases = []
    for item in test_cases:
        tokens = tokenizer.encode(item["sentence"], add_special_tokens=False)
        decoded = [safe_decode(tokenizer, t) for t in tokens]
        words = item["sentence"].split()

        # 检查BOS后的token数是否等于词数
        all_tokens = [safe_decode(tokenizer, t) for t in tokenizer.encode(item["sentence"])]
        print(f"  [{item['name']}] {item['sentence']}")
        print(f"    All tokens (with BOS): {all_tokens}")
        print(f"    Token count: {len(all_tokens)}")

        valid_cases.append(item)

    # 运行测试
    print("\n\n--- Attention Tracking Results ---")
    for item in valid_cases:
        sentence = item["sentence"]
        deps = item["deps"]
        name = item["name"]

        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue

        seq_len = len(tokens)
        print(f"\n  [{name}] {sentence}")
        print(f"  Tokens: {[(i,t) for i,t in enumerate(tokens)]}")

        for head_pos, dep_pos, rel_type in deps:
            if head_pos >= seq_len or dep_pos >= seq_len:
                print(f"    {rel_type}: pos out of range (head={head_pos}, dep={dep_pos}, seq={seq_len})")
                continue

            results = compute_rank_for_dep(attn_dict, head_pos, dep_pos, info.n_layers)
            if results:
                best_rank = min(r["rank"] for r in results)
                k1_count = sum(1 for r in results if r["rank"] == 1)
                total = len(results)
                k3_count = sum(1 for r in results if r["rank"] <= 3)

                print(f"    {tokens[dep_pos] if dep_pos < len(tokens) else '?'}({dep_pos})→"
                      f"{tokens[head_pos] if head_pos < len(tokens) else '?'}({head_pos}) [{rel_type}] "
                      f"best_rank={best_rank} k1={k1_count}/{total} k3={k3_count}/{total}")

                # Permutation test for this specific dep
                valid_positions = [p for p in range(1, seq_len) if p != head_pos and p != dep_pos]
                if valid_positions:
                    perm_ranks = []
                    for _ in range(200):
                        fake_dep = random.choice(valid_positions)
                        perm_results = compute_rank_for_dep(attn_dict, head_pos, fake_dep, info.n_layers)
                        if perm_results:
                            perm_ranks.append(min(r["rank"] for r in perm_results))

                    perm_avg = np.mean(perm_ranks) if perm_ranks else 999
                    perm_k1 = sum(1 for r in perm_ranks if r == 1) / len(perm_ranks) if perm_ranks else 0
                    signal = "★★★" if best_rank == 1 and perm_k1 < 0.2 else ("★" if best_rank == 1 else "✗")
                    print(f"      perm: avg_rank={perm_avg:.1f} k1_rate={perm_k1:.3f} {signal}")

        del attn_dict, inputs
        gc.collect()


# ============================================================
# ExpD: Targeted Ablation (简化版)
# ============================================================
def expD_targeted_ablation(model, tokenizer, device, info):
    """
    简化版: 找语法head → ablate → 看next-token prediction变化
    
    方法: 比较模型在语法正确 vs 错误句上的surprise
    - "The cat chased" → 预期下一词是名词/det
    - ablate语法head后 → surprise应该变化
    """
    print("\n" + "="*70)
    print("★★★ ExpD: Targeted Ablation ★★★")
    print("="*70)

    # Step 1: 找在多个SVO句子上都能找到nsubj的head
    print("\n--- Step 1: Identifying reliable nsubj-tracking heads ---")

    svo_sentences = [
        "The cat chased the mouse",
        "The dog bit the man",
        "The girl saw the boy",
        "The king ruled the land",
        "The bird ate the fish",
        "The wolf killed the deer",
    ]

    head_success = defaultdict(int)
    head_total = defaultdict(int)

    for sentence in svo_sentences:
        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue

        seq_len = len(tokens)
        # nsubj: pos2 → pos3
        results = compute_rank_for_dep(attn_dict, 3, 2, info.n_layers)
        for r in results:
            key = (r["layer"], r["head"])
            head_total[key] += 1
            if r["rank"] == 1:
                head_success[key] += 1

        del attn_dict, inputs
        gc.collect()

    # 找6/6都成功的head
    perfect_heads = [(li, h) for (li, h), cnt in head_success.items()
                     if cnt >= 5 and head_total[(li,h)] >= 5]
    perfect_heads.sort(key=lambda x: head_success[x], reverse=True)

    print(f"  Heads with nsubj rank=1 in ≥5/6 sentences:")
    for li, h in perfect_heads[:10]:
        print(f"    L{li} H{h}: {head_success[(li,h)]}/{head_total[(li,h)]}")

    if not perfect_heads:
        print("  No perfect heads found, using top-5 by success count")
        all_heads_sorted = sorted(head_success.keys(), key=lambda x: head_success[x], reverse=True)
        perfect_heads = all_heads_sorted[:5]
        for li, h in perfect_heads:
            print(f"    L{li} H{h}: {head_success[(li,h)]}/{head_total[(li,h)]}")

    # Step 2: Ablation test
    print("\n--- Step 2: Ablation Test (logit difference) ---")

    # 测试: 模型对 "chased" vs "chase" 的偏好 (在 "The cat ___" 之后)
    test_prompts = [
        ("The cat", "chased", "chase"),      # 正确: chased (过去时)
        ("The cats", "chased", "chase"),     # 正确: chased
        ("The dog", "bit", "bite"),          # 正确: bit
        ("The girl", "saw", "see"),          # 正确: saw
    ]

    def get_logit_diff(prompt, correct, wrong, model, tokenizer, device):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"])
            logits = outputs.logits[0]

        correct_id = tokenizer.encode(correct, add_special_tokens=False)[0]
        wrong_id = tokenizer.encode(wrong, add_special_tokens=False)[0]

        # 最后位置的logit
        last_logits = logits[-1]
        diff = float(last_logits[correct_id] - last_logits[wrong_id])
        return diff

    # Baseline
    print("\n  Baseline (no ablation):")
    baseline_diffs = []
    for prompt, correct, wrong in test_prompts:
        diff = get_logit_diff(prompt, correct, wrong, model, tokenizer, device)
        baseline_diffs.append(diff)
        print(f"    '{prompt}' → {correct} vs {wrong}: logit_diff = {diff:.2f}")

    avg_baseline = np.mean(baseline_diffs)
    print(f"    Average: {avg_baseline:.2f}")

    # Ablate each syntax head
    print("\n  Ablating syntax heads:")
    ablation_results = []

    for li, h in perfect_heads[:5]:
        try:
            layer = model.model.layers[li]
            attn = layer.self_attn

            # Get n_heads
            n_heads = attn.num_heads if hasattr(attn, 'num_heads') else 28
            d_head = attn.hidden_size // n_heads if hasattr(attn, 'hidden_size') else 128

            def make_zero_hook(target_head_idx, n_h, d_h):
                def hook_fn(module, input, output):
                    attn_output = output[0].clone()
                    start = target_head_idx * d_h
                    end = (target_head_idx + 1) * d_h
                    if end <= attn_output.shape[-1]:
                        attn_output[:, :, start:end] = 0
                    return (attn_output,) + output[1:]
                return hook_fn

            hook = attn.register_forward_hook(make_zero_hook(h, n_heads, d_head))

            ablation_diffs = []
            for prompt, correct, wrong in test_prompts:
                diff = get_logit_diff(prompt, correct, wrong, model, tokenizer, device)
                ablation_diffs.append(diff)

            hook.remove()

            avg_ablation = np.mean(ablation_diffs)
            delta = avg_ablation - avg_baseline
            ablation_results.append((li, h, avg_ablation, delta))

            sig = "★★★" if delta < -2.0 else ("★" if delta < -0.5 else "")
            print(f"    L{li} H{h}: avg={avg_ablation:.2f} Δ={delta:+.2f} {sig}")

        except Exception as e:
            print(f"    L{li} H{h}: FAILED - {e}")

    # Control: random heads
    print("\n  Control: Ablating random heads:")
    random_results = []
    for _ in range(5):
        li = random.randint(0, info.n_layers - 1)
        h = random.randint(0, 27)

        try:
            layer = model.model.layers[li]
            attn = layer.self_attn
            n_heads = attn.num_heads if hasattr(attn, 'num_heads') else 28
            d_head = attn.hidden_size // n_heads if hasattr(attn, 'hidden_size') else 128

            hook = attn.register_forward_hook(make_zero_hook(h, n_heads, d_head))

            ablation_diffs = []
            for prompt, correct, wrong in test_prompts:
                diff = get_logit_diff(prompt, correct, wrong, model, tokenizer, device)
                ablation_diffs.append(diff)

            hook.remove()

            avg_ablation = np.mean(ablation_diffs)
            delta = avg_ablation - avg_baseline
            random_results.append(delta)
            print(f"    L{li} H{h} (random): Δ={delta:+.2f}")

        except Exception as e:
            print(f"    L{li} H{h} (random): FAILED - {e}")

    # Final comparison
    if ablation_results and random_results:
        syntax_deltas = [d for _, _, _, d in ablation_results]
        print(f"\n  Syntax head ablation: mean Δ = {np.mean(syntax_deltas):+.2f}")
        print(f"  Random head ablation: mean Δ = {np.mean(random_results):+.2f}")

        if np.mean(syntax_deltas) < np.mean(random_results) - 1.0:
            print("  ★★★ Syntax heads are CAUSALLY involved in grammar ★★★")
        else:
            print("  ✗ No significant causal effect from syntax heads")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b",
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all",
                        help="A=permutation_by_role, B=distance_role, C=template_fixed, D=ablation, all")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"Model: {args.model}, Layers: {info.n_layers}, d_model: {info.d_model}")

    if args.exp in ["all", "A"]:
        expA_permutation_by_role(model, tokenizer, device, info)

    if args.exp in ["all", "B"]:
        expB_distance_controlled_role(model, tokenizer, device, info)

    if args.exp in ["all", "C"]:
        expC_template_control_fixed(model, tokenizer, device, info)

    if args.exp in ["all", "D"]:
        expD_targeted_ablation(model, tokenizer, device, info)

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("\nPhase 56B Complete!")


if __name__ == "__main__":
    main()
