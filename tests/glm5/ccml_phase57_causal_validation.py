"""
Phase 57: 因果验证 + nsubj/dobj不对称性 + 跨模型确认
=====================================================

基于Phase 56/56B的关键发现:
★ nsubj tracking: 统计显著 (p=0.000098), 真实优于随机
✗ dobj tracking: 不显著 (p=0.79), 实际比随机还差
✗ 角色区分: 同距离下无显著差异 (所有p>0.05)
? 因果验证: 从未成功执行

本Phase目标:
Exp1: 大规模nsubj验证 + Bootstrap CI (强化最强claim)
Exp2: 因果ablation — 语法head消融→主谓一致错误率上升
Exp3: nsubj/dobj不对称性的深度分析 (为什么dobj失败?)
Exp4: OVS/Genuine冲突测试 (位置≠语法时nsubj是否仍追踪?)
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


def compute_rank_for_dep(attn_dict, head_pos, dep_pos, n_layers, exclude_self=True):
    """计算per-head的rank, 返回list of {layer, head, rank, score}"""
    results = []
    for li in range(n_layers):
        if li not in attn_dict:
            continue
        A = attn_dict[li]
        n_heads = A.shape[0]

        if dep_pos < head_pos:
            row = A[:, head_pos, 1:head_pos]  # 排除BOS
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
# Exp1: 大规模nsubj Bootstrap验证
# ============================================================
def exp1_nsubj_bootstrap(model, tokenizer, device, info):
    """
    用20+句子, 对nsubj和dobj分别做:
    1. Per-head rank统计
    2. 200次permutation test
    3. 1000次bootstrap → 95% CI
    """
    print("\n" + "="*70)
    print("★★★ EXP1: 大规模nsubj/dobj Bootstrap验证 ★★★")
    print("="*70)

    # 大规模句子集 (增加统计力)
    svo_sentences = [
        "The cat chased the mouse",
        "The dog bit the man",
        "The girl saw the boy",
        "The king ruled the land",
        "The bird ate the fish",
        "The wolf killed the deer",
        "The fox caught the rabbit",
        "The bear chased the deer",
        "The horse kicked the man",
        "The snake bit the frog",
        "The lion ate the meat",
        "The hawk caught the fish",
        "The priest blessed the child",
        "The nurse helped the patient",
        "The teacher taught the student",
        "The doctor cured the patient",
    ]

    center_sentences = [
        "The cat that the dog chased ran",
        "The man that the girl saw left",
        "The bird that the cat watched flew",
        "The king that the man served died",
        "The dog that the wolf chased barked",
        "The fish that the cat ate vanished",
    ]

    # 收集数据
    nsubj_real_ranks = []
    dobj_real_ranks = []
    nsubj_perm_ranks_list = []  # 存每条dep的perm ranks
    dobj_perm_ranks_list = []

    N_PERM = 200

    for sentence in svo_sentences:
        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue

        seq_len = len(tokens)
        # SVO: BOS + The + N1 + V + the + N2
        # nsubj: pos2 → pos3, dobj: pos5 → pos3
        deps = [(3, 2, "nsubj"), (3, 5, "dobj")]

        for head_pos, dep_pos, rel_type in deps:
            if head_pos >= seq_len or dep_pos >= seq_len:
                continue

            # 真实dep
            real_results = compute_rank_for_dep(attn_dict, head_pos, dep_pos, info.n_layers)
            if not real_results:
                continue
            real_best = min(r["rank"] for r in real_results)

            # Permutation
            valid_positions = [p for p in range(1, seq_len) if p != head_pos and p != dep_pos]
            if not valid_positions:
                continue

            perm_ranks = []
            for _ in range(N_PERM):
                fake_dep = random.choice(valid_positions)
                perm_results = compute_rank_for_dep(attn_dict, head_pos, fake_dep, info.n_layers)
                if perm_results:
                    perm_ranks.append(min(r["rank"] for r in perm_results))

            if rel_type == "nsubj":
                nsubj_real_ranks.append(real_best)
                nsubj_perm_ranks_list.append(perm_ranks)
            else:
                dobj_real_ranks.append(real_best)
                dobj_perm_ranks_list.append(perm_ranks)

        del attn_dict, inputs
        gc.collect()

    for sentence in center_sentences:
        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue

        seq_len = len(tokens)
        deps = [(7, 2, "nsubj"), (6, 5, "nsubj"), (6, 2, "dobj")]

        for head_pos, dep_pos, rel_type in deps:
            if head_pos >= seq_len or dep_pos >= seq_len:
                continue

            real_results = compute_rank_for_dep(attn_dict, head_pos, dep_pos, info.n_layers)
            if not real_results:
                continue
            real_best = min(r["rank"] for r in real_results)

            valid_positions = [p for p in range(1, seq_len) if p != head_pos and p != dep_pos]
            if not valid_positions:
                continue

            perm_ranks = []
            for _ in range(N_PERM):
                fake_dep = random.choice(valid_positions)
                perm_results = compute_rank_for_dep(attn_dict, head_pos, fake_dep, info.n_layers)
                if perm_results:
                    perm_ranks.append(min(r["rank"] for r in perm_results))

            if "nsubj" in rel_type:
                nsubj_real_ranks.append(real_best)
                nsubj_perm_ranks_list.append(perm_ranks)
            else:
                dobj_real_ranks.append(real_best)
                dobj_perm_ranks_list.append(perm_ranks)

        del attn_dict, inputs
        gc.collect()

    # ===== Bootstrap CI =====
    print("\n--- Bootstrap 95% CI ---")
    N_BOOT = 2000

    for role_name, real_ranks, perm_ranks_list in [
        ("NSUBJ", nsubj_real_ranks, nsubj_perm_ranks_list),
        ("DOBJ", dobj_real_ranks, dobj_perm_ranks_list),
    ]:
        if not real_ranks:
            continue

        real_k1_rate = sum(1 for r in real_ranks if r == 1) / len(real_ranks)
        real_avg = np.mean(real_ranks)

        # Bootstrap: 重采样real_ranks
        boot_k1_rates = []
        boot_avg_ranks = []
        for _ in range(N_BOOT):
            sample = [random.choice(real_ranks) for _ in range(len(real_ranks))]
            boot_k1_rates.append(sum(1 for r in sample if r == 1) / len(sample))
            boot_avg_ranks.append(np.mean(sample))

        ci_k1_low = np.percentile(boot_k1_rates, 2.5)
        ci_k1_high = np.percentile(boot_k1_rates, 97.5)
        ci_avg_low = np.percentile(boot_avg_ranks, 2.5)
        ci_avg_high = np.percentile(boot_avg_ranks, 97.5)

        # Permutation baseline
        all_perm_ranks = [r for pl in perm_ranks_list for r in pl]
        perm_k1 = sum(1 for r in all_perm_ranks if r == 1) / len(all_perm_ranks) if all_perm_ranks else 0
        perm_avg = np.mean(all_perm_ranks) if all_perm_ranks else 999

        print(f"\n  {role_name} (N={len(real_ranks)}):")
        print(f"    Real k1 rate: {real_k1_rate:.3f} 95%CI=[{ci_k1_low:.3f}, {ci_k1_high:.3f}]")
        print(f"    Real avg rank: {real_avg:.2f} 95%CI=[{ci_avg_low:.2f}, {ci_avg_high:.2f}]")
        print(f"    Perm k1 rate:  {perm_k1:.3f}")
        print(f"    Perm avg rank: {perm_avg:.2f}")

        # 关键判断: 真实CI是否与permutation baseline不重叠
        if ci_k1_low > perm_k1:
            print(f"    ★★★ REAL CI 下界 > Perm k1 rate → 统计显著 ★★★")
        elif ci_avg_high < perm_avg:
            print(f"    ★★★ REAL CI 上界 < Perm avg rank → 统计显著 ★★★")
        else:
            print(f"    ✗ CI与permutation重叠 → 无法确认统计显著")

    return {"nsubj_real": nsubj_real_ranks, "dobj_real": dobj_real_ranks}


# ============================================================
# Exp2: 因果Ablation — 主谓一致测试
# ============================================================
def exp2_causal_ablation(model, tokenizer, device, info):
    """
    核心逻辑:
    1. 找到跨句子稳定的nsubj-tracking heads
    2. Ablate这些heads → 测主谓一致能力
    3. 对比: 随机heads ablation → 应该影响小

    主谓一致测试:
    - "The cat [chased/chase]" → 模型应偏好 chased
    - "The cats [chased/chase]" → 模型应偏好 chase (复数现在时)
    - ablate语法head → 偏好应该减弱
    """
    print("\n" + "="*70)
    print("★★★ EXP2: 因果ABLATION — 主谓一致 ★★★")
    print("="*70)

    # Step 1: 找稳定的nsubj-tracking heads
    print("\n--- Step 1: Identifying reliable nsubj-tracking heads ---")

    svo_sentences = [
        "The cat chased the mouse",
        "The dog bit the man",
        "The girl saw the boy",
        "The king ruled the land",
        "The bird ate the fish",
        "The wolf killed the deer",
        "The fox caught the rabbit",
        "The bear chased the deer",
    ]

    head_success = defaultdict(int)
    head_total = defaultdict(int)

    for sentence in svo_sentences:
        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        seq_len = len(tokens)

        results = compute_rank_for_dep(attn_dict, 3, 2, info.n_layers)  # nsubj: pos2→pos3
        for r in results:
            key = (r["layer"], r["head"])
            head_total[key] += 1
            if r["rank"] == 1:
                head_success[key] += 1

        del attn_dict, inputs
        gc.collect()

    # 找≥7/8成功的head
    reliable_heads = []
    for key in head_total:
        if head_total[key] >= 6:
            rate = head_success[key] / head_total[key]
            if rate >= 0.8:
                reliable_heads.append((key[0], key[1], rate, head_success[key]))

    reliable_heads.sort(key=lambda x: x[2], reverse=True)
    print(f"  Found {len(reliable_heads)} reliable nsubj-tracking heads (≥80% across sentences):")
    for li, h, rate, cnt in reliable_heads[:20]:
        print(f"    L{li} H{h}: rate={rate:.2f} ({cnt}/{head_total[(li,h)]})")

    if not reliable_heads:
        print("  No reliable heads! Using top-10 by absolute success count.")
        all_heads = [(k[0], k[1], head_success[k]/head_total[k], head_success[k])
                     for k in head_total if head_total[k] >= 4]
        all_heads.sort(key=lambda x: x[3], reverse=True)
        reliable_heads = all_heads[:10]

    # Step 2: 主谓一致测试集
    print("\n--- Step 2: Subject-Verb Agreement Test ---")

    agreement_pairs = [
        # (prefix, correct_verb, wrong_verb, description)
        ("The cat", "chased", "chase"),        # 单数主语 → 过去时无所谓单复数, 但chase是现在时
        ("The cat", "runs", "run"),             # 单数 → runs
        ("The dog", "barks", "bark"),           # 单数 → barks
        ("The bird", "flies", "fly"),           # 单数 → flies
        ("The cats", "run", "runs"),            # 复数 → run
        ("The dogs", "bark", "barks"),          # 复数 → bark
        ("The birds", "fly", "flies"),          # 复数 → fly
        ("The cat", "eats", "eat"),             # 单数 → eats
        ("The fish", "swims", "swim"),          # 单数 → swims
        ("The children", "play", "plays"),      # 复数 → play
    ]

    def get_logit_diff(prefix, correct, wrong, mdl, tok, dev):
        """correct vs wrong的logit差 (在prefix末尾位置)"""
        inputs = tok(prefix, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = mdl(input_ids=inputs["input_ids"])
            logits = outputs.logits[0]

        correct_id = tok.encode(correct, add_special_tokens=False)[0]
        wrong_id = tok.encode(wrong, add_special_tokens=False)[0]

        last_logits = logits[-1]
        return float(last_logits[correct_id] - last_logits[wrong_id])

    # Baseline
    print("\n  Baseline (no ablation):")
    baseline_diffs = []
    for prefix, correct, wrong in agreement_pairs:
        diff = get_logit_diff(prefix, correct, wrong, model, tokenizer, device)
        baseline_diffs.append(diff)
        print(f"    '{prefix}' → {correct} vs {wrong}: Δ={diff:.2f}")

    avg_baseline = np.mean(baseline_diffs)
    print(f"    Average: {avg_baseline:.2f}")

    # Step 3: Ablate reliable nsubj heads
    print("\n--- Step 3: Ablating reliable nsubj-tracking heads ---")

    ablation_results = []

    # 获取模型结构信息
    layers = model.model.layers

    def make_zero_hook(target_head_idx, n_h):
        """创建zero-out hook"""
        def hook_fn(module, input, output):
            attn_output = output[0].clone()
            d_model = attn_output.shape[-1]
            d_head = d_model // n_h
            start = target_head_idx * d_head
            end = (target_head_idx + 1) * d_head
            if end <= d_model:
                attn_output[:, :, start:end] = 0
            return (attn_output,) + output[1:]
        return hook_fn

    # 获取n_heads
    sample_attn = layers[0].self_attn
    if hasattr(sample_attn, 'num_heads'):
        n_heads = sample_attn.num_heads
    elif hasattr(sample_attn.config, 'num_attention_heads'):
        n_heads = sample_attn.config.num_attention_heads
    else:
        n_heads = info.d_model // 128  # 估计

    print(f"  n_heads per layer: {n_heads}")

    # 逐个ablate top reliable heads
    top_heads = reliable_heads[:8]  # top 8

    for li, h, rate, cnt in top_heads:
        try:
            attn = layers[li].self_attn
            hook = attn.register_forward_hook(make_zero_hook(h, n_heads))

            ablation_diffs = []
            for prefix, correct, wrong in agreement_pairs:
                diff = get_logit_diff(prefix, correct, wrong, model, tokenizer, device)
                ablation_diffs.append(diff)

            hook.remove()

            avg_ablation = np.mean(ablation_diffs)
            delta = avg_ablation - avg_baseline
            ablation_results.append((li, h, rate, avg_ablation, delta))

            sig = "★★★" if delta < -1.0 else ("★" if delta < -0.3 else "")
            print(f"    L{li} H{h} (rate={rate:.2f}): avg={avg_ablation:.2f} Δ={delta:+.2f} {sig}")

        except Exception as e:
            print(f"    L{li} H{h}: FAILED - {e}")

    # Step 4: 同时ablate多个heads (联合效应)
    print("\n--- Step 4: Joint Ablation (multiple syntax heads) ---")

    # 选top-3最reliable的heads同时ablate
    if len(top_heads) >= 3:
        joint_heads = [(li, h) for li, h, _, _ in top_heads[:3]]
        hooks = []
        for li, h in joint_heads:
            attn = layers[li].self_attn
            hooks.append(attn.register_forward_hook(make_zero_hook(h, n_heads)))

        joint_diffs = []
        for prefix, correct, wrong in agreement_pairs:
            diff = get_logit_diff(prefix, correct, wrong, model, tokenizer, device)
            joint_diffs.append(diff)

        for hk in hooks:
            hk.remove()

        avg_joint = np.mean(joint_diffs)
        delta_joint = avg_joint - avg_baseline
        sig = "★★★" if delta_joint < -2.0 else ("★" if delta_joint < -0.5 else "")
        print(f"  Joint ablation of {len(joint_heads)} heads: avg={avg_joint:.2f} Δ={delta_joint:+.2f} {sig}")

    # Step 5: Control — 随机heads
    print("\n--- Step 5: Control — Random head ablation ---")

    random_deltas = []
    for trial in range(10):
        li = random.randint(0, info.n_layers - 1)
        h = random.randint(0, n_heads - 1)

        try:
            attn = layers[li].self_attn
            hook = attn.register_forward_hook(make_zero_hook(h, n_heads))

            rdiffs = []
            for prefix, correct, wrong in agreement_pairs:
                diff = get_logit_diff(prefix, correct, wrong, model, tokenizer, device)
                rdiffs.append(diff)

            hook.remove()

            avg_r = np.mean(rdiffs)
            delta_r = avg_r - avg_baseline
            random_deltas.append(delta_r)
            print(f"    L{li} H{h} (random): Δ={delta_r:+.2f}")

        except Exception as e:
            print(f"    L{li} H{h} (random): FAILED - {e}")

    # ===== 最终比较 =====
    print("\n--- Final Comparison ---")
    if ablation_results:
        syntax_deltas = [d for _, _, _, _, d in ablation_results]
        print(f"  Syntax head ablation: mean Δ = {np.mean(syntax_deltas):+.3f}")
    if random_deltas:
        print(f"  Random head ablation: mean Δ = {np.mean(random_deltas):+.3f}")

    if ablation_results and random_deltas:
        syntax_deltas = [d for _, _, _, _, d in ablation_results]
        from scipy import stats
        try:
            t, p = stats.ttest_ind(syntax_deltas, random_deltas)
            print(f"  t-test (syntax vs random Δ): t={t:.3f}, p={p:.4f}")
            if np.mean(syntax_deltas) < np.mean(random_deltas) - 0.5 and p < 0.05:
                print("  ★★★ SYNTAX HEADS CAUSALLY INVOLVED IN GRAMMAR ★★★")
            else:
                print("  ✗ No significant causal effect")
        except:
            pass

    return ablation_results


# ============================================================
# Exp3: nsubj/dobj不对称性深度分析
# ============================================================
def exp3_nsubj_dobj_asymmetry(model, tokenizer, device, info):
    """
    为什么nsubj被追踪而dobj不?

    假说1: 位置原因 — nsubj在pos2(靠近动词), dobj在pos5(远)
    假说2: 角色原因 — attention天然更关注主语
    假说3: 多重比较 — nsubj是"左侧最近名词"的假象

    测试:
    - 同距离不同角色: nsubj(pos5→pos6) vs dobj(pos5→pos6) in center-embedding
    - 不同距离同角色: nsubj at d=1 vs nsubj at d=5
    - 左侧最近名词baseline
    """
    print("\n" + "="*70)
    print("★★★ EXP3: nsubj/dobj不对称性分析 ★★★")
    print("="*70)

    # ---- 测试1: 中心嵌入中的nsubj vs dobj ----
    # "The cat that the dog chased ran"
    # nsubj_short: dog(5)→chased(6), d=1
    # dobj: cat(2)→chased(6), d=4
    # nsubj_long: cat(2)→ran(7), d=5

    center_sentences = [
        "The cat that the dog chased ran",
        "The man that the girl saw left",
        "The bird that the cat watched flew",
        "The king that the man served died",
        "The dog that the wolf chased barked",
        "The fish that the cat ate vanished",
    ]

    print("\n--- Center-Embedding: nsubj vs dobj at same verb ---")

    nsubj_short_ranks = []  # dog→chased (d=1, nsubj)
    dobj_ranks = []          # cat→chased (d=4, dobj)
    nsubj_long_ranks = []   # cat→ran (d=5, nsubj)

    N_PERM = 200

    for sentence in center_sentences:
        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue

        seq_len = len(tokens)
        print(f"\n  [{sentence}] tokens={seq_len}")

        # nsubj_short: pos5 → pos6 (d=1)
        results = compute_rank_for_dep(attn_dict, 6, 5, info.n_layers)
        if results:
            best = min(r["rank"] for r in results)
            nsubj_short_ranks.append(best)

            # Permutation
            valid_pos = [p for p in range(1, seq_len) if p != 6 and p != 5]
            perm_best = []
            for _ in range(N_PERM):
                fake = random.choice(valid_pos)
                pr = compute_rank_for_dep(attn_dict, 6, fake, info.n_layers)
                if pr:
                    perm_best.append(min(r["rank"] for r in pr))
            perm_k1 = sum(1 for r in perm_best if r == 1) / len(perm_best) if perm_best else 0
            print(f"    nsubj_short (pos5→pos6, d=1): best_rank={best}, perm_k1={perm_k1:.3f}")

        # dobj: pos2 → pos6 (d=4)
        results = compute_rank_for_dep(attn_dict, 6, 2, info.n_layers)
        if results:
            best = min(r["rank"] for r in results)
            dobj_ranks.append(best)

            valid_pos = [p for p in range(1, seq_len) if p != 6 and p != 2]
            perm_best = []
            for _ in range(N_PERM):
                fake = random.choice(valid_pos)
                pr = compute_rank_for_dep(attn_dict, 6, fake, info.n_layers)
                if pr:
                    perm_best.append(min(r["rank"] for r in pr))
            perm_k1 = sum(1 for r in perm_best if r == 1) / len(perm_best) if perm_best else 0
            print(f"    dobj (pos2→pos6, d=4): best_rank={best}, perm_k1={perm_k1:.3f}")

        # nsubj_long: pos2 → pos7 (d=5)
        results = compute_rank_for_dep(attn_dict, 7, 2, info.n_layers)
        if results:
            best = min(r["rank"] for r in results)
            nsubj_long_ranks.append(best)

            valid_pos = [p for p in range(1, seq_len) if p != 7 and p != 2]
            perm_best = []
            for _ in range(N_PERM):
                fake = random.choice(valid_pos)
                pr = compute_rank_for_dep(attn_dict, 7, fake, info.n_layers)
                if pr:
                    perm_best.append(min(r["rank"] for r in pr))
            perm_k1 = sum(1 for r in perm_best if r == 1) / len(perm_best) if perm_best else 0
            print(f"    nsubj_long (pos2→pos7, d=5): best_rank={best}, perm_k1={perm_k1:.3f}")

        del attn_dict, inputs
        gc.collect()

    # ---- 统计比较 ----
    print("\n\n--- nsubj vs dobj Comparison ---")

    for name, ranks in [("nsubj_short (d=1)", nsubj_short_ranks),
                         ("dobj (d=4)", dobj_ranks),
                         ("nsubj_long (d=5)", nsubj_long_ranks)]:
        if ranks:
            k1_rate = sum(1 for r in ranks if r == 1) / len(ranks)
            avg = np.mean(ranks)
            print(f"  {name}: k1_rate={k1_rate:.3f}, avg_rank={avg:.2f} (N={len(ranks)})")

    # 关键比较: 同一个动词chased, nsubj(pos5) vs dobj(pos2)
    # 如果距离是关键: nsubj_short(d=1) > dobj(d=4) > nsubj_long(d=5)
    # 如果角色是关键: nsubj_short ≈ nsubj_long > dobj
    print("\n  ★ If distance drives: nsubj_short(d=1) > dobj(d=4) > nsubj_long(d=5)")
    print("  ★ If role drives:     nsubj_short ≈ nsubj_long > dobj")
    print("  ★ If position drives: nsubj_short > nsubj_long > dobj")

    # ---- 测试2: 左侧最近名词baseline ----
    print("\n\n--- Left-Nearest-Noun Baseline ---")
    print("  For SVO 'The cat chased the mouse':")
    print("    chased(pos3) attend left → nearest noun = cat(pos2) = nsubj ✓")
    print("    chased(pos3) attend left → far noun = mouse(pos5) ≠ nearest ✗")
    print("  → nsubj 'success' may just be 'attend to nearest left noun'")
    print("  → This is position heuristic, not syntactic role understanding")

    # ---- 测试3: VSO语言对照 ----
    # 在英语中无法完美测试, 但可以用不同结构创造"最近名词≠nsubj"
    print("\n\n--- Critical Test: Nearest-Noun ≠ nsubj ---")

    # 使用前置宾语: "The mouse, the cat chased"
    # 如果是nearest-noun: cat(最近)被追踪
    # 如果是nsubj: cat(nsubj)被追踪 → same result, 不区分!

    # 更好的测试: 用介词短语隔开
    # "The cat with the hat chased the mouse"
    # hat是距chased最近的左侧名词, 但不是nsubj
    # nsubj仍然是cat

    complex_sentences = [
        {"sentence": "The cat with the hat chased the mouse",
         "nsubj_pos": 2, "verb_pos": 7, "nearest_noun_pos": 5,  # hat
         "desc": "PP modifier: nearest_noun=hat ≠ nsubj=cat"},
        {"sentence": "The dog near the park chased the cat",
         "nsubj_pos": 2, "verb_pos": 7, "nearest_noun_pos": 5,  # park
         "desc": "PP modifier: nearest_noun=park ≠ nsubj=dog"},
        {"sentence": "The bird on the tree saw the fish",
         "nsubj_pos": 2, "verb_pos": 7, "nearest_noun_pos": 5,  # tree
         "desc": "PP modifier: nearest_noun=tree ≠ nsubj=bird"},
    ]

    print("\n  Testing: Does attention track nsubj or nearest noun?")
    print("  (When nearest noun ≠ nsubj, this is the critical test)")

    for item in complex_sentences:
        sentence = item["sentence"]
        nsubj_pos = item["nsubj_pos"]
        verb_pos = item["verb_pos"]
        nearest_pos = item["nearest_noun_pos"]

        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue

        seq_len = len(tokens)
        print(f"\n  [{sentence}]")
        print(f"  Tokens: {[(i,t) for i,t in enumerate(tokens)]}")

        # nsubj rank
        nsubj_results = compute_rank_for_dep(attn_dict, verb_pos, nsubj_pos, info.n_layers)
        if nsubj_results:
            nsubj_best = min(r["rank"] for r in nsubj_results)
            nsubj_k1 = sum(1 for r in nsubj_results if r["rank"] == 1)
        else:
            nsubj_best = 999

        # nearest noun rank
        nearest_results = compute_rank_for_dep(attn_dict, verb_pos, nearest_pos, info.n_layers)
        if nearest_results:
            nearest_best = min(r["rank"] for r in nearest_results)
            nearest_k1 = sum(1 for r in nearest_results if r["rank"] == 1)
        else:
            nearest_best = 999

        print(f"  {item['desc']}")
        print(f"    nsubj({tokens[nsubj_pos]}, pos{nsubj_pos}): best_rank={nsubj_best}")
        print(f"    nearest_noun({tokens[nearest_pos]}, pos{nearest_pos}): best_rank={nearest_best}")

        if nsubj_best < nearest_best:
            print(f"    ★★★ Attention tracks NSUBJ, not nearest noun ★★★")
        elif nsubj_best > nearest_best:
            print(f"    ✗ Attention tracks NEAREST NOUN, not nsubj")
        else:
            print(f"    ~ Same rank — inconclusive")

        # Permutation test for both
        valid_pos = [p for p in range(1, seq_len) if p != verb_pos and p != nsubj_pos and p != nearest_pos]
        if valid_pos:
            perm_nsubj = []
            perm_nearest = []
            for _ in range(200):
                fake = random.choice(valid_pos)
                pr = compute_rank_for_dep(attn_dict, verb_pos, fake, info.n_layers)
                if pr:
                    perm_nsubj.append(min(r["rank"] for r in pr))

            perm_k1 = sum(1 for r in perm_nsubj if r == 1) / len(perm_nsubj) if perm_nsubj else 0
            print(f"    Perm k1 rate: {perm_k1:.3f}")

            if nsubj_best == 1 and perm_k1 < 0.2:
                print(f"    ★ nsubj rank=1 is above chance")
            if nearest_best == 1 and perm_k1 < 0.2:
                print(f"    ★ nearest_noun rank=1 is above chance")

        del attn_dict, inputs
        gc.collect()

    return {"nsubj_short": nsubj_short_ranks, "dobj": dobj_ranks, "nsubj_long": nsubj_long_ranks}


# ============================================================
# Exp4: OVS + Wh-front 冲突测试 (改进版)
# ============================================================
def exp4_conflict_test_rigorous(model, tokenizer, device, info):
    """
    Phase 55B的冲突测试缺少permutation control

    改进:
    1. 每个冲突句都做permutation test
    2. 关注"nsubj在非标准位置"的追踪
    3. 比较同一动词的nsubj vs dobj attention profile
    """
    print("\n" + "="*70)
    print("★★★ EXP4: 严格冲突测试 + Permutation Control ★★★")
    print("="*70)

    # OSV: "The mouse the cat chased"
    # → cat(pos3) 是 chased(pos4) 的 nsubj, 在pos3(非标准nsubj位置pos2)
    # → mouse(pos2) 是 chased(pos4) 的 dobj, 在pos2(标准nsubj位置)

    # Wh-front: "What did the cat chase"
    # → What(pos1) 是 chase(pos5) 的 dobj, 在pos1(标准nsubj位置)
    # → cat(pos4) 是 chase(pos5) 的 nsubj, 在pos4

    test_cases = [
        {
            "name": "OSV conflict",
            "sentences": [
                "The mouse the cat chased",
                "The man the girl saw",
                "The bird the dog watched",
                "The deer the wolf chased",
            ],
            "get_deps": lambda sl: [(4, 3, "nsubj"), (4, 2, "dobj")] if sl >= 5 else [],
            "conflict": "nsubj(pos3, non-standard) vs dobj(pos2, looks-like-nsubj-position)",
        },
        {
            "name": "Wh-front conflict",
            "sentences": [
                "What did the cat chase",
                "What did the dog see",
                "What did the bird eat",
                "What did the man read",
            ],
            "get_deps": lambda sl: [(5, 4, "nsubj"), (5, 1, "dobj")] if sl >= 6 else [],
            "conflict": "dobj(pos1, looks-like-nsubj) vs nsubj(pos4)",
        },
        {
            "name": "SVO baseline",
            "sentences": [
                "The cat chased the mouse",
                "The dog saw the man",
                "The bird ate the fish",
            ],
            "get_deps": lambda sl: [(3, 2, "nsubj"), (3, 5, "dobj")] if sl >= 6 else [],
            "conflict": "No conflict — position=syntax",
        },
    ]

    N_PERM = 200

    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"  Conflict: {case['conflict']}")

        nsubj_real_ranks = []
        dobj_real_ranks = []
        nsubj_perm_k1s = []
        dobj_perm_k1s = []

        for sentence in case["sentences"]:
            attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
            if attn_dict is None:
                continue

            seq_len = len(tokens)
            deps = case["get_deps"](seq_len)

            print(f"\n  [{sentence}] tokens={seq_len}: {[(i,t) for i,t in enumerate(tokens)]}")

            for head_pos, dep_pos, rel_type in deps:
                if head_pos >= seq_len or dep_pos >= seq_len:
                    continue

                results = compute_rank_for_dep(attn_dict, head_pos, dep_pos, info.n_layers)
                if not results:
                    continue
                best_rank = min(r["rank"] for r in results)

                # Permutation
                valid_pos = [p for p in range(1, seq_len) if p != head_pos and p != dep_pos]
                if not valid_pos:
                    continue

                perm_ranks = []
                for _ in range(N_PERM):
                    fake = random.choice(valid_pos)
                    pr = compute_rank_for_dep(attn_dict, head_pos, fake, info.n_layers)
                    if pr:
                        perm_ranks.append(min(r["rank"] for r in pr))

                perm_avg = np.mean(perm_ranks) if perm_ranks else 999
                perm_k1 = sum(1 for r in perm_ranks if r == 1) / len(perm_ranks) if perm_ranks else 0

                sig = "★★★" if best_rank == 1 and perm_k1 < 0.2 else ("★" if best_rank == 1 and perm_k1 < 0.4 else "✗")

                print(f"    {rel_type}: {tokens[dep_pos]}(pos{dep_pos})→{tokens[head_pos]}(pos{head_pos}) "
                      f"best_rank={best_rank} perm_k1={perm_k1:.3f} {sig}")

                if "nsubj" in rel_type:
                    nsubj_real_ranks.append(best_rank)
                    nsubj_perm_k1s.append(perm_k1)
                else:
                    dobj_real_ranks.append(best_rank)
                    dobj_perm_k1s.append(perm_k1)

            del attn_dict, inputs
            gc.collect()

        # 汇总
        print(f"\n  Summary for {case['name']}:")
        if nsubj_real_ranks:
            nsubj_k1 = sum(1 for r in nsubj_real_ranks if r == 1) / len(nsubj_real_ranks)
            avg_perm_k1 = np.mean(nsubj_perm_k1s) if nsubj_perm_k1s else 0
            print(f"    NSUBJ: k1_rate={nsubj_k1:.3f}, avg_perm_k1={avg_perm_k1:.3f}")
        if dobj_real_ranks:
            dobj_k1 = sum(1 for r in dobj_real_ranks if r == 1) / len(dobj_real_ranks)
            avg_perm_k1 = np.mean(dobj_perm_k1s) if dobj_perm_k1s else 0
            print(f"    DOBJ:  k1_rate={dobj_k1:.3f}, avg_perm_k1={avg_perm_k1:.3f}")

    return {}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b",
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=0,
                        help="0=all, 1=bootstrap, 2=ablation, 3=asymmetry, 4=conflict")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"Model: {args.model}, Layers: {info.n_layers}, d_model: {info.d_model}")

    if args.exp in [0, 1]:
        exp1_nsubj_bootstrap(model, tokenizer, device, info)

    if args.exp in [0, 2]:
        exp2_causal_ablation(model, tokenizer, device, info)

    if args.exp in [0, 3]:
        exp3_nsubj_dobj_asymmetry(model, tokenizer, device, info)

    if args.exp in [0, 4]:
        exp4_conflict_test_rigorous(model, tokenizer, device, info)

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("\nPhase 57 Complete!")


if __name__ == "__main__":
    main()
