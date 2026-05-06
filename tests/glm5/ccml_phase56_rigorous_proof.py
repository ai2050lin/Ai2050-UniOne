"""
Phase 56: 论文级严格验证 — 解决4大硬伤
==========================================

硬伤1: 784 heads "any hit" = 多重比较问题 → Permutation Test
硬伤2: Wh-front可能是pattern非role → Role-level Discrimination
硬伤3: 语义剥离过强 → 修正表述 + 模板控制
硬伤4: correlational≠causal → Targeted Ablation

实验设计:
Exp1: Permutation Test — 打乱dependency labels, 评估统计显著性
Exp2: Role Discrimination — What/Who/Where/When区分测试
Exp3: Targeted Ablation — 语法head消融→语法错误率
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
# 测试句集合
# ============================================================

# ---- 基础SVO (用于permutation test) ----
BASIC_SVO = [
    {"sentence": "The cat chased the mouse", "deps": [(3,2,"nsubj"), (3,5,"dobj")]},
    {"sentence": "The dog bit the man", "deps": [(3,2,"nsubj"), (3,5,"dobj")]},
    {"sentence": "The girl saw the boy", "deps": [(3,2,"nsubj"), (3,5,"dobj")]},
    {"sentence": "The king ruled the land", "deps": [(3,2,"nsubj"), (3,5,"dobj")]},
    {"sentence": "The bird ate the fish", "deps": [(3,2,"nsubj"), (3,5,"dobj")]},
    {"sentence": "The wolf killed the deer", "deps": [(3,2,"nsubj"), (3,5,"dobj")]},
]

# ---- 中心嵌入 (用于permutation test) ----
CENTER_EMBED = [
    {"sentence": "The cat that the dog chased ran", "deps": [(7,2,"nsubj"), (6,5,"nsubj"), (6,2,"dobj")]},
    {"sentence": "The man that the girl saw left", "deps": [(7,2,"nsubj"), (6,5,"nsubj"), (6,2,"dobj")]},
    {"sentence": "The bird that the cat watched flew", "deps": [(7,2,"nsubj"), (6,5,"nsubj"), (6,2,"dobj")]},
    {"sentence": "The king that the man served died", "deps": [(7,2,"nsubj"), (6,5,"nsubj"), (6,2,"dobj")]},
]

# ---- ★★★ Role Discrimination: 不同Wh-词的语法角色区分 ★★★ ----
WH_ROLE_DISCRIM = [
    # What = dobj (chase的直接宾语)
    {"name": "what=dobj", "sentence": "What did the cat chase",
     "verb_pos": 4, "wh_pos": 1, "nsubj_pos": 3,
     "expected_role": "dobj", "wh_word": "What"},
    # Who = nsubj (chase的主语, 虽然语义上who也可以是obj, 但这里测试结构)
    # 改用更清晰的: "Who chased the cat" → who是nsubj
    {"name": "who=nsubj", "sentence": "Who chased the cat",
     "verb_pos": 2, "wh_pos": 1, "dobj_pos": 4,
     "expected_role": "nsubj", "wh_word": "Who"},
    # Where = adjunct (不是论元)
    {"name": "where=adjunct", "sentence": "Where did the cat sleep",
     "verb_pos": 4, "wh_pos": 1, "nsubj_pos": 3,
     "expected_role": "adjunct", "wh_word": "Where"},
    # When = adjunct
    {"name": "when=adjunct", "sentence": "When did the cat sleep",
     "verb_pos": 4, "wh_pos": 1, "nsubj_pos": 3,
     "expected_role": "adjunct", "wh_word": "When"},
    # How = adjunct (方式)
    {"name": "how=adjunct", "sentence": "How did the cat escape",
     "verb_pos": 4, "wh_pos": 1, "nsubj_pos": 3,
     "expected_role": "adjunct", "wh_word": "How"},
    # ★ 关键对照: What(主语) vs What(宾语)
    {"name": "what=nsubj", "sentence": "What chased the cat",
     "verb_pos": 2, "wh_pos": 1, "dobj_pos": 4,
     "expected_role": "nsubj", "wh_word": "What"},
]

# ---- Ablation测试句 (简单的语法错误检测) ----
ABLATION_SENTENCES = [
    # 正确语法
    {"sentence": "The cat chased the mouse", "correct": True},
    {"sentence": "The dog bit the man", "correct": True},
    # 语法错误 (主谓不一致)
    {"sentence": "The cat chase the mouse", "correct": False},
    {"sentence": "The dog bites the man", "correct": True},  # 这其实是正确的...
]


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


def compute_rank_for_dep(attn_dict, head_pos, dep_pos, n_layers):
    """对给定的dependency (head_pos, dep_pos), 计算per-head的rank"""
    results = []
    for li in range(n_layers):
        if li not in attn_dict:
            continue
        A = attn_dict[li]
        n_heads = A.shape[0]

        # head_pos attend to dep_pos
        if dep_pos < head_pos:
            # dep在head左边, 看head_pos行的左边部分
            row = A[:, head_pos, 1:head_pos]  # [n_heads, head_pos-1]
            target = dep_pos - 1
        elif dep_pos > head_pos:
            # dep在head右边 (罕见但可能)
            row = A[:, head_pos, 1:]  # [n_heads, seq_len-1]
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
# Exp1: Permutation Test
# ============================================================
def exp1_permutation_test(model, tokenizer, device, info):
    """
    核心逻辑:
    1. 对真实句子, 计算"真实dependency"的per-head rank
    2. 打乱dependency labels (随机选一个非正确的position作为"假dep")
    3. 计算假dep的per-head rank
    4. 比较: 真实dep的命中率是否显著高于随机dep

    如果: 真实dep的best_rank=1率 >> 随机dep的best_rank=1率
    则: 排除多重比较导致的false positive
    """
    print("\n" + "="*70)
    print("★★★ EXP1: PERMUTATION TEST ★★★")
    print("测试: 真实dependency的命中率是否显著高于随机baseline")
    print("="*70)

    all_sentences = BASIC_SVO + CENTER_EMBED

    real_best_ranks = []  # 真实dep的best_rank
    perm_best_ranks = []  # 随机dep的best_rank (多次permutation平均)

    N_PERM = 100  # permutation次数

    for item in all_sentences:
        sentence = item["sentence"]
        deps = item["deps"]

        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue

        seq_len = len(tokens)
        print(f"\n  [{sentence}] tokens={seq_len}")

        for head_pos, dep_pos, rel_type in deps:
            if head_pos >= seq_len or dep_pos >= seq_len:
                continue

            # 1. 真实dep的rank
            real_results = compute_rank_for_dep(attn_dict, head_pos, dep_pos, info.n_layers)
            if real_results:
                real_best = min(r["rank"] for r in real_results)
                real_k1_count = sum(1 for r in real_results if r["rank"] == 1)
            else:
                continue

            real_best_ranks.append(real_best)

            # 2. Permutation: 随机选择非正确的position作为"假dep"
            #    只选合理的position (不是head自身, 不是特殊token)
            valid_positions = [p for p in range(1, seq_len) if p != head_pos and p != dep_pos]
            if not valid_positions:
                continue

            perm_ranks_for_this_dep = []
            for _ in range(N_PERM):
                fake_dep = random.choice(valid_positions)
                perm_results = compute_rank_for_dep(attn_dict, head_pos, fake_dep, info.n_layers)
                if perm_results:
                    perm_best = min(r["rank"] for r in perm_results)
                    perm_ranks_for_this_dep.append(perm_best)

            if perm_ranks_for_this_dep:
                perm_best_ranks.append(np.mean(perm_ranks_for_this_dep))

            # 打印详情
            perm_k1_rate = sum(1 for r in perm_ranks_for_this_dep if r == 1) / len(perm_ranks_for_this_dep) if perm_ranks_for_this_dep else 0
            print(f"    {tokens[dep_pos]}({dep_pos})→{tokens[head_pos]}({head_pos}) [{rel_type}] "
                  f"real_best_rank={real_best} perm_avg_best_rank={np.mean(perm_ranks_for_this_dep):.1f} "
                  f"perm_k1_rate={perm_k1_rate:.3f}")

        del attn_dict, inputs
        gc.collect()

    # ===== 统计检验 =====
    print("\n\n" + "="*70)
    print("★★★ PERMUTATION TEST RESULTS ★★★")
    print("="*70)

    if not real_best_ranks:
        print("No results!")
        return

    real_k1_rate = sum(1 for r in real_best_ranks if r == 1) / len(real_best_ranks)
    real_avg_rank = np.mean(real_best_ranks)

    print(f"\n  真实dependency: best_rank=1 率 = {real_k1_rate:.3f} ({sum(1 for r in real_best_ranks if r == 1)}/{len(real_best_ranks)})")
    print(f"  真实dependency: 平均best_rank = {real_avg_rank:.2f}")

    if perm_best_ranks:
        perm_avg = np.mean(perm_best_ranks)
        perm_k1_rate_est = max(0, 1 - perm_avg / (perm_avg + 1))  # 估计
        print(f"  随机dependency: 平均best_rank = {perm_avg:.2f}")

        # 效应量: 真实 vs 随机的差异
        # 更精确: 计算per-dep的real vs perm差异
        print(f"\n  ★★★ 如果 real_avg_rank << perm_avg → 统计显著 ★★★")

        # 简单p值估计: 用rank sum test
        from scipy import stats
        if len(real_best_ranks) >= 2 and len(perm_best_ranks) >= 2:
            # real vs perm的best_rank比较
            try:
                stat, pval = stats.mannwhitneyu(real_best_ranks, perm_best_ranks, alternative='less')
                print(f"  Mann-Whitney U test: p = {pval:.6f}")
                if pval < 0.05:
                    print(f"  ★★★ p < 0.05 → 真实dep的rank显著低于随机dep ★★★")
                else:
                    print(f"  ✗ p >= 0.05 → 无法排除多重比较假阳性")
            except:
                print("  (stats test failed, too few samples)")

    # 额外: 逐层分析permutation
    print("\n\n--- Per-Layer Permutation Analysis ---")
    # 重新运行, 按层统计
    layer_real_k1 = defaultdict(int)
    layer_real_total = defaultdict(int)
    layer_perm_k1 = defaultdict(int)
    layer_perm_total = defaultdict(int)

    for item in all_sentences:
        sentence = item["sentence"]
        deps = item["deps"]

        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue

        seq_len = len(tokens)

        for head_pos, dep_pos, rel_type in deps:
            if head_pos >= seq_len or dep_pos >= seq_len:
                continue

            valid_positions = [p for p in range(1, seq_len) if p != head_pos and p != dep_pos]
            if not valid_positions:
                continue

            for li in range(min(info.n_layers, 28)):
                if li not in attn_dict:
                    continue
                A = attn_dict[li]
                n_heads = A.shape[0]

                if dep_pos < head_pos:
                    row = A[:, head_pos, 1:head_pos]
                    target = dep_pos - 1
                else:
                    row = A[:, head_pos, 1:]
                    target = dep_pos - 1

                if target < 0 or target >= row.shape[1]:
                    continue

                for h in range(n_heads):
                    sorted_idx = np.argsort(row[h])[::-1]
                    real_rank = int(np.where(sorted_idx == target)[0][0]) + 1 if target in sorted_idx else 999

                    layer_real_total[li] += 1
                    if real_rank == 1:
                        layer_real_k1[li] += 1

                    # Permutation
                    fake_dep = random.choice(valid_positions)
                    fake_target = fake_dep - 1
                    if fake_target < 0 or fake_target >= row.shape[1]:
                        continue
                    fake_rank = int(np.where(sorted_idx == fake_target)[0][0]) + 1 if fake_target in sorted_idx else 999

                    layer_perm_total[li] += 1
                    if fake_rank == 1:
                        layer_perm_k1[li] += 1

        del attn_dict, inputs
        gc.collect()

    print(f"\n  {'Layer':>6} {'Real k1%':>10} {'Perm k1%':>10} {'Ratio':>8} {'Signal?':>8}")
    print(f"  {'-'*50}")
    for li in sorted(layer_real_total.keys()):
        if li not in [0, 1, 3, 6, 10, 15, 20]:
            continue
        real_rate = layer_real_k1[li] / layer_real_total[li] * 100 if layer_real_total[li] > 0 else 0
        perm_rate = layer_perm_k1[li] / layer_perm_total[li] * 100 if layer_perm_total[li] > 0 else 0
        ratio = real_rate / perm_rate if perm_rate > 0 else float('inf')
        signal = "★" if ratio > 3 else "✗"
        print(f"  L{li:>4} {real_rate:>9.1f}% {perm_rate:>9.1f}% {ratio:>7.1f}x {signal:>7}")

    return {
        "real_k1_rate": real_k1_rate,
        "real_avg_rank": real_avg_rank,
        "perm_avg_rank": np.mean(perm_best_ranks) if perm_best_ranks else None,
    }


# ============================================================
# Exp2: Role-Level Discrimination
# ============================================================
def exp2_role_discrimination(model, tokenizer, device, info):
    """
    核心逻辑:
    如果attention只编码"wh-结构"pattern:
      → What/Who/Where/When的attention pattern相同
    如果attention编码语法角色:
      → dobj(What) vs nsubj(Who) vs adjunct(Where/When/How)的pattern不同

    测量: 每个wh-word到verb的attention profile
    - 如果不同wh-word有不同的attention distribution → 角色区分
    """
    print("\n" + "="*70)
    print("★★★ EXP2: ROLE-LEVEL DISCRIMINATION ★★★")
    print("测试: What/Who/Where/When/How → attention是否区分语法角色?")
    print("="*70)

    role_profiles = {}  # role → {layer: {head: attn_score_to_verb}}

    for item in WH_ROLE_DISCRIM:
        name = item["name"]
        sentence = item["sentence"]
        wh_pos = item["wh_pos"]
        verb_pos = item["verb_pos"]
        expected_role = item["expected_role"]
        wh_word = item["wh_word"]

        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue

        seq_len = len(tokens)
        print(f"\n  [{name}] {sentence}")
        print(f"  Tokens: {[(i,t) for i,t in enumerate(tokens)]}")
        print(f"  {wh_word}(pos{wh_pos}) → verb(pos{verb_pos}) expected_role={expected_role}")

        # 对每个层和head, 计算verb(verb_pos) attend to wh(wh_pos)的分数
        # 以及verb attend to nsubj/other的分数 (作为对比)
        role_profiles[expected_role] = {}

        for li in range(min(info.n_layers, 28)):
            if li not in attn_dict:
                continue
            A = attn_dict[li]
            n_heads = A.shape[0]

            # verb行: verb_pos attend to 各位置
            verb_row = A[:, verb_pos, :]  # [n_heads, seq_len]

            # wh到verb的attention分数
            wh_attn = verb_row[:, wh_pos]  # [n_heads]

            # 找verb attend to 各位置的top-1
            for h in range(n_heads):
                sorted_pos = np.argsort(verb_row[h])[::-1]
                top1_pos = int(sorted_pos[0])
                top1_score = float(verb_row[h, top1_pos])
                wh_score = float(verb_row[h, wh_pos])

                if li not in role_profiles[expected_role]:
                    role_profiles[expected_role][li] = []

                role_profiles[expected_role][li].append({
                    "head": h,
                    "wh_score": wh_score,
                    "top1_pos": top1_pos,
                    "top1_score": top1_score,
                    "wh_is_top1": top1_pos == wh_pos,
                })

        del attn_dict, inputs
        gc.collect()

    # ===== 分析: 不同role的attention pattern是否不同 =====
    print("\n\n" + "="*70)
    print("★★★ ROLE DISCRIMINATION ANALYSIS ★★★")
    print("="*70)

    # 对关键层, 统计: 多少head把wh作为verb的top-1 attention
    print(f"\n  {'Layer':>6} {'dobj(What)':>12} {'nsubj(Who/What)':>16} {'adjunct(Where/When/How)':>24}")
    print(f"  {'-'*65}")

    for li in [0, 1, 3, 6, 10, 15, 20]:
        dobj_rate = 0
        dobj_total = 0
        nsubj_rate = 0
        nsubj_total = 0
        adjunct_rate = 0
        adjunct_total = 0

        for role in ["dobj", "nsubj", "adjunct"]:
            if role not in role_profiles or li not in role_profiles[role]:
                continue
            heads = role_profiles[role][li]
            k1_count = sum(1 for h in heads if h["wh_is_top1"])
            total = len(heads)

            if role == "dobj":
                dobj_rate = k1_count / total * 100 if total > 0 else 0
                dobj_total = total
            elif role == "nsubj":
                nsubj_rate = k1_count / total * 100 if total > 0 else 0
                nsubj_total = total
            else:
                adjunct_rate = k1_count / total * 100 if total > 0 else 0
                adjunct_total = total

        print(f"  L{li:>4} {dobj_rate:>10.1f}%({dobj_total:>3}) {nsubj_rate:>13.1f}%({nsubj_total:>3}) {adjunct_rate:>21.1f}%({adjunct_total:>3})")

    # ===== 关键测试: What(dobj) vs What(nsubj) 的pattern差异 =====
    print("\n\n--- ★★★ Critical Test: Same Wh-Word, Different Role ★★★ ---")
    print("  If attention tracks role → What-as-dobj ≠ What-as-nsubj")
    print("  If attention tracks word → What-as-dobj ≈ What-as-nsubj")

    # 找到What(dobj)和What(nsubj)的per-head top-1 profile
    dobj_wh_top1 = defaultdict(int)  # (layer, head) → count of top-1
    nsubj_wh_top1 = defaultdict(int)

    for role in ["dobj", "nsubj"]:
        if role not in role_profiles:
            continue
        for li, heads in role_profiles[role].items():
            for h_info in heads:
                key = (li, h_info["head"])
                if h_info["wh_is_top1"]:
                    if role == "dobj":
                        dobj_wh_top1[key] += 1
                    else:
                        nsubj_wh_top1[key] += 1

    # 计算两个set的重叠度
    dobj_set = set(dobj_wh_top1.keys())
    nsubj_set = set(nsubj_wh_top1.keys())

    if dobj_set and nsubj_set:
        overlap = dobj_set & nsubj_set
        union = dobj_set | nsubj_set
        jaccard = len(overlap) / len(union) if union else 0

        print(f"\n  What-as-dobj top-1 heads: {len(dobj_set)}")
        print(f"  What-as-nsubj top-1 heads: {len(nsubj_set)}")
        print(f"  Overlap: {len(overlap)}")
        print(f"  Jaccard similarity: {jaccard:.3f}")

        if jaccard < 0.3:
            print(f"  ★★★ LOW overlap → attention tracks ROLE, not word ★★★")
        elif jaccard > 0.7:
            print(f"  ✗ HIGH overlap → attention tracks WORD, not role")
        else:
            print(f"  ~ MODERATE overlap → partial role sensitivity")

    # ===== Per-head分析: 找区分dobj和nsubj的head =====
    print("\n\n--- Role-Discriminating Heads (per-layer) ---")
    for li in [0, 1, 3, 6, 10, 15, 20]:
        dobj_scores = []
        nsubj_scores = []
        adjunct_scores = []

        for role in ["dobj", "nsubj", "adjunct"]:
            if role not in role_profiles or li not in role_profiles[role]:
                continue
            for h_info in role_profiles[role][li]:
                if role == "dobj":
                    dobj_scores.append(h_info["wh_score"])
                elif role == "nsubj":
                    nsubj_scores.append(h_info["wh_score"])
                else:
                    adjunct_scores.append(h_info["wh_score"])

        if dobj_scores and nsubj_scores:
            dobj_mean = np.mean(dobj_scores)
            nsubj_mean = np.mean(nsubj_scores)
            adjunct_mean = np.mean(adjunct_scores) if adjunct_scores else 0

            diff = abs(dobj_mean - nsubj_mean)
            print(f"  L{li}: dobj_mean={dobj_mean:.4f} nsubj_mean={nsubj_mean:.4f} "
                  f"adjunct_mean={adjunct_mean:.4f} |dobj-nsubj|={diff:.4f}")


# ============================================================
# Exp3: Targeted Ablation
# ============================================================
def exp3_targeted_ablation(model, tokenizer, device, info):
    """
    核心逻辑:
    1. 先找出"语法head" (在多个句子上都能正确恢复dependency的head)
    2. 消融这些head → 看语法错误率是否上升
    3. 对照: 消融随机head → 语法错误率应该不变

    语法错误检测方法:
    - 用模型的next-token预测, 看主谓一致
    - "The cat ___" → 应该预测"chased"而非"chase"
    """
    print("\n" + "="*70)
    print("★★★ EXP3: TARGETED ABLATION ★★★")
    print("测试: 消融语法head → 语法能力是否下降?")
    print("="*70)

    # Step 1: 找语法head
    print("\n--- Step 1: Finding syntax heads ---")

    # 用基础SVO句找"在多个句子上都rank=1"的head
    head_success_count = defaultdict(int)  # (layer, head) → 成功次数
    head_total_count = defaultdict(int)    # (layer, head) → 总测试次数

    for item in BASIC_SVO:
        sentence = item["sentence"]
        deps = item["deps"]

        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue

        seq_len = len(tokens)

        for head_pos, dep_pos, rel_type in deps:
            if head_pos >= seq_len or dep_pos >= seq_len:
                continue

            results = compute_rank_for_dep(attn_dict, head_pos, dep_pos, info.n_layers)
            for r in results:
                key = (r["layer"], r["head"])
                head_total_count[key] += 1
                if r["rank"] == 1:
                    head_success_count[key] += 1

        del attn_dict, inputs
        gc.collect()

    # 找"在≥4/6句子上rank=1"的head
    syntax_heads = []
    for key in head_total_count:
        if head_total_count[key] >= 4 and head_success_count[key] / head_total_count[key] >= 0.8:
            li, h = key
            success_rate = head_success_count[key] / head_total_count[key]
            syntax_heads.append((li, h, success_rate))

    syntax_heads.sort(key=lambda x: x[2], reverse=True)

    print(f"  Found {len(syntax_heads)} syntax heads (success≥80% across sentences):")
    for li, h, rate in syntax_heads[:15]:
        print(f"    L{li} H{h}: success_rate={rate:.2f} ({head_success_count[(li,h)]}/{head_total_count[(li,h)]})")

    if not syntax_heads:
        print("  ✗ No reliable syntax heads found! Using top-5 by success count instead.")
        # 用绝对成功次数排序
        all_heads = [(li, h, head_success_count[(li,h)] / head_total_count[(li,h)])
                     for (li,h) in head_total_count if head_total_count[(li,h)] >= 2]
        all_heads.sort(key=lambda x: x[2], reverse=True)
        syntax_heads = all_heads[:5]
        print(f"  Using top-5: {[(li,h,f'{r:.2f}') for li,h,r in syntax_heads]}")

    # Step 2: Ablation test - 使用logit lens方法
    print("\n\n--- Step 2: Ablation via logit difference ---")
    print("  Method: Compare model's logit for correct vs incorrect verb form")

    # 测试句: "The cat [chased/chase] the mouse" → 模型应该偏好chased
    test_pairs = [
        ("The cat chased the mouse", "chased", "chase"),     # 正确: chased
        ("The cats chased the mouse", "chased", "chase"),    # 正确: chased (复数主语)
        ("The cat has chased the mouse", "chased", "chasing"), # 正确: chased
        ("The dog bit the man", "bit", "bite"),              # 正确: bit
        ("The bird ate the fish", "ate", "eat"),             # 正确: ate
    ]

    def get_logit_diff(sentence, correct_token, wrong_token, model, tokenizer, device):
        """获取correct_token vs wrong_token的logit差"""
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"])
            logits = outputs.logits[0]  # [seq_len, vocab]

        # 找verb位置 (简化: 句子的第3个token通常是verb)
        tokens = [safe_decode(tokenizer, tid) for tid in inputs["input_ids"][0].cpu().numpy()]

        # 找correct_token和wrong_token的位置
        correct_id = tokenizer.encode(correct_token, add_special_tokens=False)[0]
        wrong_id = tokenizer.encode(wrong_token, add_special_tokens=False)[0]

        # 在verb位置(第3个token位置)的logit
        verb_pos = 3  # 简化假设
        if verb_pos >= len(logits):
            return None

        # 用前一个位置的logit预测verb位置
        pred_pos = verb_pos - 1
        correct_logit = float(logits[pred_pos, correct_id])
        wrong_logit = float(logits[pred_pos, wrong_id])

        return correct_logit - wrong_logit

    # Baseline: 正常模型的logit diff
    print("\n  Baseline (no ablation):")
    baseline_diffs = []
    for sentence, correct, wrong in test_pairs:
        diff = get_logit_diff(sentence, correct, wrong, model, tokenizer, device)
        if diff is not None:
            baseline_diffs.append(diff)
            print(f"    {sentence}: logit_diff({correct} vs {wrong}) = {diff:.2f}")

    avg_baseline = np.mean(baseline_diffs) if baseline_diffs else 0
    print(f"    Average baseline logit diff: {avg_baseline:.2f}")

    # Step 3: Ablate syntax heads
    print("\n  Ablating syntax heads:")
    # 这里用zero-out attention pattern的方法
    # 由于直接修改model的attention比较复杂, 我们用hook方法

    # 只ablate最top的几个head
    top_syntax = syntax_heads[:5]

    for li, h, rate in top_syntax:
        print(f"\n    Ablating L{li} H{h} (success_rate={rate:.2f})...")

        # 创建hook
        ablation_diffs = []

        def make_ablation_hook(target_layer, target_head):
            def hook_fn(module, input, output):
                # output 是 (attn_output, attn_weights, ...) 或类似
                # 我们需要zero-out特定head的attention
                if len(output) >= 2 and output[1] is not None:
                    attn_weights = output[1].clone()
                    attn_weights[:, target_head, :, :] = 1.0 / attn_weights.shape[-1]  # uniform
                    # 重新计算attn_output
                    # 这很复杂, 暂时跳过
                return output
            return hook_fn

        # 简化方法: 直接用attention output zero-out
        # 获取attention layer
        try:
            layer = model.model.layers[li]
            attn = layer.self_attn

            # 注册hook
            hooks = []

            def make_zero_hook(target_head_idx, n_heads):
                def hook_fn(module, input, output):
                    # output[0] 是 attn_output: [batch, seq, d_model]
                    attn_output = output[0].clone()
                    d_head = attn_output.shape[-1] // n_heads
                    # zero out target head
                    start = target_head_idx * d_head
                    end = (target_head_idx + 1) * d_head
                    attn_output[:, :, start:end] = 0
                    return (attn_output,) + output[1:]
                return hook_fn

            # 获取n_heads
            n_heads = info.n_layers  # 这不对, 需要正确的n_heads
            if hasattr(attn.config, 'num_attention_heads'):
                n_heads = attn.config.num_attention_heads
            elif hasattr(attn, 'num_heads'):
                n_heads = attn.num_heads
            else:
                n_heads = 28  # 默认

            hook = attn.register_forward_hook(make_zero_hook(h, n_heads))
            hooks.append(hook)

            # 运行测试
            ablation_diffs = []
            for sentence, correct, wrong in test_pairs:
                diff = get_logit_diff(sentence, correct, wrong, model, tokenizer, device)
                if diff is not None:
                    ablation_diffs.append(diff)

            # 移除hook
            for hk in hooks:
                hk.remove()

            if ablation_diffs:
                avg_ablation = np.mean(ablation_diffs)
                delta = avg_ablation - avg_baseline
                print(f"      Avg logit diff after ablation: {avg_ablation:.2f} (Δ={delta:+.2f})")
                if delta < -1.0:
                    print(f"      ★★★ SIGNIFICANT DEGRADATION → This head is CAUSALLY involved ★★★")
                elif delta < -0.5:
                    print(f"      ★ MODERATE degradation")
                else:
                    print(f"      ✗ No significant degradation")

        except Exception as e:
            print(f"      Hook failed: {e}")

    # Step 4: Control ablation (random heads)
    print("\n\n  Control: Ablating random heads:")
    random_heads = [(random.randint(0, info.n_layers-1), random.randint(0, 27)) for _ in range(5)]

    for li, h in random_heads:
        print(f"\n    Ablating L{li} H{h} (random)...")

        try:
            layer = model.model.layers[li]
            attn = layer.self_attn

            n_heads = 28
            if hasattr(attn.config, 'num_attention_heads'):
                n_heads = attn.config.num_attention_heads
            elif hasattr(attn, 'num_heads'):
                n_heads = attn.num_heads

            hook = attn.register_forward_hook(make_zero_hook(h, n_heads))

            ablation_diffs = []
            for sentence, correct, wrong in test_pairs:
                diff = get_logit_diff(sentence, correct, wrong, model, tokenizer, device)
                if diff is not None:
                    ablation_diffs.append(diff)

            hook.remove()

            if ablation_diffs:
                avg_ablation = np.mean(ablation_diffs)
                delta = avg_ablation - avg_baseline
                print(f"      Avg logit diff after ablation: {avg_ablation:.2f} (Δ={delta:+.2f})")
                if delta < -1.0:
                    print(f"      ✗ Random head also causes degradation → not specific")
                else:
                    print(f"      ✓ No degradation from random head")

        except Exception as e:
            print(f"      Hook failed: {e}")


# ============================================================
# Exp4: Template Control (修正硬伤3)
# ============================================================
def exp4_template_control(model, tokenizer, device, info):
    """
    核心逻辑:
    伪词句保留了function words和词序 → 可能是模板而非语法

    对照: 完全打乱function words
    - "The cat chased the mouse" → 正常
    - "The blarg flommed the snarp" → 伪词, 保留function words
    - "Blarg the snarp flommed the" → 完全打乱

    如果attention在完全打乱中仍"恢复语法" → 纯位置模式
    如果attention在完全打乱中失败 → 不是纯位置, 需要词序信息
    """
    print("\n" + "="*70)
    print("★★★ EXP4: TEMPLATE CONTROL ★★★")
    print("测试: 伪词成功是因为模板还是语法?")
    print("="*70)

    test_cases = [
        {"name": "Normal SVO", "sentence": "The cat chased the mouse",
         "deps": [(3,2,"nsubj"), (3,5,"dobj")]},
        {"name": "Nonsense (keep function)", "sentence": "The blarg flommed the snarp",
         "deps": [(3,2,"nsubj"), (3,5,"dobj")]},
        {"name": "Scrambled (all random)", "sentence": "Flommed the blarg snarp the",
         "deps": [(3,2,"nsubj"), (3,5,"dobj")]},  # 标注无效, 纯位置测试
        {"name": "No function words", "sentence": "Cat chased mouse",
         "deps": [(1,0,"nsubj"), (1,2,"dobj")]},
        {"name": "Wrong word order", "sentence": "Mouse chased cat the",
         "deps": [(1,0,"nsubj"), (1,2,"dobj")]},
    ]

    for item in test_cases:
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
                continue

            results = compute_rank_for_dep(attn_dict, head_pos, dep_pos, info.n_layers)
            if results:
                best_rank = min(r["rank"] for r in results)
                k1_count = sum(1 for r in results if r["rank"] == 1)
                total = len(results)
                print(f"    {tokens[dep_pos]}({dep_pos})→{tokens[head_pos]}({head_pos}) [{rel_type}] "
                      f"best_rank={best_rank} k1_rate={k1_count/total:.3f}")

        del attn_dict, inputs
        gc.collect()

    print("\n\n--- Interpretation ---")
    print("  If scrambled sentences still show high k1_rate → position template")
    print("  If scrambled sentences show low k1_rate → requires word-order information")
    print("  ★ Key distinction: 'form-based syntax' vs 'position template'")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b",
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=0,
                        help="0=all, 1=permutation, 2=role, 3=ablation, 4=template")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"Model: {args.model}, Layers: {info.n_layers}, d_model: {info.d_model}")

    if args.exp in [0, 1]:
        perm_results = exp1_permutation_test(model, tokenizer, device, info)

    if args.exp in [0, 2]:
        exp2_role_discrimination(model, tokenizer, device, info)

    if args.exp in [0, 3]:
        exp3_targeted_ablation(model, tokenizer, device, info)

    if args.exp in [0, 4]:
        exp4_template_control(model, tokenizer, device, info)

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("\nPhase 56 Complete!")


if __name__ == "__main__":
    main()
