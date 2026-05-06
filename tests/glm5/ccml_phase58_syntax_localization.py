"""
Phase 58: 语法信息定位 + 分布式指标 + 强Baseline
================================================

核心问题转变:
  旧: "attention是否编码语法?"
  新: "语法信息在哪里? 如何被使用?"

5个实验:
  Exp1: Attention Mass Analysis — 不用top-1, 用gold位置的attention权重占比
        + distance-matched baseline + POS-matched baseline
  Exp2: Top-K Recall — k=1,3,5时gold位置的recall
  Exp3: Probing Classifier — residual stream/attention_output/MLP_output上的语法角色分类
  Exp4: Attention Intervention — 替换attention为uniform/random, 测语法是否崩溃
  Exp5: Cross-component Comparison — 各组件的语法信息量对比

关键修正:
  - 不再用 top-1 hit rate (太严苛, 丢失分布信息)
  - 不再用 uniform null baseline (不反映真实attention分布)
  - 新增 distance-matched 和 POS-matched baseline
  - 新增 probing classifier 直接测试representation中的语法信息
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np, gc, argparse, random, time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model_utils import load_model, get_model_info, release_model, safe_decode


# ============================================================
# 工具函数
# ============================================================
def get_model_outputs(model, tokenizer, sentence, device, output_hidden_states=True):
    """获取完整的模型输出: attention + hidden_states + MLP输出"""
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model(
                input_ids=inputs["input_ids"],
                output_attentions=True,
                output_hidden_states=output_hidden_states,
            )
            all_attn = outputs.attentions
            all_hidden = outputs.hidden_states  # tuple of (1, seq_len, d_model)
        except Exception as e:
            print(f"  [Error] {e}")
            return None, None, None, None

    # 提取attention
    attn_dict = {}
    for li, attn in enumerate(all_attn):
        if attn is not None:
            attn_dict[li] = attn.detach().float().cpu().numpy()[0]  # (n_heads, seq, seq)

    # 提取hidden states (residual stream at each layer)
    hidden_dict = {}
    if all_hidden is not None:
        for li, h in enumerate(all_hidden):
            hidden_dict[li] = h.detach().float().cpu().numpy()[0]  # (seq_len, d_model)

    tokens = [safe_decode(tokenizer, tid) for tid in inputs["input_ids"][0].cpu().numpy()]
    return attn_dict, hidden_dict, inputs, tokens


def get_mlp_outputs(model, tokenizer, inputs, device, info):
    """获取MLP层的输出 (通过hook)"""
    layers = model.model.layers if hasattr(model, 'model') and hasattr(model.model, 'layers') else None
    if layers is None:
        return {}

    mlp_outputs = {}
    hooks = []

    def hook_fn(li):
        def fn(module, input, output):
            mlp_outputs[li] = output.detach().float().cpu().numpy()[0]  # (seq_len, d_model)
        return fn

    for li in range(min(info.n_layers, len(layers))):
        if hasattr(layers[li], 'mlp'):
            h = layers[li].mlp.register_forward_hook(hook_fn(li))
            hooks.append(h)

    # 重新forward
    with torch.no_grad():
        try:
            model(input_ids=inputs["input_ids"])
        except:
            pass

    for h in hooks:
        h.remove()

    return mlp_outputs


# ============================================================
# Exp1: Attention Mass Analysis (修正版指标)
# ============================================================
def exp1_attention_mass(model, tokenizer, device, info):
    """
    核心改进:
    1. 不用 top-1 hit rate, 用 gold token 的 attention mass (权重占比)
    2. 对比3种baseline:
       a. Uniform: 1/N (之前的null)
       b. Distance-matched: 同距离位置的mean attention mass
       c. POS-matched: 同词类位置的mean attention mass
    """
    print("\n" + "="*70)
    print("★★★ Exp1: Attention Mass Analysis (修正版指标 + 强Baseline) ★★★")
    print("="*70)

    test_cases = [
        # SVO: nsubj at pos2, verb at pos3, dobj at pos5
        {"name": "SVO nsubj→verb", "sentences": [
            "The cat chased the mouse",
            "The dog bit the man",
            "The girl saw the boy",
            "The bird ate the fish",
            "The wolf killed the deer",
            "The fox caught the rabbit",
            "The king ruled the land",
            "The bear chased the deer",
        ], "from_pos": 3, "gold_pos": 2,
         "same_dist_pos": [1],  # distance=1 from verb: pos2 and pos4(但4是the)
         "same_pos_tags": [5],  # other nouns: pos5(dobj)
         "context_start": 1},  # skip BOS

        # SVO: dobj at pos5, from verb pos3
        {"name": "SVO dobj→verb", "sentences": [
            "The cat chased the mouse",
            "The dog bit the man",
            "The girl saw the boy",
            "The bird ate the fish",
            "The wolf killed the deer",
            "The fox caught the rabbit",
            "The king ruled the land",
            "The bear chased the deer",
        ], "from_pos": 3, "gold_pos": 5,
         "same_dist_pos": [1],  # distance=2 from verb: pos1(nsubj) is dist=2
         "same_pos_tags": [2],  # other nouns: pos2(nsubj)
         "context_start": 1},

        # Center-embedding: nsubj_long at pos2, from verb pos7
        {"name": "CE nsubj_long→verb", "sentences": [
            "The cat that the dog chased ran",
            "The man that the girl saw left",
            "The bird that the cat watched flew",
            "The fish that the bear caught swam",
            "The boy that the king saw ran",
        ], "from_pos": 7, "gold_pos": 2,
         "same_dist_pos": [5],  # same distance: pos5 is dist=2 from pos7
         "same_pos_tags": [5],  # other noun at pos5
         "context_start": 1},

        # PP: nsubj at pos2, from verb pos7
        {"name": "PP nsubj→verb", "sentences": [
            "The cat with the hat chased the mouse",
            "The dog near the park chased the cat",
            "The bird on the tree saw the fish",
            "The girl with the dog saw the boy",
        ], "from_pos": 7, "gold_pos": 2,
         "same_dist_pos": [5],  # distance=5 from pos7: pos2 and pos5
         "same_pos_tags": [5],  # pos5 = "hat"/"park" = noun
         "context_start": 1},
    ]

    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        from_pos = case["from_pos"]
        gold_pos = case["gold_pos"]
        same_dist = case.get("same_dist_pos", [])
        same_pos_tags = case.get("same_pos_tags", [])
        ctx_start = case["context_start"]

        # 收集所有句子的attention mass
        gold_masses = []      # gold position的attention mass
        uniform_baselines = []  # 1/N baseline
        dist_baselines = []   # distance-matched baseline
        pos_baselines = []    # POS-matched baseline

        for sentence in case["sentences"]:
            attn_dict, _, inputs, tokens = get_model_outputs(model, tokenizer, sentence, device, output_hidden_states=False)
            if attn_dict is None:
                continue

            seq_len = len(tokens)
            # 确保位置有效
            if from_pos >= seq_len or gold_pos >= seq_len:
                continue

            for li in sorted(attn_dict.keys()):
                A = attn_dict[li]  # (n_heads, seq, seq)
                n_heads = A.shape[0]

                # 看from_pos行的attention分布 (只看左边context)
                for h in range(n_heads):
                    attn_row = A[h, from_pos, ctx_start:from_pos]  # 只看左边token
                    if len(attn_row) == 0:
                        continue
                    attn_row = attn_row / attn_row.sum()  # normalize

                    # Gold position的attention mass
                    gold_idx = gold_pos - ctx_start
                    if gold_idx < 0 or gold_idx >= len(attn_row):
                        continue

                    gold_mass = attn_row[gold_idx]
                    gold_masses.append(gold_mass)

                    # Uniform baseline
                    uniform_baselines.append(1.0 / len(attn_row))

                    # Distance-matched baseline: 同距离位置的mean attention
                    dist_from_gold = abs(gold_pos - from_pos)
                    dist_masses = []
                    for p in range(ctx_start, from_pos):
                        if abs(p - from_pos) == dist_from_gold and p != gold_pos:
                            idx = p - ctx_start
                            if 0 <= idx < len(attn_row):
                                dist_masses.append(attn_row[idx])
                    if dist_masses:
                        dist_baselines.append(np.mean(dist_masses))
                    else:
                        # 没有同距离位置, 用最近距离
                        for p in range(ctx_start, from_pos):
                            if p != gold_pos:
                                idx = p - ctx_start
                                if 0 <= idx < len(attn_row):
                                    dist_masses.append(attn_row[idx])
                        dist_baselines.append(np.mean(dist_masses) if dist_masses else 1.0/len(attn_row))

                    # POS-matched baseline: 同词类位置的mean attention
                    pos_masses = []
                    for p in same_pos_tags:
                        if p != gold_pos and ctx_start <= p < from_pos:
                            idx = p - ctx_start
                            if 0 <= idx < len(attn_row):
                                pos_masses.append(attn_row[idx])
                    if pos_masses:
                        pos_baselines.append(np.mean(pos_masses))
                    else:
                        pos_baselines.append(1.0/len(attn_row))

            del attn_dict, inputs
            gc.collect()

        if not gold_masses:
            print("  No data collected!")
            continue

        gold_arr = np.array(gold_masses)
        uniform_arr = np.array(uniform_baselines)
        dist_arr = np.array(dist_baselines)
        pos_arr = np.array(pos_baselines)

        # 统计检验: gold vs each baseline
        from scipy.stats import wilcoxon, mannwhitneyu

        def safe_wilcoxon(x, y):
            diff = x - y
            if np.all(diff == 0):
                return 1.0
            try:
                _, p = wilcoxon(x, y, alternative='greater')
                return p
            except:
                try:
                    _, p = mannwhitneyu(x, y, alternative='greater')
                    return p
                except:
                    return 1.0

        p_uniform = safe_wilcoxon(gold_arr, uniform_arr)
        p_dist = safe_wilcoxon(gold_arr, dist_arr)
        p_pos = safe_wilcoxon(gold_arr, pos_arr)

        print(f"  Attention mass on gold position: {gold_arr.mean():.4f} ± {gold_arr.std():.4f}")
        print(f"  Baselines:")
        print(f"    Uniform (1/N):              {uniform_arr.mean():.4f} ± {uniform_arr.std():.4f}  p={p_uniform:.6f} {'★★★' if p_uniform<0.001 else ('★' if p_uniform<0.01 else ('~' if p_uniform<0.05 else ''))}")
        print(f"    Distance-matched:           {dist_arr.mean():.4f} ± {dist_arr.std():.4f}  p={p_dist:.6f} {'★★★' if p_dist<0.001 else ('★' if p_dist<0.01 else ('~' if p_dist<0.05 else ''))}")
        print(f"    POS-matched:                {pos_arr.mean():.4f} ± {pos_arr.std():.4f}  p={p_pos:.6f} {'★★★' if p_pos<0.001 else ('★' if p_pos<0.01 else ('~' if p_pos<0.05 else ''))}")
        print(f"  Ratios (gold/baseline):")
        print(f"    vs Uniform:   {gold_arr.mean()/uniform_arr.mean():.2f}x")
        print(f"    vs Distance:  {gold_arr.mean()/dist_arr.mean():.2f}x")
        print(f"    vs POS:       {gold_arr.mean()/pos_arr.mean():.2f}x")

        # Per-layer breakdown (选取关键层)
        print(f"\n  Per-layer breakdown (mean gold mass):")
        layer_gold = defaultdict(list)
        layer_dist = defaultdict(list)

        for sentence in case["sentences"]:
            attn_dict, _, inputs, tokens = get_model_outputs(model, tokenizer, sentence, device, output_hidden_states=False)
            if attn_dict is None:
                continue
            seq_len = len(tokens)
            if from_pos >= seq_len or gold_pos >= seq_len:
                continue

            for li in sorted(attn_dict.keys()):
                A = attn_dict[li]
                n_heads = A.shape[0]
                for h in range(n_heads):
                    attn_row = A[h, from_pos, ctx_start:from_pos]
                    if len(attn_row) == 0:
                        continue
                    attn_row = attn_row / attn_row.sum()
                    gold_idx = gold_pos - ctx_start
                    if gold_idx < 0 or gold_idx >= len(attn_row):
                        continue
                    layer_gold[li].append(attn_row[gold_idx])

                    # distance-matched for this layer
                    dist_masses = []
                    for p in range(ctx_start, from_pos):
                        if abs(p - from_pos) == abs(gold_pos - from_pos) and p != gold_pos:
                            idx = p - ctx_start
                            if 0 <= idx < len(attn_row):
                                dist_masses.append(attn_row[idx])
                    if dist_masses:
                        layer_dist[li].append(np.mean(dist_masses))

            del attn_dict, inputs
            gc.collect()

        print(f"  {'Layer':>6} {'GoldMass':>9} {'DistBase':>9} {'Ratio':>6} {'p-value':>9} {'Sig':>5}")
        print(f"  {'-'*55}")
        for li in [0, 1, 2, 5, 10, 15, 20, 25]:
            if li not in layer_gold or not layer_gold[li]:
                continue
            gm = np.mean(layer_gold[li])
            dm = np.mean(layer_dist[li]) if li in layer_dist and layer_dist[li] else 1.0/max(1, from_pos-ctx_start)
            ratio = gm / dm if dm > 0 else float('inf')
            p = safe_wilcoxon(np.array(layer_gold[li]), np.array(layer_dist[li]) if li in layer_dist and layer_dist[li] else np.full(len(layer_gold[li]), dm))
            sig = "★★★" if p < 0.001 else ("★" if p < 0.01 else ("~" if p < 0.05 else ""))
            print(f"  L{li:>4} {gm:>8.4f} {dm:>8.4f} {ratio:>5.2f}x {p:>8.5f} {sig:>5}")

    return {}


# ============================================================
# Exp2: Top-K Recall (比top-1更宽容的指标)
# ============================================================
def exp2_topk_recall(model, tokenizer, device, info):
    """
    不用 top-1 hit rate, 用 top-k recall:
    - gold position在top-k中的比例
    - k=1,3,5
    
    同时对比distance-matched baseline
    """
    print("\n" + "="*70)
    print("★★★ Exp2: Top-K Recall (k=1,3,5) ★★★")
    print("="*70)

    test_cases = [
        {"name": "SVO nsubj", "sentences": [
            "The cat chased the mouse",
            "The dog bit the man",
            "The girl saw the boy",
            "The bird ate the fish",
            "The wolf killed the deer",
            "The fox caught the rabbit",
        ], "from_pos": 3, "gold_pos": 2, "context_start": 1},

        {"name": "SVO dobj", "sentences": [
            "The cat chased the mouse",
            "The dog bit the man",
            "The girl saw the boy",
            "The bird ate the fish",
        ], "from_pos": 3, "gold_pos": 5, "context_start": 1},

        {"name": "CE nsubj_long", "sentences": [
            "The cat that the dog chased ran",
            "The man that the girl saw left",
            "The bird that the cat watched flew",
        ], "from_pos": 7, "gold_pos": 2, "context_start": 1},

        {"name": "PP nsubj", "sentences": [
            "The cat with the hat chased the mouse",
            "The dog near the park chased the cat",
            "The bird on the tree saw the fish",
        ], "from_pos": 7, "gold_pos": 2, "context_start": 1},
    ]

    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        from_pos = case["from_pos"]
        gold_pos = case["gold_pos"]
        ctx_start = case["context_start"]

        # Collect per-layer top-k stats
        layer_stats = defaultdict(lambda: {"gold_in_k": defaultdict(int), "total": 0,
                                            "dist_in_k": defaultdict(int)})

        for sentence in case["sentences"]:
            attn_dict, _, inputs, tokens = get_model_outputs(model, tokenizer, sentence, device, output_hidden_states=False)
            if attn_dict is None:
                continue
            seq_len = len(tokens)
            if from_pos >= seq_len or gold_pos >= seq_len:
                continue

            # Find distance-matched position
            dist_from_gold = abs(gold_pos - from_pos)
            dist_pos = None
            for p in range(ctx_start, from_pos):
                if abs(p - from_pos) == dist_from_gold and p != gold_pos:
                    dist_pos = p
                    break

            for li in sorted(attn_dict.keys()):
                A = attn_dict[li]
                n_heads = A.shape[0]
                for h in range(n_heads):
                    attn_row = A[h, from_pos, ctx_start:from_pos]
                    if len(attn_row) == 0:
                        continue
                    attn_row = attn_row / attn_row.sum()

                    gold_idx = gold_pos - ctx_start
                    if gold_idx < 0 or gold_idx >= len(attn_row):
                        continue

                    # Get ranking of gold position
                    ranking = np.argsort(-attn_row)
                    gold_rank = int(np.where(ranking == gold_idx)[0][0]) + 1

                    layer_stats[li]["total"] += 1
                    for k in [1, 3, 5]:
                        if gold_rank <= k:
                            layer_stats[li]["gold_in_k"][k] += 1

                    # Distance-matched baseline
                    if dist_pos is not None:
                        dist_idx = dist_pos - ctx_start
                        if 0 <= dist_idx < len(attn_row):
                            dist_rank = int(np.where(ranking == dist_idx)[0][0]) + 1
                            for k in [1, 3, 5]:
                                if dist_rank <= k:
                                    layer_stats[li]["dist_in_k"][k] += 1

            del attn_dict, inputs
            gc.collect()

        # Report
        print(f"  {'Layer':>6} {'Top1_G':>7} {'Top1_D':>7} {'Top3_G':>7} {'Top3_D':>7} {'Top5_G':>7} {'Top5_D':>7}")
        print(f"  {'-'*55}")

        for li in [0, 2, 5, 10, 15, 20, 25]:
            if li not in layer_stats:
                continue
            s = layer_stats[li]
            n = max(s["total"], 1)
            row = f"  L{li:>4}"
            for k in [1, 3, 5]:
                gr = s["gold_in_k"][k] / n
                dr = s["dist_in_k"][k] / n if s["dist_in_k"][k] > 0 else float('nan')
                row += f" {gr:>6.3f} {dr:>6.3f}"
            print(row)

        # Overall
        total_gold_k = defaultdict(int)
        total_dist_k = defaultdict(int)
        total_n = 0
        for li in layer_stats:
            s = layer_stats[li]
            total_n += s["total"]
            for k in [1, 3, 5]:
                total_gold_k[k] += s["gold_in_k"][k]
                total_dist_k[k] += s["dist_in_k"][k]

        print(f"\n  Overall (all layers):")
        for k in [1, 3, 5]:
            gr = total_gold_k[k] / max(total_n, 1)
            dr = total_dist_k[k] / max(total_n, 1)
            print(f"    Top-{k}: Gold={gr:.3f}, DistMatched={dr:.3f}, Ratio={gr/dr:.2f}x" if dr > 0 else f"    Top-{k}: Gold={gr:.3f}")

    return {}


# ============================================================
# Exp3: Probing Classifier (语法信息定位)
# ============================================================
def exp3_probing_classifier(model, tokenizer, device, info):
    """
    核心实验: 训练线性探针, 判断语法角色信息在哪个组件中最强
    
    输入: 
      - Residual stream h_i (at each layer)
      - Attention output (at each layer)
      - MLP output (at each layer)
    
    输出: 
      - 语法角色 (nsubj / dobj / other)
    
    对比:
      - 各层的分类准确率
      - 哪个组件信息最强
    """
    print("\n" + "="*70)
    print("★★★ Exp3: Probing Classifier — 语法信息在哪里? ★★★")
    print("="*70)

    # 构造训练和测试数据
    print("\n--- 构造训练数据 ---")

    # 角色标注数据: 每个token标记为 nsubj/dobj/other
    # 格式: (sentence, {pos: role})
    train_data = [
        ("The cat chased the mouse", {2: "nsubj", 5: "dobj"}),
        ("The dog bit the man", {2: "nsubj", 5: "dobj"}),
        ("The girl saw the boy", {2: "nsubj", 5: "dobj"}),
        ("The bird ate the fish", {2: "nsubj", 5: "dobj"}),
        ("The wolf killed the deer", {2: "nsubj", 5: "dobj"}),
        ("The fox caught the rabbit", {2: "nsubj", 5: "dobj"}),
        ("The king ruled the land", {2: "nsubj", 5: "dobj"}),
        ("The bear chased the deer", {2: "nsubj", 5: "dobj"}),
        ("The teacher helped the student", {2: "nsubj", 5: "dobj"}),
        ("The doctor treated the patient", {2: "nsubj", 5: "dobj"}),
    ]

    test_data = [
        ("The cat watched the dog", {2: "nsubj", 5: "dobj"}),
        ("The man found the key", {2: "nsubj", 5: "dobj"}),
        ("The horse pulled the cart", {2: "nsubj", 5: "dobj"}),
        ("The child broke the toy", {2: "nsubj", 5: "dobj"}),
        ("The woman read the book", {2: "nsubj", 5: "dobj"}),
    ]

    # Center-embedding (nsubj远距离)
    ce_train = [
        ("The cat that the dog chased ran", {2: "nsubj", 5: "dobj"}),
        ("The man that the girl saw left", {2: "nsubj", 5: "dobj"}),
        ("The bird that the cat watched flew", {2: "nsubj", 5: "dobj"}),
    ]
    ce_test = [
        ("The fish that the bear caught swam", {2: "nsubj", 5: "dobj"}),
        ("The boy that the king saw ran", {2: "nsubj", 5: "dobj"}),
    ]

    # 合并
    all_train = train_data + ce_train
    all_test = test_data + ce_test

    role_to_id = {"nsubj": 0, "dobj": 1, "other": 2}

    # 提取features
    print("  Extracting features from residual stream, attention output, MLP output...")

    def extract_features(data, model, tokenizer, device, info):
        """提取三种组件的representation"""
        resid_features = defaultdict(list)  # li → [(vec, role)]
        attn_features = defaultdict(list)
        mlp_features = defaultdict(list)

        for sentence, roles in data:
            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            seq_len = inputs["input_ids"].shape[1]
            tokens = [safe_decode(tokenizer, tid) for tid in inputs["input_ids"][0].cpu().numpy()]

            # Get hidden states
            with torch.no_grad():
                try:
                    outputs = model(
                        input_ids=inputs["input_ids"],
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                    all_hidden = outputs.hidden_states
                    all_attn = outputs.attentions
                except Exception as e:
                    print(f"  [Error] {e}")
                    continue

            # Residual stream: hidden_states[li] = (1, seq, d_model)
            for li in range(len(all_hidden)):
                h = all_hidden[li].detach().float().cpu().numpy()[0]  # (seq, d)
                for pos in range(seq_len):
                    role = roles.get(pos, "other")
                    resid_features[li].append((h[pos], role_to_id[role]))

            # Attention output: for each layer, compute output = sum(head_i * V_i) * W_o
            # Approximation: use the attention-weighted values
            for li, attn in enumerate(all_attn):
                if attn is None:
                    continue
                A = attn.detach().float().cpu().numpy()[0]  # (n_heads, seq, seq)
                # Use mean across heads as attention output representation
                mean_attn = A.mean(axis=0)  # (seq, seq)
                for pos in range(seq_len):
                    role = roles.get(pos, "other")
                    # Attention row at this position (who it attends to)
                    attn_row = mean_attn[pos, :pos+1]  # causal: only look left
                    if len(attn_row) > 0:
                        # Pad to fixed size
                        padded = np.zeros(64)
                        attn_row_norm = attn_row / max(attn_row.sum(), 1e-8)
                        padded[:min(len(attn_row_norm), 64)] = attn_row_norm[:64]
                        attn_features[li].append((padded, role_to_id[role]))

            # MLP output: need hooks
            mlp_out = get_mlp_outputs(model, tokenizer, inputs, device, info)
            for li in mlp_out:
                mlp_h = mlp_out[li]  # (seq, d)
                for pos in range(min(seq_len, mlp_h.shape[0])):
                    role = roles.get(pos, "other")
                    mlp_features[li].append((mlp_h[pos], role_to_id[role]))

            del outputs, all_hidden, all_attn
            if 'mlp_out' in dir():
                del mlp_out
            gc.collect()
            torch.cuda.empty_cache()

        return resid_features, attn_features, mlp_features

    # Extract train features
    train_resid, train_attn, train_mlp = extract_features(all_train, model, tokenizer, device, info)
    # Extract test features
    test_resid, test_attn, test_mlp = extract_features(all_test, model, tokenizer, device, info)

    # Train and evaluate linear probes
    print("\n--- Training linear probes ---")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def train_probe(train_features, test_features, name):
        """训练线性探针并返回结果"""
        results = {}
        for li in sorted(set(list(train_features.keys()) + list(test_features.keys()))):
            if li not in train_features or li not in test_features:
                continue
            if len(train_features[li]) < 10:
                continue

            X_train = np.array([f[0] for f in train_features[li]])
            y_train = np.array([f[1] for f in train_features[li]])
            X_test = np.array([f[0] for f in test_features[li]])
            y_test = np.array([f[1] for f in test_features[li]])

            # Subsample if too large
            if X_train.shape[0] > 500:
                idx = np.random.choice(len(X_train), 500, replace=False)
                X_train, y_train = X_train[idx], y_train[idx]

            # Check class balance
            classes = np.unique(y_train)
            if len(classes) < 2:
                continue

            # Standardize
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Train
            clf = LogisticRegression(max_iter=1000, C=1.0)
            try:
                clf.fit(X_train_s, y_train)
                acc = clf.score(X_test_s, y_test)
                # Per-class accuracy
                per_class = {}
                for c in classes:
                    mask = y_test == c
                    if mask.sum() > 0:
                        per_class[c] = clf.score(X_test_s[mask], y_test[mask])
                    else:
                        per_class[c] = float('nan')
                results[li] = {"acc": acc, "per_class": per_class}
            except:
                continue

        return results

    # Train probes on each component
    resid_results = train_probe(train_resid, test_resid, "residual")
    attn_results = train_probe(train_attn, test_attn, "attention")
    mlp_results = train_probe(train_mlp, test_mlp, "mlp")

    # Report
    print(f"\n  {'Layer':>6} {'Resid':>7} {'Attn':>7} {'MLP':>7}")
    print(f"  {'-'*35}")

    role_names = {0: "nsubj", 1: "dobj", 2: "other"}

    for li in sorted(set(list(resid_results.keys()) + list(attn_results.keys()) + list(mlp_results.keys()))):
        r = resid_results.get(li, {}).get("acc", float('nan'))
        a = attn_results.get(li, {}).get("acc", float('nan'))
        m = mlp_results.get(li, {}).get("acc", float('nan'))
        print(f"  L{li:>4} {r:>6.3f} {a:>6.3f} {m:>6.3f}")

    # Best layers for each component
    print(f"\n  Best layer per component:")
    for comp_name, comp_results in [("Residual", resid_results), ("Attention", attn_results), ("MLP", mlp_results)]:
        if comp_results:
            best_li = max(comp_results, key=lambda x: comp_results[x]["acc"])
            best_acc = comp_results[best_li]["acc"]
            per_class = comp_results[best_li]["per_class"]
            print(f"    {comp_name}: L{best_li} (acc={best_acc:.3f}, "
                  f"nsubj={per_class.get(0, float('nan')):.3f}, "
                  f"dobj={per_class.get(1, float('nan')):.3f}, "
                  f"other={per_class.get(2, float('nan')):.3f})")

    # Majority baseline
    n_other = sum(1 for _, roles in all_test for pos in range(20) if roles.get(pos, "other") == "other")
    n_total_test = sum(1 for _, roles in all_test for pos in range(20))
    print(f"\n  Majority baseline (always 'other'): {n_other/n_total_test:.3f}")

    return {"resid": resid_results, "attn": attn_results, "mlp": mlp_results}


# ============================================================
# Exp4: Attention Intervention (因果验证)
# ============================================================
def exp4_attention_intervention(model, tokenizer, device, info):
    """
    替换attention pattern, 测语法是否崩溃
    
    三种intervention:
    1. Uniform attention: 所有位置等权重
    2. Reverse attention: 翻转attention权重 (低→高)
    3. Shuffle attention: 在同层同head内shuffle
    
    测量: PPL变化 + 语法正确率
    """
    print("\n" + "="*70)
    print("★★★ Exp4: Attention Intervention — 语法是否依赖attention? ★★★")
    print("="*70)

    # 语法测试句对 (正确 vs 错误)
    grammar_pairs = [
        ("The cat chases the mouse", "The cat chase the mouse"),     # SV agreement
        ("The dogs chase the cat", "The dogs chases the cat"),       # SV agreement
        ("The cat that the dog chased ran", "The cat that the dog chased run"),  # long-distance agreement
        ("The girls that the boy saw left", "The girls that the boy saw leaves"),
        ("The cat with the hat chased the mouse", "The cat with the hat chase the mouse"),
    ]

    layers = model.model.layers if hasattr(model, 'model') and hasattr(model.model, 'layers') else None
    if layers is None:
        print("  Cannot access model layers!")
        return {}

    # 定义intervention函数
    def make_uniform_attn_hook():
        def hook_fn(module, input, output):
            attn_output, attn_weights = output
            # Replace with uniform
            B, H, S1, S2 = attn_weights.shape
            uniform = torch.ones_like(attn_weights) / attn_weights.shape[-1]
            # Apply causal mask
            mask = torch.tril(torch.ones(S1, S2, device=attn_weights.device)).unsqueeze(0).unsqueeze(0)
            uniform = uniform * mask
            uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            return attn_output, uniform
        return hook_fn

    def compute_ppl(model, tokenizer, sentence, device):
        """计算句子的PPL"""
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            try:
                outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
                return torch.exp(outputs.loss).item()
            except:
                return float('inf')

    # Baseline PPL
    print("\n--- Baseline PPL ---")
    baseline_results = {}
    for correct, wrong in grammar_pairs:
        ppl_c = compute_ppl(model, tokenizer, correct, device)
        ppl_w = compute_ppl(model, tokenizer, wrong, device)
        diff = ppl_w - ppl_c
        ratio = ppl_w / ppl_c if ppl_c > 0 else float('inf')
        baseline_results[correct] = (ppl_c, ppl_w, diff, ratio)
        print(f"  Correct: {ppl_c:.1f} | Wrong: {ppl_w:.1f} | Diff: {diff:.1f} | Ratio: {ratio:.2f}x")
        print(f"    '{correct}' vs '{wrong}'")

    # Uniform attention intervention (per layer group)
    print("\n--- Uniform Attention Intervention (per layer group) ---")

    n_layers = info.n_layers
    layer_groups = [
        ("Early (0-6)", list(range(0, min(7, n_layers)))),
        ("Mid (7-17)", list(range(7, min(18, n_layers)))),
        ("Late (18-27)", list(range(18, min(28, n_layers)))),
        ("All", list(range(n_layers))),
    ]

    for group_name, layer_indices in layer_groups:
        print(f"\n  [{group_name}] — layers {layer_indices[0]}-{layer_indices[-1]}")

        # Register hooks
        hooks = []
        for li in layer_indices:
            if li < len(layers) and hasattr(layers[li], 'self_attn'):
                h = layers[li].self_attn.register_forward_hook(make_uniform_attn_hook())
                hooks.append(h)

        # Measure PPL
        for correct, wrong in grammar_pairs[:3]:  # Only test 3 pairs for speed
            ppl_c = compute_ppl(model, tokenizer, correct, device)
            ppl_w = compute_ppl(model, tokenizer, wrong, device)
            diff = ppl_w - ppl_c
            ratio = ppl_w / ppl_c if ppl_c > 0 else float('inf')
            base_diff = baseline_results[correct][2]
            base_ratio = baseline_results[correct][3]
            print(f"    Correct: {ppl_c:.1f} | Wrong: {ppl_w:.1f} | Diff: {diff:.1f} (base: {base_diff:.1f}) | Ratio: {ratio:.2f}x (base: {base_ratio:.2f}x)")

        # Remove hooks
        for h in hooks:
            h.remove()

        gc.collect()
        torch.cuda.empty_cache()

    return {}


# ============================================================
# Exp5: Cross-Component Comparison
# ============================================================
def exp5_cross_component(model, tokenizer, device, info):
    """
    直接对比三种组件对语法判断的贡献
    
    方法: 测量每层各组件输出中, nsubj和dobj的表征可分性
    用cosine distance between nsubj and dobj representations
    """
    print("\n" + "="*70)
    print("★★★ Exp5: Cross-Component 语法可分性对比 ★★★")
    print("="*70)

    sentences = [
        "The cat chased the mouse",
        "The dog bit the man",
        "The girl saw the boy",
        "The bird ate the fish",
        "The wolf killed the deer",
    ]

    nsubj_pos = 2
    dobj_pos = 5

    # Collect representations
    resid_nsubj = defaultdict(list)  # li → [vectors]
    resid_dobj = defaultdict(list)
    attn_nsubj = defaultdict(list)
    attn_dobj = defaultdict(list)

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            try:
                outputs = model(
                    input_ids=inputs["input_ids"],
                    output_attentions=True,
                    output_hidden_states=True,
                )
                all_hidden = outputs.hidden_states
                all_attn = outputs.attentions
            except:
                continue

        # Residual stream
        for li in range(len(all_hidden)):
            h = all_hidden[li].detach().float().cpu().numpy()[0]
            if nsubj_pos < h.shape[0]:
                resid_nsubj[li].append(h[nsubj_pos])
            if dobj_pos < h.shape[0]:
                resid_dobj[li].append(h[dobj_pos])

        # Attention patterns
        for li, attn in enumerate(all_attn):
            if attn is None:
                continue
            A = attn.detach().float().cpu().numpy()[0]  # (n_heads, seq, seq)
            # For nsubj and dobj positions, get their attention distributions
            if nsubj_pos < A.shape[2]:
                # Mean attention FROM nsubj position
                attn_nsubj[li].append(A[:, nsubj_pos, :nsubj_pos+1].mean(axis=0))
            if dobj_pos < A.shape[2]:
                attn_dobj[li].append(A[:, dobj_pos, :dobj_pos+1].mean(axis=0))

        del outputs, all_hidden, all_attn
        gc.collect()

    # Compute separability: cosine distance between nsubj and dobj
    print(f"\n  Residual Stream Separability (nsubj vs dobj):")
    print(f"  {'Layer':>6} {'CosDist':>8} {'RandomDist':>11} {'Ratio':>6}")
    print(f"  {'-'*40}")

    for li in [0, 1, 2, 5, 10, 15, 20, 25]:
        if li not in resid_nsubj or li not in resid_dobj:
            continue
        if len(resid_nsubj[li]) < 2 or len(resid_dobj[li]) < 2:
            continue

        # Mean vectors
        mean_nsubj = np.mean(resid_nsubj[li], axis=0)
        mean_dobj = np.mean(resid_dobj[li], axis=0)

        # Cosine distance
        dot = np.dot(mean_nsubj, mean_dobj)
        norm_n = np.linalg.norm(mean_nsubj)
        norm_d = np.linalg.norm(mean_dobj)
        cos_sim = dot / max(norm_n * norm_d, 1e-8)
        cos_dist = 1 - cos_sim

        # Random baseline: distance between random pairs of same-class vectors
        random_dists = []
        nsubj_arr = np.array(resid_nsubj[li])
        for i in range(min(10, len(nsubj_arr))):
            for j in range(i+1, min(10, len(nsubj_arr))):
                d = 1 - np.dot(nsubj_arr[i], nsubj_arr[j]) / max(np.linalg.norm(nsubj_arr[i]) * np.linalg.norm(nsubj_arr[j]), 1e-8)
                random_dists.append(d)
        random_dist = np.mean(random_dists) if random_dists else 0.0

        ratio = cos_dist / max(random_dist, 1e-8)
        print(f"  L{li:>4} {cos_dist:>7.4f} {random_dist:>10.4f} {ratio:>5.2f}x")

    # Attention separability
    print(f"\n  Attention Pattern Separability (nsubj vs dobj distributions):")
    print(f"  {'Layer':>6} {'KLDiv':>8} {'Explanation':>30}")
    print(f"  {'-'*50}")

    for li in [0, 1, 2, 5, 10, 15, 20, 25]:
        if li not in attn_nsubj or li not in attn_dobj:
            continue
        if len(attn_nsubj[li]) < 2 or len(attn_dobj[li]) < 2:
            continue

        mean_nsubj = np.mean(attn_nsubj[li], axis=0)
        mean_dobj = np.mean(attn_dobj[li], axis=0)

        # Pad to same length for comparison
        max_len = max(len(mean_nsubj), len(mean_dobj))
        ns = np.zeros(max_len)
        db = np.zeros(max_len)
        ns[:len(mean_nsubj)] = mean_nsubj
        db[:len(mean_dobj)] = mean_dobj

        # Normalize as distributions
        ns = ns / max(ns.sum(), 1e-8)
        db = db / max(db.sum(), 1e-8)

        # KL divergence (use symmetric version for robustness)
        eps = 1e-8
        kl_div = 0.5 * (np.sum(ns * np.log((ns + eps) / (db + eps))) +
                        np.sum(db * np.log((db + eps) / (ns + eps))))

        explanation = ""
        if kl_div > 1.0:
            explanation = "Strong differentiation ★★★"
        elif kl_div > 0.3:
            explanation = "Moderate differentiation ★"
        else:
            explanation = "Weak differentiation"

        print(f"  L{li:>4} {kl_div:>7.4f} {explanation:>30}")

    return {}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all",
                       choices=["1", "2", "3", "4", "5", "all"])
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"Model: {info.name}, Layers: {info.n_layers}, d_model: {info.d_model}, class: {info.model_class}")

    if args.exp in ["1", "all"]:
        exp1_attention_mass(model, tokenizer, device, info)
    if args.exp in ["2", "all"]:
        exp2_topk_recall(model, tokenizer, device, info)
    if args.exp in ["3", "all"]:
        exp3_probing_classifier(model, tokenizer, device, info)
    if args.exp in ["4", "all"]:
        exp4_attention_intervention(model, tokenizer, device, info)
    if args.exp in ["5", "all"]:
        exp5_cross_component(model, tokenizer, device, info)

    release_model(model)
    print("\n[Phase 58 Complete]")


if __name__ == "__main__":
    main()
