"""
Phase 58B: 位置vs语法分离实验 — 核心突破
==========================================

Phase 58的关键问题: 所有probing都被位置信息污染
  - SVO中nsubj=pos2, dobj=pos5 → probe可能只学位置

解决方案: 用多种句式训练, 强制probe泛化
  - SVO: "The cat chased the mouse" → nsubj=pos2
  - Adv-SVO: "Quickly the cat chased the mouse" → nsubj=pos3
  - PP-SVO: "In the garden the cat chased the mouse" → nsubj=pos5
  - CE: "The cat that the dog chased ran" → nsubj(pos2) + dobj(pos5)
  - Question: "What did the cat chase" → nsubj=pos4

如果probe能在不同位置识别出nsubj → 语法信息确实存在
如果probe只能在固定位置识别 → 只是位置信息

实验:
  ExpA: Position-controlled probing (核心)
  ExpB: Surgical attention intervention (逐head替换)
  ExpC: Residual stream direction analysis (语法子空间)
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# ExpA: Position-Controlled Probing (核心实验)
# ============================================================
def expA_position_controlled_probing(model, tokenizer, device, info):
    """
    关键设计:
    1. 训练集: 多种句式, nsubj在不同位置
    2. 测试集: 新句式, nsubj在未见过的位置
    3. 如果能泛化 → 语法信息存在
    
    两种训练策略:
    a. Same-position训练: 只用SVO训练, SVO测试 (位置混淆)
    b. Cross-position训练: 用多种句式训练, 新句式测试 (位置分离)
    """
    print("\n" + "="*70)
    print("★★★ ExpA: Position-Controlled Probing ★★★")
    print("="*70)

    # 数据设计: (sentence, {pos: role})
    # nsubj出现在不同位置
    role_to_id = {"nsubj": 0, "dobj": 1, "verb": 2, "other": 3}

    # SVO: nsubj=pos2, dobj=pos5, verb=pos3
    svo_data = [
        ("The cat chased the mouse", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ("The dog bit the man", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ("The girl saw the boy", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ("The bird ate the fish", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ("The wolf killed the deer", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ("The fox caught the rabbit", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ("The king ruled the land", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ("The bear chased the deer", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ("The teacher helped the student", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ("The doctor treated the patient", {2: "nsubj", 3: "verb", 5: "dobj"}),
    ]

    # Adv-SVO: nsubj=pos3, dobj=pos6, verb=pos4
    # "Quickly the cat chased the mouse"
    adv_svo_data = [
        ("Quickly the cat chased the mouse", {3: "nsubj", 4: "verb", 6: "dobj"}),
        ("Slowly the dog bit the man", {3: "nsubj", 4: "verb", 6: "dobj"}),
        ("Quietly the girl saw the boy", {3: "nsubj", 4: "verb", 6: "dobj"}),
        ("Gently the bird ate the fish", {3: "nsubj", 4: "verb", 6: "dobj"}),
        ("Fiercely the wolf killed the deer", {3: "nsubj", 4: "verb", 6: "dobj"}),
        ("Swiftly the fox caught the rabbit", {3: "nsubj", 4: "verb", 6: "dobj"}),
        ("Softly the king ruled the land", {3: "nsubj", 4: "verb", 6: "dobj"}),
        ("Boldly the bear chased the deer", {3: "nsubj", 4: "verb", 6: "dobj"}),
    ]

    # PP-front: nsubj=pos5, dobj=pos8, verb=pos7
    # "In the garden the cat chased the mouse"
    pp_front_data = [
        ("In the garden the cat chased the mouse", {5: "nsubj", 7: "verb", 9: "dobj"}),
        ("In the forest the dog bit the man", {5: "nsubj", 7: "verb", 9: "dobj"}),
        ("On the hill the girl saw the boy", {5: "nsubj", 7: "verb", 9: "dobj"}),
        ("By the river the bird ate the fish", {5: "nsubj", 7: "verb", 9: "dobj"}),
        ("In the field the wolf killed the deer", {5: "nsubj", 7: "verb", 9: "dobj"}),
        ("At the park the fox caught the rabbit", {5: "nsubj", 7: "verb", 9: "dobj"}),
    ]

    # Center-embedding: nsubj(pos2)=far, dobj(pos5), verb(pos6)
    ce_data = [
        ("The cat that the dog chased ran", {2: "nsubj", 5: "dobj", 6: "verb"}),
        ("The man that the girl saw left", {2: "nsubj", 5: "dobj", 6: "verb"}),
        ("The bird that the cat watched flew", {2: "nsubj", 5: "dobj", 6: "verb"}),
        ("The fish that the bear caught swam", {2: "nsubj", 5: "dobj", 6: "verb"}),
        ("The boy that the king saw ran", {2: "nsubj", 5: "dobj", 6: "verb"}),
    ]

    # 测试集 (新句式)
    test_svo = [
        ("The horse pulled the cart", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ("The child broke the toy", {2: "nsubj", 3: "verb", 5: "dobj"}),
        ("The woman read the book", {2: "nsubj", 3: "verb", 5: "dobj"}),
    ]
    test_adv = [
        ("Loudly the horse pulled the cart", {3: "nsubj", 4: "verb", 6: "dobj"}),
        ("Quietly the child broke the toy", {3: "nsubj", 4: "verb", 6: "dobj"}),
    ]
    test_pp = [
        ("Near the lake the horse pulled the cart", {5: "nsubj", 7: "verb", 9: "dobj"}),
        ("On the road the woman read the book", {5: "nsubj", 7: "verb", 9: "dobj"}),
    ]

    # ============================================================
    # Strategy 1: Same-position training (SVO only)
    # ============================================================
    print("\n--- Strategy 1: Same-position training (SVO only) ---")
    print("    Train on SVO, test on SVO/Adv/PP → if drops on non-SVO, it's position-based")

    # Extract residual stream features
    def extract_resid_features(data, model, tokenizer, device):
        layer_features = defaultdict(list)  # li → [(vector, role_id)]
        for sentence, roles in data:
            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            with torch.no_grad():
                try:
                    outputs = model(input_ids=inputs["input_ids"], output_hidden_states=True)
                    all_hidden = outputs.hidden_states
                except:
                    continue
            
            seq_len = len(inputs["input_ids"][0])
            for li in range(len(all_hidden)):
                h = all_hidden[li].detach().float().cpu().numpy()[0]  # (seq, d)
                for pos in range(seq_len):
                    role = roles.get(pos, "other")
                    layer_features[li].append((h[pos], role_to_id[role]))
            
            del outputs, all_hidden
            gc.collect()
        return layer_features

    # Train on SVO
    train_svo_features = extract_resid_features(svo_data, model, tokenizer, device)
    test_svo_features = extract_resid_features(test_svo, model, tokenizer, device)
    test_adv_features = extract_resid_features(test_adv, model, tokenizer, device)
    test_pp_features = extract_resid_features(test_pp, model, tokenizer, device)

    # Evaluate
    def evaluate_probe(train_features, test_features, test_name, layers_to_test=None):
        """Train probe and evaluate"""
        if layers_to_test is None:
            layers_to_test = [0, 2, 5, 10, 15, 20, 25]
        
        results = {}
        for li in layers_to_test:
            if li not in train_features or li not in test_features:
                continue
            if len(train_features[li]) < 10:
                continue

            X_train = np.array([f[0] for f in train_features[li]])
            y_train = np.array([f[1] for f in train_features[li]])
            X_test = np.array([f[0] for f in test_features[li]])
            y_test = np.array([f[1] for f in test_features[li]])

            # Subsample
            if X_train.shape[0] > 500:
                idx = np.random.choice(len(X_train), 500, replace=False)
                X_train, y_train = X_train[idx], y_train[idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, C=1.0)
            try:
                clf.fit(X_train_s, y_train)
                acc = clf.score(X_test_s, y_test)
                
                # Per-class for nsubj and dobj
                nsubj_acc = dobj_acc = float('nan')
                for c, name in [(0, "nsubj"), (1, "dobj")]:
                    mask = y_test == c
                    if mask.sum() > 0:
                        ca = clf.score(X_test_s[mask], y_test[mask])
                        if name == "nsubj": nsubj_acc = ca
                        if name == "dobj": dobj_acc = ca
                
                results[li] = {"acc": acc, "nsubj": nsubj_acc, "dobj": dobj_acc}
            except:
                pass
        return results

    svo_only_results = evaluate_probe(train_svo_features, test_svo_features, "SVO→SVO")
    svo_to_adv_results = evaluate_probe(train_svo_features, test_adv_features, "SVO→Adv")
    svo_to_pp_results = evaluate_probe(train_svo_features, test_pp_features, "SVO→PP")

    # ============================================================
    # Strategy 2: Cross-position training
    # ============================================================
    print("\n--- Strategy 2: Cross-position training (SVO + Adv + PP + CE) ---")
    print("    Train on mixed positions, test on all → if generalizes, syntax info exists")

    cross_train_data = svo_data[:6] + adv_svo_data[:4] + pp_front_data[:3] + ce_data[:3]
    cross_train_features = extract_resid_features(cross_train_data, model, tokenizer, device)

    cross_svo_results = evaluate_probe(cross_train_features, test_svo_features, "Cross→SVO")
    cross_adv_results = evaluate_probe(cross_train_features, test_adv_features, "Cross→Adv")
    cross_pp_results = evaluate_probe(cross_train_features, test_pp_features, "Cross→PP")

    # Report
    print(f"\n{'='*80}")
    print(f"  RESULTS: Position-Controlled Probing")
    print(f"{'='*80}")
    print(f"  Strategy 1: Trained on SVO only (nsubj always at pos2)")
    print(f"  {'Layer':>6} {'SVO→SVO':>10} {'SVO→Adv':>10} {'SVO→PP':>10}  | nsubj_acc (SVO/Adv/PP)")
    print(f"  {'-'*75}")

    for li in [0, 2, 5, 10, 15, 20, 25]:
        r1 = svo_only_results.get(li, {})
        r2 = svo_to_adv_results.get(li, {})
        r3 = svo_to_pp_results.get(li, {})
        
        a1 = f"{r1.get('acc', 0):.3f}" if r1 else "  -  "
        a2 = f"{r2.get('acc', 0):.3f}" if r2 else "  -  "
        a3 = f"{r3.get('acc', 0):.3f}" if r3 else "  -  "
        
        ns1 = f"{r1.get('nsubj', 0):.2f}" if r1 and not np.isnan(r1.get('nsubj', float('nan'))) else " -"
        ns2 = f"{r2.get('nsubj', 0):.2f}" if r2 and not np.isnan(r2.get('nsubj', float('nan'))) else " -"
        ns3 = f"{r3.get('nsubj', 0):.2f}" if r3 and not np.isnan(r3.get('nsubj', float('nan'))) else " -"
        
        print(f"  L{li:>4} {a1:>10} {a2:>10} {a3:>10}  | {ns1:>4} / {ns2:>4} / {ns3:>4}")

    print(f"\n  Strategy 2: Trained on mixed positions (SVO+Adv+PP+CE)")
    print(f"  {'Layer':>6} {'Cross→SVO':>10} {'Cross→Adv':>10} {'Cross→PP':>10}  | nsubj_acc (SVO/Adv/PP)")
    print(f"  {'-'*75}")

    for li in [0, 2, 5, 10, 15, 20, 25]:
        r1 = cross_svo_results.get(li, {})
        r2 = cross_adv_results.get(li, {})
        r3 = cross_pp_results.get(li, {})
        
        a1 = f"{r1.get('acc', 0):.3f}" if r1 else "  -  "
        a2 = f"{r2.get('acc', 0):.3f}" if r2 else "  -  "
        a3 = f"{r3.get('acc', 0):.3f}" if r3 else "  -  "
        
        ns1 = f"{r1.get('nsubj', 0):.2f}" if r1 and not np.isnan(r1.get('nsubj', float('nan'))) else " -"
        ns2 = f"{r2.get('nsubj', 0):.2f}" if r2 and not np.isnan(r2.get('nsubj', float('nan'))) else " -"
        ns3 = f"{r3.get('nsubj', 0):.2f}" if r3 and not np.isnan(r3.get('nsubj', float('nan'))) else " -"
        
        print(f"  L{li:>4} {a1:>10} {a2:>10} {a3:>10}  | {ns1:>4} / {ns2:>4} / {ns3:>4}")

    # Key comparison
    print(f"\n  KEY COMPARISON: SVO→Adv vs Cross→Adv (nsubj accuracy)")
    print(f"  If SVO→Adv drops AND Cross→Adv recovers → position was the main feature")
    print(f"  If both low → syntax information is genuinely weak in residual stream")
    print(f"  If both high → syntax information is strong and position-invariant")
    print(f"  {'Layer':>6} {'SVO→Adv':>10} {'Cross→Adv':>10} {'Diff':>10}")
    print(f"  {'-'*40}")

    for li in [0, 2, 5, 10, 15, 20, 25]:
        ns_svo = svo_to_adv_results.get(li, {}).get('nsubj', float('nan'))
        ns_cross = cross_adv_results.get(li, {}).get('nsubj', float('nan'))
        diff = ns_cross - ns_svo if not (np.isnan(ns_svo) or np.isnan(ns_cross)) else float('nan')
        s1 = f"{ns_svo:.2f}" if not np.isnan(ns_svo) else " -"
        s2 = f"{ns_cross:.2f}" if not np.isnan(ns_cross) else " -"
        sd = f"{diff:+.2f}" if not np.isnan(diff) else " -"
        print(f"  L{li:>4} {s1:>10} {s2:>10} {sd:>10}")

    return {}


# ============================================================
# ExpB: Surgical Attention Intervention (逐head替换)
# ============================================================
def expB_surgical_attention(model, tokenizer, device, info):
    """
    比Exp4更精细的intervention:
    1. 逐层替换attention为distance-preserving random
    2. 测量语法判断(正确vs错误句子的PPL差异)的变化
    3. 找到哪些层的attention对语法判断最关键
    """
    print("\n" + "="*70)
    print("★★★ ExpB: Surgical Attention Intervention (逐层) ★★★")
    print("="*70)

    # 语法测试句对
    grammar_pairs = [
        ("The cat chases the mouse", "The cat chase the mouse"),
        ("The dogs chase the cat", "The dogs chases the cat"),
        ("The cat that the dog chased ran", "The cat that the dog chased run"),
    ]

    layers = model.model.layers if hasattr(model, 'model') and hasattr(model.model, 'layers') else None
    if layers is None:
        print("  Cannot access model layers!")
        return {}

    def compute_ppl(model, tokenizer, sentence, device):
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            try:
                outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
                return torch.exp(outputs.loss).item()
            except:
                return float('inf')

    # Baseline grammar sensitivity
    print("\n--- Baseline Grammar Sensitivity ---")
    baseline_sensitivities = []
    for correct, wrong in grammar_pairs:
        ppl_c = compute_ppl(model, tokenizer, correct, device)
        ppl_w = compute_ppl(model, tokenizer, wrong, device)
        sensitivity = ppl_w / ppl_c if ppl_c > 0 else 0
        baseline_sensitivities.append(sensitivity)
        print(f"  '{correct}': PPL_correct={ppl_c:.1f}, PPL_wrong={ppl_w:.1f}, Ratio={sensitivity:.2f}x")
    
    mean_baseline = np.mean(baseline_sensitivities)
    print(f"  Mean grammar sensitivity: {mean_baseline:.2f}x")

    # Per-layer ablation: zero out attention output (more stable than uniform replacement)
    print("\n--- Per-Layer Attention Zero-Ablation ---")
    print(f"  (Zeroing out attention output at each layer, measuring grammar sensitivity)")
    print(f"  {'Layer':>6} {'MeanSens':>10} {'vsBaseline':>10} {'GrammarDrop':>12}")
    print(f"  {'-'*45}")

    n_layers = info.n_layers

    for li in range(n_layers):
        # Register hook to zero out attention output
        hooks = []
        
        def zero_attn_hook(module, input, output):
            # Zero out the attention output
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        
        if hasattr(layers[li], 'self_attn'):
            h = layers[li].self_attn.register_forward_hook(zero_attn_hook)
            hooks.append(h)

        # Measure grammar sensitivity
        ablation_sensitivities = []
        for correct, wrong in grammar_pairs:
            ppl_c = compute_ppl(model, tokenizer, correct, device)
            ppl_w = compute_ppl(model, tokenizer, wrong, device)
            if ppl_c > 0 and not np.isinf(ppl_c) and not np.isinf(ppl_w):
                sensitivity = ppl_w / ppl_c
                ablation_sensitivities.append(sensitivity)

        # Remove hooks
        for h in hooks:
            h.remove()

        if ablation_sensitivities:
            mean_ablation = np.mean(ablation_sensitivities)
            diff = mean_ablation - mean_baseline
            grammar_drop = "GRAMMAR LOST" if mean_ablation < 1.0 else ("WEAKENED" if diff < -0.2 else "")
            print(f"  L{li:>4} {mean_ablation:>9.2f}x {diff:>+9.2f}x {grammar_drop:>12}")
        else:
            print(f"  L{li:>4} {'inf':>9} {'N/A':>10}")

        gc.collect()
        torch.cuda.empty_cache()

    return {}


# ============================================================
# ExpC: Residual Stream语法子空间分析
# ============================================================
def expC_syntax_subspace(model, tokenizer, device, info):
    """
    用PCA分析residual stream中的语法子空间
    
    核心思路:
    1. 收集大量token的residual stream表示
    2. 标注语法角色 (nsubj/dobj/verb/other)
    3. 用PCA降维, 看语法角色是否在低维空间可分
    4. 计算语法子空间的方差解释率
    
    如果存在低维语法子空间 → 语法信息被结构化编码
    如果需要高维 → 语法信息是分布式/隐式编码
    """
    print("\n" + "="*70)
    print("★★★ ExpC: Residual Stream 语法子空间分析 ★★★")
    print("="*70)

    role_to_id = {"nsubj": 0, "dobj": 1, "verb": 2, "det": 3, "other": 4}

    # 收集数据
    sentence_data = [
        ("The cat chased the mouse", {1: "det", 2: "nsubj", 3: "verb", 4: "det", 5: "dobj"}),
        ("The dog bit the man", {1: "det", 2: "nsubj", 3: "verb", 4: "det", 5: "dobj"}),
        ("The girl saw the boy", {1: "det", 2: "nsubj", 3: "verb", 4: "det", 5: "dobj"}),
        ("The bird ate the fish", {1: "det", 2: "nsubj", 3: "verb", 4: "det", 5: "dobj"}),
        ("The wolf killed the deer", {1: "det", 2: "nsubj", 3: "verb", 4: "det", 5: "dobj"}),
        ("Quickly the cat chased the mouse", {2: "det", 3: "nsubj", 4: "verb", 5: "det", 6: "dobj"}),
        ("Slowly the dog bit the man", {2: "det", 3: "nsubj", 4: "verb", 5: "det", 6: "dobj"}),
        ("In the garden the cat chased the mouse", {2: "det", 4: "det", 5: "nsubj", 7: "verb", 8: "det", 9: "dobj"}),
        ("The cat that the dog chased ran", {1: "det", 2: "nsubj", 4: "det", 5: "dobj", 6: "verb", 7: "verb"}),
        ("The man that the girl saw left", {1: "det", 2: "nsubj", 4: "det", 5: "dobj", 6: "verb", 7: "verb"}),
    ]

    # 提取residual stream features (只关注关键层)
    target_layers = [0, 5, 10, 15, 20, 25]
    layer_features = defaultdict(lambda: {"vectors": [], "roles": [], "positions": []})

    for sentence, roles in sentence_data:
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            try:
                outputs = model(input_ids=inputs["input_ids"], output_hidden_states=True)
                all_hidden = outputs.hidden_states
            except:
                continue

        seq_len = len(inputs["input_ids"][0])

        for li in target_layers:
            if li >= len(all_hidden):
                continue
            h = all_hidden[li].detach().float().cpu().numpy()[0]  # (seq, d)
            for pos in range(seq_len):
                role = roles.get(pos, "other")
                layer_features[li]["vectors"].append(h[pos])
                layer_features[li]["roles"].append(role_to_id[role])
                layer_features[li]["positions"].append(pos)

        del outputs, all_hidden
        gc.collect()

    # PCA分析
    from sklearn.decomposition import PCA

    print(f"\n  PCA Analysis of Residual Stream:")
    print(f"  {'Layer':>6} {'Top3Var%':>10} {'nsubj/dobj sep (PC1)':>22} {'nsubj/dobj sep (PC2)':>22}")
    print(f"  {'-'*70}")

    for li in target_layers:
        if li not in layer_features or len(layer_features[li]["vectors"]) < 10:
            continue

        X = np.array(layer_features[li]["vectors"])
        y = np.array(layer_features[li]["roles"])
        pos = np.array(layer_features[li]["positions"])

        # Subsample if too large
        if X.shape[0] > 300:
            idx = np.random.choice(len(X), 300, replace=False)
            X, y, pos = X[idx], y[idx], pos[idx]

        # PCA to 10 components
        pca = PCA(n_components=min(10, X.shape[1], X.shape[0]))
        X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

        # Variance explained
        top3_var = sum(pca.explained_variance_ratio_[:3]) * 100

        # nsubj/dobj separability on PC1 and PC2
        nsubj_mask = y == 0
        dobj_mask = y == 1
        verb_mask = y == 2

        sep_pc1 = sep_pc2 = float('nan')
        if nsubj_mask.sum() > 0 and dobj_mask.sum() > 0:
            nsubj_pc1 = X_pca[nsubj_mask, 0].mean()
            dobj_pc1 = X_pca[dobj_mask, 0].mean()
            sep_pc1 = abs(nsubj_pc1 - dobj_pc1) / max(np.std(X_pca[:, 0]), 1e-8)

            nsubj_pc2 = X_pca[nsubj_mask, 1].mean()
            dobj_pc2 = X_pca[dobj_mask, 1].mean()
            sep_pc2 = abs(nsubj_pc2 - dobj_pc2) / max(np.std(X_pca[:, 1]), 1e-8)

        s1 = f"{sep_pc1:.2f}σ" if not np.isnan(sep_pc1) else "N/A"
        s2 = f"{sep_pc2:.2f}σ" if not np.isnan(sep_pc2) else "N/A"
        print(f"  L{li:>4} {top3_var:>8.1f}% {s1:>22} {s2:>22}")

        # Also check: are nsubj at different positions close in PCA space?
        if li >= 10:  # Only check mid/late layers
            nsubj_pos2 = X_pca[(y == 0) & (pos == 2)]
            nsubj_pos3 = X_pca[(y == 0) & (pos == 3)]
            nsubj_pos5 = X_pca[(y == 0) & (pos == 5)]
            
            if len(nsubj_pos2) > 0 and len(nsubj_pos3) > 0:
                # Cosine similarity between nsubj at different positions
                mean_p2 = nsubj_pos2.mean(axis=0)
                mean_p3 = nsubj_pos3.mean(axis=0)
                cos_sim = np.dot(mean_p2, mean_p3) / max(np.linalg.norm(mean_p2) * np.linalg.norm(mean_p3), 1e-8)
                print(f"         nsubj@pos2 ↔ nsubj@pos3 cosine: {cos_sim:.3f}")

    return {}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all",
                       choices=["A", "B", "C", "all"])
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"Model: {info.name}, Layers: {info.n_layers}, d_model: {info.d_model}")

    if args.exp in ["A", "all"]:
        expA_position_controlled_probing(model, tokenizer, device, info)
    if args.exp in ["B", "all"]:
        expB_surgical_attention(model, tokenizer, device, info)
    if args.exp in ["C", "all"]:
        expC_syntax_subspace(model, tokenizer, device, info)

    release_model(model)
    print("\n[Phase 58B Complete]")


if __name__ == "__main__":
    main()
