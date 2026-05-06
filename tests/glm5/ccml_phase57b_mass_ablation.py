"""
Phase 57B: 修正指标 + 大规模Ablation + nsubj真伪验证
=====================================================

核心发现需要修正:
1. best_rank=1 over 784 heads 在短句中完全失效 (perm_k1>0.65)
2. 因果ablation单head无效 → 需要mass ablation
3. nsubj在SVO中"成功"可能是pos2启发式, 不是语法理解

新指标: Per-layer hit rate (比per-head best_rank更统计可靠)
新测试: Mass ablation (44 heads同时)
关键验证: nsubj是语法角色还是position 2启发式?
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


# ============================================================
# ExpA: Per-layer Hit Rate (修正版指标)
# ============================================================
def expA_per_layer_hit_rate(model, tokenizer, device, info):
    """
    不用"784个head中有没有一个命中", 改用:
    "每层28个head中, 多少个把目标位置作为top-1"
    
    这个指标不受多重比较影响, 因为每层独立评估
    
    Null hypothesis: 每个head随机选top-1 → hit rate ≈ 1/num_left_positions
    Alternative:     语法位置的hit rate显著高于null
    """
    print("\n" + "="*70)
    print("★★★ ExpA: Per-Layer Hit Rate (修正版指标) ★★★")
    print("="*70)

    test_cases = [
        # SVO
        {"name": "SVO nsubj", "sentences": [
            "The cat chased the mouse",
            "The dog bit the man",
            "The girl saw the boy",
            "The bird ate the fish",
            "The wolf killed the deer",
            "The fox caught the rabbit",
        ], "head_pos": 3, "dep_pos": 2, "num_left_pos": 2},
        {"name": "SVO dobj", "sentences": [
            "The cat chased the mouse",
            "The dog bit the man",
            "The girl saw the boy",
            "The bird ate the fish",
            "The wolf killed the deer",
            "The fox caught the rabbit",
        ], "head_pos": 3, "dep_pos": 5, "num_left_pos": 2},  # dobj在右边, 看左边的只有2个pos
        # Center-embedding
        {"name": "CE nsubj_short", "sentences": [
            "The cat that the dog chased ran",
            "The man that the girl saw left",
            "The bird that the cat watched flew",
        ], "head_pos": 6, "dep_pos": 5, "num_left_pos": 5},
        {"name": "CE nsubj_long", "sentences": [
            "The cat that the dog chased ran",
            "The man that the girl saw left",
            "The bird that the cat watched flew",
        ], "head_pos": 7, "dep_pos": 2, "num_left_pos": 6},
        {"name": "CE dobj", "sentences": [
            "The cat that the dog chased ran",
            "The man that the girl saw left",
            "The bird that the cat watched flew",
        ], "head_pos": 6, "dep_pos": 2, "num_left_pos": 5},
        # PP modifier (critical: nearest_noun ≠ nsubj)
        {"name": "PP nsubj (cat, pos2)", "sentences": [
            "The cat with the hat chased the mouse",
            "The dog near the park chased the cat",
            "The bird on the tree saw the fish",
        ], "head_pos": 7, "dep_pos": 2, "num_left_pos": 6},
        {"name": "PP nearest_noun (hat, pos5)", "sentences": [
            "The cat with the hat chased the mouse",
            "The dog near the park chased the cat",
            "The bird on the tree saw the fish",
        ], "head_pos": 7, "dep_pos": 5, "num_left_pos": 6},
    ]

    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        name = case["name"]
        sentences = case["sentences"]
        head_pos = case["head_pos"]
        dep_pos = case["dep_pos"]
        num_left = case["num_left_pos"]

        # 收集per-layer hit rates
        layer_hit_rates = defaultdict(list)  # li → [hit_rate_per_sentence]

        for sentence in sentences:
            attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
            if attn_dict is None:
                continue

            seq_len = len(tokens)

            for li in range(min(info.n_layers, 28)):
                if li not in attn_dict:
                    continue
                A = attn_dict[li]
                n_heads = A.shape[0]

                # 只看左边的attention
                if dep_pos < head_pos:
                    row = A[:, head_pos, 1:head_pos]
                    target = dep_pos - 1
                elif dep_pos > head_pos:
                    # dobj在右边 → 只看左边context (causal)
                    row = A[:, head_pos, 1:head_pos]
                    target_in_row = dep_pos - 1
                    if target_in_row >= row.shape[1]:
                        continue
                    target = target_in_row
                else:
                    continue

                if target < 0 or target >= row.shape[1]:
                    continue

                # 多少个head把target作为top-1
                for h in range(n_heads):
                    top1 = int(np.argmax(row[h]))
                    hit = 1 if top1 == target else 0
                    layer_hit_rates[li].append(hit)

            del attn_dict, inputs
            gc.collect()

        # 汇总per-layer
        print(f"  Null expected hit rate: 1/{row.shape[1] if 'row' in dir() else num_left} "
              f"≈ {1.0/max(num_left, 1):.3f}")
        print(f"  {'Layer':>6} {'HitRate':>8} {'NullRate':>8} {'Ratio':>6} {'p-value':>8} {'Sig':>5}")
        print(f"  {'-'*50}")

        significant_layers = 0
        for li in [0, 1, 2, 3, 5, 7, 10, 15, 20, 25]:
            if li not in layer_hit_rates or not layer_hit_rates[li]:
                continue

            hits = layer_hit_rates[li]
            hit_rate = np.mean(hits)
            n_total = len(hits)
            # Null: each head randomly picks top-1 from num_left positions
            null_rate = 1.0 / max(num_left, 1)

            ratio = hit_rate / null_rate if null_rate > 0 else float('inf')

            # Binomial test: is hit_rate significantly above null_rate?
            from scipy.stats import binomtest
            result = binomtest(int(sum(hits)), int(n_total), null_rate, alternative='greater')
            pval = result.pvalue

            sig = "★★★" if pval < 0.001 else ("★" if pval < 0.01 else ("" if pval > 0.05 else "~"))
            if pval < 0.05:
                significant_layers += 1

            print(f"  L{li:>4} {hit_rate:>7.3f} {null_rate:>7.3f} {ratio:>5.1f}x {pval:>8.4f} {sig:>5}")

        print(f"\n  显著层数 (p<0.05): {significant_layers}/{len([li for li in [0,1,2,3,5,7,10,15,20,25] if li in layer_hit_rates])}")

    return {}


# ============================================================
# ExpB: Mass Ablation (44 heads同时)
# ============================================================
def expB_mass_ablation(model, tokenizer, device, info):
    """
    单head ablation无效 → 尝试同时消融所有44个nsubj-tracking heads
    
    同时测试:
    1. 44个syntax heads同时ablation
    2. 44个随机heads同时ablation (控制)
    3. 全部784个heads同时ablation (极限测试)
    
    Grammar test: 完整句子perplexity + subject-verb agreement
    """
    print("\n" + "="*70)
    print("★★★ ExpB: Mass Ablation (多heads同时消融) ★★★")
    print("="*70)

    # Step 1: 找所有reliable nsubj heads
    print("\n--- Step 1: Identifying ALL reliable nsubj-tracking heads ---")

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
        results = compute_rank_for_dep_simple(attn_dict, 3, 2, info.n_layers)
        for r in results:
            key = (r["layer"], r["head"])
            head_total[key] += 1
            if r["rank"] == 1:
                head_success[key] += 1
        del attn_dict, inputs
        gc.collect()

    # ≥6/8 success
    reliable_heads = []
    for key in head_total:
        if head_total[key] >= 6 and head_success[key] / head_total[key] >= 0.75:
            reliable_heads.append((key[0], key[1], head_success[key] / head_total[key]))

    reliable_heads.sort(key=lambda x: x[2], reverse=True)
    print(f"  Found {len(reliable_heads)} reliable nsubj-tracking heads:")
    for li, h, rate in reliable_heads[:10]:
        print(f"    L{li} H{h}: rate={rate:.2f}")
    print(f"  ... and {len(reliable_heads)-10} more" if len(reliable_heads) > 10 else "")

    # Step 2: Grammar test — Perplexity on grammatical vs ungrammatical
    print("\n--- Step 2: Grammar Test via Perplexity ---")

    grammatical = [
        "The cat chased the mouse quickly",
        "The dog saw the man yesterday",
        "The girl ate the fish slowly",
        "The birds fly in the sky",
        "The cat runs very fast",
        "The children play in the park",
        "The dog barks at the cat",
        "The student reads the book",
    ]

    ungrammatical = [
        "The cat chase the mouse quickly",  # 主谓不一致
        "The dog see the man yesterday",    # 主谓不一致
        "The girl eat the fish slowly",     # 主谓不一致
        "The birds flies in the sky",       # 主谓不一致
        "The cat run very fast",            # 主谓不一致
        "The children plays in the park",   # 主谓不一致
    ]

    def compute_perplexity(sentences, mdl, tok, dev):
        """计算句子集的平均perplexity"""
        ppls = []
        for sent in sentences:
            inputs = tok(sent, return_tensors="pt").to(dev)
            with torch.no_grad():
                outputs = mdl(input_ids=inputs["input_ids"])
                logits = outputs.logits
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = inputs["input_ids"][:, 1:]
            # Perplexity
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )
            ppl = torch.exp(loss).item()
            ppls.append(ppl)
        return np.mean(ppls), ppls

    # Step 3: Baseline perplexity
    print("\n  Baseline perplexity:")
    gram_ppl_base, gram_ppls = compute_perplexity(grammatical, model, tokenizer, device)
    ungram_ppl_base, ungram_ppls = compute_perplexity(ungrammatical, model, tokenizer, device)
    diff_base = ungram_ppl_base - gram_ppl_base
    print(f"    Grammatical: {gram_ppl_base:.2f}")
    print(f"    Ungrammatical: {ungram_ppl_base:.2f}")
    print(f"    Difference (ungram-gram): {diff_base:.2f}")
    print(f"    → If model detects grammar: ungram PPL > gram PPL (positive diff)")

    # Step 4: Mass ablation of syntax heads
    print("\n--- Step 3: Mass Ablation of Syntax Heads ---")

    layers = model.model.layers
    n_heads = layers[0].self_attn.num_heads if hasattr(layers[0].self_attn, 'num_heads') else 28

    def make_zero_hook(target_head_idx, n_h):
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

    # Ablate ALL reliable syntax heads
    print(f"\n  Ablating all {len(reliable_heads)} reliable nsubj-tracking heads...")
    hooks = []
    for li, h, rate in reliable_heads:
        attn = layers[li].self_attn
        hooks.append(attn.register_forward_hook(make_zero_hook(h, n_heads)))

    gram_ppl_syntax, _ = compute_perplexity(grammatical, model, tokenizer, device)
    ungram_ppl_syntax, _ = compute_perplexity(ungrammatical, model, tokenizer, device)
    diff_syntax = ungram_ppl_syntax - gram_ppl_syntax

    for hk in hooks:
        hk.remove()

    print(f"    Grammatical: {gram_ppl_syntax:.2f}")
    print(f"    Ungrammatical: {ungram_ppl_syntax:.2f}")
    print(f"    Difference: {diff_syntax:.2f} (baseline: {diff_base:.2f}, Δ={diff_syntax-diff_base:+.2f})")

    if diff_syntax < diff_base - 2.0:
        print(f"    ★★★ SIGNIFICANT: Grammar sensitivity DECREASED after syntax head ablation ★★★")
    elif diff_syntax < diff_base - 0.5:
        print(f"    ★ MODERATE: Some decrease in grammar sensitivity")
    else:
        print(f"    ✗ No significant change in grammar sensitivity")

    # Step 5: Control — random heads (same count)
    print(f"\n  Control: Ablating {len(reliable_heads)} RANDOM heads...")
    random_heads = [(random.randint(0, info.n_layers-1), random.randint(0, n_heads-1))
                    for _ in range(len(reliable_heads))]

    hooks = []
    for li, h in random_heads:
        attn = layers[li].self_attn
        hooks.append(attn.register_forward_hook(make_zero_hook(h, n_heads)))

    gram_ppl_random, _ = compute_perplexity(grammatical, model, tokenizer, device)
    ungram_ppl_random, _ = compute_perplexity(ungrammatical, model, tokenizer, device)
    diff_random = ungram_ppl_random - gram_ppl_random

    for hk in hooks:
        hk.remove()

    print(f"    Grammatical: {gram_ppl_random:.2f}")
    print(f"    Ungrammatical: {ungram_ppl_random:.2f}")
    print(f"    Difference: {diff_random:.2f} (baseline: {diff_base:.2f}, Δ={diff_random-diff_base:+.2f})")

    # Step 6: Per-layer group ablation
    print("\n--- Step 4: Per-Layer Group Ablation ---")
    print("  (Ablate all syntax heads within each layer group)")

    for layer_group, layer_name in [([0,1,2], "Early (L0-2)"), ([3,4,5,6,7], "Mid-early (L3-7)"),
                                      ([8,9,10,11,12], "Mid (L8-12)"), ([13,14,15,16,17], "Mid-late (L13-17)"),
                                      ([18,19,20,21,22,23,24,25,26,27], "Late (L18-27)")]:
        group_heads = [(li, h, r) for li, h, r in reliable_heads if li in layer_group]
        if not group_heads:
            print(f"  {layer_name}: 0 syntax heads → skip")
            continue

        hooks = []
        for li, h, rate in group_heads:
            attn = layers[li].self_attn
            hooks.append(attn.register_forward_hook(make_zero_hook(h, n_heads)))

        gram_ppl, _ = compute_perplexity(grammatical[:4], model, tokenizer, device)  # 少量句子加速
        ungram_ppl, _ = compute_perplexity(ungrammatical[:4], model, tokenizer, device)
        diff = ungram_ppl - gram_ppl

        for hk in hooks:
            hk.remove()

        delta = diff - diff_base
        print(f"  {layer_name}: {len(group_heads)} heads, diff={diff:.2f} (Δ={delta:+.2f})")

    # Final summary
    print("\n--- Final Summary ---")
    print(f"  Baseline grammar sensitivity: {diff_base:.2f}")
    print(f"  After syntax head ablation:   {diff_syntax:.2f} (Δ={diff_syntax-diff_base:+.2f})")
    print(f"  After random head ablation:   {diff_random:.2f} (Δ={diff_random-diff_base:+.2f})")

    return {"baseline_diff": diff_base, "syntax_diff": diff_syntax, "random_diff": diff_random}


def compute_rank_for_dep_simple(attn_dict, head_pos, dep_pos, n_layers):
    """简化版rank计算"""
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
            row = A[:, head_pos, 1:head_pos]  # 只看左边 (causal)
            target = dep_pos - 1
            if target >= row.shape[1]:
                continue
        else:
            continue

        if target < 0 or target >= row.shape[1]:
            continue

        for h in range(n_heads):
            sorted_idx = np.argsort(row[h])[::-1]
            rank = int(np.where(sorted_idx == target)[0][0]) + 1 if target in sorted_idx else 999
            results.append({"layer": li, "head": h, "rank": rank})

    return results


# ============================================================
# ExpC: nsubj是语法角色还是position 2启发式?
# ============================================================
def expC_position2_heuristic(model, tokenizer, device, info):
    """
    关键测试: SVO中nsubj"成功"是因为:
    A) attention理解"pos2 = 主语" (语法角色)
    B) attention只是"attend to position 2" (位置启发式)
    C) attention只是"attend to nearest left noun" (距离启发式)

    区分方法: 构造position 2 ≠ nsubj的句子

    测试1: "Did the cat chase the mouse?" → cat仍在pos2但不是句首名词
    测试2: "Quickly the cat chased the mouse" → cat在pos3(不是pos2)
    测试3: "In the garden the cat chased the mouse" → cat更远
    
    但这些改变了pos2的语法角色, 看attention是否仍追踪pos2
    """
    print("\n" + "="*70)
    print("★★★ ExpC: Position-2 Heuristic vs Syntactic nsubj ★★★")
    print("="*70)

    test_cases = [
        # 标准SVO: pos2 = nsubj
        {"name": "SVO (pos2=nsubj)", "sentence": "The cat chased the mouse",
         "head_pos": 3, "dep_pos": 2, "expected": "pos2 is nsubj"},
        
        # 副词开头: pos2 = adverb, nsubj在pos3
        {"name": "Adv-SVO (pos2=adv, pos3=nsubj)", "sentence": "Quickly the cat chased the mouse",
         "head_pos": 4, "dep_pos": 3, "expected": "pos2 is NOT nsubj, pos3 is"},
        
        # 介词短语开头: pos2-4 = PP, nsubj在pos5
        {"name": "PP-SVO (nsubj at pos5)", "sentence": "In the park the cat chased the mouse",
         "head_pos": 6, "dep_pos": 5, "expected": "nsubj far from verb"},
        
        # 疑问句: pos2 = auxiliary "did", nsubj在pos3
        {"name": "Q-SVO (pos2=aux, pos3=nsubj)", "sentence": "Did the cat chase the mouse",
         "head_pos": 4, "dep_pos": 3, "expected": "pos2 is auxiliary, pos3 is nsubj"},
        
        # 另一个SVO对照
        {"name": "SVO-2 (pos2=nsubj)", "sentence": "The dog bit the man",
         "head_pos": 3, "dep_pos": 2, "expected": "pos2 is nsubj (control)"},
    ]

    print("\n  Key question: When nsubj is NOT at position 2, does attention still find it?")
    print("  If YES → genuine nsubj tracking (beyond position heuristic)")
    print("  If NO  → just position 2 heuristic\n")

    for case in test_cases:
        sentence = case["sentence"]
        head_pos = case["head_pos"]
        dep_pos = case["dep_pos"]

        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue

        seq_len = len(tokens)
        print(f"\n  [{case['name']}] {sentence}")
        print(f"  Tokens: {[(i,t) for i,t in enumerate(tokens)]}")
        print(f"  {case['expected']}")
        print(f"  Testing: {tokens[dep_pos]}(pos{dep_pos}) → {tokens[head_pos]}(pos{head_pos})")

        # Per-layer analysis: fraction of heads with target as top-1
        n_left = head_pos - 1  # number of left positions (excluding BOS)
        null_rate = 1.0 / n_left if n_left > 0 else 0

        print(f"  Left context positions: {n_left}, Null hit rate: {null_rate:.3f}")

        for li in [0, 1, 3, 7, 10, 15, 20]:
            if li not in attn_dict:
                continue
            A = attn_dict[li]
            n_heads = A.shape[0]

            row = A[:, head_pos, 1:head_pos]
            target = dep_pos - 1

            if target < 0 or target >= row.shape[1]:
                continue

            hit_count = 0
            for h in range(n_heads):
                top1 = int(np.argmax(row[h]))
                if top1 == target:
                    hit_count += 1

            hit_rate = hit_count / n_heads
            ratio = hit_rate / null_rate if null_rate > 0 else float('inf')

            sig = "★★★" if ratio > 5 else ("★" if ratio > 2 else "")
            print(f"    L{li}: hit_rate={hit_rate:.3f} ({hit_count}/{n_heads}), "
                  f"null={null_rate:.3f}, ratio={ratio:.1f}x {sig}")

        # Also check: is pos2 always getting high attention regardless?
        print(f"  Position 2 attention check:")
        for li in [0, 1, 3]:
            if li not in attn_dict:
                continue
            A = attn_dict[li]
            row = A[:, head_pos, 1:head_pos]
            # pos2 in the row → index 1
            pos2_idx = 1  # pos2 in the row (pos1=0, pos2=1, ...)
            if pos2_idx >= row.shape[1]:
                continue
            pos2_top1_count = sum(1 for h in range(n_heads) if int(np.argmax(row[h])) == pos2_idx)
            print(f"    L{li}: pos2 as top-1: {pos2_top1_count}/{n_heads} = {pos2_top1_count/n_heads:.3f}")

        del attn_dict, inputs
        gc.collect()

    return {}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b",
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all",
                        help="A=per_layer_hitrate, B=mass_ablation, C=position2_test, all")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"Model: {args.model}, Layers: {info.n_layers}, d_model: {info.d_model}")

    if args.exp in ["all", "A"]:
        expA_per_layer_hit_rate(model, tokenizer, device, info)

    if args.exp in ["all", "B"]:
        expB_mass_ablation(model, tokenizer, device, info)

    if args.exp in ["all", "C"]:
        expC_position2_heuristic(model, tokenizer, device, info)

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("\nPhase 57B Complete!")


if __name__ == "__main__":
    main()
