"""
Phase 54C-v2: 严格复杂句验证 + Per-Layer评估 + 强Baseline对抗
============================================================

基于tokenization调试的修正版本:
- 所有dep_list位置已验证匹配token位置
- 加入per-layer UAS分析 (不只看"第一层hit")
- 加入强baseline对比 (nearest_noun等)
- 使用k=1严格评估 + k=3宽松评估
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse, torch, numpy as np, gc
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model_utils import load_model, get_model_info, release_model, safe_decode


# ============================================================
# ★ 精确token对齐的句子数据
# ============================================================
# 所有位置已验证匹配DS7B tokenizer输出
# 格式: (head_pos, dep_pos, rel_type) 位置0=BOS

# ---- 简单SVO (1:1 word:token) ----
SIMPLE_SVO = [
    {"sentence": "The cat chased the mouse",
     "tokens": ["<BOS>","The"," cat"," chased"," the"," mouse"],
     "deps": [(3,2,"nsubj"),(3,5,"dobj"),(2,1,"det"),(5,4,"det")]},
    {"sentence": "The dog bit the man",
     "tokens": ["<BOS>","The"," dog"," bit"," the"," man"],
     "deps": [(3,2,"nsubj"),(3,5,"dobj"),(2,1,"det"),(5,4,"det")]},
    {"sentence": "The boy ate the apple",
     "tokens": ["<BOS>","The"," boy"," ate"," the"," apple"],
     "deps": [(3,2,"nsubj"),(3,5,"dobj"),(2,1,"det"),(5,4,"det")]},
    {"sentence": "The woman drove the car",
     "tokens": ["<BOS>","The"," woman"," drove"," the"," car"],
     "deps": [(3,2,"nsubj"),(3,5,"dobj"),(2,1,"det"),(5,4,"det")]},
]

# ---- 语义异常SVO (1:1 word:token) ----
ANOMALOUS_SVO = [
    {"sentence": "The rock ate the concept",
     "tokens": ["<BOS>","The"," rock"," ate"," the"," concept"],
     "deps": [(3,2,"nsubj"),(3,5,"dobj"),(2,1,"det"),(5,4,"det")]},
    {"sentence": "The fish ate the cat",
     "tokens": ["<BOS>","The"," fish"," ate"," the"," cat"],
     "deps": [(3,2,"nsubj"),(3,5,"dobj"),(2,1,"det"),(5,4,"det")]},
    {"sentence": "The idea chased the theory",
     "tokens": ["<BOS>","The"," idea"," chased"," the"," theory"],
     "deps": [(3,2,"nsubj"),(3,5,"dobj"),(2,1,"det"),(5,4,"det")]},
    {"sentence": "The mountain drank the silence",
     "tokens": ["<BOS>","The"," mountain"," drank"," the"," silence"],
     "deps": [(3,2,"nsubj"),(3,5,"dobj"),(2,1,"det"),(5,4,"det")]},
]

# ---- 中心嵌入 (1:1 word:token) ----
CENTER_EMBED = [
    {"sentence": "The cat that the dog chased ran",
     "tokens": ["<BOS>","The"," cat"," that"," the"," dog"," chased"," ran"],
     "deps": [(7,2,"nsubj"),(6,5,"nsubj"),(6,2,"dobj")]},
    # ★关键: ran的nsubj=cat(距离5!), 不是dog(距离1)
    # ★关键: chased的nsubj=dog, chased的dobj=cat
    
    {"sentence": "The man who the woman saw left",
     "tokens": ["<BOS>","The"," man"," who"," the"," woman"," saw"," left"],
     "deps": [(7,2,"nsubj"),(6,5,"nsubj"),(6,2,"dobj")]},
    # ★关键: left的nsubj=man(距离5!), 不是woman
]

# ---- 被动句 (1:1 word:token) ----
PASSIVE = [
    {"sentence": "The mouse was chased by the cat",
     "tokens": ["<BOS>","The"," mouse"," was"," chased"," by"," the"," cat"],
     "deps": [(4,2,"nsubj_pass")]},
    # ★关键: chased的语法主语=mouse(语义宾语!)
    
    {"sentence": "The car was repaired by the mechanic",
     "tokens": ["<BOS>","The"," car"," was"," repaired"," by"," the"," mechanic"],
     "deps": [(4,2,"nsubj_pass")]},
]

# ---- 关系从句 (1:1 word:token) ----
REL_CLAUSE = [
    {"sentence": "The woman who saw the man left",
     "tokens": ["<BOS>","The"," woman"," who"," saw"," the"," man"," left"],
     "deps": [(7,2,"nsubj"),(4,3,"nsubj"),(4,6,"dobj")]},
    # ★关键: left的nsubj=woman, saw的nsubj=who
    
    {"sentence": "The dog saw the cat that ran",
     "tokens": ["<BOS>","The"," dog"," saw"," the"," cat"," that"," ran"],
     "deps": [(3,2,"nsubj"),(3,5,"dobj"),(7,5,"nsubj")]},
]

# ---- 多从句 (1:1 word:token) ----
MULTI_CLAUSE = [
    {"sentence": "John thinks that Mary believes that the cat ran",
     "tokens": ["<BOS>","John"," thinks"," that"," Mary"," believes"," that"," the"," cat"," ran"],
     "deps": [(2,1,"nsubj"),(5,4,"nsubj"),(9,8,"nsubj")]},
]

# ---- ★★★ 关键对比: swap句 (同词不同角色) ★★★ ----
SWAP_PAIRS = [
    ("The cat ate the fish", 
     [(3,2,"nsubj"),(3,5,"dobj")],  # cat=nsubj, fish=dobj
     "正常语义"),
    ("The fish ate the cat",
     [(3,2,"nsubj"),(3,5,"dobj")],  # fish=nsubj, cat=dobj (异常!)
     "语义异常"),
]


def get_attention_weights(model, tokenizer, sentence, device):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model(input_ids=inputs["input_ids"], output_attentions=True)
            all_attn = outputs.attentions
        except:
            return None
    attn_dict = {}
    for li, attn in enumerate(all_attn):
        if attn is not None:
            attn_dict[li] = attn.detach().float().cpu().numpy()[0]
    return attn_dict


def evaluate_attn_per_layer(attn_dict, head_pos, dep_pos, seq_len):
    """Per-layer UAS评估, 返回每层的hit情况 (k=1严格 + k=3宽松)"""
    layer_results = {}
    for li in sorted(attn_dict.keys()):
        A = attn_dict[li].mean(axis=0)  # mean over heads
        
        # 确定哪个token可以attend到另一个
        if head_pos < dep_pos:
            # dep可以attend到head → A[dep, 1:dep]
            row = A[dep_pos, 1:dep_pos]
            target = head_pos - 1  # 相对位置
        elif head_pos > dep_pos:
            # head可以attend到dep → A[head, 1:head]
            row = A[head_pos, 1:head_pos]
            target = dep_pos - 1
        else:
            continue
        
        if len(row) == 0 or target < 0 or target >= len(row):
            layer_results[li] = {"k1": False, "k3": False, "score": 0.0}
            continue
        
        score = float(row[target])
        sorted_indices = np.argsort(row)[::-1]
        
        k1_hit = sorted_indices[0] == target if len(sorted_indices) > 0 else False
        k3_hit = target in sorted_indices[:min(3, len(sorted_indices))]
        
        layer_results[li] = {"k1": k1_hit, "k3": k3_hit, "score": score,
                             "rank": int(np.where(sorted_indices == target)[0][0]) + 1 if target in sorted_indices else -1}
    
    return layer_results


def nearest_noun_baseline(head_pos, dep_pos, tokens, rel_type, seq_len):
    """强baseline: 找最近noun/verb"""
    # 简化POS推断
    def is_verb(t):
        t = t.lower().strip()
        return t.endswith("ed") or t.endswith("s") or t.endswith("ing") or t in ["was","is","are","were","had","has"]
    
    def is_noun(t):
        t = t.lower().strip()
        if t in ["the","a","an"]: return False
        if is_verb(t): return False
        if t in ["that","who","which","by","and","or"]: return False
        return True  # 默认是名词
    
    if rel_type == "nsubj":
        # 名词的head=右边最近verb
        for j in range(dep_pos + 1, seq_len):
            if is_verb(tokens[j]):
                return j
        return dep_pos + 1
    
    elif rel_type == "dobj":
        # 名词的head=左边最近verb
        for j in range(dep_pos - 1, 0, -1):
            if is_verb(tokens[j]):
                return j
        return dep_pos - 1
    
    elif rel_type == "nsubj_pass":
        # 被动主语: 同nsubj
        for j in range(dep_pos + 1, seq_len):
            if is_verb(tokens[j]):
                return j
        return dep_pos + 1
    
    elif rel_type == "det":
        return dep_pos + 1
    
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek7b", choices=["deepseek7b","glm4","qwen3"])
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase 54C-v2: Strict Complex Sentence Verification")
    print(f"Model={args.model}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"  Layers: {info.n_layers}, d_model: {info.d_model}")
    
    # ============================================================
    # 实验1: Per-layer UAS分析 (k=1严格)
    # ============================================================
    print("\n\n" + "="*70)
    print("Experiment 1: Per-Layer UAS (STRICT k=1)")
    print("="*70)
    
    all_groups = {
        "simple_svo": SIMPLE_SVO,
        "anomalous_svo": ANOMALOUS_SVO,
        "center_embed": CENTER_EMBED,
        "passive": PASSIVE,
        "rel_clause": REL_CLAUSE,
        "multi_clause": MULTI_CLAUSE,
    }
    
    # 收集per-layer数据
    group_layer_k1 = defaultdict(lambda: defaultdict(list))  # group -> layer -> [hit/miss]
    group_layer_k3 = defaultdict(lambda: defaultdict(list))
    
    for group_name, sentences in all_groups.items():
        for sent_data in sentences:
            sentence = sent_data["sentence"]
            dep_list = sent_data["deps"]
            
            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            tokens = [safe_decode(tokenizer, tid) for tid in inputs["input_ids"][0].cpu().numpy()]
            seq_len = len(tokens)
            
            attn_dict = get_attention_weights(model, tokenizer, sentence, device)
            if attn_dict is None:
                continue
            
            for head_pos, dep_pos, rel_type in dep_list:
                if head_pos >= seq_len or dep_pos >= seq_len or head_pos == dep_pos:
                    continue
                
                layer_res = evaluate_attn_per_layer(attn_dict, head_pos, dep_pos, seq_len)
                
                for li, res in layer_res.items():
                    group_layer_k1[group_name][li].append(float(res["k1"]))
                    group_layer_k3[group_name][li].append(float(res["k3"]))
            
            del attn_dict
            gc.collect()
    
    # 打印per-layer UAS (关键层)
    key_layers = [0, 1, 2, 3, 5, 10, 15, 20, info.n_layers-1]
    key_layers = [l for l in key_layers if l < info.n_layers]
    
    print(f"\n{'Group':<18}", end="")
    for l in key_layers:
        print(f" {'L'+str(l):>5}", end="")
    print()
    print("-" * 80)
    
    for group_name in ["simple_svo", "anomalous_svo", "center_embed", "passive", "rel_clause"]:
        print(f"  {group_name:<16}", end="")
        for l in key_layers:
            vals = group_layer_k1[group_name].get(l, [])
            rate = np.mean(vals) if vals else float('nan')
            print(f" {rate:>5.2f}", end="")
        print()
    
    # ★★★ 关键对比: Simple vs Complex at each layer
    print("\n\n★★★ KEY: Simple SVO vs Complex (k=1 strict UAS) ★★★")
    for l in key_layers:
        simple_vals = group_layer_k1["simple_svo"].get(l, [])
        center_vals = group_layer_k1["center_embed"].get(l, [])
        passive_vals = group_layer_k1["passive"].get(l, [])
        
        s_rate = np.mean(simple_vals) if simple_vals else float('nan')
        c_rate = np.mean(center_vals) if center_vals else float('nan')
        p_rate = np.mean(passive_vals) if passive_vals else float('nan')
        
        print(f"  Layer {l:>2}: Simple={s_rate:.2f}  CenterEmbed={c_rate:.2f}  Passive={p_rate:.2f}")
    
    # ============================================================
    # 实验2: 强Baseline对抗 (复杂句)
    # ============================================================
    print("\n\n" + "="*70)
    print("Experiment 2: Strong Baselines on Complex Sentences")
    print("="*70)
    
    all_test_sents = []
    for group_name, sentences in all_groups.items():
        for sent_data in sentences:
            sent_data["group"] = group_name
            all_test_sents.append(sent_data)
    
    baseline_results = defaultdict(lambda: defaultdict(list))  # method -> group -> [hit/miss]
    
    for sent_data in all_test_sents:
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        group_name = sent_data["group"]
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        tokens = [safe_decode(tokenizer, tid) for tid in inputs["input_ids"][0].cpu().numpy()]
        seq_len = len(tokens)
        
        attn_dict = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos >= seq_len or dep_pos >= seq_len or head_pos == dep_pos:
                continue
            
            # Baseline 1: nearest_noun
            pred_head = nearest_noun_baseline(head_pos, dep_pos, tokens, rel_type, seq_len)
            baseline_results["nearest_noun"][group_name].append(float(pred_head == head_pos))
            
            # Baseline 2: left_attach
            baseline_results["left_attach"][group_name].append(float(max(1, dep_pos-1) == head_pos))
            
            # Baseline 3: distance_decay
            tau = 2.0
            scores = np.array([np.exp(-abs(dep_pos-j)/tau) if j != dep_pos else -np.inf 
                              for j in range(1, seq_len)])
            pred_head = np.argmax(scores) + 1
            baseline_results["distance_decay"][group_name].append(float(pred_head == head_pos))
            
            # Baseline 4: random
            choices = [j for j in range(1, seq_len) if j != dep_pos]
            if choices:
                pred_head = np.random.choice(choices)
                baseline_results["random"][group_name].append(float(pred_head == head_pos))
            
            # Attention: best layer (k=1 strict)
            layer_res = evaluate_attn_per_layer(attn_dict, head_pos, dep_pos, seq_len)
            # 取所有层中最好的
            any_layer_hit = any(res["k1"] for res in layer_res.values())
            baseline_results["attn_best_layer_k1"][group_name].append(float(any_layer_hit))
            
            # Attention: specific layers (L3, L10)
            l3_hit = layer_res.get(3, {}).get("k1", False)
            l10_hit = layer_res.get(10, {}).get("k1", False)
            baseline_results["attn_L3_k1"][group_name].append(float(l3_hit))
            baseline_results["attn_L10_k1"][group_name].append(float(l10_hit))
        
        del attn_dict
        gc.collect()
    
    # 打印结果
    print(f"\n{'Method':<25} {'Simple':<10} {'Anom':<10} {'Center':<10} {'Passive':<10} {'RelCl':<10}")
    print("-" * 75)
    
    for method in ["random", "left_attach", "distance_decay", "nearest_noun",
                   "attn_L3_k1", "attn_L10_k1", "attn_best_layer_k1"]:
        rates = {}
        for g in ["simple_svo", "anomalous_svo", "center_embed", "passive", "rel_clause"]:
            vals = baseline_results[method].get(g, [])
            rates[g] = np.mean(vals) if vals else float('nan')
        
        marker = " ★" if "attn" in method else ""
        print(f"  {method:<23} {rates.get('simple_svo',0):<10.3f} {rates.get('anomalous_svo',0):<10.3f} {rates.get('center_embed',0):<10.3f} {rates.get('passive',0):<10.3f} {rates.get('rel_clause',0):<10.3f}{marker}")
    
    # ============================================================
    # 实验3: Swap句精细分析 (语法vs语义)
    # ============================================================
    print("\n\n" + "="*70)
    print("Experiment 3: Role Swap Analysis (Syntax vs Semantics)")
    print("="*70)
    
    for sentence, dep_list, note in SWAP_PAIRS:
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        tokens = [safe_decode(tokenizer, tid) for tid in inputs["input_ids"][0].cpu().numpy()]
        seq_len = len(tokens)
        
        attn_dict = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        print(f"\n  {sentence} ({note})")
        print(f"  Tokens: {tokens}")
        
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos >= seq_len or dep_pos >= seq_len:
                continue
            
            dep_token = tokens[dep_pos]
            head_token = tokens[head_pos]
            
            layer_res = evaluate_attn_per_layer(attn_dict, head_pos, dep_pos, seq_len)
            
            # 找最佳层
            best_k1_layer = -1
            best_k1_score = 0
            for li, res in layer_res.items():
                if res["k1"] and res["score"] > best_k1_score:
                    best_k1_layer = li
                    best_k1_score = res["score"]
            
            # 打印关键层的attention分数
            key_l = [0, 3, 6, 10, 15, 20]
            key_l = [l for l in key_l if l in layer_res]
            
            print(f"  {dep_token}→{head_token} ({rel_type}): Best L{best_k1_layer} (score={best_k1_score:.3f})")
            for l in key_l[:4]:
                res = layer_res[l]
                print(f"    L{l}: k1={'✓' if res['k1'] else '✗'} rank={res['rank']} score={res['score']:.3f}")
        
        del attn_dict
        gc.collect()
    
    # ============================================================
    # 实验4: 中心嵌入的head分析 — ★★★最关键★★★
    # ============================================================
    print("\n\n" + "="*70)
    print("Experiment 4: Center Embedding Head Analysis")
    print("="*70)
    
    for sent_data in CENTER_EMBED:
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        tokens = [safe_decode(tokenizer, tid) for tid in inputs["input_ids"][0].cpu().numpy()]
        seq_len = len(tokens)
        
        attn_dict = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        print(f"\n  Sentence: {sentence}")
        print(f"  Tokens: {[(i,t) for i,t in enumerate(tokens)]}")
        
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos >= seq_len or dep_pos >= seq_len:
                continue
            
            dep_token = tokens[dep_pos]
            head_token = tokens[head_pos]
            distance = abs(head_pos - dep_pos)
            
            # 找每个层中, dep在head的attention中的排名
            print(f"\n  ★ {dep_token}({dep_pos}) → {head_token}({head_pos}) [{rel_type}] distance={distance}")
            
            # Baseline: nearest_noun能找到吗?
            pred_head = nearest_noun_baseline(head_pos, dep_pos, tokens, rel_type, seq_len)
            baseline_hit = pred_head == head_pos
            pred_token = tokens[pred_head] if pred_head < len(tokens) else "?"
            print(f"    Nearest noun baseline: predict {pred_token}({pred_head}) {'✓' if baseline_hit else '✗'}")
            
            # Attention: per-head分析 (采样关键层)
            for li in [0, 3, 6, 10, 15]:
                if li not in attn_dict:
                    continue
                A = attn_dict[li]  # [n_heads, seq, seq]
                n_heads = A.shape[0]
                
                # 对每个head, 找dep对head的attention
                head_scores = []
                for h_idx in range(n_heads):
                    if head_pos < dep_pos:
                        if dep_pos < A.shape[1] and head_pos < A.shape[2]:
                            score = float(A[h_idx, dep_pos, head_pos])
                        else:
                            score = 0.0
                    else:
                        if head_pos < A.shape[1] and dep_pos < A.shape[2]:
                            score = float(A[h_idx, head_pos, dep_pos])
                        else:
                            score = 0.0
                    head_scores.append((h_idx, score))
                
                head_scores.sort(key=lambda x: x[1], reverse=True)
                top3 = head_scores[:3]
                
                # 检查top head的attention分布: head是否在top-k
                best_h = top3[0][0]
                if head_pos < dep_pos:
                    row = A[best_h, dep_pos, 1:dep_pos]
                else:
                    row = A[best_h, head_pos, 1:head_pos]
                
                target = head_pos - 1 if head_pos < dep_pos else dep_pos - 1
                if len(row) > 0 and 0 <= target < len(row):
                    rank = int(np.where(np.argsort(row)[::-1] == target)[0][0]) + 1
                    top3_tokens = [(idx+1, tokens[idx+1] if idx+1 < len(tokens) else "?") for idx in np.argsort(row)[::-1][:3]]
                else:
                    rank = -1
                    top3_tokens = []
                
                k1_hit = rank == 1
                print(f"    L{li} H{best_h}: score={top3[0][1]:.3f} rank={rank} k1={'✓' if k1_hit else '✗'} top3={top3_tokens}")
        
        del attn_dict
        gc.collect()
    
    # ============================================================
    # 最终汇总
    # ============================================================
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # 对所有关系计算overall UAS
    for method in ["nearest_noun", "attn_best_layer_k1", "attn_L3_k1"]:
        all_vals = []
        for g in baseline_results[method]:
            all_vals.extend(baseline_results[method][g])
        overall = np.mean(all_vals) if all_vals else 0
        print(f"  {method}: Overall UAS = {overall:.3f}")
    
    # 中心嵌入单独统计
    for method in ["nearest_noun", "attn_best_layer_k1", "attn_L3_k1"]:
        vals = baseline_results[method].get("center_embed", [])
        rate = np.mean(vals) if vals else 0
        print(f"  {method}: Center Embed UAS = {rate:.3f}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nPhase 54C-v2 Complete!")


if __name__ == "__main__":
    main()
