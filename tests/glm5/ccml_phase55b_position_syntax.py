"""
Phase 55B: Position vs Syntax Conflict Test
==========================================

★★★ 最关键实验: 构造"位置和语法冲突"的句子 ★★★

核心逻辑:
- 英语SVO: 位置2 = nsubj, 位置5 = dobj → 位置和语法一致, 无法区分
- OSV/倒装: 位置2 = dobj(语义), 位置5 = nsubj(语义) → 位置和语法冲突
- 如果attention指向位置 → 位置模板
- 如果attention指向语法角色 → 语法理解

测试句子:
1. "The cat, the dog chased" (OSV) → cat(pos2)=dobj, dog(pos4)=nsubj
2. "What did the cat chase" (wh-fronting) → what(pos1)=dobj of chase(pos4)
3. "By the cat, the mouse was chased" (fronted PP) → cat(pos3)=oblique
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np, gc
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model_utils import load_model, get_model_info, release_model, safe_decode


# ============================================================
# ★ 位置-语法冲突测试句
# ============================================================

# 所有位置已验证为1:1 word:token

# ---- 主动SVO (位置=语法, baseline) ----
ACTIVE_SVO = [
    {
        "name": "Active SVO: pos2=nsubj, pos5=dobj",
        "sentence": "The cat chased the mouse",
        "deps": [(3,2,"nsubj"), (3,5,"dobj")],
        "position_syntax_align": True,  # 位置和语法对齐
    },
    {
        "name": "Active SVO 2",
        "sentence": "The dog bit the man",
        "deps": [(3,2,"nsubj"), (3,5,"dobj")],
        "position_syntax_align": True,
    },
]

# ---- OSV/倒装 (位置≠语法, 冲突!) ----
# "The mouse, the cat chased" → mouse(pos2)是dobj, cat(pos4)是nsubj
# ★ 关键: 位置2通常是nsubj, 但这里位置2是dobj!
OSV_SENTENCES = [
    {
        "name": "OSV: pos2=OBJ, pos4=SUBJ (CONFLICT!)",
        "sentence": "The mouse the cat chased",
        "deps": [(4,3,"nsubj"), (4,2,"dobj")],  # chased←cat(nsubj), chased←mouse(dobj)
        "position_syntax_align": False,
    },
    {
        "name": "OSV 2: pos2=OBJ, pos4=SUBJ",
        "sentence": "The man the dog bit",
        "deps": [(4,3,"nsubj"), (4,2,"dobj")],
        "position_syntax_align": False,
    },
    # 非常关键的对比: 语义相同, 结构不同
    {
        "name": "Active SVO (same semantics as OSV 1)",
        "sentence": "The cat chased the mouse",
        "deps": [(3,2,"nsubj"), (3,5,"dobj")],
        "position_syntax_align": True,
    },
]

# ---- Wh-fronting (位置1=宾语, 非主语!) ----
WH_FRONT = [
    {
        "name": "Wh-front: what(pos1)=dobj of chase(pos4)",
        "sentence": "What did the cat chase",
        "deps": [(4,3,"nsubj"), (4,1,"dobj")],  # chase←cat(nsubj), chase←what(dobj)
        "position_syntax_align": False,  # 位置1是dobj!
    },
]

# ---- 被动 vs 主动对比 ----
VOICE_CONTRAST = [
    {
        "name": "Passive: mouse(pos2)=nsubj_pass of chased(pos4)",
        "sentence": "The mouse was chased by the cat",
        "deps": [(4,2,"nsubj_pass")],
        "position_syntax_align": True,  # pos2仍然是syntactic subject
    },
    {
        "name": "Active: mouse(pos2)=nsubj of chased(pos3)",
        "sentence": "The mouse chased the cat",
        "deps": [(3,2,"nsubj")],
        "position_syntax_align": True,  # pos2仍然是syntactic subject
    },
]

# ---- 中心嵌入 (长距离, 位置≠简单模板) ----
CENTER_EMBED = [
    {
        "name": "Center-embed: cat(pos2)=nsubj of ran(pos7), dist=5",
        "sentence": "The cat that the dog chased ran",
        "deps": [(7,2,"nsubj"), (6,5,"nsubj"), (6,2,"dobj")],
        "position_syntax_align": False,  # 位置模板无法处理
    },
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


def analyze_per_head(attn_dict, head_pos, dep_pos, seq_len, n_layers):
    """对每个层的每个head, 检查dep是否在head的top-k attention中"""
    results = []
    for li in range(min(n_layers, 28)):
        if li not in attn_dict:
            continue
        A = attn_dict[li]
        n_heads = A.shape[0]
        
        if head_pos < dep_pos:
            rows = A[:, dep_pos, 1:dep_pos]  # [n_heads, dep_pos-1]
            target = head_pos - 1
        else:
            rows = A[:, head_pos, 1:head_pos]
            target = dep_pos - 1
        
        if len(rows) == 0 or target < 0 or target >= rows.shape[1]:
            continue
        
        for h in range(n_heads):
            score = float(rows[h, target])
            sorted_idx = np.argsort(rows[h])[::-1]
            rank = int(np.where(sorted_idx == target)[0][0]) + 1 if target in sorted_idx else -1
            top1_token_idx = int(sorted_idx[0]) + 1  # 1-indexed
            results.append({
                "layer": li, "head": h,
                "score": score, "rank": rank,
                "k1": rank == 1, "k3": rank <= 3,
                "top1_idx": top1_token_idx,
            })
    
    return results


def main():
    model, tokenizer, device = load_model("deepseek7b")
    info = get_model_info(model, "deepseek7b")
    print(f"Layers: {info.n_layers}, d_model: {info.d_model}")
    
    all_test_groups = {
        "active_svo": ACTIVE_SVO,
        "osv": OSV_SENTENCES,
        "wh_front": WH_FRONT,
        "voice_contrast": VOICE_CONTRAST,
        "center_embed": CENTER_EMBED,
    }
    
    # ============================================================
    # 核心: Position vs Syntax 冲突测试
    # ============================================================
    print("\n" + "="*70)
    print("★★★ POSITION vs SYNTAX CONFLICT TEST ★★★")
    print("="*70)
    
    # 对每个测试组, 统计: 位置预测 vs 语法预测的一致性
    position_correct = 0
    syntax_correct = 0
    total_deps = 0
    
    # 详细记录: 对每个dep, 记录top-1 attention指向谁
    conflict_results = []
    
    for group_name, sentences in all_test_groups.items():
        print(f"\n--- {group_name} ---")
        
        for item in sentences:
            sentence = item["sentence"]
            deps = item["deps"]
            aligned = item["position_syntax_align"]
            
            attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
            if attn_dict is None:
                continue
            
            seq_len = len(tokens)
            
            print(f"\n  [{item['name']}]")
            print(f"  Tokens: {[(i,t) for i,t in enumerate(tokens)]}")
            
            for head_pos, dep_pos, rel_type in deps:
                if head_pos >= seq_len or dep_pos >= seq_len:
                    continue
                
                dep_token = tokens[dep_pos]
                head_token = tokens[head_pos]
                distance = abs(head_pos - dep_pos)
                
                # 找最佳per-head结果
                results = analyze_per_head(attn_dict, head_pos, dep_pos, seq_len, info.n_layers)
                
                # 找rank=1的head
                best_k1_results = [r for r in results if r["k1"]]
                
                # 统计: 多少head指向正确位置 vs 指向"位置模板预测"位置
                # 位置模板预测: nsubj→左边最近noun, dobj→右边最近noun, 或基于词类序列
                
                # 简化的位置预测: 
                # 对于SVO: nsubj = pos(verb-1), dobj = pos(verb+2)
                # 对于OSV: 位置预测仍是 pos(verb-1) = nsubj
                # 但语法上 OSV 的 pos2 = dobj
                
                if rel_type in ["nsubj", "nsubj_pass"]:
                    # 位置模板预测: verb左边的noun = nsubj
                    # 在SVO中: 位置verb-1就是nsubj (正确)
                    # 在OSV中: 位置verb-1是nsubj (语法正确, 但pos2是dobj)
                    pos_template_target = head_pos - 1  # verb左边的词
                elif rel_type == "dobj":
                    # 位置模板预测: verb右边的noun = dobj
                    pos_template_target = head_pos + 1  # verb右边的词
                else:
                    pos_template_target = -1
                
                # 在所有head中, 统计指向语法正确位置 vs 位置模板位置的比例
                syntax_heads = sum(1 for r in results if r["k1"])  # 指向语法正确
                pos_template_heads = sum(1 for r in results 
                                        if r["top1_idx"] == pos_template_target and r["top1_idx"] != head_pos - (1 if head_pos < dep_pos else -1))
                
                # 找最好的head
                if best_k1_results:
                    best = min(best_k1_results, key=lambda x: x["layer"])
                    best_info = f"L{best['layer']} H{best['head']} rank=1 ✓"
                else:
                    best_k3 = [r for r in results if r["k3"]]
                    if best_k3:
                        best = min(best_k3, key=lambda x: x["rank"])
                        best_info = f"L{best['layer']} H{best['head']} rank={best['rank']} (k3✓)"
                    else:
                        best_info = "no k3 hit"
                
                # ★ 关键: 在最佳层中, top-1指向谁?
                # 对关键层检查
                key_layer_results = {}
                for li in [0, 1, 3, 6, 10, 15, 20]:
                    layer_results = [r for r in results if r["layer"] == li]
                    if layer_results:
                        best_h = max(layer_results, key=lambda x: x["score"])
                        key_layer_results[li] = best_h
                
                print(f"  {dep_token}({dep_pos})→{head_token}({head_pos}) [{rel_type}] dist={distance} align={aligned}")
                print(f"    Best: {best_info}")
                print(f"    Syntax-correct heads: {syntax_heads}/{len(results)}")
                
                for li in [0, 3, 10, 15]:
                    if li in key_layer_results:
                        r = key_layer_results[li]
                        top1_tok = tokens[r["top1_idx"]] if r["top1_idx"] < len(tokens) else "?"
                        print(f"    L{li} best: H{r['head']} rank={r['rank']} top1→{top1_tok}({r['top1_idx']}) "
                              f"{'✓' if r['k1'] else '✗'}")
                
                # 记录结果
                conflict_results.append({
                    "group": group_name,
                    "dep_token": dep_token,
                    "head_token": head_token,
                    "rel_type": rel_type,
                    "distance": distance,
                    "aligned": aligned,
                    "syntax_heads": syntax_heads,
                    "total_heads": len(results),
                    "best_rank": min((r["rank"] for r in results), default=999),
                })
                
                total_deps += 1
                if aligned:
                    position_correct += 1  # 位置和语法对齐, 两者预测相同
                    syntax_correct += 1
                else:
                    # 位置和语法冲突
                    # 检查best head是否指向语法正确位置
                    if best_k1_results:
                        syntax_correct += 1  # 如果有head指向正确位置, 说明语法追踪存在
            
            del attn_dict, inputs
            gc.collect()
    
    # ============================================================
    # ★★★ 最终判断
    # ============================================================
    print("\n\n" + "="*70)
    print("★★★ VERDICT: Position Template vs Syntax ★★★")
    print("="*70)
    
    # 分组统计
    aligned_cases = [r for r in conflict_results if r["aligned"]]
    conflict_cases = [r for r in conflict_results if not r["aligned"]]
    
    print(f"\nPosition-Syntax Aligned cases: {len(aligned_cases)}")
    print(f"  Best rank = 1 (any head): {sum(1 for r in aligned_cases if r['best_rank'] == 1)}/{len(aligned_cases)}")
    
    print(f"\nPosition-Syntax CONFLICT cases: {len(conflict_cases)}")
    print(f"  Best rank = 1 (any head): {sum(1 for r in conflict_cases if r['best_rank'] == 1)}/{len(conflict_cases)}")
    
    # ★ 最关键的判断
    conflict_k1_rate = sum(1 for r in conflict_cases if r['best_rank'] == 1) / len(conflict_cases) if conflict_cases else 0
    aligned_k1_rate = sum(1 for r in aligned_cases if r['best_rank'] == 1) / len(aligned_cases) if aligned_cases else 0
    
    print(f"\n★★★ CONFLICT case k1 rate: {conflict_k1_rate:.3f}")
    print(f"★★★ ALIGNED case k1 rate: {aligned_k1_rate:.3f}")
    
    if conflict_k1_rate > 0.5:
        print("\n★★★★★ ATTENTION TRACKS SYNTAX, NOT JUST POSITION ★★★★★")
        print("  → Even when position and syntax conflict, attention finds the correct syntactic relation")
    elif conflict_k1_rate > 0:
        print("\n★★★ PARTIAL: Some syntax tracking, but position still dominates")
    else:
        print("\n✗ ATTENTION IS PURELY POSITION-DRIVEN")
    
    # 详细打印conflict cases
    print("\n\n--- Conflict Case Details ---")
    for r in conflict_cases:
        marker = "✓" if r['best_rank'] == 1 else "✗"
        print(f"  {r['group']}: {r['dep_token']}→{r['head_token']} [{r['rel_type']}] "
              f"dist={r['distance']} best_rank={r['best_rank']} {marker}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("\nPhase 55B Complete!")


if __name__ == "__main__":
    main()
