"""
Phase 54: 语法 vs 语义 分离 + 强Baseline对抗 + 复杂句验证 + 分布式Head组合
========================================================================

Phase 53剩余4个硬伤:
  A. 90% UAS可能是"弱语法+强统计"混合 → 语义剥离测试
  B. dobj baseline太弱 → 加入强baseline (verb→最近右noun)
  C. Layer 3集中可能是假象 → 复杂句验证
  D. attention≠edge → Head组合/分布式circuit

4个实验:
  54A: 语义剥离测试 — Chomsky句 + 语义异常但语法合法句
       ★ 关键: 排除"语义共现"替代解释
  54B: 强Baseline对抗 — verb→nearest right noun, POS-pattern等
       ★ 关键: 排除"弱启发式"替代解释
  54C: 复杂句验证 — 从句/被动/嵌套/中心嵌入
       ★ 关键: 排除"简单SVO"数据偏差
  54D: 分布式Head组合 — top-k heads ensemble vs 单head
       ★ 关键: 语法=distributed circuit而非单边
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import torch
import numpy as np
import gc
import time
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model_utils import (load_model, get_layers, get_model_info, release_model,
                          safe_decode, get_W_U)


# ============================================================
# ★★★ 数据集1: 语义剥离测试 (Chomsky + 语义异常句)
# ============================================================
# 核心逻辑: 如果attention在语义异常但语法合法的句子中
#           仍然正确找到nsubj/dobj → 说明它学的是语法不是语义

SEMANTIC_STRIPPING_SENTENCES = [
    # --- Group A: Chomsky经典 ---
    {
        "id": "chomsky_1",
        "sentence": "Colorless green ideas sleep furiously",
        "deps": [
            (3, 2, "amod"),   # ideas ← green
            (3, 1, "amod"),   # ideas ← colorless
            (4, 3, "nsubj"),  # sleep ← ideas
            (4, 5, "advmod"), # sleep ← furiously
        ],
        "note": "Chomsky经典: 完全无意义但语法正确, nsubj=ideas"
    },
    
    # --- Group B: 语义异常但语法合法 (主语) ---
    {
        "id": "anomalous_nsubj_1",
        "sentence": "The rock ate the concept",
        "deps": [
            (2, 1, "det"),    # rock ← The
            (3, 2, "nsubj"),  # ate ← rock (语义异常!)
            (5, 4, "det"),    # concept ← the
            (3, 5, "dobj"),   # ate ← concept
        ],
        "note": "rock吃concept: 语义异常, 但语法上rock=nsubj"
    },
    {
        "id": "anomalous_nsubj_2",
        "sentence": "The idea chased the theory",
        "deps": [
            (2, 1, "det"),
            (3, 2, "nsubj"),  # chased ← idea (抽象词作主语)
            (5, 4, "det"),
            (3, 5, "dobj"),
        ],
        "note": "idea追theory: 抽象词SVO, 测试是否仍能分nsubj/dobj"
    },
    {
        "id": "anomalous_nsubj_3",
        "sentence": "The mountain drank the silence",
        "deps": [
            (2, 1, "det"),
            (3, 2, "nsubj"),  # drank ← mountain (语义不可能!)
            (5, 4, "det"),
            (3, 5, "dobj"),   # drank ← silence
        ],
        "note": "山喝沉默: 最极端语义异常"
    },
    
    # --- Group C: 角色互换对照 (关键!) ---
    {
        "id": "swap_normal_1",
        "sentence": "The cat ate the fish",
        "deps": [
            (2, 1, "det"),
            (3, 2, "nsubj"),  # ate ← cat
            (5, 4, "det"),
            (3, 5, "dobj"),   # ate ← fish
        ],
        "note": "正常语义: cat=nsubj, fish=dobj"
    },
    {
        "id": "swap_anomalous_1",
        "sentence": "The fish ate the cat",
        "deps": [
            (2, 1, "det"),
            (3, 2, "nsubj"),  # ate ← fish (语义异常! 鱼吃猫)
            (5, 4, "det"),
            (3, 5, "dobj"),   # ate ← cat
        ],
        "note": "语义异常: fish=nsubj, cat=dobj → 如果attention跟fish而非cat → 学的是语法不是语义"
    },
    {
        "id": "swap_normal_2",
        "sentence": "The dog bit the man",
        "deps": [
            (2, 1, "det"),
            (3, 2, "nsubj"),
            (5, 4, "det"),
            (3, 5, "dobj"),
        ],
        "note": "正常: dog=nsubj, man=dobj"
    },
    {
        "id": "swap_anomalous_2",
        "sentence": "The man bit the dog",
        "deps": [
            (2, 1, "det"),
            (3, 2, "nsubj"),  # bit ← man (可以但罕见)
            (5, 4, "det"),
            (3, 5, "dobj"),
        ],
        "note": "反转: man=nsubj, dog=dobj"
    },
    
    # --- Group D: 无生命主语 ---
    {
        "id": "inanimate_1",
        "sentence": "The storm destroyed the building",
        "deps": [
            (2, 1, "det"),
            (3, 2, "nsubj"),
            (5, 4, "det"),
            (3, 5, "dobj"),
        ],
        "note": "无生命主语: storm=nsubj"
    },
    {
        "id": "inanimate_2",
        "sentence": "The building destroyed the storm",
        "deps": [
            (2, 1, "det"),
            (3, 2, "nsubj"),  # 语义异常
            (5, 4, "det"),
            (3, 5, "dobj"),
        ],
        "note": "反转: building=nsubj (异常)"
    },
    
    # --- Group E: 完全无关词组合 ---
    {
        "id": "nonsense_1",
        "sentence": "The zimble flarbed the glonk",
        "deps": [
            (2, 1, "det"),
            (3, 2, "nsubj"),  # flarbed ← zimble (伪词!)
            (5, 4, "det"),
            (3, 5, "dobj"),   # flarbed ← glonk
        ],
        "note": "★ 伪词! 完全排除语义, 只剩语法骨架"
    },
    {
        "id": "nonsense_2",
        "sentence": "A blorp crunted the snark",
        "deps": [
            (2, 1, "det"),
            (3, 2, "nsubj"),
            (5, 4, "det"),
            (3, 5, "dobj"),
        ],
        "note": "★ 伪词2: 测试POS推断(词形暗示词类)"
    },
]


# ============================================================
# 数据集2: 复杂句验证 (从句/被动/嵌套/中心嵌入)
# ============================================================

COMPLEX_SENTENCES = [
    # --- 中心嵌入 (最难) ---
    {
        "id": "center_embed_1",
        "sentence": "The cat that the dog chased ran",
        "deps": [
            (2, 1, "det"),     # cat ← The
            (7, 2, "nsubj"),   # ran ← cat (距离5!)
            (5, 4, "det"),     # dog ← the
            (6, 5, "nsubj"),   # chased ← dog
            (6, 2, "dobj"),    # chased ← cat
            (7, 6, "acl"),     # ran ← chased (关系从句)
        ],
        "note": "中心嵌入: cat同时是ran的nsubj和chased的dobj"
    },
    {
        "id": "center_embed_2",
        "sentence": "The book that John said Mary likes is good",
        "deps": [
            (2, 1, "det"),     # book ← The
            (9, 2, "nsubj"),   # is ← book (距离7!)
            (4, 3, "nsubj"),   # said ← John
            (4, 2, "dobj"),    # said ← book
            (6, 5, "nsubj"),   # likes ← Mary
            (6, 2, "dobj"),    # likes ← book
            (9, 8, "acomp"),   # is ← good
        ],
        "note": "双重嵌套: book同时是said/likes的dobj和is的nsubj"
    },
    
    # --- 被动句 ---
    {
        "id": "passive_1",
        "sentence": "The mouse was chased by the cat",
        "deps": [
            (2, 1, "det"),     # mouse ← The
            (3, 2, "nsubj_pass"), # was ← mouse (被动主语!)
            (3, 4, "aux"),     # was ← chased
            (4, 5, "agent"),   # chased ← by
            (7, 6, "det"),     # cat ← the
            (5, 7, "pobj"),    # by ← cat
        ],
        "note": "被动句: mouse是语法主语但语义宾语"
    },
    {
        "id": "passive_2",
        "sentence": "The car was repaired by the mechanic",
        "deps": [
            (2, 1, "det"),
            (3, 2, "nsubj_pass"),
            (3, 4, "aux"),
            (4, 5, "agent"),
            (7, 6, "det"),
            (5, 7, "pobj"),
        ],
        "note": "被动句2"
    },
    
    # --- 关系从句 ---
    {
        "id": "relclause_1",
        "sentence": "The woman who saw the man left",
        "deps": [
            (2, 1, "det"),
            (7, 2, "nsubj"),   # left ← woman (距离5!)
            (4, 3, "nsubj"),   # saw ← who
            (4, 6, "dobj"),    # saw ← man
            (6, 5, "det"),
        ],
        "note": "关系从句: who是saw的nsubj"
    },
    {
        "id": "relclause_2",
        "sentence": "The man who the woman saw left",
        "deps": [
            (2, 1, "det"),
            (7, 2, "nsubj"),   # left ← man
            (5, 4, "nsubj"),   # saw ← woman
            (4, 2, "dobj"),    # saw ← man (man是saw的dobj)
            (7, 5, "acl"),     # left ← saw
        ],
        "note": "关系从句: man是saw的dobj"
    },
    
    # --- 并列结构 ---
    {
        "id": "coordination_1",
        "sentence": "The cat and the dog played together",
        "deps": [
            (2, 1, "det"),
            (4, 2, "conj"),    # dog ← cat (并列)
            (4, 3, "cc"),      # and
            (5, 4, "nsubj"),   # played ← dog
            (5, 2, "nsubj"),   # played ← cat
            (5, 6, "advmod"),  # played ← together
        ],
        "note": "并列: cat和dog都是nsubj"
    },
    
    # --- 多从句 ---
    {
        "id": "multi_clause_1",
        "sentence": "John thinks that Mary believes that the cat ran",
        "deps": [
            (2, 1, "nsubj"),   # thinks ← John
            (4, 3, "mark"),    # believes ← that
            (4, 2, "ccomp"),   # thinks ← believes (补语从句)
            (6, 5, "nsubj"),   # believes ← Mary
            (8, 7, "mark"),    # ran ← that
            (8, 4, "ccomp"),   # believes ← ran
            (10, 9, "det"),
            (8, 10, "nsubj"),  # ran ← cat
        ],
        "note": "多级从句嵌套"
    },
]


# ============================================================
# 数据集3: 正常SVO句子 (作为对照组)
# ============================================================

NORMAL_SVO_SENTENCES = [
    {"sentence": "The cat chased the mouse", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(3,5,"dobj")]},
    {"sentence": "The dog bit the man", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(3,5,"dobj")]},
    {"sentence": "The boy ate the apple", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(3,5,"dobj")]},
    {"sentence": "The girl read the book", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(3,5,"dobj")]},
    {"sentence": "The man pushed the door", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(3,5,"dobj")]},
    {"sentence": "The woman drove the car", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(3,5,"dobj")]},
    {"sentence": "The teacher helped the student", "deps": [(2,1,"det"),(3,2,"nsubj"),(6,5,"det"),(3,6,"dobj")]},
    {"sentence": "The doctor cured the patient", "deps": [(2,1,"det"),(3,2,"nsubj"),(6,5,"det"),(3,6,"dobj")]},
]


# ============================================================
# 辅助函数
# ============================================================
def get_attention_weights(model, tokenizer, sentence, device):
    """获取所有层的attention权重"""
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    
    with torch.no_grad():
        try:
            outputs = model(input_ids=inputs["input_ids"], output_attentions=True)
            all_attn = outputs.attentions
        except:
            return None, None
    
    attn_dict = {}
    for li, attn in enumerate(all_attn):
        if attn is not None:
            attn_dict[li] = attn.detach().float().cpu().numpy()[0]  # [n_heads, seq, seq]
    
    tokens = [safe_decode(tokenizer, tid) for tid in inputs["input_ids"][0].cpu().numpy()]
    
    return attn_dict, tokens


def evaluate_single_dep_attn(attn_dict, head_pos, dep_pos, seq_len, k=3):
    """评估单条依赖在attention中的可恢复性 (因果方向, 遍历所有层, 取最佳)"""
    best_hit = False
    best_layer = -1
    best_score = 0.0
    
    for li in sorted(attn_dict.keys()):
        A = attn_dict[li].mean(axis=0)  # mean over heads → [seq, seq]
        
        if head_pos < dep_pos:
            # dep可以attend到head → 检查A[dep, 1:dep]中head是否在top-k
            row = A[dep_pos, 1:dep_pos]
        elif head_pos > dep_pos:
            # head可以attend到dep → 检查A[head, 1:head]中dep是否在top-k
            row = A[head_pos, 1:head_pos]
        else:
            continue
        
        if len(row) == 0:
            continue
        
        # 找target位置
        target = head_pos - 1 if head_pos < dep_pos else dep_pos - 1
        if target < 0 or target >= len(row):
            continue
        
        target_score = row[target]
        k_actual = min(k, len(row))
        top_k_indices = np.argsort(row)[-k_actual:]
        
        if target in top_k_indices:
            if not best_hit:
                best_hit = True
                best_layer = li
            best_score = max(best_score, target_score)
    
    return best_hit, best_layer, best_score


def evaluate_single_dep_attn_per_head(attn_dict, head_pos, dep_pos, seq_len, top_k_heads=5):
    """评估单条依赖在每个head中的分数, 返回top-k heads"""
    head_scores = []  # [(layer, head_idx, score)]
    
    for li in sorted(attn_dict.keys()):
        A = attn_dict[li]  # [n_heads, seq, seq]
        n_heads = A.shape[0]
        
        for h_idx in range(n_heads):
            if head_pos < dep_pos:
                if dep_pos < A.shape[1] and head_pos < A.shape[2]:
                    score = float(A[h_idx, dep_pos, head_pos])
                else:
                    score = 0.0
            elif head_pos > dep_pos:
                if head_pos < A.shape[1] and dep_pos < A.shape[2]:
                    score = float(A[h_idx, head_pos, dep_pos])
                else:
                    score = 0.0
            else:
                score = 0.0
            
            head_scores.append((li, h_idx, score))
    
    # 按score排序
    head_scores.sort(key=lambda x: x[2], reverse=True)
    return head_scores[:top_k_heads]


# ============================================================
# 54A: 语义剥离测试
# ============================================================
def exp_54a_semantic_stripping(model, tokenizer, info, model_name):
    """
    ★★★ 最关键实验: 语法 vs 语义分离 ★★★
    
    核心逻辑:
    - 正常句: "The cat ate the fish" → cat=nsubj, fish=dobj
    - 异常句: "The fish ate the cat" → fish=nsubj, cat=dobj (语义异常!)
    - 伪词句: "The zimble flarbed the glonk" → zimble=nsubj
    
    如果attention在异常句和伪词句中仍正确:
      → 语法, 不是语义共现
    """
    print("\n" + "="*70)
    print("54A: Semantic Stripping Test — 语法 vs 语义分离")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # ---- 测试所有语义剥离句 ----
    print("\n--- Semantic Stripping Results ---")
    print(f"{'ID':<25} {'Rel':<10} {'Attn Hit':<10} {'Best Layer':<12} {'Note'}")
    print("-" * 80)
    
    results_by_group = defaultdict(lambda: defaultdict(list))  # group -> rel -> [hit/miss]
    
    for sent_data in SEMANTIC_STRIPPING_SENTENCES:
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        sid = sent_data["id"]
        note = sent_data.get("note", "")
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
        seq_len = len(tokens)
        
        resolved_deps = []
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos < seq_len and dep_pos < seq_len and head_pos != dep_pos:
                resolved_deps.append((head_pos, dep_pos, rel_type))
        
        if len(resolved_deps) == 0:
            continue
        
        attn_dict, _ = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        # 确定group
        if sid.startswith("chomsky"):
            group = "chomsky"
        elif sid.startswith("anomalous"):
            group = "anomalous"
        elif sid.startswith("swap_normal"):
            group = "swap_normal"
        elif sid.startswith("swap_anomalous"):
            group = "swap_anomalous"
        elif sid.startswith("inanimate"):
            group = "inanimate"
        elif sid.startswith("nonsense"):
            group = "nonsense"
        else:
            group = "other"
        
        for head_pos, dep_pos, rel_type in resolved_deps:
            hit, best_layer, best_score = evaluate_single_dep_attn(
                attn_dict, head_pos, dep_pos, seq_len, k=3)
            
            results_by_group[group][rel_type].append(float(hit))
            
            dep_token = tokens[dep_pos] if dep_pos < len(tokens) else "?"
            head_token = tokens[head_pos] if head_pos < len(tokens) else "?"
            
            print(f"  {sid:<23} {rel_type:<10} {'✓' if hit else '✗':<10} L{best_layer:<10} {dep_token}→{head_token} ({best_score:.3f})")
        
        del attn_dict
        gc.collect()
    
    # ---- 汇总: 按组统计 ----
    print("\n\n--- Summary: Attn UAS by Semantic Group ---")
    print(f"{'Group':<20} {'nsubj':<12} {'dobj':<12} {'Overall':<12} {'N'}")
    print("-" * 60)
    
    for group in ["chomsky", "swap_normal", "swap_anomalous", "anomalous", "inanimate", "nonsense"]:
        group_data = results_by_group[group]
        all_hits = []
        nsubj_hits = group_data.get("nsubj", [])
        dobj_hits = group_data.get("dobj", [])
        all_hits = nsubj_hits + dobj_hits + group_data.get("amod", []) + group_data.get("advmod", [])
        
        nsubj_rate = np.mean(nsubj_hits) if nsubj_hits else float('nan')
        dobj_rate = np.mean(dobj_hits) if dobj_hits else float('nan')
        overall_rate = np.mean(all_hits) if all_hits else float('nan')
        
        n_total = len(all_hits)
        print(f"  {group:<18} {nsubj_rate:<12.3f} {dobj_rate:<12.3f} {overall_rate:<12.3f} {n_total}")
    
    # ---- ★★★ 关键对比: swap_normal vs swap_anomalous ★★★
    print("\n\n★★★ KEY COMPARISON: Normal vs Anomalous Role Swap ★★★")
    normal_nsubj = results_by_group["swap_normal"].get("nsubj", [])
    anomalous_nsubj = results_by_group["swap_anomalous"].get("nsubj", [])
    normal_dobj = results_by_group["swap_normal"].get("dobj", [])
    anomalous_dobj = results_by_group["swap_anomalous"].get("dobj", [])
    
    print(f"  Normal sentences:     nsubj UAS = {np.mean(normal_nsubj):.3f}, dobj UAS = {np.mean(normal_dobj):.3f}")
    print(f"  Anomalous sentences:  nsubj UAS = {np.mean(anomalous_nsubj):.3f}, dobj UAS = {np.mean(anomalous_dobj):.3f}")
    
    if normal_nsubj and anomalous_nsubj:
        diff = abs(np.mean(normal_nsubj) - np.mean(anomalous_nsubj))
        if diff < 0.15:
            print(f"\n  ★★★ CONCLUSION: nsubj difference = {diff:.3f} < 0.15")
            print(f"  → Attention tracks SYNTACTIC role, not semantic plausibility!")
            print(f"  → This is the STRONGEST evidence for syntax-over-semantics")
        else:
            print(f"\n  ⚠️ nsubj difference = {diff:.3f} ≥ 0.15")
            print(f"  → Attention may be partially influenced by semantics")
    
    # ---- 伪词句测试 ----
    nonsense_data = results_by_group["nonsense"]
    nonsense_all = []
    for v in nonsense_data.values():
        nonsense_all.extend(v)
    if nonsense_all:
        nonsense_rate = np.mean(nonsense_all)
        print(f"\n  Nonsense words: Overall UAS = {nonsense_rate:.3f}")
        if nonsense_rate > 0.5:
            print(f"  ★★★ Attention works even with COMPLETELY MEANINGLESS words!")
            print(f"  → This proves syntax is independent of word semantics")
        else:
            print(f"  ⚠️ Nonsense UAS low → word form/POS may be needed")
    
    return results_by_group


# ============================================================
# 54B: 强Baseline对抗
# ============================================================
def exp_54b_strong_baselines(model, tokenizer, info, model_name):
    """
    ★★★ 加入更强的baseline, 排除"弱启发式"解释 ★★★
    
    新增baseline:
    1. nearest_right_noun: verb → 右边最近noun (dobj启发式)
    2. nearest_left_verb: noun → 左边最近verb (nsubj启发式)
    3. pos_pattern: 基于POS的完整启发式规则
    4. freq_cooccurrence: 高频共现模式 (The+noun)
    """
    print("\n" + "="*70)
    print("54B: Strong Baseline Adversarial — 更强baseline对抗")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # 合并所有句子
    all_sents = NORMAL_SVO_SENTENCES + [
        {"sentence": s["sentence"], "deps": s["deps"]} 
        for s in SEMANTIC_STRIPPING_SENTENCES[:8]  # 前8个有意义词的句子
    ]
    
    print(f"\n  Total test sentences: {len(all_sents)}")
    
    # ---- 简单POS推断 (基于token特征) ----
    # 由于我们没有真正的POS tagger, 我们用启发式:
    # - 以"The"/"A"/"An"开头 → 后面是NOUN
    # - 在句子中间, 前面有DET → NOUN
    # - 在NOUN后面 → VERB (如果还没到句末)
    # 这个方法很粗糙, 但足以展示baseline的天花板
    
    def infer_pos_simple(tokens):
        """简单POS推断"""
        pos_tags = ["UNK"] * len(tokens)
        for i, t in enumerate(tokens):
            t_lower = t.lower().strip()
            if t_lower in ["the", "a", "an"]:
                pos_tags[i] = "DET"
            elif t_lower in ["and", "or", "but"]:
                pos_tags[i] = "CC"
            elif t_lower in ["in", "on", "at", "over", "by", "with", "to"]:
                pos_tags[i] = "PREP"
            elif t_lower in ["that", "which", "who", "whom"]:
                pos_tags[i] = "RELPRON"
            elif t_lower.endswith("ly"):
                pos_tags[i] = "ADV"
            elif t_lower.endswith("ed") or t_lower.endswith("s") or t_lower.endswith("ing"):
                pos_tags[i] = "VERB"
            elif t_lower.endswith("tion") or t_lower.endswith("ness") or t_lower.endswith("ment"):
                pos_tags[i] = "NOUN"
            else:
                pos_tags[i] = "UNK"
        return pos_tags
    
    results = defaultdict(lambda: defaultdict(list))  # method -> rel -> [hit/miss]
    
    for sent_idx, sent_data in enumerate(all_sents):
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
        seq_len = len(tokens)
        pos_tags = infer_pos_simple(tokens)
        
        resolved_deps = []
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos < seq_len and dep_pos < seq_len and head_pos != dep_pos:
                resolved_deps.append((head_pos, dep_pos, rel_type))
        
        if len(resolved_deps) == 0:
            continue
        
        attn_dict, _ = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        for head_pos, dep_pos, rel_type in resolved_deps:
            # === Baseline 1: left_attach ===
            pred_head = max(1, dep_pos - 1)
            results["left_attach"][rel_type].append(float(pred_head == head_pos))
            
            # === Baseline 2: distance_decay ===
            tau = 2.0
            scores = np.array([np.exp(-abs(dep_pos - j) / tau) if j != dep_pos else -np.inf 
                              for j in range(1, seq_len)])
            pred_head = np.argmax(scores) + 1
            results["distance_decay"][rel_type].append(float(pred_head == head_pos))
            
            # === Baseline 3: nearest_right_noun (★ 新增! 关键!) ===
            # 对于dobj: verb的head预测 = dep的head = verb,
            #           但从dep(noun)视角: dep的head = 左边最近verb
            # 等价: verb → 右边最近noun = dobj
            if rel_type == "dobj":
                # dep(noun)的head = 左边最近verb
                pred_head = -1
                for j in range(dep_pos - 1, 0, -1):
                    if pos_tags[j] in ["VERB", "UNK"] and j != dep_pos:
                        pred_head = j
                        break
                if pred_head == -1:
                    pred_head = max(1, dep_pos - 1)
            elif rel_type == "nsubj":
                # dep(noun)的head = 右边最近verb
                pred_head = -1
                for j in range(dep_pos + 1, seq_len):
                    if pos_tags[j] in ["VERB", "UNK"] and j != dep_pos:
                        pred_head = j
                        break
                if pred_head == -1:
                    pred_head = min(seq_len - 1, dep_pos + 1)
            elif rel_type == "det":
                pred_head = min(dep_pos + 1, seq_len - 1)
            else:
                pred_head = max(1, dep_pos - 1)
            results["nearest_pos"][rel_type].append(float(pred_head == head_pos))
            
            # === Baseline 4: nearest_right_noun_strict (★ 更强!) ===
            # 对于dobj: 从verb位置找右边最近noun
            # 这个是"verb → right nearest noun"的直接实现
            if rel_type == "dobj":
                # head=verb, dep=noun
                # 预测: verb的dobj = verb右边第一个non-DET token
                pred_dep = -1
                for j in range(head_pos + 1, seq_len):
                    if pos_tags[j] != "DET" and pos_tags[j] != "CC":
                        pred_dep = j
                        break
                if pred_dep == -1:
                    pred_dep = head_pos + 1
                # UAS: 检查预测的dep位置是否匹配实际dep位置
                # 但UAS是head预测, 所以我们要从dep视角预测head
                # 重做: dep(noun)的head预测 = 左边最近的非DET非NOUN token (即verb)
                pred_head = -1
                for j in range(dep_pos - 1, 0, -1):
                    if pos_tags[j] not in ["DET", "CC", "PREP"] and j != dep_pos:
                        pred_head = j
                        break
                if pred_head == -1:
                    pred_head = max(1, dep_pos - 2)
            elif rel_type == "nsubj":
                # dep(noun)的head = 右边最近verb (同上)
                pred_head = -1
                for j in range(dep_pos + 1, seq_len):
                    if pos_tags[j] not in ["DET", "CC", "PREP"] and j != dep_pos:
                        pred_head = j
                        break
                if pred_head == -1:
                    pred_head = min(seq_len - 1, dep_pos + 2)
            elif rel_type == "det":
                pred_head = min(dep_pos + 1, seq_len - 1)
            else:
                pred_head = max(1, dep_pos - 1)
            results["nearest_content_word"][rel_type].append(float(pred_head == head_pos))
            
            # === Baseline 5: pos_heuristic_old (Phase 53的) ===
            if rel_type == "det":
                pred_head = min(dep_pos + 1, seq_len - 1)
            elif rel_type in ["nsubj", "dobj"]:
                pred_head = max(1, dep_pos - 1) if dep_pos > 1 else dep_pos + 1
            elif rel_type == "amod":
                pred_head = min(dep_pos + 1, seq_len - 1)
            else:
                pred_head = max(1, dep_pos - 1)
            results["pos_heuristic"][rel_type].append(float(pred_head == head_pos))
            
            # === Baseline 6: random ===
            choices = [j for j in range(1, seq_len) if j != dep_pos]
            if choices:
                pred_head = np.random.choice(choices)
                results["random"][rel_type].append(float(pred_head == head_pos))
            
            # === Attention (causal, k=3) ===
            hit, _, _ = evaluate_single_dep_attn(attn_dict, head_pos, dep_pos, seq_len, k=3)
            results["attn_causal_k3"][rel_type].append(float(hit))
        
        del attn_dict
        gc.collect()
        
        if (sent_idx + 1) % 5 == 0:
            print(f"  Processed {sent_idx+1}/{len(all_sents)}...", flush=True)
    
    # ---- 汇总 ----
    print("\n\n--- Strong Baseline Comparison ---")
    print(f"{'Method':<25} {'nsubj':<10} {'dobj':<10} {'det':<10} {'Overall':<10}")
    print("-" * 60)
    
    key_rels = ["nsubj", "dobj", "det"]
    for method in ["random", "left_attach", "distance_decay", "pos_heuristic", 
                   "nearest_pos", "nearest_content_word", "attn_causal_k3"]:
        rel_rates = {}
        all_vals = []
        for rel in key_rels:
            vals = results[method].get(rel, [])
            rate = np.mean(vals) if vals else float('nan')
            rel_rates[rel] = rate
            all_vals.extend(vals)
        
        overall = np.mean(all_vals) if all_vals else float('nan')
        marker = " ★" if method == "attn_causal_k3" else ""
        print(f"  {method:<23} {rel_rates.get('nsubj',0):<10.3f} {rel_rates.get('dobj',0):<10.3f} {rel_rates.get('det',0):<10.3f} {overall:<10.3f}{marker}")
    
    # ---- 关键: dobj的强baseline vs attention ----
    print("\n\n★★★ KEY: dobj Strong Baseline vs Attention ★★★")
    for method in ["random", "left_attach", "pos_heuristic", "nearest_pos", 
                   "nearest_content_word", "attn_causal_k3"]:
        dobj_vals = results[method].get("dobj", [])
        if dobj_vals:
            print(f"  {method:<25} dobj UAS = {np.mean(dobj_vals):.3f} (n={len(dobj_vals)})")
    
    best_baseline_dobj = max(
        np.mean(results[m].get("dobj", [0])) 
        for m in ["nearest_pos", "nearest_content_word", "pos_heuristic"]
        if results[m].get("dobj", [])
    )
    attn_dobj = np.mean(results["attn_causal_k3"].get("dobj", [0]))
    
    print(f"\n  Best baseline dobj: {best_baseline_dobj:.3f}")
    print(f"  Attention dobj:     {attn_dobj:.3f}")
    print(f"  Margin:             {attn_dobj - best_baseline_dobj:.3f}")
    
    if attn_dobj > best_baseline_dobj + 0.15:
        print(f"\n  ★★★ Attention >> Strong Baseline for dobj!")
        print(f"  → Even with verb→nearest_right_noun, attention wins by {attn_dobj - best_baseline_dobj:.3f}")
    else:
        print(f"\n  ⚠️ Margin small → need more careful analysis")
    
    return results


# ============================================================
# 54C: 复杂句验证
# ============================================================
def exp_54c_complex_sentences(model, tokenizer, info, model_name):
    """
    ★★★ 复杂句验证: 排除"简单SVO"数据偏差 ★★★
    
    测试:
    - 中心嵌入 (cat that dog chased ran)
    - 被动句 (mouse was chased by cat)
    - 关系从句 (woman who saw man left)
    - 多从句嵌套
    """
    print("\n" + "="*70)
    print("54C: Complex Sentence Verification — 排除简单SVO偏差")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # ---- 对比: 简单SVO vs 复杂句 ----
    print("\n--- Attention UAS: Simple SVO vs Complex Sentences ---")
    print(f"{'Type':<20} {'Sentence':<50} {'nsubj':<8} {'dobj':<8} {'Other':<8} {'Overall':<8}")
    print("-" * 100)
    
    all_results = {}  # sentence_type -> {rel: [hit/miss]}
    
    # 简单SVO
    simple_results = defaultdict(list)
    for sent_data in NORMAL_SVO_SENTENCES:
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
        seq_len = len(tokens)
        
        resolved_deps = []
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos < seq_len and dep_pos < seq_len and head_pos != dep_pos:
                resolved_deps.append((head_pos, dep_pos, rel_type))
        
        if len(resolved_deps) == 0:
            continue
        
        attn_dict, _ = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        for head_pos, dep_pos, rel_type in resolved_deps:
            hit, _, _ = evaluate_single_dep_attn(attn_dict, head_pos, dep_pos, seq_len, k=3)
            simple_results[rel_type].append(float(hit))
        
        del attn_dict
        gc.collect()
    
    # 复杂句
    complex_results_by_type = defaultdict(lambda: defaultdict(list))
    
    for sent_data in COMPLEX_SENTENCES:
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        sid = sent_data["id"]
        
        # 判断类型
        if "center" in sid:
            stype = "center_embed"
        elif "passive" in sid:
            stype = "passive"
        elif "relclause" in sid:
            stype = "rel_clause"
        elif "coordination" in sid:
            stype = "coordination"
        elif "multi_clause" in sid:
            stype = "multi_clause"
        else:
            stype = "other_complex"
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
        seq_len = len(tokens)
        
        resolved_deps = []
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos < seq_len and dep_pos < seq_len and head_pos != dep_pos:
                resolved_deps.append((head_pos, dep_pos, rel_type))
        
        if len(resolved_deps) == 0:
            continue
        
        attn_dict, _ = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        rel_hits = defaultdict(list)
        for head_pos, dep_pos, rel_type in resolved_deps:
            hit, best_layer, best_score = evaluate_single_dep_attn(
                attn_dict, head_pos, dep_pos, seq_len, k=3)
            complex_results_by_type[stype][rel_type].append(float(hit))
            rel_hits[rel_type].append(float(hit))
            
            dep_token = tokens[dep_pos] if dep_pos < len(tokens) else "?"
            head_token = tokens[head_pos] if head_pos < len(tokens) else "?"
        
        # 打印每句结果
        nsubj_rate = np.mean(rel_hits.get("nsubj", [])) if rel_hits.get("nsubj") else float('nan')
        dobj_rate = np.mean(rel_hits.get("dobj", [])) if rel_hits.get("dobj") else float('nan')
        other_vals = []
        for k, v in rel_hits.items():
            if k not in ["nsubj", "dobj"]:
                other_vals.extend(v)
        other_rate = np.mean(other_vals) if other_vals else float('nan')
        all_vals = []
        for v in rel_hits.values():
            all_vals.extend(v)
        overall_rate = np.mean(all_vals) if all_vals else float('nan')
        
        sent_short = sentence[:47] + "..." if len(sentence) > 50 else sentence
        print(f"  {stype:<18} {sent_short:<50} {nsubj_rate:<8.3f} {dobj_rate:<8.3f} {other_rate:<8.3f} {overall_rate:<8.3f}")
        
        del attn_dict
        gc.collect()
    
    # ---- 汇总对比 ----
    simple_all = []
    for v in simple_results.values():
        simple_all.extend(v)
    simple_overall = np.mean(simple_all) if simple_all else 0.0
    
    print(f"\n  Simple SVO overall: {simple_overall:.3f}")
    
    for stype in ["center_embed", "passive", "rel_clause", "multi_clause"]:
        type_data = complex_results_by_type[stype]
        type_all = []
        for v in type_data.values():
            type_all.extend(v)
        type_overall = np.mean(type_all) if type_all else float('nan')
        print(f"  {stype:<18} overall: {type_overall:.3f}")
    
    # ---- ★★★ Head层分布: 简单句 vs 复杂句 ★★★
    print("\n\n--- Head Layer Distribution: Simple vs Complex ---")
    
    # 重新收集: 找最佳层
    simple_best_layers = []
    complex_best_layers = []
    
    for sent_data in NORMAL_SVO_SENTENCES[:4]:
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        seq_len = len(inputs["input_ids"][0])
        
        attn_dict, _ = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos < seq_len and dep_pos < seq_len and head_pos != dep_pos:
                _, best_layer, _ = evaluate_single_dep_attn(
                    attn_dict, head_pos, dep_pos, seq_len, k=3)
                if best_layer >= 0:
                    simple_best_layers.append(best_layer)
        
        del attn_dict
        gc.collect()
    
    for sent_data in COMPLEX_SENTENCES:
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        seq_len = len(inputs["input_ids"][0])
        
        attn_dict, _ = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos < seq_len and dep_pos < seq_len and head_pos != dep_pos:
                _, best_layer, _ = evaluate_single_dep_attn(
                    attn_dict, head_pos, dep_pos, seq_len, k=3)
                if best_layer >= 0:
                    complex_best_layers.append(best_layer)
        
        del attn_dict
        gc.collect()
    
    if simple_best_layers:
        print(f"  Simple SVO best layers: mean={np.mean(simple_best_layers):.1f}, median={np.median(simple_best_layers):.1f}")
        # 分布
        layer_counts = defaultdict(int)
        for l in simple_best_layers:
            layer_counts[l] += 1
        for l in sorted(layer_counts.keys()):
            print(f"    Layer {l}: {layer_counts[l]} hits")
    
    if complex_best_layers:
        print(f"  Complex sentences best layers: mean={np.mean(complex_best_layers):.1f}, median={np.median(complex_best_layers):.1f}")
        layer_counts = defaultdict(int)
        for l in complex_best_layers:
            layer_counts[l] += 1
        for l in sorted(layer_counts.keys()):
            print(f"    Layer {l}: {layer_counts[l]} hits")
    
    # 判断Layer 3集中是否假象
    if simple_best_layers and complex_best_layers:
        simple_l3_rate = np.mean([1 for l in simple_best_layers if l <= 5])
        complex_l3_rate = np.mean([1 for l in complex_best_layers if l <= 5])
        print(f"\n  Simple L<=5 rate: {simple_l3_rate:.3f}")
        print(f"  Complex L<=5 rate: {complex_l3_rate:.3f}")
        if abs(simple_l3_rate - complex_l3_rate) > 0.2:
            print(f"  ⚠️ Layer distribution differs! L3 concentration may be SVO artifact")
        else:
            print(f"  ★ Layer distribution similar → L3 concentration is genuine")
    
    return complex_results_by_type


# ============================================================
# 54D: 分布式Head组合
# ============================================================
def exp_54d_distributed_circuit(model, tokenizer, info, model_name):
    """
    ★★★ 语法 = 分布式circuit, 不是单边 ★★★
    
    对比:
    1. 单head (best single head)
    2. Top-3 heads ensemble
    3. Top-5 heads ensemble
    4. 所有heads平均 (Phase 53方法)
    
    Ensemble方法: 对多个head的attention取加权平均
    """
    print("\n" + "="*70)
    print("54D: Distributed Head Circuit — 语法 = 多head组合")
    print("="*70)
    
    device = next(model.parameters()).device
    n_layers = info.n_layers
    
    # 使用swap句做精细分析
    test_pairs = [
        ("The cat ate the fish", 3, 2, "nsubj", "cat"),
        ("The fish ate the cat", 3, 2, "nsubj", "fish"),
        ("The cat ate the fish", 3, 5, "dobj", "fish"),
        ("The fish ate the cat", 3, 5, "dobj", "cat"),
    ]
    
    # ---- 收集每个head对每条依赖的分数 ----
    print("\n--- Per-Head Attention Scores for Key Dependencies ---")
    
    # 选取4个关键句
    key_sents = [
        {"sentence": "The cat ate the fish", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(3,5,"dobj")]},
        {"sentence": "The fish ate the cat", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(3,5,"dobj")]},
        {"sentence": "The rock ate the concept", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(3,5,"dobj")]},
        {"sentence": "Colorless green ideas sleep furiously", "deps": [(3,2,"amod"),(3,1,"amod"),(4,3,"nsubj"),(4,5,"advmod")]},
    ]
    
    # 采样关键层
    sample_layers = sorted(set([0, 3, 6, 10, 15, 20, n_layers-1]))
    sample_layers = [l for l in sample_layers if l < n_layers]
    
    # 对每个句子, 收集每个head对nsubj/dobj的分数
    all_head_data = []  # list of {sentence, rel_type, head_scores: [(li, hi, score)]}
    
    for sent_data in key_sents:
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
        seq_len = len(tokens)
        
        attn_dict, _ = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos >= seq_len or dep_pos >= seq_len or head_pos == dep_pos:
                continue
            
            # 收集每个head的分数
            head_scores = evaluate_single_dep_attn_per_head(
                attn_dict, head_pos, dep_pos, seq_len, top_k_heads=10)
            
            all_head_data.append({
                "sentence": sentence,
                "rel_type": rel_type,
                "dep_token": tokens[dep_pos] if dep_pos < len(tokens) else "?",
                "head_token": tokens[head_pos] if head_pos < len(tokens) else "?",
                "head_scores": head_scores,
            })
        
        del attn_dict
        gc.collect()
    
    # ---- 打印每个head的贡献 ----
    print(f"\n{'Sentence':<40} {'Rel':<8} {'Top Heads (layer:head:score)'}")
    print("-" * 100)
    
    for hd in all_head_data:
        if hd["rel_type"] not in ["nsubj", "dobj"]:
            continue
        top_str = ", ".join([f"L{li}:H{hi}:{sc:.3f}" for li, hi, sc in hd["head_scores"][:5]])
        sent_short = hd["sentence"][:37] + "..." if len(hd["sentence"]) > 40 else hd["sentence"]
        print(f"  {sent_short:<38} {hd['rel_type']:<8} {top_str}")
    
    # ---- ★★★ Ensemble评估 ★★★
    print("\n\n--- Ensemble Evaluation: Single Head vs Top-K Heads ---")
    
    # 重新评估: 用不同数量的top heads做ensemble
    ensemble_results = defaultdict(lambda: defaultdict(list))  # method -> rel -> [hit/miss]
    
    for sent_data in key_sents + NORMAL_SVO_SENTENCES[:4]:
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
        seq_len = len(tokens)
        
        attn_dict, _ = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos >= seq_len or dep_pos >= seq_len or head_pos == dep_pos:
                continue
            
            # 对每条依赖, 在每个层中评估
            # 方法1: 所有heads平均 (baseline, Phase 53)
            for li in sorted(attn_dict.keys()):
                A = attn_dict[li].mean(axis=0)  # [seq, seq]
                
                if head_pos < dep_pos:
                    row = A[dep_pos, 1:dep_pos]
                    target = head_pos - 1
                else:
                    row = A[head_pos, 1:head_pos]
                    target = dep_pos - 1
                
                if len(row) == 0 or target < 0 or target >= len(row):
                    continue
                
                k_actual = min(3, len(row))
                top_k = np.argsort(row)[-k_actual:]
                hit_mean = target in top_k
                
                ensemble_results["all_heads_mean"][rel_type].append(float(hit_mean))
                break  # 只用第一层(跟Phase 53一致)
            
            # 方法2-4: Top-K heads ensemble
            # 对每个head独立评分, 然后取top-k heads的attention平均
            head_attn_rows = []  # [(score, row)] for each head
            best_single_hit = False
            best_single_score = 0.0
            
            for li in sorted(attn_dict.keys()):
                A = attn_dict[li]  # [n_heads, seq, seq]
                n_heads = A.shape[0]
                
                for h_idx in range(n_heads):
                    if head_pos < dep_pos:
                        if dep_pos < A.shape[1] and head_pos < A.shape[2]:
                            row = A[h_idx, dep_pos, 1:dep_pos]
                            target = head_pos - 1
                            score = float(A[h_idx, dep_pos, head_pos])
                        else:
                            continue
                    else:
                        if head_pos < A.shape[1] and dep_pos < A.shape[2]:
                            row = A[h_idx, head_pos, 1:head_pos]
                            target = dep_pos - 1
                            score = float(A[h_idx, head_pos, dep_pos])
                        else:
                            continue
                    
                    if len(row) == 0 or target < 0 or target >= len(row):
                        continue
                    
                    head_attn_rows.append((score, li, h_idx, row, target))
                    
                    # 单head评估
                    k_actual = min(3, len(row))
                    top_k = np.argsort(row)[-k_actual:]
                    if target in top_k:
                        best_single_hit = True
                        best_single_score = max(best_single_score, score)
            
            # 记录单head最佳
            ensemble_results["best_single_head"][rel_type].append(float(best_single_hit))
            
            # Top-3 heads ensemble
            if len(head_attn_rows) >= 3:
                head_attn_rows.sort(key=lambda x: x[0], reverse=True)
                top3_rows = head_attn_rows[:3]
                ensemble_row = np.mean([r[3] for r in top3_rows], axis=0)
                target = top3_rows[0][4]
                k_actual = min(3, len(ensemble_row))
                top_k = np.argsort(ensemble_row)[-k_actual:]
                hit_top3 = target in top_k
                ensemble_results["top3_ensemble"][rel_type].append(float(hit_top3))
            
            # Top-5 heads ensemble
            if len(head_attn_rows) >= 5:
                head_attn_rows.sort(key=lambda x: x[0], reverse=True)
                top5_rows = head_attn_rows[:5]
                ensemble_row = np.mean([r[3] for r in top5_rows], axis=0)
                target = top5_rows[0][4]
                k_actual = min(3, len(ensemble_row))
                top_k = np.argsort(ensemble_row)[-k_actual:]
                hit_top5 = target in top_k
                ensemble_results["top5_ensemble"][rel_type].append(float(hit_top5))
        
        del attn_dict
        gc.collect()
    
    # ---- 汇总 ----
    print(f"\n{'Method':<25} {'nsubj':<10} {'dobj':<10} {'Overall':<10}")
    print("-" * 55)
    
    for method in ["all_heads_mean", "best_single_head", "top3_ensemble", "top5_ensemble"]:
        nsubj_vals = ensemble_results[method].get("nsubj", [])
        dobj_vals = ensemble_results[method].get("dobj", [])
        all_vals = nsubj_vals + dobj_vals
        
        nsubj_rate = np.mean(nsubj_vals) if nsubj_vals else float('nan')
        dobj_rate = np.mean(dobj_vals) if dobj_vals else float('nan')
        overall_rate = np.mean(all_vals) if all_vals else float('nan')
        
        print(f"  {method:<23} {nsubj_rate:<10.3f} {dobj_rate:<10.3f} {overall_rate:<10.3f}")
    
    # ---- 关键结论 ----
    mean_rate = np.mean(ensemble_results["all_heads_mean"].get("nsubj", []) + 
                        ensemble_results["all_heads_mean"].get("dobj", []))
    top3_rate = np.mean(ensemble_results["top3_ensemble"].get("nsubj", []) + 
                        ensemble_results["top3_ensemble"].get("dobj", []))
    single_rate = np.mean(ensemble_results["best_single_head"].get("nsubj", []) + 
                          ensemble_results["best_single_head"].get("dobj", []))
    
    print(f"\n★★★ Key Comparison ★★★")
    print(f"  All heads mean: {mean_rate:.3f}")
    print(f"  Best single head: {single_rate:.3f}")
    print(f"  Top-3 ensemble: {top3_rate:.3f}")
    
    if top3_rate > single_rate + 0.05:
        print(f"\n  ★★★ Ensemble > Single Head by {top3_rate - single_rate:.3f}")
        print(f"  → Syntax is a DISTRIBUTED CIRCUIT, not a single edge")
        print(f"  → This is a crucial upgrade from Phase 53")
    elif single_rate >= top3_rate:
        print(f"\n  → Single head suffices → syntax may be modular")
    else:
        print(f"\n  → Marginal ensemble benefit")
    
    return ensemble_results


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 54: Syntax vs Semantic + Strong Baselines")
    parser.add_argument("--model", type=str, default="deepseek7b", 
                        choices=["deepseek7b", "glm4", "qwen3"])
    parser.add_argument("--exp", type=int, default=0,
                        help="Experiment: 1=54A, 2=54B, 3=54C, 4=54D, 0=all")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase 54: Model={args.model}")
    print(f"{'='*70}")
    
    # 加载模型
    print("\nLoading model...")
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"  Model: {args.model}, Layers: {info.n_layers}, d_model: {info.d_model}")
    
    try:
        if args.exp in [0, 1]:
            results_54a = exp_54a_semantic_stripping(model, tokenizer, info, args.model)
        
        if args.exp in [0, 2]:
            results_54b = exp_54b_strong_baselines(model, tokenizer, info, args.model)
        
        if args.exp in [0, 3]:
            results_54c = exp_54c_complex_sentences(model, tokenizer, info, args.model)
        
        if args.exp in [0, 4]:
            results_54d = exp_54d_distributed_circuit(model, tokenizer, info, args.model)
    
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print("Phase 54 Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
