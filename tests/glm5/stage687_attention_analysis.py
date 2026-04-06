#!/usr/bin/env python3
"""
P41: 注意力权重分析（Stage687）

核心问题：注意力如何引导信息流动？注意力分布与层动力学模式的关系？

实验设计：
1. 分析跨层注意力分布（各层对哪些位置最关注）
2. 测量注意力的"信息聚焦度"（entropy）
3. 分析注意力与Silhouette变化的相关性
4. 对比不同模型的注意力模式

用法：python tests/glm5/stage687_attention_analysis.py <model_name>
"""
import sys, os, math, json, statistics, time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

def load_model(model_name):
    path = MODEL_MAP.get(model_name, _Path(model_name))
    print(f"  loading model: {path.name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto",
        attn_implementation="eager"
    )
    model.eval()
    return model, tokenizer

def get_attention_and_hidden(model, tokenizer, text):
    """获取attention weights和hidden states"""
    model_device = next(model.parameters()).device
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=32)
    tokens = tokens.to(model_device)
    
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True, output_attentions=True)
    
    hidden_states = {}
    for i, hs in enumerate(outputs.hidden_states):
        hidden_states[i] = hs[0, -1, :].float().cpu()
    
    # attention: tuple of (batch, heads, seq, seq) for each layer
    attn_weights = []
    if outputs.attentions:
        for layer_attn in outputs.attentions:
            # layer_attn: (1, n_heads, T, T)
            # 取最后一个query token对所有key的attention
            last_token_attn = layer_attn[0, :, -1, :].float().cpu()  # (n_heads, T)
            attn_weights.append(last_token_attn)
    
    return hidden_states, attn_weights

def compute_attention_entropy(attn_weights):
    """计算注意力的信息熵 (entropy)
    高entropy -> 均匀分布(关注所有位置)
    低entropy -> 集中分布(聚焦于少数位置)
    """
    entropies = []
    for layer_attn in attn_weights:
        # layer_attn: (n_heads, T)
        eps = 1e-10
        probs = layer_attn + eps
        probs = probs / probs.sum(dim=-1, keepdim=True)
        log_probs = torch.log(probs)
        entropy = -(probs * log_probs).sum(dim=-1)  # (n_heads,)
        entropies.append(entropy.mean().item())  # average over heads
    return entropies

def compute_attention_focus_position(attn_weights):
    """分析注意力聚焦在哪个位置
    返回每层的平均聚焦位置索引
    """
    focus_positions = []
    for layer_attn in attn_weights:
        # layer_attn: (n_heads, T)
        mean_attn = layer_attn.mean(dim=0)  # (T,)
        focus_pos = torch.argmax(mean_attn).item()
        focus_positions.append(focus_pos)
    return focus_positions

def compute_attention_self_ratio(attn_weights):
    """计算自注意力的比例(最后一个token关注自身的比例)"""
    self_ratios = []
    for layer_attn in attn_weights:
        # layer_attn: (n_heads, T)
        self_attn = layer_attn[:, -1]  # 最后token关注自身的权重
        total_attn = layer_attn.sum(dim=-1)
        ratio = (self_attn / (total_attn + 1e-10)).mean().item()
        self_ratios.append(ratio)
    return self_ratios

def compute_cross_sample_attention_variability(model, tokenizer, texts):
    """测量不同文本的注意力分布差异
    如果注意力模式对文本敏感 -> 高variability
    如果注意力模式固定 -> 低variability
    """
    all_entropies = []
    all_focus_pos = []
    
    for text in texts:
        hs, attn_w = get_attention_and_hidden(model, tokenizer, text)
        if not attn_w:
            continue
        ent = compute_attention_entropy(attn_w)
        fp = compute_attention_focus_position(attn_w)
        all_entropies.append(ent)
        all_focus_pos.append(fp)
    
    # 计算跨文本entropy的变异系数
    n_layers = min(len(e) for e in all_entropies) if all_entropies else 0
    entropy_cv = []
    for l in range(n_layers):
        vals = [e[l] for e in all_entropies if l < len(e) and not math.isnan(e[l])]
        if len(vals) > 1:
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            cv = std_v / max(mean_v, 1e-10)
            if not math.isnan(cv) and not math.isinf(cv):
                entropy_cv.append(cv)
    
    return {
        "avg_entropy_cv": statistics.mean(entropy_cv) if entropy_cv else 0,
        "entropy_cv_by_layer": entropy_cv,
    }

def analyze_attention_patterns(model, tokenizer, texts):
    """P41核心分析"""
    
    print("\n" + "="*60)
    print("  A: cross-layer attention entropy")
    print("="*60)
    
    # 用一个样本分析
    hs, attn_w = get_attention_and_hidden(model, tokenizer, texts[0])
    
    if not attn_w:
        print("  [WARN] attention weights not available for this model")
        return None
    
    n_layers = len(attn_w)
    print(f"  total attention layers: {n_layers}")
    
    entropies = compute_attention_entropy(attn_w)
    focus_pos = compute_attention_focus_position(attn_w)
    self_ratio = compute_attention_self_ratio(attn_w)
    
    # 分段统计
    n_q = n_layers // 4
    segments = [
        ("early", 0, n_q),
        ("mid-early", n_q, 2*n_q),
        ("mid-late", 2*n_q, 3*n_q),
        ("late", 3*n_q, n_layers),
    ]
    
    for seg_name, start, end in segments:
        seg_ent = entropies[start:end]
        seg_self = self_ratio[start:end]
        avg_ent = statistics.mean(seg_ent) if seg_ent else 0
        avg_self = statistics.mean(seg_self) if seg_self else 0
        print(f"  {seg_name:12s} (L{start}-{end-1}): entropy={avg_ent:.3f}, self_ratio={avg_self:.3f}")
    
    # 找entropy最高和最低的层
    max_ent_layer = entropies.index(max(entropies))
    min_ent_layer = entropies.index(min(entropies))
    print(f"  max entropy: L{max_ent_layer} ({max(entropies):.3f})")
    print(f"  min entropy: L{min_ent_layer} ({min(entropies):.3f})")
    
    print("\n" + "="*60)
    print("  B: attention focus position analysis")
    print("="*60)
    
    # 统计关注的位置分布
    pos_counts = {}
    for p in focus_pos:
        pos_counts[p] = pos_counts.get(p, 0) + 1
    
    sorted_pos = sorted(pos_counts.items(), key=lambda x: -x[1])
    print(f"  top-3 focused positions:")
    for pos, count in sorted_pos[:3]:
        print(f"    pos={pos}: {count}/{n_layers} layers ({count/n_layers*100:.0f}%)")
    
    # 最后token自关注比例的趋势
    early_self = statistics.mean(self_ratio[:n_q])
    late_self = statistics.mean(self_ratio[3*n_q:])
    print(f"\n  early self_ratio: {early_self:.3f}")
    print(f"  late self_ratio: {late_self:.3f}")
    if late_self > early_self * 1.5:
        print(f"  -> [INV-336] later layers increase self-attention [OK]")
    else:
        print(f"  -> [INV-336] self-attention does not increase in late layers")
    
    print("\n" + "="*60)
    print("  C: cross-text attention variability")
    print("="*60)
    
    variability = compute_cross_sample_attention_variability(model, tokenizer, texts[:10])
    print(f"  avg entropy CV (cross-text): {variability['avg_entropy_cv']:.4f}")
    
    if variability['avg_entropy_cv'] < 0.05:
        print(f"  -> attention patterns are FIXED across texts (CV<5%)")
    elif variability['avg_entropy_cv'] < 0.15:
        print(f"  -> attention patterns are MODERATELY text-dependent")
    else:
        print(f"  -> attention patterns are HIGHLY text-dependent (CV>{0.15:.0%})")
    
    print("\n" + "="*60)
    print("  D: attention-head diversity")
    print("="*60)
    
    # 分析各层head之间的差异
    head_diversities = []
    for layer_attn in attn_w[:8]:  # 只分析前8层
        # layer_attn: (n_heads, T)
        mean_attn = layer_attn.mean(dim=0)  # (T,)
        # 每个head与mean的cos
        cos_sims = []
        for h in range(layer_attn.shape[0]):
            cos_v = F.cosine_similarity(layer_attn[h:h+1], mean_attn.unsqueeze(0)).item()
            cos_sims.append(abs(cos_v))
        head_diversities.append(1 - statistics.mean(cos_sims))
    
    avg_div = statistics.mean(head_diversities) if head_diversities else 0
    print(f"  avg head diversity (1 - mean_cos_with_avg): {avg_div:.4f}")
    if avg_div > 0.3:
        print(f"  -> attention heads are HIGHLY diverse (different roles)")
    elif avg_div > 0.1:
        print(f"  -> attention heads are MODERATELY diverse")
    else:
        print(f"  -> attention heads are SIMILAR (redundant)")
    
    print("\n" + "="*60)
    print("  E: attention vs hidden-state direction correlation")
    print("="*60)
    
    # 分析注意力entropy变化与方向变化的相关性
    direction_changes = []
    entropy_changes = []
    
    for i in range(min(n_layers - 1, len(hs) - 2)):
        h1 = hs[i + 1]
        h2 = hs[i + 2]
        cos_val = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
        direction_changes.append(abs(math.acos(max(-1, min(1, cos_val)))))
        entropy_changes.append(abs(entropies[i+1] - entropies[i]))
    
    if len(direction_changes) >= 3:
        # Pearson correlation
        mean_dc = statistics.mean(direction_changes)
        mean_ec = statistics.mean(entropy_changes)
        cov = sum((d - mean_dc) * (e - mean_ec) for d, e in zip(direction_changes, entropy_changes)) / len(direction_changes)
        std_dc = statistics.stdev(direction_changes)
        std_ec = statistics.stdev(entropy_changes)
        corr = cov / (std_dc * std_ec) if std_dc > 0 and std_ec > 0 else 0
        
        print(f"  direction_change vs entropy_change correlation: {corr:.4f}")
        if abs(corr) > 0.3:
            print(f"  -> [INV-337] attention entropy and direction change are {'positively' if corr > 0 else 'negatively'} correlated")
        else:
            print(f"  -> [INV-337] weak correlation between attention and direction change")
    
    return {
        "n_attn_layers": n_layers,
        "max_entropy": max(entropies),
        "min_entropy": min(entropies),
        "avg_entropy_cv": variability['avg_entropy_cv'],
        "head_diversity": avg_div,
        "early_self_ratio": early_self,
        "late_self_ratio": late_self,
    }

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    print(f"\n{'='*60}")
    print(f"  P41: Attention Weight Analysis")
    print(f"  model: {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer = load_model(model_name)
    
    texts = [
        "The cat sat on the mat and looked around.",
        "Paris is the capital of France in Europe.",
        "She carefully folded the origami crane.",
        "The derivative of x squared is two x.",
        "DNA contains genetic instructions for life.",
        "Gravity causes objects to fall downward.",
        "The orchestra played a beautiful symphony.",
        "He solved the equation step by step.",
        "Yesterday it rained heavily all day long.",
        "The neural network learned complex patterns.",
    ]
    
    t0 = time.time()
    results = analyze_attention_patterns(model, tokenizer, texts)
    elapsed = time.time() - t0
    
    if results:
        print(f"\n{'='*60}")
        print(f"  P41 Summary")
        print(f"{'='*60}")
        for k, v in results.items():
            print(f"  {k}: {v}")
        print(f"  elapsed: {elapsed:.1f}s")
    
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
