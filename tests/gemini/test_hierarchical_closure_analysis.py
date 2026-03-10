#!/usr/bin/env python3
"""
==========================================================
步骤4: 层级闭包分析 — SPDM 的 T(层级变换) 组件验证
==========================================================
核心问题:
  共享字典模型中,不同层是否系统性地使用不同的字典子集?
  如果存在 Micro→Meso→Macro 的层级位移规律,
  就证明了 SPDM 的 T 组件: 同一字典在不同深度的读出模式不同。

实验设计:
  1. 训练共享字典骨架模型
  2. 用三类概念探针输入:
     - Micro: 属性描述 ("red color", "sweet taste", "round shape")
     - Meso: 实体指称 ("an apple", "a cat", "the king")
     - Macro: 抽象关系 ("justice for all", "cause and effect", "freedom of thought")
  3. 在每层的 SparseOffsetDict 中记录被激活的字典原子分布
  4. 在每层的 FamilyBasisPool 中记录被激活的原型分布
  5. 检测:
     - 不同层对三类概念的字典使用是否有系统差异
     - 是否存在从浅层→深层的特征漂移(Micro→Meso→Macro)

Author: Gemini AGI Research
Date:   2026-03-10
GPU:    Required (CUDA)
"""

import os
import json
import math
import time
import datetime
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 导入共享字典模型
import sys
sys.path.insert(0, os.path.dirname(__file__))
from test_shared_dict_ablation import (
    UnifiedDictionaryCore, SharedFamilyBasisPool, SharedSparseOffsetDict,
    SharedRelationProtocolLayer, ContextGatedRouter, HierarchicalClosureNorm,
    SkeletonBlock, SkeletonLM, TextDataset
)


# ================================================================
# 1. 带钩子的模型包装器 (用于记录中间层字典激活)
# ================================================================

class DictionaryActivationTracker:
    """追踪每层字典激活模式的钩子系统"""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.layer_activations = {}  # {layer_idx: {'offset_coeffs': [], 'family_scores': []}}
        self._install_hooks()
    
    def _install_hooks(self):
        """在每层的 sparse_offset 和 family_basis 上安装钩子"""
        for i, block in enumerate(self.model.blocks):
            self.layer_activations[i] = {'offset_coeffs': [], 'family_scores': []}
            
            # Hook for sparse offset encoder output (before top-k)
            def make_offset_hook(layer_idx):
                def hook_fn(module, input, output):
                    # output = (offset_vector, sparsity_loss)
                    # 我们需要在 forward 中间截获 coeffs
                    pass
                return hook_fn
            
            # Hook for family basis gate output
            def make_family_hook(layer_idx):
                def hook_fn(module, input, output):
                    # output = (basis, scores)
                    scores = output[1]  # (B, T, K)
                    self.layer_activations[layer_idx]['family_scores'].append(
                        scores.detach().cpu()
                    )
                return hook_fn
            
            h = block.family_basis.register_forward_hook(make_family_hook(i))
            self.hooks.append(h)
    
    def clear(self):
        for i in self.layer_activations:
            self.layer_activations[i] = {'offset_coeffs': [], 'family_scores': []}
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


class InstrumentedSkeletonBlock(nn.Module):
    """带字典激活记录的骨架Block"""
    def __init__(self, original_block):
        super().__init__()
        self.block = original_block
        self.last_offset_coeffs = None
        self.last_family_scores = None
    
    def forward(self, x, mask=None):
        aux_losses = {}
        
        # Family basis
        basis, family_scores = self.block.family_basis(x)
        self.last_family_scores = family_scores.detach()
        
        # Sparse offset - 手动展开以截获 coefficients
        residual_from_basis = x - basis
        coeffs = self.block.sparse_offset.encoder(residual_from_basis)
        topk_vals, topk_idx = coeffs.topk(self.block.sparse_offset.top_k, dim=-1)
        sparse_coeffs = torch.zeros_like(coeffs)
        sparse_coeffs.scatter_(-1, topk_idx, topk_vals)
        self.last_offset_coeffs = sparse_coeffs.detach()
        
        offset = torch.einsum('btm,md->btd', sparse_coeffs, 
                             self.block.sparse_offset.dictionary)
        sparsity_loss = sparse_coeffs.abs().mean()
        aux_losses['sparsity'] = sparsity_loss
        
        x_reconstructed = basis + offset + self.block.bias_correction
        
        normed, e1 = self.block.norm1(x_reconstructed)
        aux_losses['energy_1'] = e1
        relation_out, attn_weights = self.block.relation_layer(normed, mask)
        x = x + self.block.dropout(relation_out)
        
        normed2, e2 = self.block.norm2(x)
        aux_losses['energy_2'] = e2
        routed = self.block.gated_router(normed2)
        x = x + self.block.dropout(routed)
        
        return x, attn_weights, family_scores, aux_losses


# ================================================================
# 2. 概念探针生成
# ================================================================

def generate_concept_probes():
    """生成三类概念探针"""
    probes = {
        'micro': {
            'description': '属性级(颜色/味道/形状/大小)',
            'prompts': [
                "The color is red",
                "The taste is sweet",
                "The shape is round",
                "The size is small",
                "It feels cold",
                "The texture is smooth",
                "The weight is heavy",
                "It smells fresh",
                "The sound is loud",
                "The surface is rough",
                "It looks bright",
                "The temperature is warm",
            ]
        },
        'meso': {
            'description': '实体级(具体名词)',
            'prompts': [
                "This is an apple",
                "This is a cat",
                "This is the king",
                "This is a car",
                "This is a banana",
                "This is a dog",
                "This is a tree",
                "This is a house",
                "This is a bird",
                "This is a fish",
                "This is a book",
                "This is a river",
            ]
        },
        'macro': {
            'description': '抽象关系级(概念/因果/价值)',
            'prompts': [
                "The concept of justice",
                "The idea of freedom",
                "Cause and effect relationship",
                "The meaning of truth",
                "The nature of beauty",
                "The principle of logic",
                "The value of wisdom",
                "The essence of courage",
                "The pursuit of knowledge",
                "The struggle for equality",
                "The foundation of trust",
                "The power of memory",
            ]
        }
    }
    return probes


# ================================================================
# 3. 层级字典激活分析
# ================================================================

def analyze_layer_activations(model, tokenizer, device, probes):
    """在每层提取字典激活分布,按概念类型分组"""
    model.eval()
    n_layers = len(model.blocks)
    
    # 包装blocks以记录中间激活
    instrumented_blocks = []
    for block in model.blocks:
        instrumented_blocks.append(InstrumentedSkeletonBlock(block))
    
    results = {}
    
    for concept_type, probe_data in probes.items():
        print(f"\n  处理 {concept_type} ({probe_data['description']})...")
        
        layer_offset_profiles = {i: [] for i in range(n_layers)}
        layer_family_profiles = {i: [] for i in range(n_layers)}
        
        for prompt in probe_data['prompts']:
            ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            B, T = ids.shape
            
            # 手动前向传播以截获中间层
            pos = torch.arange(0, T, device=device).unsqueeze(0)
            x = model.token_emb(ids) + model.pos_emb(pos)
            mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                for i, block in enumerate(model.blocks):
                    # 展开 forward 以截获 coefficients
                    basis, family_scores = block.family_basis(x)
                    residual = x - basis
                    
                    # 截获偏移系数
                    coeffs = block.sparse_offset.encoder(residual)
                    topk_vals, topk_idx = coeffs.topk(block.sparse_offset.top_k, dim=-1)
                    sparse_coeffs = torch.zeros_like(coeffs)
                    sparse_coeffs.scatter_(-1, topk_idx, topk_vals)
                    
                    offset = torch.einsum('btm,md->btd', sparse_coeffs,
                                         block.sparse_offset.dictionary)
                    
                    x_recon = basis + offset + block.bias_correction
                    normed, _ = block.norm1(x_recon)
                    rel_out, _ = block.relation_layer(normed, mask)
                    x = x + block.dropout(rel_out)
                    normed2, _ = block.norm2(x)
                    routed = block.gated_router(normed2)
                    x = x + block.dropout(routed)
                    
                    # 记录激活: 取最后token的激活 (最有语义信息)
                    offset_profile = sparse_coeffs[0, -1, :].cpu()  # (dict_size,)
                    family_profile = family_scores[0, -1, :].cpu()  # (num_families,)
                    
                    layer_offset_profiles[i].append(offset_profile)
                    layer_family_profiles[i].append(family_profile)
        
        # 聚合每层的激活分布
        type_result = {}
        for i in range(n_layers):
            offsets = torch.stack(layer_offset_profiles[i])  # (n_probes, dict_size)
            families = torch.stack(layer_family_profiles[i])  # (n_probes, num_families)
            
            # 字典原子使用频率 (哪些被激活)
            offset_usage = (offsets.abs() > 0).float().mean(dim=0)  # (dict_size,)
            # 字典原子平均激活强度
            offset_intensity = offsets.abs().mean(dim=0)  # (dict_size,)
            # 家族分配的熵 (越高=越分散)
            family_mean = families.mean(dim=0)
            family_entropy = -(family_mean * (family_mean + 1e-10).log()).sum().item()
            # 主导家族
            dominant_family = family_mean.argmax().item()
            dominant_family_score = family_mean.max().item()
            
            type_result[f'layer{i}'] = {
                'offset_active_ratio': offset_usage.mean().item(),
                'offset_top_atoms': offset_usage.topk(8)[1].tolist(),
                'offset_concentration': offset_intensity.topk(8)[0].sum().item() / (offset_intensity.sum().item() + 1e-10),
                'family_entropy': family_entropy,
                'dominant_family': dominant_family,
                'dominant_family_score': dominant_family_score,
                'offset_usage_vector': offset_usage.tolist(),
                'family_distribution': family_mean.tolist(),
            }
        
        results[concept_type] = type_result
    
    return results


def compute_hierarchical_metrics(activation_results, n_layers):
    """计算层级闭包的核心指标"""
    metrics = {}
    concept_types = ['micro', 'meso', 'macro']
    
    print(f"\n{'='*60}")
    print("层级闭包分析指标")
    print(f"{'='*60}")
    
    # 指标1: 层间字典使用差异 (每对概念类型, 每层)
    print("\n--- 指标 1: 概念类型间字典使用差异 (余弦距离) ---")
    for i in range(n_layers):
        layer_diffs = {}
        for t1_idx, t1 in enumerate(concept_types):
            for t2 in concept_types[t1_idx+1:]:
                u1 = torch.tensor(activation_results[t1][f'layer{i}']['offset_usage_vector'])
                u2 = torch.tensor(activation_results[t2][f'layer{i}']['offset_usage_vector'])
                cos_dist = 1 - F.cosine_similarity(u1.unsqueeze(0), u2.unsqueeze(0)).item()
                layer_diffs[f'{t1}_vs_{t2}'] = cos_dist
                print(f"  Layer {i}: {t1} vs {t2} = {cos_dist:.4f}")
        metrics[f'layer{i}_type_diffs'] = layer_diffs
    
    # 指标2: 层间字典漂移 (同一概念类型在不同层的使用变化)
    print("\n--- 指标 2: 同一概念类型的层间漂移 ---")
    for t in concept_types:
        drift_chain = []
        for i in range(n_layers - 1):
            u_curr = torch.tensor(activation_results[t][f'layer{i}']['offset_usage_vector'])
            u_next = torch.tensor(activation_results[t][f'layer{i+1}']['offset_usage_vector'])
            drift = 1 - F.cosine_similarity(u_curr.unsqueeze(0), u_next.unsqueeze(0)).item()
            drift_chain.append(drift)
        avg_drift = sum(drift_chain) / len(drift_chain) if drift_chain else 0
        metrics[f'{t}_layer_drift'] = {'per_layer': drift_chain, 'avg': avg_drift}
        print(f"  {t}: 逐层漂移 = {drift_chain}, 平均 = {avg_drift:.4f}")
    
    # 指标3: 层级分化梯度 — Micro应该在浅层主导, Macro在深层主导
    print("\n--- 指标 3: 层级分化梯度 (核心指标) ---")
    # 对每层, 计算哪种概念类型的字典使用最"独特"(与其他两类距离最大)
    for i in range(n_layers):
        dominance = {}
        for t in concept_types:
            u_t = torch.tensor(activation_results[t][f'layer{i}']['offset_usage_vector'])
            others = [torch.tensor(activation_results[ot][f'layer{i}']['offset_usage_vector'])
                      for ot in concept_types if ot != t]
            avg_other = torch.stack(others).mean(dim=0)
            specificity = 1 - F.cosine_similarity(u_t.unsqueeze(0), avg_other.unsqueeze(0)).item()
            dominance[t] = specificity
        
        dominant = max(dominance, key=dominance.get)
        metrics[f'layer{i}_dominant_type'] = dominant
        metrics[f'layer{i}_specificity'] = dominance
        print(f"  Layer {i}: 特异性 micro={dominance['micro']:.4f} "
              f"meso={dominance['meso']:.4f} macro={dominance['macro']:.4f} "
              f"→ 主导: {dominant}")
    
    # 指标4: 家族分配的层间熵变化
    print("\n--- 指标 4: 家族分配熵的层间变化 ---")
    for t in concept_types:
        entropies = [activation_results[t][f'layer{i}']['family_entropy'] for i in range(n_layers)]
        metrics[f'{t}_family_entropy_chain'] = entropies
        trend = 'ascending' if entropies[-1] > entropies[0] else 'descending'
        print(f"  {t}: 熵链 = {[f'{e:.3f}' for e in entropies]} ({trend})")
    
    # 指标5: 是否存在Micro→Meso→Macro的层级位移
    print("\n--- 指标 5: 层级位移检测 ---")
    expected_order = ['micro', 'meso', 'macro']
    actual_order = [metrics.get(f'layer{i}_dominant_type', '?') for i in range(n_layers)]
    
    # 计算与期望顺序的相关性
    # 简化: 给 micro=0, meso=1, macro=2, 检测是否随层递增
    type_to_score = {'micro': 0, 'meso': 1, 'macro': 2}
    scores = [type_to_score.get(actual_order[i], 1) for i in range(n_layers)]
    
    is_ascending = all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
    avg_score_first_half = sum(scores[:n_layers//2]) / max(n_layers//2, 1)
    avg_score_second_half = sum(scores[n_layers//2:]) / max(n_layers - n_layers//2, 1)
    hierarchy_gradient = avg_score_second_half - avg_score_first_half
    
    metrics['actual_dominant_order'] = actual_order
    metrics['hierarchy_gradient'] = hierarchy_gradient
    metrics['is_ascending'] = is_ascending
    metrics['hierarchy_scores'] = scores
    
    print(f"  层级主导类型序列: {actual_order}")
    print(f"  层级评分: {scores}")
    print(f"  梯度(后半-前半): {hierarchy_gradient:.3f} (>0 = 支持Micro→Macro)")
    print(f"  严格递增: {'是' if is_ascending else '否'}")
    
    # 最终判定
    print(f"\n{'='*60}")
    H_hierarchy = 'PASS' if hierarchy_gradient > 0 else 'FAIL'
    print(f"  H_hierarchy (层级位移存在): {H_hierarchy}")
    metrics['H_hierarchy'] = H_hierarchy
    
    return metrics


# ================================================================
# 4. 主函数
# ================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='步骤4: 层级闭包分析')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--num-families', type=int, default=16)
    parser.add_argument('--dict-size', type=int, default=64)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--json-out', type=str,
                        default='tests/gemini_temp/hierarchical_closure_20260310.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # --- Tokenizer ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # --- 训练数据 ---
    random.seed(42)
    templates = [
        "The {adj} {noun} is a type of {category}.",
        "A {noun} can be {adj} and {adj2}.",
        "In the garden, a {adj} {noun} grows near the {noun2}.",
        "The {category} includes {noun}, {noun2}, and {noun3}.",
        "{noun} is related to {noun2} but different from {noun3}.",
        "We think about {abstract} and {abstract2} every day.",
        "The concept of {abstract} is important for understanding {abstract2}.",
        "A {adj} {noun} tastes {taste} in the morning.",
        "The {animal} ran quickly across the field chasing a {animal2}.",
        "Unlike a {animal}, a {noun} cannot move on its own.",
        "The idea of {abstract} connects deeply to {abstract2} and {abstract3}.",
        "Every {category} has its own unique characteristics.",
        "She picked a {adj} {noun} from the tree and smiled.",
        "The {animal} and the {animal2} lived in the forest together.",
        "Thinking about {abstract} makes us question {abstract2}.",
        "A basket full of {noun}, {noun2}, and {noun3} was on the table.",
    ]
    fruits = ['apple', 'banana', 'orange', 'grape', 'pear', 'lemon', 'mango', 'peach']
    animals = ['cat', 'dog', 'rabbit', 'horse', 'tiger', 'bird', 'fish', 'deer']
    abstracts = ['justice', 'truth', 'logic', 'memory', 'beauty', 'freedom', 'wisdom', 'courage']
    adjectives = ['red', 'sweet', 'big', 'small', 'bright', 'dark', 'fresh', 'old', 'young', 'round']
    tastes = ['sweet', 'sour', 'bitter', 'delicious', 'fresh', 'wonderful']
    categories = ['fruit', 'animal', 'food', 'object', 'creature', 'living thing']

    def gen():
        t = random.choice(templates)
        return t.format(
            noun=random.choice(fruits + animals), noun2=random.choice(fruits + animals),
            noun3=random.choice(fruits + animals), adj=random.choice(adjectives),
            adj2=random.choice(adjectives), category=random.choice(categories),
            abstract=random.choice(abstracts), abstract2=random.choice(abstracts),
            abstract3=random.choice(abstracts), animal=random.choice(animals),
            animal2=random.choice(animals), taste=random.choice(tastes),
        )

    print("生成训练语料...")
    train_text = ' '.join([gen() for _ in range(20000)])
    val_text = ' '.join([gen() for _ in range(2000)])
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    print(f"训练tokens: {len(train_ids):,}, 验证tokens: {len(val_ids):,}")

    train_loader = DataLoader(TextDataset(train_ids, args.seq_len),
                             batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TextDataset(val_ids, args.seq_len),
                           batch_size=args.batch_size, shuffle=False, drop_last=True)

    # --- 训练共享字典模型 ---
    print("\n" + "="*60)
    print("训练共享字典骨架模型...")
    print("="*60)
    model = SkeletonLM(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, num_families=args.num_families,
        dict_size=args.dict_size, top_k=args.top_k, max_seq_len=args.seq_len,
        use_unified_dict=True,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"参数量: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        n = 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            out = model(x, labels=y)
            loss = out['loss']
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n += 1
            if step % 50 == 0:
                elapsed = time.time() - start_time
                ppl = math.exp(min(loss.item(), 20))
                progress = (epoch * len(train_loader) + step) / total_steps * 100
                print(f"  Epoch {epoch+1}/{args.epochs} Step {step}/{len(train_loader)} "
                      f"Loss={loss.item():.4f} PPL={ppl:.1f} 进度={progress:.1f}% 耗时={elapsed:.0f}s")
        
        # 验证
        model.eval()
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += model(x, labels=y)['loss'].item()
                vn += 1
        avg_vl = val_loss / max(vn, 1)
        print(f"  Epoch {epoch+1} | Train Loss={epoch_loss/n:.4f} | Val Loss={avg_vl:.4f} | Val PPL={math.exp(min(avg_vl,20)):.1f}")
    
    train_time = time.time() - start_time
    print(f"训练完成, 耗时 {train_time:.1f}s")
    
    # --- 层级闭包分析 ---
    print("\n" + "="*60)
    print("层级闭包分析: 逐层字典激活探测")
    print("="*60)
    
    probes = generate_concept_probes()
    activation_results = analyze_layer_activations(model, tokenizer, device, probes)
    hierarchical_metrics = compute_hierarchical_metrics(activation_results, args.n_layers)
    
    # --- 统一字典内部分析 ---
    print("\n" + "="*60)
    print("统一字典内部结构分析")
    print("="*60)
    
    W_unified = model.unified_core.get_dictionary().detach().cpu()  # (dict_size, d_model)
    W_norm = F.normalize(W_unified, dim=-1)
    dict_similarity = (W_norm @ W_norm.T)
    off_diag = dict_similarity[~torch.eye(dict_similarity.size(0), dtype=bool)]
    
    dict_analysis = {
        'dict_mean_abs_cos': off_diag.abs().mean().item(),
        'dict_max_abs_cos': off_diag.abs().max().item(),
        'dict_orthogonality': (1 - off_diag.abs().mean()).item(),
        'dict_frobenius_norm': W_unified.norm().item(),
        'dict_rank_ratio': torch.linalg.matrix_rank(W_unified).item() / min(W_unified.shape),
    }
    print(f"  字典正交度: {dict_analysis['dict_orthogonality']:.4f}")
    print(f"  字典秩比: {dict_analysis['dict_rank_ratio']:.4f}")
    
    # --- 保存结果 ---
    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'experiment': 'hierarchical_closure_analysis_step4',
        'args': vars(args),
        'train_time_sec': train_time,
        'param_count': param_count,
        'activation_results': {},
        'hierarchical_metrics': {},
        'dict_analysis': dict_analysis,
    }
    
    # 清理不可序列化的内容
    for ct in activation_results:
        results['activation_results'][ct] = {}
        for layer_key in activation_results[ct]:
            layer_data = activation_results[ct][layer_key].copy()
            # 只保留标量和短列表
            clean = {k: v for k, v in layer_data.items()
                    if k not in ['offset_usage_vector', 'family_distribution']}
            clean['offset_top8_usage'] = layer_data.get('offset_usage_vector', [])[:8]
            clean['family_top3'] = sorted(
                enumerate(layer_data.get('family_distribution', [])),
                key=lambda x: -x[1])[:3]
            results['activation_results'][ct][layer_key] = clean
    
    for k, v in hierarchical_metrics.items():
        if isinstance(v, (int, float, str, bool, list)):
            results['hierarchical_metrics'][k] = v
        elif isinstance(v, dict):
            results['hierarchical_metrics'][k] = {
                kk: vv for kk, vv in v.items() 
                if isinstance(vv, (int, float, str, bool, list))
            }
    
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {args.json_out}")
    
    print("\n" + "="*60)
    print("步骤4: 层级闭包分析完成")
    print("="*60)


if __name__ == '__main__':
    main()
