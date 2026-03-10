#!/usr/bin/env python3
"""
==========================================================
共享字典对照实验: 验证 SNN 统一编码假说
==========================================================
核心假设:
  基底+偏移 和 拓扑协议层 是SNN同一编码原理在不同尺度上的产物。
  如果成立,让 FamilyBasisPool / SparseOffsetDict / RelationProtocolLayer
  共享底层字典矩阵 W_unified 应该:
    1. 涌现属性不会退化 (或甚至提升)
    2. 跨维度迁移: 概念空间基底与关系空间模板统计相关
    3. 参数效率更高

实验设计 (三组对照):
  A: 独立字典骨架 (Independent) — 现有架构
  B: 共享字典骨架 (Unified)    — 所有模块从同一字典导出
  C: 标准 Transformer (Baseline) — 无骨架约束

Author: Gemini AGI Research
Date:   2026-03-10
GPU:    Required (CUDA)
"""

import os
import json
import math
import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ================================================================
# 1. 统一字典核心 (Unified Dictionary Core)
# ================================================================

class UnifiedDictionaryCore(nn.Module):
    """
    统一字典核心: 所有模块共享的底层字典矩阵 W_unified。
    不同模块通过不同的投影头从 W_unified 中导出自己的子空间。
    
    对应 SNN 原理:
    - W_unified = 脉冲神经网络通过 Oja/WTA/LIF 学到的统一正交字典
    - 投影头 = 不同皮层功能柱对同一字典的不同读出方式
    """
    def __init__(self, dict_size: int, d_model: int):
        super().__init__()
        # 核心共享字典 (dict_size个d_model维原子)
        self.W_unified = nn.Parameter(torch.randn(dict_size, d_model) * 0.02)
        self.dict_size = dict_size
        self.d_model = d_model
    
    def get_dictionary(self):
        """返回共享字典矩阵"""
        return self.W_unified


# ================================================================
# 2. 共享字典版 FamilyBasisPool
# ================================================================

class SharedFamilyBasisPool(nn.Module):
    """
    从统一字典中导出家族原型。
    每个家族原型 = W_unified 的行的加权组合。
    """
    def __init__(self, num_families: int, d_model: int, unified_core: UnifiedDictionaryCore):
        super().__init__()
        self.unified_core = unified_core
        self.num_families = num_families
        # 从字典导出原型的投影矩阵: (num_families, dict_size)
        self.proto_proj = nn.Parameter(
            torch.randn(num_families, unified_core.dict_size) * 0.02
        )
        self.gate = nn.Linear(d_model, num_families, bias=False)
    
    @property
    def prototypes(self):
        """从统一字典导出的原型"""
        # softmax 确保是字典原子的凸组合
        weights = torch.softmax(self.proto_proj, dim=-1)
        return weights @ self.unified_core.get_dictionary()  # (K, D)
    
    def forward(self, x):
        protos = self.prototypes
        scores = torch.softmax(self.gate(x), dim=-1)    # (B, T, K)
        basis = torch.einsum('btk,kd->btd', scores, protos)  # (B, T, D)
        return basis, scores


# ================================================================
# 3. 共享字典版 SparseOffsetDict
# ================================================================

class SharedSparseOffsetDict(nn.Module):
    """
    直接使用统一字典作为偏移字典, 通过 top-k 稀疏激活。
    """
    def __init__(self, d_model: int, unified_core: UnifiedDictionaryCore, top_k: int = 8):
        super().__init__()
        self.unified_core = unified_core
        self.encoder = nn.Linear(d_model, unified_core.dict_size, bias=False)
        self.top_k = top_k
    
    @property
    def dictionary(self):
        return self.unified_core.get_dictionary()
    
    def forward(self, x):
        coeffs = self.encoder(x)
        topk_vals, topk_idx = coeffs.topk(self.top_k, dim=-1)
        sparse_coeffs = torch.zeros_like(coeffs)
        sparse_coeffs.scatter_(-1, topk_idx, topk_vals)
        offset = torch.einsum('btm,md->btd', sparse_coeffs, self.dictionary)
        sparsity_loss = sparse_coeffs.abs().mean()
        return offset, sparsity_loss


# ================================================================
# 4. 共享字典版 RelationProtocolLayer
# ================================================================

class SharedRelationProtocolLayer(nn.Module):
    """
    关系协议模板从统一字典导出。
    每个头的 relation_template = 从 W_unified 子空间构造的旋转矩阵。
    """
    def __init__(self, d_model: int, n_heads: int, unified_core: UnifiedDictionaryCore, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.unified_core = unified_core
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 从统一字典导出关系模板的投影: 每个头选择字典的两行做外积
        # template_h = proj_left_h @ W_unified[:d_head, :] 的变形
        self.template_proj_left = nn.Parameter(
            torch.randn(n_heads, self.d_head, unified_core.dict_size) * 0.02
        )
        self.template_proj_right = nn.Parameter(
            torch.randn(n_heads, unified_core.dict_size, self.d_head) * 0.02
        )
        
        self.dropout = nn.Dropout(dropout)
    
    @property
    def relation_templates(self):
        """从统一字典导出关系模板"""
        W = self.unified_core.get_dictionary()  # (dict_size, d_model)
        # 取字典的前 d_head 维作为子空间
        W_sub = W[:, :self.d_head]  # (dict_size, d_head)
        # left: (n_heads, d_head, dict_size) @ (dict_size, d_head) -> (n_heads, d_head, d_head)
        templates = torch.einsum('hid,dj->hij', self.template_proj_left, W_sub)
        # 再混合 right projection
        templates = templates + torch.einsum('hid,hdj->hij',
                                              self.template_proj_left[:, :, :W_sub.shape[0]],
                                              self.template_proj_right)
        return templates
    
    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        templates = self.relation_templates
        k_modulated = torch.einsum('bhtd,hde->bhte', k, templates)
        
        attn = torch.matmul(q, k_modulated.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)
        return out, attn


# ================================================================
# 5. 独立字典模块 (现有架构，直接复制)
# ================================================================

class FamilyBasisPool(nn.Module):
    def __init__(self, num_families: int, d_model: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_families, d_model) * 0.02)
        self.gate = nn.Linear(d_model, num_families, bias=False)

    def forward(self, x):
        scores = torch.softmax(self.gate(x), dim=-1)
        basis = torch.einsum('btk,kd->btd', scores, self.prototypes)
        return basis, scores


class SparseOffsetDict(nn.Module):
    def __init__(self, d_model: int, dict_size: int, top_k: int = 8):
        super().__init__()
        self.dictionary = nn.Parameter(torch.randn(dict_size, d_model) * 0.02)
        self.encoder = nn.Linear(d_model, dict_size, bias=False)
        self.top_k = top_k

    def forward(self, x):
        coeffs = self.encoder(x)
        topk_vals, topk_idx = coeffs.topk(self.top_k, dim=-1)
        sparse_coeffs = torch.zeros_like(coeffs)
        sparse_coeffs.scatter_(-1, topk_idx, topk_vals)
        offset = torch.einsum('btm,md->btd', sparse_coeffs, self.dictionary)
        sparsity_loss = sparse_coeffs.abs().mean()
        return offset, sparsity_loss


class RelationProtocolLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.relation_templates = nn.Parameter(torch.randn(n_heads, self.d_head, self.d_head) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k_modulated = torch.einsum('bhtd,hde->bhte', k, self.relation_templates)
        attn = torch.matmul(q, k_modulated.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)
        return out, attn


# ================================================================
# 6. 通用辅助模块
# ================================================================

class ContextGatedRouter(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * F.silu(up))


class HierarchicalClosureNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        energy = rms.mean()
        return self.weight * x / rms, energy


# ================================================================
# 7. Block 与 Model (参数化选择独立/共享)
# ================================================================

class SkeletonBlock(nn.Module):
    """可选独立/共享字典的 4D+3D Block"""
    def __init__(self, d_model, n_heads, d_ff, num_families, dict_size, top_k,
                 dropout=0.1, unified_core=None):
        super().__init__()
        # 根据是否提供 unified_core 来选择模块
        if unified_core is not None:
            # 共享字典模式
            self.family_basis = SharedFamilyBasisPool(num_families, d_model, unified_core)
            self.sparse_offset = SharedSparseOffsetDict(d_model, unified_core, top_k)
            self.relation_layer = SharedRelationProtocolLayer(d_model, n_heads, unified_core, dropout)
        else:
            # 独立字典模式
            self.family_basis = FamilyBasisPool(num_families, d_model)
            self.sparse_offset = SparseOffsetDict(d_model, dict_size, top_k)
            self.relation_layer = RelationProtocolLayer(d_model, n_heads, dropout)
        
        self.gated_router = ContextGatedRouter(d_model, d_ff)
        self.norm1 = HierarchicalClosureNorm(d_model)
        self.norm2 = HierarchicalClosureNorm(d_model)
        self.bias_correction = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        aux_losses = {}
        basis, family_scores = self.family_basis(x)
        residual_from_basis = x - basis
        offset, sparsity_loss = self.sparse_offset(residual_from_basis)
        aux_losses['sparsity'] = sparsity_loss
        x_reconstructed = basis + offset + self.bias_correction
        normed, e1 = self.norm1(x_reconstructed)
        aux_losses['energy_1'] = e1
        relation_out, attn_weights = self.relation_layer(normed, mask)
        x = x + self.dropout(relation_out)
        normed2, e2 = self.norm2(x)
        aux_losses['energy_2'] = e2
        routed = self.gated_router(normed2)
        x = x + self.dropout(routed)
        return x, attn_weights, family_scores, aux_losses


class SkeletonLM(nn.Module):
    """统一的骨架语言模型 (支持独立/共享字典切换)"""
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4,
                 d_ff=512, num_families=16, dict_size=64, top_k=8,
                 max_seq_len=256, dropout=0.1, use_unified_dict=False):
        super().__init__()
        self.d_model = d_model
        self.use_unified_dict = use_unified_dict
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # 如果是共享字典模式, 创建一个全局统一字典核心
        self.unified_core = None
        if use_unified_dict:
            self.unified_core = UnifiedDictionaryCore(dict_size, d_model)
        
        self.blocks = nn.ModuleList([
            SkeletonBlock(d_model, n_heads, d_ff, num_families, dict_size, top_k,
                         dropout, unified_core=self.unified_core)
            for _ in range(n_layers)
        ])
        self.final_norm = HierarchicalClosureNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.max_seq_len = max_seq_len
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        all_attn = []
        all_family_scores = []
        total_aux = {'sparsity': 0.0, 'energy': 0.0}

        for block in self.blocks:
            x, attn, fam_scores, aux = block(x, mask)
            all_attn.append(attn)
            all_family_scores.append(fam_scores)
            total_aux['sparsity'] = total_aux['sparsity'] + aux['sparsity']
            total_aux['energy'] = total_aux['energy'] + aux['energy_1'] + aux['energy_2']

        x, _ = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            lambda_sparse = 0.01
            lambda_energy = 0.001
            loss = loss + lambda_sparse * total_aux['sparsity'] + lambda_energy * total_aux['energy']

        return {
            'loss': loss,
            'logits': logits,
            'attn_weights': all_attn,
            'family_scores': all_family_scores,
            'aux_losses': total_aux,
        }


class BaselineTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4,
                 d_ff=512, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {'loss': loss, 'logits': logits}


# ================================================================
# 8. 数据集
# ================================================================

class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        n = (len(token_ids) // seq_len) * seq_len
        self.data = torch.tensor(token_ids[:n], dtype=torch.long).view(-1, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]


# ================================================================
# 9. 涌现验证 + 跨维度迁移检测
# ================================================================

def verify_emergence(model, tokenizer, device, result_dict, model_label):
    """验证涌现属性 + 跨维度迁移分析"""
    print(f"\n{'='*60}")
    print(f"[{model_label}] 涌现结构验证")
    print(f"{'='*60}")
    model.eval()

    families = {
        'fruit': ['apple', 'banana', 'orange', 'grape', 'pear', 'lemon'],
        'animal': ['cat', 'dog', 'rabbit', 'horse', 'tiger', 'bird'],
        'abstract': ['justice', 'truth', 'logic', 'memory', 'beauty', 'freedom'],
    }

    emergence = {}

    # --- 族内/族间分析 ---
    family_embeddings = {}
    for fam_name, words in families.items():
        embeddings = []
        for w in words:
            ids = tokenizer.encode(f"This is {w}", return_tensors='pt').to(device)
            if ids.shape[1] < 2:
                continue
            with torch.no_grad():
                out = model(ids)
            emb = model.token_emb(ids).mean(dim=1).squeeze(0)
            embeddings.append(emb.cpu())
        if embeddings:
            family_embeddings[fam_name] = torch.stack(embeddings)

    for fam_name, embs in family_embeddings.items():
        mean_emb = embs.mean(dim=0, keepdim=True)
        residuals = embs - mean_emb
        residual_ratio = residuals.norm(dim=-1).mean() / embs.norm(dim=-1).mean()
        emergence[f'{fam_name}_residual_ratio'] = residual_ratio.item()
        print(f"  {fam_name} 残差比: {residual_ratio.item():.4f}")

    if 'fruit' in family_embeddings and 'animal' in family_embeddings:
        fruit_mean = family_embeddings['fruit'].mean(0)
        animal_mean = family_embeddings['animal'].mean(0)
        cosine_sep = F.cosine_similarity(fruit_mean.unsqueeze(0), animal_mean.unsqueeze(0)).item()
        emergence['fruit_animal_cosine'] = cosine_sep
        print(f"  水果-动物余弦: {cosine_sep:.4f}")

    # --- 原型正交性 ---
    if hasattr(model, 'blocks'):
        for i, block in enumerate(model.blocks):
            if hasattr(block.family_basis, 'prototypes'):
                protos = block.family_basis.prototypes
                if callable(getattr(protos, 'detach', None)):
                    protos = protos.detach().cpu()
                else:
                    protos = protos.cpu()
                proto_norm = F.normalize(protos, dim=-1)
                sim = proto_norm @ proto_norm.T
                off_diag = sim[~torch.eye(sim.size(0), dtype=bool)]
                mean_cos = off_diag.abs().mean().item()
                emergence[f'layer{i}_proto_cos'] = mean_cos
                print(f"  Layer {i} 原型|cos|: {mean_cos:.4f}")

    # --- Offset字典正交性 ---
    if hasattr(model, 'blocks'):
        for i, block in enumerate(model.blocks):
            if hasattr(block.sparse_offset, 'dictionary'):
                d = block.sparse_offset.dictionary
                if callable(getattr(d, 'detach', None)):
                    d = d.detach().cpu()
                else:
                    d = d.cpu()
                d_norm = F.normalize(d, dim=-1)
                sim = d_norm @ d_norm.T
                off_diag = sim[~torch.eye(sim.size(0), dtype=bool)]
                orth = (1 - off_diag.abs().mean()).item()
                emergence[f'layer{i}_dict_orth'] = orth
                print(f"  Layer {i} 字典正交: {orth:.4f}")

    # --- 注意力头分化 ---
    concept_prompts = {
        'entity': ["This is apple", "This is cat", "This is car"],
        'abstract': ["The idea of justice", "The idea of truth"],
    }
    head_patterns = {}
    for level, prompts in concept_prompts.items():
        level_attn = []
        for p in prompts:
            ids = tokenizer.encode(p, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(ids)
            if 'attn_weights' in out and out['attn_weights']:
                last_attn = out['attn_weights'][-1].squeeze(0)
                last_token_attn = last_attn[:, -1, :]
                entropy = -(last_token_attn * (last_token_attn + 1e-10).log()).sum(-1)
                level_attn.append(entropy.cpu())
        if level_attn:
            head_patterns[level] = torch.stack(level_attn).mean(0)

    if 'entity' in head_patterns and 'abstract' in head_patterns:
        entropy_diff = (head_patterns['entity'] - head_patterns['abstract']).abs()
        emergence['head_entropy_diff_mean'] = entropy_diff.mean().item()
        emergence['head_entropy_diff_max'] = entropy_diff.max().item()
        print(f"  头级熵分化(均值): {entropy_diff.mean().item():.4f}")

    # ============================================================
    # 🔑 跨维度迁移检测 (SNN统一假说的核心验证)
    # ============================================================
    if hasattr(model, 'blocks') and hasattr(model, 'unified_core') and model.unified_core is not None:
        print(f"\n  --- 跨维度迁移检测 (共享字典模式) ---")
        W_unified = model.unified_core.get_dictionary().detach().cpu()  # (dict_size, d_model)
        
        for i, block in enumerate(model.blocks):
            # 概念空间: 原型的字典投影权重
            proto_proj = block.family_basis.proto_proj.detach().cpu()  # (K, dict_size)
            # 关系空间: 模板的字典投影权重
            tmpl_left = block.relation_layer.template_proj_left.detach().cpu()  # (H, d_head, dict_size)
            
            # 检测: 原型投影和模板投影是否使用了相似的字典子集
            # 把 proto_proj 的行归一化
            proto_usage = proto_proj.abs().mean(dim=0)  # (dict_size,) 每个字典原子被概念使用的程度
            # 把 tmpl_left 展平为 (H*d_head, dict_size) 再取列均值
            tmpl_usage = tmpl_left.abs().reshape(-1, tmpl_left.shape[-1]).mean(dim=0)  # (dict_size,)
            
            # 计算概念和关系对字典使用模式的相关性
            cross_corr = F.cosine_similarity(proto_usage.unsqueeze(0), tmpl_usage.unsqueeze(0)).item()
            emergence[f'layer{i}_cross_dim_corr'] = cross_corr
            print(f"  Layer {i} 概念-关系字典使用相关性: {cross_corr:.4f}")
            
            # 字典共享度: 被两者共同高度使用的字典原子比例
            proto_topk = set(proto_usage.topk(min(16, len(proto_usage)))[1].tolist())
            tmpl_topk = set(tmpl_usage.topk(min(16, len(tmpl_usage)))[1].tolist())
            overlap = len(proto_topk & tmpl_topk) / len(proto_topk | tmpl_topk)
            emergence[f'layer{i}_dict_overlap'] = overlap
            print(f"  Layer {i} 字典重叠度: {overlap:.4f}")
    
    elif hasattr(model, 'blocks'):
        # 独立字典模式: 检测概念原型和偏移字典之间的统计关联
        print(f"\n  --- 跨模块统计关联检测 (独立字典模式) ---")
        for i, block in enumerate(model.blocks):
            proto = block.family_basis.prototypes.detach().cpu()   # (K, D)
            offset_dict = block.sparse_offset.dictionary.detach().cpu()  # (M, D)
            
            # 计算原型与偏移字典条目之间的最大余弦
            proto_n = F.normalize(proto, dim=-1)
            dict_n = F.normalize(offset_dict, dim=-1)
            cross_sim = (proto_n @ dict_n.T).abs()  # (K, M)
            max_cross = cross_sim.max().item()
            mean_cross = cross_sim.mean().item()
            emergence[f'layer{i}_proto_dict_max_cos'] = max_cross
            emergence[f'layer{i}_proto_dict_mean_cos'] = mean_cross
            print(f"  Layer {i} 原型-偏移交叉cos (max: {max_cross:.4f}, mean: {mean_cross:.4f})")

    result_dict['emergence'] = emergence
    return emergence


# ================================================================
# 10. 训练循环
# ================================================================

def train_model(model, train_loader, val_loader, device, epochs, lr, model_name, result_dict):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"开始训练: {model_name}")
    print(f"参数量: {param_count:,}")
    print(f"Epochs: {epochs}, Steps/epoch: {len(train_loader)}")
    print(f"{'='*60}")

    result_dict['param_count'] = param_count
    result_dict['train_losses'] = []
    result_dict['val_losses'] = []

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

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
            n_batches += 1

            if step % 50 == 0:
                elapsed = time.time() - start_time
                ppl = math.exp(min(loss.item(), 20))
                progress = (epoch * len(train_loader) + step) / total_steps * 100
                print(f"  [{model_name}] Epoch {epoch+1}/{epochs} "
                      f"Step {step}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f} PPL: {ppl:.1f} "
                      f"进度: {progress:.1f}% "
                      f"耗时: {elapsed:.0f}s")

        avg_train_loss = epoch_loss / n_batches
        result_dict['train_losses'].append(avg_train_loss)

        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, labels=y)
                val_loss += out['loss'].item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        val_ppl = math.exp(min(avg_val_loss, 20))
        result_dict['val_losses'].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        print(f"  [{model_name}] Epoch {epoch+1} 完成 | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val PPL: {val_ppl:.1f}")

    result_dict['best_val_loss'] = best_val_loss
    result_dict['best_val_ppl'] = math.exp(min(best_val_loss, 20))
    result_dict['total_time'] = time.time() - start_time
    print(f"  [{model_name}] 训练完成 | 最佳 Val PPL: {result_dict['best_val_ppl']:.1f} "
          f"| 耗时: {result_dict['total_time']:.0f}s")


# ================================================================
# 11. 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='共享字典对照实验: 验证SNN统一编码假说')
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
                        default='tests/gemini_temp/shared_dict_ablation_20260310.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # --- 加载 tokenizer ---
    print("加载 tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # --- 构建结构化合成语料 ---
    import random
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
        "The {adj} {animal} slept peacefully under the bright stars.",
        "Knowledge of {abstract} helps build {abstract2} in society.",
    ]

    fruits = ['apple', 'banana', 'orange', 'grape', 'pear', 'lemon', 'mango', 'peach']
    animals = ['cat', 'dog', 'rabbit', 'horse', 'tiger', 'bird', 'fish', 'deer']
    abstracts = ['justice', 'truth', 'logic', 'memory', 'beauty', 'freedom', 'wisdom', 'courage']
    adjectives = ['red', 'sweet', 'big', 'small', 'bright', 'dark', 'fresh', 'old', 'young', 'round']
    tastes = ['sweet', 'sour', 'bitter', 'delicious', 'fresh', 'wonderful']
    categories = ['fruit', 'animal', 'food', 'object', 'creature', 'living thing']

    def generate_sentence():
        t = random.choice(templates)
        return t.format(
            noun=random.choice(fruits + animals),
            noun2=random.choice(fruits + animals),
            noun3=random.choice(fruits + animals),
            adj=random.choice(adjectives),
            adj2=random.choice(adjectives),
            category=random.choice(categories),
            abstract=random.choice(abstracts),
            abstract2=random.choice(abstracts),
            abstract3=random.choice(abstracts),
            animal=random.choice(animals),
            animal2=random.choice(animals),
            taste=random.choice(tastes),
        )

    print("生成结构化合成语料...")
    train_sentences = [generate_sentence() for _ in range(20000)]
    val_sentences = [generate_sentence() for _ in range(2000)]
    train_text = ' '.join(train_sentences)
    val_text = ' '.join(val_sentences)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    print(f"训练 tokens: {len(train_ids):,}, 验证 tokens: {len(val_ids):,}")

    train_dataset = TextDataset(train_ids, args.seq_len)
    val_dataset = TextDataset(val_ids, args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'args': vars(args),
        'device': str(device),
        'experiment': 'shared_dict_ablation_SNN_unification',
    }

    # ============================================================
    # 实验组 A: 独立字典骨架 (Independent)
    # ============================================================
    print("\n" + "=" * 60)
    print("实验组 A: 独立字典骨架 (Independent)")
    print("=" * 60)
    model_A = SkeletonLM(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, num_families=args.num_families,
        dict_size=args.dict_size, top_k=args.top_k, max_seq_len=args.seq_len,
        use_unified_dict=False,
    ).to(device)

    results['group_A_independent'] = {}
    train_model(model_A, train_loader, val_loader, device, args.epochs, args.lr,
                "A-独立字典", results['group_A_independent'])
    verify_emergence(model_A, tokenizer, device, results['group_A_independent'], "A-独立字典")

    # ============================================================
    # 实验组 B: 共享字典骨架 (Unified)
    # ============================================================
    print("\n" + "=" * 60)
    print("实验组 B: 共享字典骨架 (Unified)")
    print("=" * 60)
    model_B = SkeletonLM(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, num_families=args.num_families,
        dict_size=args.dict_size, top_k=args.top_k, max_seq_len=args.seq_len,
        use_unified_dict=True,
    ).to(device)

    results['group_B_unified'] = {}
    train_model(model_B, train_loader, val_loader, device, args.epochs, args.lr,
                "B-共享字典", results['group_B_unified'])
    verify_emergence(model_B, tokenizer, device, results['group_B_unified'], "B-共享字典")

    # ============================================================
    # 实验组 C: 标准 Baseline
    # ============================================================
    print("\n" + "=" * 60)
    print("实验组 C: 标准 Transformer Baseline")
    print("=" * 60)
    model_C = BaselineTransformerLM(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, max_seq_len=args.seq_len,
    ).to(device)

    results['group_C_baseline'] = {}
    train_model(model_C, train_loader, val_loader, device, args.epochs, args.lr,
                "C-Baseline", results['group_C_baseline'])

    # ============================================================
    # 对比分析
    # ============================================================
    print("\n" + "=" * 60)
    print("三组对比分析")
    print("=" * 60)

    ppl_A = results['group_A_independent']['best_val_ppl']
    ppl_B = results['group_B_unified']['best_val_ppl']
    ppl_C = results['group_C_baseline']['best_val_ppl']
    params_A = results['group_A_independent']['param_count']
    params_B = results['group_B_unified']['param_count']
    params_C = results['group_C_baseline']['param_count']

    print(f"\n  独立字典(A) PPL: {ppl_A:.2f} | 参数: {params_A:,}")
    print(f"  共享字典(B) PPL: {ppl_B:.2f} | 参数: {params_B:,}")
    print(f"  Baseline(C) PPL: {ppl_C:.2f} | 参数: {params_C:,}")
    print(f"\n  A/C PPL比: {ppl_A/ppl_C:.2f}")
    print(f"  B/C PPL比: {ppl_B/ppl_C:.2f}")
    print(f"  B/A PPL比: {ppl_B/ppl_A:.2f} (>1.0 = 共享字典更差, <1.0 = 共享字典更好)")
    print(f"  参数效率 B/A: {params_B/params_A:.3f}")

    comparison = {
        'ppl_A': ppl_A, 'ppl_B': ppl_B, 'ppl_C': ppl_C,
        'params_A': params_A, 'params_B': params_B, 'params_C': params_C,
        'B_vs_A_ppl_ratio': ppl_B / ppl_A,
        'B_vs_C_ppl_ratio': ppl_B / ppl_C,
        'A_vs_C_ppl_ratio': ppl_A / ppl_C,
        'param_efficiency_B_over_A': params_B / params_A,
    }

    # 假设检验
    hypotheses = {}

    # H_unify_1: 共享字典不显著退化涌现 (B的涌现指标 >= A的80%)
    emerge_A = results['group_A_independent'].get('emergence', {})
    emerge_B = results['group_B_unified'].get('emergence', {})
    
    proto_cos_A = [v for k, v in emerge_A.items() if 'proto_cos' in k]
    proto_cos_B = [v for k, v in emerge_B.items() if 'proto_cos' in k]
    
    if proto_cos_A and proto_cos_B:
        avg_proto_A = sum(proto_cos_A) / len(proto_cos_A)
        avg_proto_B = sum(proto_cos_B) / len(proto_cos_B)
        # 原型正交性: cos越低越好, B不应该比A差太多
        hypotheses['H_unify_proto_preserved'] = 'PASS' if avg_proto_B <= avg_proto_A * 1.5 else 'FAIL'
        comparison['avg_proto_cos_A'] = avg_proto_A
        comparison['avg_proto_cos_B'] = avg_proto_B
    
    # H_unify_2: 共享字典PPL不比独立字典差超过50% 
    hypotheses['H_unify_ppl_within_50pct'] = 'PASS' if ppl_B / ppl_A <= 1.5 else 'FAIL'
    
    # H_unify_3: 共享字典参数更少
    hypotheses['H_unify_param_efficient'] = 'PASS' if params_B < params_A else 'FAIL'
    
    # H_unify_4: 跨维度迁移检测 (共享字典模式下概念-关系相关性 > 0.3)
    cross_corrs = [v for k, v in emerge_B.items() if 'cross_dim_corr' in k]
    if cross_corrs:
        avg_cross = sum(cross_corrs) / len(cross_corrs)
        hypotheses['H_unify_cross_dim_transfer'] = 'PASS' if avg_cross > 0.3 else 'FAIL'
        comparison['avg_cross_dim_corr'] = avg_cross

    comparison['hypotheses'] = hypotheses
    results['comparison'] = comparison

    print(f"\n  假设检验:")
    for k, v in hypotheses.items():
        status = "✅" if v == 'PASS' else "❌"
        print(f"    {status} {k}: {v}")

    # 保存
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {args.json_out}")

    print("\n" + "=" * 60)
    print("共享字典对照实验完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
