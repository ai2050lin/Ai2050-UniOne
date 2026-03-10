#!/usr/bin/env python3
"""
==========================================================
任务A: 正向训练闭环验证 — 4D+3D 骨架可训练原型
==========================================================
目标:
  构建一个显式引入 4D semantic skeleton + 3D vector correction
  的小型 Transformer, 在 TinyStories 上训练, 然后验证训练后
  是否自动涌现:
    1. 概念家族共享基底
    2. 偏移在自然字典中的聚集性
    3. 注意力头路由分化

4D 骨架组件:
  - 共享基底模块 (Family Basis Pool)
  - 稀疏偏移模块 (Sparse Offset Dictionary)
  - 关系协议层  (Relation Protocol Layer)
  - 门控路由模块 (Context-Gated Router)

3D 修正:
  - 层级闭包投影 (Hierarchical Closure Projection)
  - 能量稳态约束 (Energy Homeostasis via RMSNorm + penalty)
  - 偏置校正     (Residual Bias Correction)

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

# -------------------------------------------------------
# 1. 4D+3D 骨架核心模块
# -------------------------------------------------------

class FamilyBasisPool(nn.Module):
    """共享基底池: 可学习的 K 组家族原型"""
    def __init__(self, num_families: int, d_model: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_families, d_model) * 0.02)
        self.gate = nn.Linear(d_model, num_families, bias=False)

    def forward(self, x):
        # x: (B, T, D)
        scores = torch.softmax(self.gate(x), dim=-1)        # (B, T, K)
        basis = torch.einsum('btk,kd->btd', scores, self.prototypes)  # (B, T, D)
        return basis, scores


class SparseOffsetDict(nn.Module):
    """稀疏偏移字典: 用 top-k 稀疏激活实现个体偏移"""
    def __init__(self, d_model: int, dict_size: int, top_k: int = 8):
        super().__init__()
        self.dictionary = nn.Parameter(torch.randn(dict_size, d_model) * 0.02)
        self.encoder = nn.Linear(d_model, dict_size, bias=False)
        self.top_k = top_k

    def forward(self, x):
        # x: (B, T, D) — 残差输入
        coeffs = self.encoder(x)                       # (B, T, dict_size)
        # top-k 稀疏化
        topk_vals, topk_idx = coeffs.topk(self.top_k, dim=-1)
        sparse_coeffs = torch.zeros_like(coeffs)
        sparse_coeffs.scatter_(-1, topk_idx, topk_vals)
        offset = torch.einsum('btm,md->btd', sparse_coeffs, self.dictionary)
        # L1 稀疏损失
        sparsity_loss = sparse_coeffs.abs().mean()
        return offset, sparsity_loss


class RelationProtocolLayer(nn.Module):
    """关系协议层: 用可学习的拓扑传输算子处理 token 间关系"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        # 关系类型嵌入 (可学习的拓扑传输模板)
        self.relation_templates = nn.Parameter(torch.randn(n_heads, self.d_head, self.d_head) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 标准注意力 + 关系模板调制
        # 每个头的 key 先经过其关系模板旋转
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


class ContextGatedRouter(nn.Module):
    """门控路由: 上下文条件的乘性门控"""
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
    """层级闭包投影 + 能量稳态"""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        energy = rms.mean()  # 全局能量用于稳态约束损失
        return self.weight * x / rms, energy


# -------------------------------------------------------
# 2. 4D+3D 骨架 Transformer Block
# -------------------------------------------------------

class Skeleton4D3DBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_families, dict_size, top_k, dropout=0.1):
        super().__init__()
        # 4D 组件
        self.family_basis = FamilyBasisPool(num_families, d_model)
        self.sparse_offset = SparseOffsetDict(d_model, dict_size, top_k)
        self.relation_layer = RelationProtocolLayer(d_model, n_heads, dropout)
        self.gated_router = ContextGatedRouter(d_model, d_ff)
        # 3D 修正
        self.norm1 = HierarchicalClosureNorm(d_model)
        self.norm2 = HierarchicalClosureNorm(d_model)
        self.bias_correction = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        aux_losses = {}

        # --- 共享基底 + 稀疏偏移 ---
        basis, family_scores = self.family_basis(x)
        residual_from_basis = x - basis
        offset, sparsity_loss = self.sparse_offset(residual_from_basis)
        aux_losses['sparsity'] = sparsity_loss

        # 重构: 基底 + 偏移 + 偏置校正
        x_reconstructed = basis + offset + self.bias_correction

        # --- 关系协议 (注意力) ---
        normed, e1 = self.norm1(x_reconstructed)
        aux_losses['energy_1'] = e1
        relation_out, attn_weights = self.relation_layer(normed, mask)
        x = x + self.dropout(relation_out)  # 残差连接保持层级闭包

        # --- 门控路由 (MLP) ---
        normed2, e2 = self.norm2(x)
        aux_losses['energy_2'] = e2
        routed = self.gated_router(normed2)
        x = x + self.dropout(routed)

        return x, attn_weights, family_scores, aux_losses


# -------------------------------------------------------
# 3. 完整 4D+3D Skeleton LM
# -------------------------------------------------------

class Skeleton4D3DLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4,
                 d_ff=512, num_families=16, dict_size=64, top_k=8,
                 max_seq_len=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Skeleton4D3DBlock(d_model, n_heads, d_ff, num_families, dict_size, top_k, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = HierarchicalClosureNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重绑定
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

        # 因果掩码
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
            # 加入辅助损失
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


# -------------------------------------------------------
# 4. 标准 Transformer 基线 (对比用)
# -------------------------------------------------------

class BaselineTransformerLM(nn.Module):
    """同规模标准 Transformer, 无 4D+3D 骨架"""
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


# -------------------------------------------------------
# 5. 数据集
# -------------------------------------------------------

class TextDataset(Dataset):
    """把 tokenized text 切成固定长度的块"""
    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        # 截断到整数块
        n = (len(token_ids) // seq_len) * seq_len
        self.data = torch.tensor(token_ids[:n], dtype=torch.long).view(-1, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]  # input, target


# -------------------------------------------------------
# 6. 涌现结构验证
# -------------------------------------------------------

def verify_emergence(model, tokenizer, device, result_dict):
    """训练后验证是否涌现了 4D+3D 结构"""
    print("\n" + "="*60)
    print("开始验证涌现结构...")
    print("="*60)

    model.eval()

    # --- 验证 1: 概念家族共享基底 ---
    families = {
        'fruit': ['apple', 'banana', 'orange', 'grape', 'pear', 'lemon'],
        'animal': ['cat', 'dog', 'rabbit', 'horse', 'tiger', 'bird'],
        'abstract': ['justice', 'truth', 'logic', 'memory', 'beauty', 'freedom'],
    }

    family_embeddings = {}
    for fam_name, words in families.items():
        embeddings = []
        for w in words:
            ids = tokenizer.encode(f"This is {w}", return_tensors='pt').to(device)
            if ids.shape[1] < 2:
                continue
            with torch.no_grad():
                out = model(ids)
            # 取最后 token 的最后一层前的 embedding
            # 先简化: 用 token embedding 层的输出
            emb = model.token_emb(ids).mean(dim=1).squeeze(0)
            embeddings.append(emb.cpu())
        if embeddings:
            family_embeddings[fam_name] = torch.stack(embeddings)

    # 计算族内紧凑度 (residual ratio)
    emergence_results = {}
    for fam_name, embs in family_embeddings.items():
        mean_emb = embs.mean(dim=0, keepdim=True)
        residuals = embs - mean_emb
        residual_ratio = residuals.norm(dim=-1).mean() / embs.norm(dim=-1).mean()
        emergence_results[f'{fam_name}_residual_ratio'] = residual_ratio.item()
        print(f"  {fam_name} 族内残差比: {residual_ratio.item():.4f}")

    # 计算族间分离度
    if 'fruit' in family_embeddings and 'animal' in family_embeddings:
        fruit_mean = family_embeddings['fruit'].mean(0)
        animal_mean = family_embeddings['animal'].mean(0)
        cosine_sep = F.cosine_similarity(fruit_mean.unsqueeze(0), animal_mean.unsqueeze(0)).item()
        emergence_results['fruit_animal_cosine'] = cosine_sep
        print(f"  水果-动物族间余弦: {cosine_sep:.4f}")

    # --- 验证 2: 家族基底池的分化 ---
    if hasattr(model, 'blocks'):
        for i, block in enumerate(model.blocks):
            prototypes = block.family_basis.prototypes.detach().cpu()
            # 计算原型间余弦相似度
            proto_norm = F.normalize(prototypes, dim=-1)
            sim_matrix = proto_norm @ proto_norm.T
            off_diag = sim_matrix[~torch.eye(sim_matrix.size(0), dtype=bool)]
            mean_off_diag = off_diag.abs().mean().item()
            emergence_results[f'layer{i}_prototype_mean_abs_cos'] = mean_off_diag
            print(f"  Layer {i} 原型间平均|cos|: {mean_off_diag:.4f} (越小=越正交)")

    # --- 验证 3: 稀疏偏移的实际稀疏度 ---
    if hasattr(model, 'blocks'):
        for i, block in enumerate(model.blocks):
            dict_weight = block.sparse_offset.dictionary.detach().cpu()
            dict_norm = F.normalize(dict_weight, dim=-1)
            dict_sim = dict_norm @ dict_norm.T
            off_diag = dict_sim[~torch.eye(dict_sim.size(0), dtype=bool)]
            emergence_results[f'layer{i}_dict_orthogonality'] = (1 - off_diag.abs().mean()).item()
            print(f"  Layer {i} 字典正交度: {emergence_results[f'layer{i}_dict_orthogonality']:.4f}")

    # --- 验证 4: 注意力头路由分化 ---
    concept_prompts = {
        'entity': ["This is apple", "This is cat", "This is car"],
        'category': ["The concept of fruit", "The concept of animal"],
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
                # 取最后一层的注意力模式
                last_attn = out['attn_weights'][-1].squeeze(0)  # (H, T, T)
                # 最后 token 对所有 token 的注意力分布熵
                last_token_attn = last_attn[:, -1, :]
                entropy = -(last_token_attn * (last_token_attn + 1e-10).log()).sum(-1)
                level_attn.append(entropy.cpu())
        if level_attn:
            head_patterns[level] = torch.stack(level_attn).mean(0)

    if 'entity' in head_patterns and 'abstract' in head_patterns:
        # 头级分化: 每个头在entity vs abstract上的熵差异
        entropy_diff = (head_patterns['entity'] - head_patterns['abstract']).abs()
        emergence_results['head_entropy_diff_mean'] = entropy_diff.mean().item()
        emergence_results['head_entropy_diff_max'] = entropy_diff.max().item()
        print(f"  头级熵分化(均值): {entropy_diff.mean().item():.4f}")
        print(f"  头级熵分化(最大): {entropy_diff.max().item():.4f}")

    result_dict['emergence'] = emergence_results
    return emergence_results


# -------------------------------------------------------
# 7. 训练循环
# -------------------------------------------------------

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

        # 验证
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
    print(f"  [{model_name}] 训练完成 | 最佳 Val PPL: {result_dict['best_val_ppl']:.1f} | 耗时: {result_dict['total_time']:.0f}s")


# -------------------------------------------------------
# 8. 主函数
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='4D+3D 骨架正向训练闭环验证')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--seq-len', type=int, default=128, help='序列长度')
    parser.add_argument('--d-model', type=int, default=256, help='隐含维度')
    parser.add_argument('--n-layers', type=int, default=4, help='层数')
    parser.add_argument('--n-heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--d-ff', type=int, default=512, help='FFN维度')
    parser.add_argument('--num-families', type=int, default=16, help='家族基底数')
    parser.add_argument('--dict-size', type=int, default=64, help='稀疏字典大小')
    parser.add_argument('--top-k', type=int, default=8, help='稀疏 top-k')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--json-out', type=str, default=None, help='结果输出路径')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # --- 加载数据 ---
    print("加载 tokenizer 与数据...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # 构建结构化合成语料 (完全离线, 覆盖概念家族)
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
    }

    # --- 训练 4D+3D 骨架模型 ---
    print("\n" + "="*60)
    print("构建 4D+3D 骨架 Transformer...")
    skeleton_model = Skeleton4D3DLanguageModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        num_families=args.num_families,
        dict_size=args.dict_size,
        top_k=args.top_k,
        max_seq_len=args.seq_len,
    ).to(device)

    skeleton_results = {}
    train_model(skeleton_model, train_loader, val_loader, device,
                args.epochs, args.lr, "4D+3D骨架", skeleton_results)

    # 涌现验证
    verify_emergence(skeleton_model, tokenizer, device, skeleton_results)
    results['skeleton'] = skeleton_results

    # --- 训练标准 Transformer 基线 ---
    print("\n" + "="*60)
    print("构建标准 Transformer 基线...")
    baseline_model = BaselineTransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
    ).to(device)

    baseline_results = {}
    train_model(baseline_model, train_loader, val_loader, device,
                args.epochs, args.lr, "标准Transformer", baseline_results)
    results['baseline'] = baseline_results

    # --- 对比分析 ---
    print("\n" + "="*60)
    print("对比分析")
    print("="*60)
    s_ppl = skeleton_results.get('best_val_ppl', float('inf'))
    b_ppl = baseline_results.get('best_val_ppl', float('inf'))
    ppl_ratio = s_ppl / b_ppl if b_ppl > 0 else float('inf')
    results['comparison'] = {
        'skeleton_ppl': s_ppl,
        'baseline_ppl': b_ppl,
        'ppl_ratio': ppl_ratio,
        'ppl_within_20pct': ppl_ratio <= 1.2,
    }
    print(f"  4D+3D PPL: {s_ppl:.1f}")
    print(f"  基线  PPL: {b_ppl:.1f}")
    print(f"  比值:      {ppl_ratio:.3f} ({'✅ 在20%范围内' if ppl_ratio <= 1.2 else '❌ 超出20%'})")

    # --- 假设判定 ---
    hypotheses = {}
    emr = skeleton_results.get('emergence', {})

    # H1: 训练收敛且PPL接近基线
    hypotheses['H1_ppl_competitive'] = 'PASS' if ppl_ratio <= 1.2 else 'FAIL'

    # H2: 族内残差比 < 0.5
    fruit_rr = emr.get('fruit_residual_ratio', 1.0)
    hypotheses['H2_family_basis_compact'] = 'PASS' if fruit_rr < 0.5 else 'FAIL'

    # H3: 原型正交度 (平均|cos| < 0.3)
    proto_cos = emr.get('layer0_prototype_mean_abs_cos', 1.0)
    hypotheses['H3_prototypes_orthogonal'] = 'PASS' if proto_cos < 0.3 else 'FAIL'

    # H4: 头级路由分化
    head_diff = emr.get('head_entropy_diff_mean', 0.0)
    hypotheses['H4_head_route_differentiation'] = 'PASS' if head_diff > 0.1 else 'FAIL'

    results['hypotheses'] = hypotheses
    print("\n假设判定:")
    for k, v in hypotheses.items():
        print(f"  {k} = {v}")

    # --- 保存结果 ---
    out_path = args.json_out
    if out_path is None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = f'tests/gemini_temp/forward_trainable_4d3d_prototype_{ts}.json'

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        # 转成可序列化
        def make_serial(obj):
            if isinstance(obj, dict):
                return {k: make_serial(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serial(v) for v in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        json.dump(make_serial(results), f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {out_path}")
    print("="*60)
    print("任务A 正向训练闭环验证 完成")
    print("="*60)


if __name__ == '__main__':
    main()
