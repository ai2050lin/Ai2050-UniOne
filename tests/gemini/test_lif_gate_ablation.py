#!/usr/bin/env python3
"""
==========================================================
步骤3: LIF门控统一 — 用脉冲神经元替换sigmoid门控
==========================================================
核心假设:
  SNN中的门控路由由LIF(Leaky Integrate-and-Fire)神经元实现,
  而非sigmoid连续门控。用LIF替换sigmoid后:
  1. 稀疏性应该提升(LIF天然产生稀疏脉冲)
  2. 头分化应该增强(硬阈值产生更强的WTA效果)
  3. PPL可能会变差(梯度通过脉冲更困难)

实验设计 (四组对照):
  A: sigmoid门控 (现有)
  B: LIF门控 (替换sigmoid为可微LIF)
  C: 硬阈值门控 (ReLU + 阈值截断)
  D: Baseline (无骨架)

Author: Gemini AGI Research
Date:   2026-03-10
"""

import os, sys, json, math, time, random, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from test_shared_dict_ablation import (
    UnifiedDictionaryCore, SharedFamilyBasisPool, SharedSparseOffsetDict,
    SharedRelationProtocolLayer, HierarchicalClosureNorm,
    SkeletonBlock, TextDataset, BaselineTransformerLM
)


# ================================================================
# 1. LIF 神经元门控 (可微分版)
# ================================================================

class SurrogateSpikeFunction(torch.autograd.Function):
    """替代梯度: 前向用阶跃, 反向用sigmoid近似"""
    @staticmethod
    def forward(ctx, membrane, threshold):
        ctx.save_for_backward(membrane, torch.tensor(threshold))
        return (membrane >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold = ctx.saved_tensors
        # sigmoid替代梯度
        sigmoid = torch.sigmoid(4.0 * (membrane - threshold.item()))
        return grad_output * sigmoid * (1 - sigmoid) * 4.0, None


def surrogate_spike(membrane, threshold=1.0):
    return SurrogateSpikeFunction.apply(membrane, threshold)


class LIFGatedRouter(nn.Module):
    """
    LIF门控路由: 用漏电积分-点火神经元替代sigmoid
    
    膜电位动力学:
      V(t+1) = beta * V(t) + W * x(t)     # 漏电积分
      spike = 1 if V >= theta else 0         # 阈值点火
      V = V - theta * spike                  # 重置
    
    对应SNN原理:
      - beta(漏电系数) = 膜时间常数, 控制记忆衰减
      - theta(阈值) = 点火门限, 产生稀疏脉冲
      - 替代梯度 = 使BP可通过脉冲层
    """
    def __init__(self, d_model: int, d_ff: int, beta: float = 0.8, threshold: float = 1.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.beta = beta
        self.threshold = threshold
        # 可学习的阈值
        self.theta = nn.Parameter(torch.ones(d_ff) * threshold)
    
    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        gate_input = self.gate_proj(x)  # (B, T, d_ff)
        
        # LIF时序积分: 沿token维度模拟时序积分
        membrane = torch.zeros(B, gate_input.shape[-1], device=x.device)
        spikes_all = []
        
        for t in range(T):
            # 漏电积分
            membrane = self.beta * membrane + gate_input[:, t, :]
            # 点火 (替代梯度)
            spike = surrogate_spike(membrane, self.threshold)
            # 重置
            membrane = membrane - self.theta * spike
            spikes_all.append(spike)
        
        gate = torch.stack(spikes_all, dim=1)  # (B, T, d_ff)
        up = self.up_proj(x)
        return self.down_proj(gate * F.silu(up))


class HardThresholdRouter(nn.Module):
    """硬阈值门控: ReLU + top-k截断"""
    def __init__(self, d_model: int, d_ff: int, top_k_ratio: float = 0.25):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.top_k = max(1, int(d_ff * top_k_ratio))
    
    def forward(self, x):
        gate_raw = F.relu(self.gate_proj(x))  # (B, T, d_ff)
        # Top-k稀疏化
        topk_vals, topk_idx = gate_raw.topk(self.top_k, dim=-1)
        gate = torch.zeros_like(gate_raw)
        gate.scatter_(-1, topk_idx, topk_vals)
        up = self.up_proj(x)
        return self.down_proj(gate * F.silu(up))


class SigmoidGatedRouter(nn.Module):
    """原始sigmoid门控"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * F.silu(up))


# ================================================================
# 2. 可切换门控类型的Block和Model
# ================================================================

class GateTestBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_families, dict_size, top_k,
                 dropout=0.1, unified_core=None, gate_type='sigmoid'):
        super().__init__()
        self.family_basis = SharedFamilyBasisPool(num_families, d_model, unified_core)
        self.sparse_offset = SharedSparseOffsetDict(d_model, unified_core, top_k)
        self.relation_layer = SharedRelationProtocolLayer(d_model, n_heads, unified_core, dropout)
        
        if gate_type == 'sigmoid':
            self.gated_router = SigmoidGatedRouter(d_model, d_ff)
        elif gate_type == 'lif':
            self.gated_router = LIFGatedRouter(d_model, d_ff, beta=0.8, threshold=1.0)
        elif gate_type == 'hard_threshold':
            self.gated_router = HardThresholdRouter(d_model, d_ff)
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
        
        self.norm1 = HierarchicalClosureNorm(d_model)
        self.norm2 = HierarchicalClosureNorm(d_model)
        self.bias_correction = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        aux_losses = {}
        basis, family_scores = self.family_basis(x)
        residual = x - basis
        offset, sparsity_loss = self.sparse_offset(residual)
        aux_losses['sparsity'] = sparsity_loss
        x_recon = basis + offset + self.bias_correction
        normed, e1 = self.norm1(x_recon)
        aux_losses['energy_1'] = e1
        rel_out, attn = self.relation_layer(normed, mask)
        x = x + self.dropout(rel_out)
        normed2, e2 = self.norm2(x)
        aux_losses['energy_2'] = e2
        routed = self.gated_router(normed2)
        x = x + self.dropout(routed)
        return x, attn, family_scores, aux_losses


class GateTestLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4,
                 d_ff=512, num_families=16, dict_size=64, top_k=8,
                 max_seq_len=256, dropout=0.1, gate_type='sigmoid'):
        super().__init__()
        self.d_model = d_model
        self.gate_type = gate_type
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.unified_core = UnifiedDictionaryCore(dict_size, d_model)
        self.blocks = nn.ModuleList([
            GateTestBlock(d_model, n_heads, d_ff, num_families, dict_size, top_k,
                         dropout, self.unified_core, gate_type)
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
        all_attn, all_fam = [], []
        total_aux = {'sparsity': 0.0, 'energy': 0.0}
        for block in self.blocks:
            x, attn, fam, aux = block(x, mask)
            all_attn.append(attn)
            all_fam.append(fam)
            total_aux['sparsity'] = total_aux['sparsity'] + aux['sparsity']
            total_aux['energy'] = total_aux['energy'] + aux['energy_1'] + aux['energy_2']
        x, _ = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss + 0.01 * total_aux['sparsity'] + 0.001 * total_aux['energy']
        return {'loss': loss, 'logits': logits, 'attn_weights': all_attn,
                'family_scores': all_fam, 'aux_losses': total_aux}


# ================================================================
# 3. 门控激活稀疏性分析
# ================================================================

def analyze_gate_sparsity(model, tokenizer, device, label):
    """分析门控层的激活稀疏性"""
    model.eval()
    prompts = [
        "This is a red apple on the table",
        "The concept of justice is important",
        "A cat ran quickly across the field",
        "We think about truth and beauty",
    ]
    
    results = {}
    for i, block in enumerate(model.blocks):
        gate_activations = []
        for p in prompts:
            ids = tokenizer.encode(p, return_tensors='pt').to(device)
            B, T = ids.shape
            pos = torch.arange(0, T, device=device).unsqueeze(0)
            x = model.token_emb(ids) + model.pos_emb(pos)
            mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                # 前向到当前block
                for j in range(i):
                    x, _, _, _ = model.blocks[j](x, mask)
                
                # 获取gate前的normed输入
                basis, fam = block.family_basis(x)
                residual = x - basis
                offset, _ = block.sparse_offset(residual)
                x_recon = basis + offset + block.bias_correction
                normed, _ = block.norm1(x_recon)
                rel_out, _ = block.relation_layer(normed, mask)
                x_after_rel = x + block.dropout(rel_out)
                normed2, _ = block.norm2(x_after_rel)
                
                # 获取gate输出
                gate_raw = block.gated_router.gate_proj(normed2)
                if hasattr(block.gated_router, 'beta'):  # LIF
                    gate_activation = torch.zeros_like(gate_raw[:, 0, :])
                    membrane = torch.zeros_like(gate_raw[:, 0, :])
                    for t in range(gate_raw.shape[1]):
                        membrane = block.gated_router.beta * membrane + gate_raw[:, t, :]
                        spike = (membrane >= block.gated_router.threshold).float()
                        membrane = membrane - block.gated_router.theta * spike
                        gate_activation = gate_activation + spike
                    gate_activation = gate_activation / gate_raw.shape[1]
                elif hasattr(block.gated_router, 'top_k'):  # HardThreshold
                    gate_activation = F.relu(gate_raw).mean(dim=1)
                else:  # Sigmoid
                    gate_activation = torch.sigmoid(gate_raw).mean(dim=1)
                
                gate_activations.append(gate_activation.cpu())
        
        gates = torch.cat(gate_activations, dim=0)  # (n_prompts, d_ff)
        sparsity = (gates < 0.01).float().mean().item()
        mean_val = gates.mean().item()
        max_val = gates.max().item()
        
        results[f'layer{i}'] = {
            'sparsity': sparsity,
            'mean_activation': mean_val,
            'max_activation': max_val,
        }
        print(f"  [{label}] Layer {i}: 稀疏度={sparsity:.4f} 均值={mean_val:.4f} 最大={max_val:.4f}")
    
    return results


def analyze_head_differentiation(model, tokenizer, device, label):
    """分析注意力头分化"""
    model.eval()
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
            if out['attn_weights']:
                last_attn = out['attn_weights'][-1].squeeze(0)
                last_token_attn = last_attn[:, -1, :]
                entropy = -(last_token_attn * (last_token_attn + 1e-10).log()).sum(-1)
                level_attn.append(entropy.cpu())
        if level_attn:
            head_patterns[level] = torch.stack(level_attn).mean(0)
    
    results = {}
    if 'entity' in head_patterns and 'abstract' in head_patterns:
        diff = (head_patterns['entity'] - head_patterns['abstract']).abs()
        results['entropy_diff_mean'] = diff.mean().item()
        results['entropy_diff_max'] = diff.max().item()
        print(f"  [{label}] 头分化: mean={diff.mean().item():.4f} max={diff.max().item():.4f}")
    return results


# ================================================================
# 4. 训练+评估循环
# ================================================================

def train_and_evaluate(model, train_loader, val_loader, device, epochs, lr, label):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  [{label}] 参数: {params:,}")
    
    result = {'param_count': params, 'train_losses': [], 'val_losses': []}
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        eloss, n = 0, 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            loss = model(x, labels=y)['loss']
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            eloss += loss.item()
            n += 1
            if step % 50 == 0:
                progress = (epoch * len(train_loader) + step) / total_steps * 100
                print(f"  [{label}] Epoch {epoch+1}/{epochs} Step {step} "
                      f"Loss={loss.item():.4f} 进度={progress:.1f}%")
        
        result['train_losses'].append(eloss / n)
        model.eval()
        vloss, vn = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vloss += model(x, labels=y)['loss'].item()
                vn += 1
        avg_vl = vloss / max(vn, 1)
        result['val_losses'].append(avg_vl)
        print(f"  [{label}] Epoch {epoch+1} | Val PPL={math.exp(min(avg_vl,20)):.1f}")
    
    result['best_val_ppl'] = math.exp(min(min(result['val_losses']), 20))
    result['time'] = time.time() - start
    return result


# ================================================================
# 5. 主函数
# ================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='步骤3: LIF门控统一实验')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--json-out', type=str,
                        default='tests/gemini_temp/lif_gate_ablation_20260310.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # 数据
    random.seed(42)
    templates = [
        "The {adj} {noun} is a type of {cat}.", "A {noun} can be {adj} and {adj2}.",
        "{noun} is related to {noun2} but different from {noun3}.",
        "We think about {abs} and {abs2} every day.",
        "The {anim} ran quickly across the field.", "She picked a {adj} {noun} from the tree.",
        "The concept of {abs} is important.",  "A basket of {noun}, {noun2} on the table.",
    ]
    fruits = ['apple','banana','orange','grape','pear','lemon','mango','peach']
    animals = ['cat','dog','rabbit','horse','tiger','bird','fish','deer']
    abstracts = ['justice','truth','logic','memory','beauty','freedom','wisdom','courage']
    adjs = ['red','sweet','big','small','bright','dark','fresh','old']
    cats = ['fruit','animal','food','object']

    def gen():
        return random.choice(templates).format(
            noun=random.choice(fruits+animals), noun2=random.choice(fruits+animals),
            noun3=random.choice(fruits+animals), adj=random.choice(adjs),
            adj2=random.choice(adjs), cat=random.choice(cats),
            abs=random.choice(abstracts), abs2=random.choice(abstracts),
            anim=random.choice(animals),
        )

    train_text = ' '.join([gen() for _ in range(20000)])
    val_text = ' '.join([gen() for _ in range(2000)])
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    train_loader = DataLoader(TextDataset(train_ids, args.seq_len),
                             batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TextDataset(val_ids, args.seq_len),
                           batch_size=args.batch_size, shuffle=False, drop_last=True)
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'experiment': 'lif_gate_ablation_step3',
    }

    gate_types = ['sigmoid', 'lif', 'hard_threshold']
    labels = {'sigmoid': 'A-Sigmoid', 'lif': 'B-LIF', 'hard_threshold': 'C-HardThreshold'}

    for gt in gate_types:
        lb = labels[gt]
        print(f"\n{'='*60}")
        print(f"实验组 {lb}")
        print(f"{'='*60}")
        
        model = GateTestLM(
            vocab_size=vocab_size, d_model=args.d_model, n_layers=4, n_heads=4,
            d_ff=512, num_families=16, dict_size=64, top_k=8,
            max_seq_len=args.seq_len, gate_type=gt,
        ).to(device)
        
        train_result = train_and_evaluate(model, train_loader, val_loader, device,
                                          args.epochs, 3e-4, lb)
        gate_sparsity = analyze_gate_sparsity(model, tokenizer, device, lb)
        head_diff = analyze_head_differentiation(model, tokenizer, device, lb)
        
        results[f'group_{gt}'] = {
            'training': train_result,
            'gate_sparsity': gate_sparsity,
            'head_differentiation': head_diff,
        }

    # Baseline
    print(f"\n{'='*60}")
    print("D-Baseline")
    print(f"{'='*60}")
    baseline = BaselineTransformerLM(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=4, n_heads=4,
        d_ff=512, max_seq_len=args.seq_len,
    ).to(device)
    bl_result = train_and_evaluate(baseline, train_loader, val_loader, device,
                                   args.epochs, 3e-4, 'D-Baseline')
    results['group_baseline'] = {'training': bl_result}

    # 对比
    print(f"\n{'='*60}")
    print("对比分析")
    print(f"{'='*60}")
    
    comparison = {}
    for gt in gate_types:
        lb = labels[gt]
        ppl = results[f'group_{gt}']['training']['best_val_ppl']
        gs = results[f'group_{gt}'].get('gate_sparsity', {})
        hd = results[f'group_{gt}'].get('head_differentiation', {})
        avg_sparsity = sum(v['sparsity'] for v in gs.values()) / max(len(gs), 1) if gs else 0
        head_diff_mean = hd.get('entropy_diff_mean', 0)
        comparison[gt] = {'ppl': ppl, 'avg_gate_sparsity': avg_sparsity,
                         'head_diff_mean': head_diff_mean}
        print(f"  {lb}: PPL={ppl:.1f} 门控稀疏={avg_sparsity:.4f} 头分化={head_diff_mean:.4f}")
    
    bl_ppl = results['group_baseline']['training']['best_val_ppl']
    print(f"  D-Baseline: PPL={bl_ppl:.1f}")
    comparison['baseline'] = {'ppl': bl_ppl}

    # 假设检验
    hypotheses = {}
    s_ppl = comparison['sigmoid']['ppl']
    l_ppl = comparison['lif']['ppl']
    h_ppl = comparison['hard_threshold']['ppl']
    
    # H1: LIF门控稀疏度 > sigmoid稀疏度
    s_sparse = comparison['sigmoid']['avg_gate_sparsity']
    l_sparse = comparison['lif']['avg_gate_sparsity']
    hypotheses['H_lif_more_sparse'] = 'PASS' if l_sparse > s_sparse else 'FAIL'
    print(f"\n  H1: LIF更稀疏: LIF={l_sparse:.4f} vs Sig={s_sparse:.4f} → {hypotheses['H_lif_more_sparse']}")
    
    # H2: LIF门控头分化 >= sigmoid头分化
    s_hd = comparison['sigmoid']['head_diff_mean']
    l_hd = comparison['lif']['head_diff_mean']
    hypotheses['H_lif_better_differentiation'] = 'PASS' if l_hd >= s_hd else 'FAIL'
    print(f"  H2: LIF头分化更强: LIF={l_hd:.4f} vs Sig={s_hd:.4f} → {hypotheses['H_lif_better_differentiation']}")
    
    # H3: LIF PPL不超过sigmoid的2倍
    hypotheses['H_lif_ppl_within_2x'] = 'PASS' if l_ppl / s_ppl <= 2.0 else 'FAIL'
    print(f"  H3: LIF PPL合理: {l_ppl:.1f}/{s_ppl:.1f}={l_ppl/s_ppl:.2f} → {hypotheses['H_lif_ppl_within_2x']}")
    
    # H4: 硬阈值稀疏度 > sigmoid
    h_sparse = comparison['hard_threshold']['avg_gate_sparsity']
    hypotheses['H_hard_more_sparse'] = 'PASS' if h_sparse > s_sparse else 'FAIL'
    print(f"  H4: 硬阈值更稀疏: Hard={h_sparse:.4f} vs Sig={s_sparse:.4f} → {hypotheses['H_hard_more_sparse']}")
    
    comparison['hypotheses'] = hypotheses
    results['comparison'] = comparison

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {args.json_out}")
    print(f"\n{'='*60}")
    print("步骤3: LIF门控统一实验完成")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
