#!/usr/bin/env python3
"""
==========================================================
步骤2: HRR绑定在统一字典子空间中的衔接验证
==========================================================
核心假设:
  如果SNN的统一字典(W_unified)天然正交,
  那么HRR循环卷积在W的行空间中执行时应该:
  1. 接近酉键条件 → 回忆精度更高
  2. 绑定容量更大 (因为正交基底减少了交叉噪声)
  3. 训练完成后, 字典子空间中的绑定优于随机空间

实验设计:
  A: 在随机高斯向量空间中执行HRR (基线)
  B: 在训练好的W_unified的行空间中执行HRR
  C: 在人工正交化(QR分解)后的字典空间中执行HRR
  
  对每组测量:
  - 单绑定回忆精度 (bind + unbind)
  - 多绑定叠加回忆精度 (M=1,2,5,10,20)
  - 交叉噪声水平

Author: Gemini AGI Research
Date:   2026-03-10
GPU:    Required (CUDA)
"""

import os
import sys
import json
import math
import time
import random
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from test_shared_dict_ablation import (
    SkeletonLM, TextDataset
)


# ================================================================
# 1. HRR 核心操作
# ================================================================

def circular_convolution(x, y):
    """循环卷积 (HRR绑定): 在频域等价于逐元素相乘"""
    X = torch.fft.fft(x)
    Y = torch.fft.fft(y)
    return torch.fft.ifft(X * Y).real

def circular_correlation(x, y):
    """循环相关 (HRR解绑): y的共轭翻转"""
    X = torch.fft.fft(x)
    Y = torch.fft.fft(y)
    return torch.fft.ifft(X * Y.conj()).real

def make_unitary_key(d):
    """生成酉键 (频域幅值=1)"""
    phase = torch.rand(d) * 2 * math.pi
    spectrum = torch.exp(1j * phase)
    key = torch.fft.ifft(spectrum).real
    return key


# ================================================================
# 2. HRR绑定测试框架
# ================================================================

def test_hrr_recall(keys, values, vocab, M_list, n_trials=50):
    """
    测试HRR叠加存储+检索的回忆精度
    
    Args:
        keys: 键向量集合 (N, d)
        values: 值向量集合 (N, d) 
        vocab: 用于top-1匹配的候选词典 (N, d)
        M_list: 要测试的绑定数量列表
        n_trials: 每个M重复的试验次数
    
    Returns:
        results: {M: {'accuracy', 'avg_cosine', 'noise_level'}}
    """
    N, d = keys.shape
    results = {}
    
    for M in M_list:
        if M > N:
            continue
        
        correct = 0
        total = 0
        cosines = []
        noise_levels = []
        
        for trial in range(n_trials):
            # 随机选择 M 个键值对
            indices = torch.randperm(N)[:M]
            k_set = keys[indices]  # (M, d)
            v_set = values[indices]  # (M, d)
            
            # 叠加绑定
            memory = torch.zeros(d)
            for i in range(M):
                bound = circular_convolution(v_set[i], k_set[i])
                memory = memory + bound
            
            # 用第一个键检索
            retrieved = circular_correlation(memory, k_set[0])
            
            # 计算与目标的余弦相似度
            cos_sim = F.cosine_similarity(
                retrieved.unsqueeze(0), v_set[0].unsqueeze(0)
            ).item()
            cosines.append(cos_sim)
            
            # 计算噪声水平
            noise = retrieved - v_set[0]
            noise_level = noise.norm().item() / (v_set[0].norm().item() + 1e-10)
            noise_levels.append(noise_level)
            
            # Top-1 匹配
            sims = F.cosine_similarity(
                retrieved.unsqueeze(0), vocab, dim=-1
            )
            predicted_idx = sims.argmax().item()
            target_idx = indices[0].item()
            if predicted_idx == target_idx:
                correct += 1
            total += 1
        
        results[M] = {
            'accuracy': correct / total,
            'avg_cosine': sum(cosines) / len(cosines),
            'avg_noise': sum(noise_levels) / len(noise_levels),
            'n_trials': n_trials,
        }
    
    return results


# ================================================================
# 3. 三组对照实验
# ================================================================

def run_hrr_experiment(d, N, M_list, n_trials, trained_dict=None):
    """
    三组对照:
    A: 随机高斯空间
    B: 训练好的字典子空间
    C: 人工正交化后的字典空间
    """
    results = {}
    
    # --- 组 A: 随机高斯空间 ---
    print("\n  组A: 随机高斯空间...")
    torch.manual_seed(42)
    keys_A = F.normalize(torch.randn(N, d), dim=-1)
    values_A = F.normalize(torch.randn(N, d), dim=-1)
    results['group_A_random'] = test_hrr_recall(keys_A, values_A, values_A, M_list, n_trials)
    
    # --- 组 B: 训练好的字典子空间 ---
    if trained_dict is not None:
        print("  组B: 训练好的字典子空间...")
        dict_size, d_dict = trained_dict.shape
        
        # 在字典子空间中生成键和值:
        # 每个概念 = 字典原子的稀疏线性组合
        keys_B = []
        values_B = []
        for i in range(N):
            # 随机选择 top_k 个字典原子做线性组合
            top_k = min(8, dict_size)
            idx = torch.randperm(dict_size)[:top_k]
            coeffs = torch.randn(top_k)
            key = (coeffs.unsqueeze(-1) * trained_dict[idx]).sum(dim=0)
            
            idx2 = torch.randperm(dict_size)[:top_k]
            coeffs2 = torch.randn(top_k)
            value = (coeffs2.unsqueeze(-1) * trained_dict[idx2]).sum(dim=0)
            
            keys_B.append(F.normalize(key, dim=-1))
            values_B.append(F.normalize(value, dim=-1))
        
        keys_B = torch.stack(keys_B)
        values_B = torch.stack(values_B)
        results['group_B_trained_dict'] = test_hrr_recall(keys_B, values_B, values_B, M_list, n_trials)
    
    # --- 组 C: QR正交化后的字典空间 ---
    if trained_dict is not None:
        print("  组C: QR正交化字典空间...")
        # QR分解得到正交基
        Q, R = torch.linalg.qr(trained_dict.T)  # Q: (d, min(d,dict_size))
        ortho_dict = Q.T[:dict_size]  # (dict_size, d) 正交行
        
        keys_C = []
        values_C = []
        for i in range(N):
            top_k = min(8, dict_size)
            idx = torch.randperm(dict_size)[:top_k]
            coeffs = torch.randn(top_k)
            key = (coeffs.unsqueeze(-1) * ortho_dict[idx]).sum(dim=0)
            
            idx2 = torch.randperm(dict_size)[:top_k]
            coeffs2 = torch.randn(top_k)
            value = (coeffs2.unsqueeze(-1) * ortho_dict[idx2]).sum(dim=0)
            
            keys_C.append(F.normalize(key, dim=-1))
            values_C.append(F.normalize(value, dim=-1))
        
        keys_C = torch.stack(keys_C)
        values_C = torch.stack(values_C)
        results['group_C_ortho_dict'] = test_hrr_recall(keys_C, values_C, values_C, M_list, n_trials)
    
    # --- 组 D: 酉键空间 (理论最优) ---
    print("  组D: 酉键空间 (理论最优)...")
    keys_D = torch.stack([make_unitary_key(d) for _ in range(N)])
    values_D = F.normalize(torch.randn(N, d), dim=-1)
    results['group_D_unitary'] = test_hrr_recall(keys_D, values_D, values_D, M_list, n_trials)
    
    return results


# ================================================================
# 4. 主函数
# ================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='步骤2: HRR绑定衔接验证')
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--dict-size', type=int, default=64)
    parser.add_argument('--n-vocab', type=int, default=100, help='HRR测试的词汇量')
    parser.add_argument('--n-trials', type=int, default=100, help='每个M的重复试验数')
    parser.add_argument('--train-epochs', type=int, default=5)
    parser.add_argument('--json-out', type=str,
                        default='tests/gemini_temp/hrr_dict_binding_20260310.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # ============================================================
    # Phase 1: 训练共享字典模型以获得训练后的W_unified
    # ============================================================
    print("\n" + "="*60)
    print("Phase 1: 训练共享字典模型")
    print("="*60)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    random.seed(42)
    templates = [
        "The {adj} {noun} is a type of {category}.",
        "A {noun} can be {adj} and {adj2}.",
        "{noun} is related to {noun2} but different from {noun3}.",
        "We think about {abstract} and {abstract2} every day.",
        "The {animal} ran quickly across the field chasing a {animal2}.",
        "She picked a {adj} {noun} from the tree and smiled.",
        "The concept of {abstract} is important for understanding {abstract2}.",
        "A basket full of {noun}, {noun2}, and {noun3} was on the table.",
    ]
    fruits = ['apple', 'banana', 'orange', 'grape', 'pear', 'lemon', 'mango', 'peach']
    animals = ['cat', 'dog', 'rabbit', 'horse', 'tiger', 'bird', 'fish', 'deer']
    abstracts = ['justice', 'truth', 'logic', 'memory', 'beauty', 'freedom', 'wisdom', 'courage']
    adjectives = ['red', 'sweet', 'big', 'small', 'bright', 'dark', 'fresh', 'old']
    categories = ['fruit', 'animal', 'food', 'object', 'creature']

    def gen():
        t = random.choice(templates)
        return t.format(
            noun=random.choice(fruits+animals), noun2=random.choice(fruits+animals),
            noun3=random.choice(fruits+animals), adj=random.choice(adjectives),
            adj2=random.choice(adjectives), category=random.choice(categories),
            abstract=random.choice(abstracts), abstract2=random.choice(abstracts),
            animal=random.choice(animals), animal2=random.choice(animals),
        )

    train_text = ' '.join([gen() for _ in range(20000)])
    train_ids = tokenizer.encode(train_text)
    
    seq_len = 128
    train_loader = DataLoader(TextDataset(train_ids, seq_len),
                             batch_size=32, shuffle=True, drop_last=True)

    model = SkeletonLM(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=4, n_heads=4,
        d_ff=512, num_families=16, dict_size=args.dict_size, top_k=8,
        max_seq_len=seq_len, use_unified_dict=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    total_steps = len(train_loader) * args.train_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    start = time.time()
    for epoch in range(args.train_epochs):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            loss = model(x, labels=y)['loss']
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if step % 50 == 0:
                progress = (epoch * len(train_loader) + step) / total_steps * 100
                print(f"  Epoch {epoch+1}/{args.train_epochs} Step {step} "
                      f"Loss={loss.item():.4f} 进度={progress:.1f}%")
    
    train_time = time.time() - start
    print(f"训练完成, 耗时 {train_time:.1f}s")

    # 提取训练好的统一字典
    W_trained = model.unified_core.get_dictionary().detach().cpu()  # (dict_size, d_model)
    print(f"\n统一字典形状: {W_trained.shape}")
    
    # 计算字典质量
    W_norm = F.normalize(W_trained, dim=-1)
    sim = W_norm @ W_norm.T
    off_diag = sim[~torch.eye(sim.size(0), dtype=bool)]
    print(f"字典正交度: {(1 - off_diag.abs().mean()).item():.4f}")
    print(f"字典秩: {torch.linalg.matrix_rank(W_trained).item()}/{min(W_trained.shape)}")

    # 同时提取一个未训练的字典作为对照
    torch.manual_seed(999)
    W_random_init = torch.randn(args.dict_size, args.d_model) * 0.02

    # ============================================================
    # Phase 2: HRR绑定对照实验
    # ============================================================
    print("\n" + "="*60)
    print("Phase 2: HRR绑定对照实验")
    print("="*60)
    
    M_list = [1, 2, 3, 5, 8, 10, 15, 20]
    
    print(f"\n参数: d={args.d_model}, N={args.n_vocab}, trials={args.n_trials}")
    print(f"绑定数量: {M_list}")
    
    # 使用训练后的字典
    hrr_results_trained = run_hrr_experiment(
        d=args.d_model, N=args.n_vocab, M_list=M_list,
        n_trials=args.n_trials, trained_dict=W_trained
    )
    
    # 也测试未训练的随机初始化字典
    print("\n  组E: 随机初始化(未训练)字典空间...")
    # 在随机初始化字典子空间中生成
    keys_E = []
    values_E = []
    for i in range(args.n_vocab):
        top_k = min(8, args.dict_size)
        idx = torch.randperm(args.dict_size)[:top_k]
        coeffs = torch.randn(top_k)
        key = (coeffs.unsqueeze(-1) * W_random_init[idx]).sum(dim=0)
        idx2 = torch.randperm(args.dict_size)[:top_k]
        coeffs2 = torch.randn(top_k)
        value = (coeffs2.unsqueeze(-1) * W_random_init[idx2]).sum(dim=0)
        keys_E.append(F.normalize(key, dim=-1))
        values_E.append(F.normalize(value, dim=-1))
    keys_E = torch.stack(keys_E)
    values_E = torch.stack(values_E)
    hrr_results_trained['group_E_untrained_dict'] = test_hrr_recall(
        keys_E, values_E, values_E, M_list, args.n_trials
    )
    
    # ============================================================
    # Phase 3: 结果对比分析
    # ============================================================
    print("\n" + "="*60)
    print("Phase 3: 结果对比")
    print("="*60)
    
    groups = {
        'A-随机空间': 'group_A_random',
        'B-训练字典': 'group_B_trained_dict',
        'C-QR正交化': 'group_C_ortho_dict',
        'D-酉键': 'group_D_unitary',
        'E-未训练字典': 'group_E_untrained_dict',
    }
    
    for M in M_list:
        print(f"\n  M={M}:")
        for label, key in groups.items():
            if key in hrr_results_trained and M in hrr_results_trained[key]:
                r = hrr_results_trained[key][M]
                print(f"    {label}: acc={r['accuracy']:.3f} cos={r['avg_cosine']:.4f} noise={r['avg_noise']:.4f}")
    
    # 关键假设检验
    print(f"\n{'='*60}")
    print("假设检验")
    print(f"{'='*60}")
    
    hypotheses = {}
    
    # H_hrr_1: 训练字典优于随机空间
    if 'group_B_trained_dict' in hrr_results_trained and 'group_A_random' in hrr_results_trained:
        test_M = 5
        if test_M in hrr_results_trained['group_A_random'] and test_M in hrr_results_trained['group_B_trained_dict']:
            acc_A = hrr_results_trained['group_A_random'][test_M]['accuracy']
            acc_B = hrr_results_trained['group_B_trained_dict'][test_M]['accuracy']
            hypotheses['H_hrr_trained_better_than_random'] = 'PASS' if acc_B >= acc_A else 'FAIL'
            print(f"  H1: 训练字典(B) >= 随机(A) @M={test_M}: B={acc_B:.3f} vs A={acc_A:.3f} → "
                  f"{'PASS' if acc_B >= acc_A else 'FAIL'}")
    
    # H_hrr_2: 训练字典优于未训练字典
    if 'group_B_trained_dict' in hrr_results_trained and 'group_E_untrained_dict' in hrr_results_trained:
        test_M = 5
        if test_M in hrr_results_trained['group_E_untrained_dict'] and test_M in hrr_results_trained['group_B_trained_dict']:
            acc_B = hrr_results_trained['group_B_trained_dict'][test_M]['accuracy']
            acc_E = hrr_results_trained['group_E_untrained_dict'][test_M]['accuracy']
            hypotheses['H_hrr_trained_better_than_untrained'] = 'PASS' if acc_B >= acc_E else 'FAIL'
            print(f"  H2: 训练(B) >= 未训练(E) @M={test_M}: B={acc_B:.3f} vs E={acc_E:.3f} → "
                  f"{'PASS' if acc_B >= acc_E else 'FAIL'}")
    
    # H_hrr_3: QR正交化进一步提升
    if 'group_C_ortho_dict' in hrr_results_trained and 'group_B_trained_dict' in hrr_results_trained:
        test_M = 5
        if test_M in hrr_results_trained['group_C_ortho_dict'] and test_M in hrr_results_trained['group_B_trained_dict']:
            acc_B = hrr_results_trained['group_B_trained_dict'][test_M]['accuracy']
            acc_C = hrr_results_trained['group_C_ortho_dict'][test_M]['accuracy']
            hypotheses['H_hrr_ortho_boost'] = 'PASS' if acc_C >= acc_B else 'FAIL'
            print(f"  H3: 正交化(C) >= 训练(B) @M={test_M}: C={acc_C:.3f} vs B={acc_B:.3f} → "
                  f"{'PASS' if acc_C >= acc_B else 'FAIL'}")
    
    # H_hrr_4: 酉键是理论上限
    if 'group_D_unitary' in hrr_results_trained:
        test_M = 5
        if test_M in hrr_results_trained['group_D_unitary']:
            acc_D = hrr_results_trained['group_D_unitary'][test_M]['accuracy']
            print(f"  H4(参考): 酉键(D)上限 @M={test_M}: D={acc_D:.3f}")
    
    # H_hrr_5: 容量曲线 — 训练字典的容量退化是否更缓慢
    if 'group_B_trained_dict' in hrr_results_trained and 'group_A_random' in hrr_results_trained:
        acc_decay_A = []
        acc_decay_B = []
        for M in M_list:
            if M in hrr_results_trained['group_A_random'] and M in hrr_results_trained['group_B_trained_dict']:
                acc_decay_A.append(hrr_results_trained['group_A_random'][M]['accuracy'])
                acc_decay_B.append(hrr_results_trained['group_B_trained_dict'][M]['accuracy'])
        
        if len(acc_decay_A) > 2:
            # AUC (粗略的面积比较)
            auc_A = sum(acc_decay_A) / len(acc_decay_A)
            auc_B = sum(acc_decay_B) / len(acc_decay_B)
            hypotheses['H_hrr_capacity_advantage'] = 'PASS' if auc_B > auc_A else 'FAIL'
            print(f"  H5: 容量曲线AUC B={auc_B:.3f} vs A={auc_A:.3f} → "
                  f"{'PASS' if auc_B > auc_A else 'FAIL'}")
    
    for k, v in hypotheses.items():
        status = "✅" if v == 'PASS' else "❌"
        print(f"  {status} {k}: {v}")
    
    # 保存
    save_results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'experiment': 'hrr_dict_binding_step2',
        'args': vars(args),
        'train_time_sec': train_time,
        'dict_orthogonality': (1 - off_diag.abs().mean()).item(),
        'dict_rank': torch.linalg.matrix_rank(W_trained).item(),
        'hrr_results': {},
        'hypotheses': hypotheses,
    }
    
    for group_key, group_data in hrr_results_trained.items():
        save_results['hrr_results'][group_key] = {}
        for M, metrics in group_data.items():
            save_results['hrr_results'][group_key][str(M)] = metrics
    
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {args.json_out}")
    
    print("\n" + "="*60)
    print("步骤2: HRR绑定衔接验证完成")
    print("="*60)


if __name__ == '__main__':
    main()
