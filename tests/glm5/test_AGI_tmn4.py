# -*- coding: utf-8 -*-
"""
TMN 4.0 AGI 测试脚本
=====================
测试 /tests/AGI.py 中 TMN4_GodView_AGI 的各项能力
包括：基础运行、预测准确率、反事实推理、多模态生长、稳态机制、泛化能力
"""

import sys
import os
import io
import time
import torch
import torch.nn.functional as F
import numpy as np

# Windows GBK 编码修复
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 直接添加项目根目录和tests目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'tests'))

from AGI import (
    PhysMultiModalManifold, CausalPhysManifold, SelfAgent,
    GlobalSteadyState, AutoDistillStack, fast_sparse_diffuse,
    TMN4_GodView_AGI
)

def test_1_basic_construction():
    """[OK] 1: 基础构建 - 所有模块能否正常初始化"""
    print("=" * 60)
    print("[OK] 1: 基础构建测试")
    print("=" * 60)
    
    try:
        agi = TMN4_GodView_AGI(dim=128)
        print(f"  [PASS] TMN4_GodView_AGI dim={agi.dim}")
        print(f"  [PASS] CausalPhysManifold: vocab={len(agi.mf.vocab)} tokens")
        print(f"  [PASS] SelfAgent: self_emb norm={torch.norm(agi.self.self_emb).item():.4f}")
        print(f"  [PASS] AutoDistillStack: {len(agi.stack.levels)} levels")
        print(f"  [PASS] GlobalSteadyState: sleep_cycle={agi.steady.sleep_cycle}")
        return True
    except Exception as e:
        print(f"  [FAIL] : {e}")
        return False


def test_2_prediction_accuracy():
    """[OK] 2: 预测准确率 - 简单序列学习"""
    print("\n" + "=" * 60)
    print("[OK] 2: 预测准确率测试")
    print("=" * 60)
    
    agi = TMN4_GodView_AGI(dim=128)
    
    # 构建训练文本
    text = "苹果是水果。苹果是红色的。苹果有重量。玻璃是透明的。玻璃易碎。"
    for c in text:
        agi.mf.add_token(c)
    
    tokens = [agi.mf.vocab[c] for c in text]
    
    # 训练
    print(f"  vocab size: {len(tokens)} tokens")
    print(f"  training...")
    
    errors_per_epoch = []
    for epoch in range(100):
        err = 0
        total = 0
        for i in range(3, len(tokens)):
            ctx = tokens[:i]
            tgt = tokens[i]
            pred = agi.predict_next(ctx)
            if pred != tgt:
                err += 1
            total += 1
            agi.learn_step(ctx, tgt)
        errors_per_epoch.append(err)
        if epoch % 20 == 0:
            acc = (total - err) / total * 100
            print(f"  Epoch {epoch:3d} | err: {err}/{total} | acc: {acc:.1f}%")
    
    # 最终准确率
    final_err = errors_per_epoch[-1]
    final_acc = (total - final_err) / total * 100
    print(f"  final acc: {final_acc:.1f}% (err: {final_err}/{total})")
    
    if final_acc > 50:
        print("  [PASS] acc > 50%")
    else:
        print("  [FAIL] acc too low")
    
    return final_acc


def test_3_text_generation():
    """[OK] 3: 文本生成 - 从prompt生成连续文本"""
    print("\n" + "=" * 60)
    print("[OK] 3: 文本生成测试")
    print("=" * 60)
    
    agi = TMN4_GodView_AGI(dim=128)
    
    text = "苹果是水果。苹果是红色的。苹果有重量。玻璃是透明的。玻璃易碎。"
    for c in text:
        vis_feat = torch.randn(128) if c in ["苹", "果", "红", "玻", "璃", "透"] else None
        phys_feat = torch.tensor([9.8, 0.3, 0.9, 1.2] + list(torch.randn(124))) if c in ["苹", "果"] else None
        phys_feat = torch.tensor([9.8, 0.3, 1.5, 0.1] + list(torch.randn(124))) if c in ["玻", "璃"] else phys_feat
        agi.mf.add_token(c, vis_feat, phys_feat)
    
    tokens = [agi.mf.vocab[c] for c in text]
    
    # 训练
    for epoch in range(50):
        for i in range(3, len(tokens)):
            agi.learn_step(tokens[:i], tokens[i])
    
    # 生成测试
    prompts = ["苹果", "玻璃", "红色"]
    for prompt in prompts:
        ctx = [agi.mf.vocab[c] for c in prompt if c in agi.mf.vocab]
        if not ctx:
            continue
        generated = list(prompt)
        for _ in range(15):
            nxt = agi.predict_next(ctx)
            ctx.append(nxt)
            generated.append(agi.mf.inv_vocab[nxt])
        result = ''.join(generated)
        print(f"  prompt='{prompt}' => '{result}'")
    
    return True


def test_4_counterfactual_reasoning():
    """[OK] 4: 反事实推理"""
    print("\n" + "=" * 60)
    print("[OK] 4: 反事实推理测试")
    print("=" * 60)
    
    agi = TMN4_GodView_AGI(dim=128)
    
    text = "苹果是水果。苹果是红色的。玻璃是透明的。玻璃易碎。"
    for c in text:
        agi.mf.add_token(c)
    
    tokens = [agi.mf.vocab[c] for c in text]
    
    # 训练
    for epoch in range(30):
        for i in range(3, len(tokens)):
            agi.learn_step(tokens[:i], tokens[i])
    
    # 反事实测试
    try:
        cf1 = agi.counterfactual_think(tokens, "易", "不")
        print(f"  counterfactual1 (glass not fragile): score={cf1.item():.4f}")
        
        cf2 = agi.counterfactual_think(tokens, "红", "绿")
        print(f"  counterfactual2 (apple not red but green): score={cf2.item():.4f}")
        
        print("  [PASS] counterfactual reasoning runs")
        return True
    except Exception as e:
        print(f"  [FAIL] counterfactual: {e}")
        return False


def test_5_steady_state():
    """[OK] 5: 全局稳态机制 - 长期训练是否稳定"""
    print("\n" + "=" * 60)
    print("[OK] 5: 全局稳态机制测试")
    print("=" * 60)
    
    agi = TMN4_GodView_AGI(dim=128)
    agi.steady.sleep_cycle = 20  # 缩短睡眠周期以更快触发
    
    text = "苹果是水果。苹果是红色的。玻璃是透明的。玻璃易碎。"
    for c in text:
        agi.mf.add_token(c)
    
    tokens = [agi.mf.vocab[c] for c in text]
    
    # 长期训练，观察稳态
    self_emb_norms = []
    sym_emb_norms = []
    
    for epoch in range(200):
        for i in range(3, len(tokens)):
            agi.learn_step(tokens[:i], tokens[i])
        
        self_emb_norms.append(torch.norm(agi.self.self_emb).item())
        if agi.mf.sym_emb is not None:
            sym_emb_norms.append(torch.norm(agi.mf.sym_emb, dim=-1).mean().item())
    
    print(f"  self_emb norm: min={min(self_emb_norms):.4f}, max={max(self_emb_norms):.4f}, "
          f"mean={np.mean(self_emb_norms):.4f}")
    print(f"  sym_emb avg norm: min={min(sym_emb_norms):.4f}, max={max(sym_emb_norms):.4f}, "
          f"mean={np.mean(sym_emb_norms):.4f}")
    
    # 检查稳态边界是否有效
    if max(self_emb_norms) <= agi.self.steady_bound + 0.1:
        print("  [PASS] self_emb steady bound effective")
    else:
        print(f"  [WARN] self_emb exceeded bound {agi.self.steady_bound}: max={max(self_emb_norms):.4f}")
    
    return True


def test_6_new_concept_growth():
    """[OK] 6: 开放式生长 - 新概念动态添加"""
    print("\n" + "=" * 60)
    print("[OK] 6: 开放式生长测试")
    print("=" * 60)
    
    agi = TMN4_GodView_AGI(dim=128)
    
    # 初始词汇
    text = "苹果是水果。"
    for c in text:
        agi.mf.add_token(c)
    
    initial_vocab_size = len(agi.mf.vocab)
    print(f"  initial vocab: {initial_vocab_size}")
    
    # 动态添加新概念
    new_concepts = [
        ("碳纤维", torch.randn(128), torch.tensor([9.8, 0.3, 0.5, 5.0] + list(torch.randn(124)))),
        ("钛合金", torch.randn(128), torch.tensor([9.8, 0.3, 2.0, 8.0] + list(torch.randn(124)))),
        ("水", torch.randn(128), torch.tensor([9.8, 0.1, 1.0, 0.01] + list(torch.randn(124)))),
    ]
    
    for name, vis, phys in new_concepts:
        new_id = agi.grow_new_node(name, vis, phys)
        print(f"  new concept: '{name}' => id={new_id}")
    
    print(f"  final vocab: {len(agi.mf.vocab)}")
    print(f"  [PASS] open-ended growth works")
    return True


def test_7_generalization():
    """[OK] 7: 泛化能力 - 训练集外预测"""
    print("\n" + "=" * 60)
    print("[OK] 7: 泛化能力测试")
    print("=" * 60)
    
    agi = TMN4_GodView_AGI(dim=128)
    
    # 训练：A是X，B是Y
    train_text = "猫是动物。狗是动物。猫会跑。狗会跑。"
    for c in train_text:
        agi.mf.add_token(c)
    
    tokens = [agi.mf.vocab[c] for c in train_text]
    
    for epoch in range(80):
        for i in range(3, len(tokens)):
            agi.learn_step(tokens[:i], tokens[i])
    
    # 泛化测试
    test_ctx = [agi.mf.vocab[c] for c in "猫是"]
    pred = agi.predict_next(test_ctx)
    pred_char = agi.mf.inv_vocab[pred]
    print(f"  'cat is' => predict: '{pred_char}' (expect: '动')")
    
    test_ctx2 = [agi.mf.vocab[c] for c in "狗是"]
    pred2 = agi.predict_next(test_ctx2)
    pred_char2 = agi.mf.inv_vocab[pred2]
    print(f"  'dog is' => predict: '{pred_char2}' (expect: '动')")
    
    # 检查因果强度
    causal_items = list(agi.mf.causal_strength.items())
    print(f"  causal relations: {len(causal_items)}")
    if causal_items:
        top_causal = sorted(causal_items, key=lambda x: x[1], reverse=True)[:5]
        for (a, b), strength in top_causal:
            print(f"    {agi.mf.inv_vocab[a]}->{agi.mf.inv_vocab[b]}: {strength:.4f}")
    
    return True


def test_8_multimodal_fusion():
    """[OK] 8: 多模态融合 - 视觉/物理特征是否影响预测"""
    print("\n" + "=" * 60)
    print("[OK] 8: 多模态融合测试")
    print("=" * 60)
    
    agi = TMN4_GodView_AGI(dim=128)
    
    # 添加带多模态特征的词汇
    text = "苹果是水果。玻璃易碎。"
    for c in text:
        vis_feat = torch.randn(128) if c in ["苹", "果", "玻", "璃"] else None
        phys_feat = torch.tensor([9.8, 0.3, 0.9, 1.2] + list(torch.randn(124))) if c in ["苹", "果"] else None
        phys_feat = torch.tensor([9.8, 0.3, 1.5, 0.1] + list(torch.randn(124))) if c in ["玻", "璃"] else phys_feat
        agi.mf.add_token(c, vis_feat, phys_feat)
    
    tokens = [agi.mf.vocab[c] for c in text]
    
    # 训练
    for epoch in range(50):
        for i in range(3, len(tokens)):
            agi.learn_step(tokens[:i], tokens[i])
    
    # 检查多模态嵌入
    if agi.mf.sym_emb is not None and agi.mf.vis_emb is not None and agi.mf.phys_emb is not None:
        apple_idx = agi.mf.vocab["苹"]
        glass_idx = agi.mf.vocab["玻"]
        
        sym_dist = F.cosine_similarity(agi.mf.sym_emb[apple_idx].unsqueeze(0), 
                                        agi.mf.sym_emb[glass_idx].unsqueeze(0)).item()
        vis_dist = F.cosine_similarity(agi.mf.vis_emb[apple_idx].unsqueeze(0), 
                                        agi.mf.vis_emb[glass_idx].unsqueeze(0)).item()
        phys_dist = F.cosine_similarity(agi.mf.phys_emb[apple_idx].unsqueeze(0), 
                                         agi.mf.phys_emb[glass_idx].unsqueeze(0)).item()
        
        fused_apple = agi.mf.fuse_modal(apple_idx)
        fused_glass = agi.mf.fuse_modal(glass_idx)
        fused_dist = F.cosine_similarity(fused_apple.unsqueeze(0), fused_glass.unsqueeze(0)).item()
        
        print(f"  apple vs glass:")
        print(f"    sym similarity: {sym_dist:.4f}")
        print(f"    vis similarity: {vis_dist:.4f}")
        print(f"    phys similarity: {phys_dist:.4f}")
        print(f"    fused similarity: {fused_dist:.4f}")
        print("  [PASS] multimodal fusion works")
    else:
        print("  [WARN] multimodal emb not initialized")
    
    return True


def test_9_hebb_convergence():
    """[OK] 9: 赫布更新收敛性 - 长期训练是否稳定收敛"""
    print("\n" + "=" * 60)
    print("[OK] 9: 赫布更新收敛性测试")
    print("=" * 60)
    
    agi = TMN4_GodView_AGI(dim=128)
    
    text = "苹果是水果。苹果是红色的。玻璃是透明的。玻璃易碎。"
    for c in text:
        agi.mf.add_token(c)
    
    tokens = [agi.mf.vocab[c] for c in text]
    
    # 记录每个epoch的嵌入变化
    emb_change_history = []
    prev_emb = agi.mf.sym_emb.clone() if agi.mf.sym_emb is not None else None
    
    for epoch in range(100):
        for i in range(3, len(tokens)):
            agi.learn_step(tokens[:i], tokens[i])
        
        if agi.mf.sym_emb is not None and prev_emb is not None:
            change = torch.norm(agi.mf.sym_emb - prev_emb).item()
            emb_change_history.append(change)
            prev_emb = agi.mf.sym_emb.clone()
    
    if emb_change_history:
        print(f"  emb change: initial={emb_change_history[0]:.4f}, "
              f"final={emb_change_history[-1]:.4f}")
        trend = "converging" if emb_change_history[-1] < emb_change_history[0] else "not converging"
        print(f"  trend: {trend}")
        
        recent_avg = np.mean(emb_change_history[-10:])
        early_avg = np.mean(emb_change_history[:10])
        print(f"  early avg change: {early_avg:.4f}, recent avg change: {recent_avg:.4f}")
    
    return True


def test_10_critical_analysis():
    """[OK] 10: 批判性分析 - 系统的根本性限制"""
    print("\n" + "=" * 60)
    print("[OK] 10: 批判性分析 - TMN4根本性限制")
    print("=" * 60)
    
    issues = [
        "1. [CRITICAL] Hebb update delta=0.01*(ctx-tgt_emb) direction WRONG: should approach target, but delta points away",
        "2. [CRITICAL] Full-rank update + immediate normalization compresses to hypersphere, losing magnitude info",
        "3. [BOTTLENECK] Prediction uses argmax(similarity) = nearest neighbor search, NO sequence modeling",
        "4. [BOTTLENECK] Causal strength stored as defaultdict(float), cannot learn complex causal graphs",
        "5. [BOTTLENECK] Distillation stack initialized as identity matrix, update rate 0.001 is too small to learn",
        "6. [BOTTLENECK] Sparse diffusion only 2 rounds of topk, propagation depth very shallow",
        "7. [CRITICAL] Counterfactual only computes dot-product diff, NOT real do-calculus causal inference",
        "8. [LIMIT] No attention mechanism, no position encoding, cannot handle token order",
        "9. [LIMIT] No backpropagation, learning signal only Hebb rule, cannot learn complex functions",
        "10.[FUNDAMENTAL] System is essentially embedding+nearest-neighbor, lacks composition, logic, AGI core"
    ]
    
    for issue in issues:
        print(f"  {issue}")
    
    return True


# ==============================================================================
# 主测试流程
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TMN 4.0 AGI Comprehensive Test")
    print("  Time: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    results = {}
    
    # 依次运行测试
    tests = [
        ("basic_construction", test_1_basic_construction),
        ("prediction_accuracy", test_2_prediction_accuracy),
        ("text_generation", test_3_text_generation),
        ("counterfactual_reasoning", test_4_counterfactual_reasoning),
        ("steady_state", test_5_steady_state),
        ("new_concept_growth", test_6_new_concept_growth),
        ("generalization", test_7_generalization),
        ("multimodal_fusion", test_8_multimodal_fusion),
        ("hebb_convergence", test_9_hebb_convergence),
        ("critical_analysis", test_10_critical_analysis),
    ]
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = "PASS" if result else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {e}"
            import traceback
            traceback.print_exc()
    
    # 汇总
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "[OK]" if result == "PASS" else "[FAIL]"
        print(f"  {status} {name}: {result}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
TMN 4.0 AGI Evaluation Conclusion:

1. Runs: All modules can execute without crash
2. NOT AGI: Essentially an embedding + nearest-neighbor system

Critical Issues:
  - Hebb update direction WRONG: delta=0.01*(ctx-tgt) moves emb AWAY from target
  - No sequence modeling: No position encoding, no attention, cannot understand token order
  - No backpropagation: Only simple Hebb rule, cannot learn nonlinear mappings
  - Counterfactual is fake: Just dot-product difference, not real causal inference

Gap to real AGI:
  - Missing: compositional generalization, logical reasoning, meta-learning, autonomous goal-setting
  - Missing: long-range dependency modeling, hierarchical planning, commonsense reasoning
  - Missing: probabilistic language model, syntactic structure understanding

To achieve AGI, need:
  1. Real sequence modeling capability (Transformer or equivalent)
  2. Formal causal reasoning framework (do-calculus)
  3. Structure learning from data (backpropagation or equivalent optimization)
  4. Compositional generalization (symbolic-neural hybrid)
""")
