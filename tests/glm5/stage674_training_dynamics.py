#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage674: P28 训练动态实验——五公理从随机初始化的涌现

目标：训练一个tiny GPT-2模型，在训练过程中定期保存checkpoint，
     在每个checkpoint上测量五公理指标，观察编码结构如何从随机涌现。

五公理指标：
  A1. 高维几何公理：方向正交性（相邻层cos≈90°）
  A2. 信息域公理：单域→SEPARATED（方向低纠缠）
  A3. 归一化隔离公理：norm密度与信号残存率
  A4. 信号聚焦公理：前期层cos低，后期层cos高
  A5. Logit精确公理：margin = cos(h,u) × ||h|| × ||u||

INV-316(训练涌现假说)：五公理不是预设的，而是训练优化的涌现结果
  - 随机初始化：方向随机、无隔离、无聚焦、logit方程不成立
  - 训练早期(100步)：几何公理开始涌现（高维空间的自然结果）
  - 训练中期(1000步)：归一化隔离开始工作
  - 训练后期(10000步)：信号聚焦和Logit精确成立
"""

from __future__ import annotations

import sys
import io
import json
import copy
import math
import statistics

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


# ============================================================
# 训练数据集：简单的语言能力测试数据
# ============================================================
class SimpleLanguageDataset(Dataset):
    """简单语言数据集，包含多种语言能力"""

    def __init__(self, sentences: List[str], tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for sent in sentences:
            tokens = tokenizer.encode(sent, truncation=True, max_length=max_length)
            if len(tokens) >= 4:
                self.examples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# 训练语料——覆盖多种语言能力
TRAINING_SENTENCES = [
    # 消歧能力
    "The river bank was muddy and the water was rising fast.",
    "The bank approved the loan for the new house.",
    "The bat flew across the dark cave at night.",
    "He swung the bat and hit a home run.",
    "The plant needs water and sunlight to grow.",
    "The plant closed down and moved to another country.",
    "The mouse ran across the kitchen floor quickly.",
    "She clicked the mouse to select the file on screen.",
    # 关系能力
    "Paris is the capital of France in Europe.",
    "Berlin is the capital of Germany in central Europe.",
    "Tokyo is the capital of Japan in East Asia.",
    "London is the capital of the United Kingdom.",
    "Beijing is the capital of China in East Asia.",
    "Moscow is the capital of Russia in northern Asia.",
    "Washington D.C. is the capital of the United States.",
    "Rome is the capital of Italy in southern Europe.",
    # 语法能力
    "She ran quickly to the store because she needed milk.",
    "The dog barked loudly at the stranger who approached.",
    "He carefully opened the old wooden door with a key.",
    "They happily played together in the sunny garden all afternoon.",
    "The teacher explained the difficult math problem clearly.",
    "She quietly read her favorite book by the warm fire.",
    # 空间能力
    "The cat is under the table near the window.",
    "The bird is above the tree in the garden.",
    "The book is on the shelf beside the desk.",
    "The keys are inside the drawer under the lamp.",
    "The children played behind the house near the fence.",
    # 风格能力
    "The meeting was extremely productive and efficient.",
    "That get-together was quite fruitful and enjoyable.",
    "The presentation was remarkably clear and engaging.",
    "This discussion was incredibly insightful and valuable.",
    # 时序能力
    "Yesterday it rained heavily all day long.",
    "Tomorrow it will snow in the northern regions.",
    "Last week she finished her project ahead of schedule.",
    "Next month they will launch the new product line.",
    "Last year the company doubled its revenue significantly.",
    # 更多通用语言
    "The sun rises in the east and sets in the west.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The earth revolves around the sun once every year.",
    "Light travels faster than sound in the atmosphere.",
    "Plants convert sunlight into energy through photosynthesis.",
    "The human body contains about sixty percent water.",
    "Gravity pulls objects toward the center of the earth.",
    "Diamonds are formed under extreme pressure and heat.",
    # 推理能力
    "If it rains, the ground will be wet and slippery.",
    "All birds have feathers, and eagles are birds with large wings.",
    "When the temperature drops below zero, water freezes into ice.",
    "If you study hard, you will likely get good grades on exams.",
    "The more you practice, the better you become at any skill.",
] * 5  # 重复5遍以增加数据量


# ============================================================
# 测试用例
# ============================================================
class TestCase:
    def __init__(self, cap: str, pa: str, pb: str, label: str):
        self.capability = cap
        self.prompt_a = pa
        self.prompt_b = pb
        self.label = label


TEST_CASES = [
    TestCase("disamb", "The river bank was muddy.", "The bank approved the loan.", "消歧"),
    TestCase("syntax", "She quickly ran home.", "Home she ran quickly.", "语法"),
    TestCase("relation", "Paris is the capital of France.", "Berlin is the capital of Germany.", "关系"),
    TestCase("style", "The meeting was extremely productive.", "That get-together was quite fruitful.", "风格"),
    TestCase("spatial", "The cat is under the table.", "The bird is above the tree.", "空间"),
    TestCase("temporal", "Yesterday it rained heavily.", "Tomorrow it will snow.", "时序"),
]


# ============================================================
# 五公理测量函数
# ============================================================

def extract_hidden_at_layer(model, tokenizer, text: str, layer_idx: int) -> torch.Tensor:
    """提取指定层的hidden state (最后一个token位置)"""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx][0, -1, :].float().cpu()

    return hidden


def extract_all_layers(model, tokenizer, text: str) -> List[torch.Tensor]:
    """提取所有层的hidden state"""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hiddens = []
        for h in outputs.hidden_states:
            hiddens.append(h[0, -1, :].float().cpu())

    return hiddens


def measure_axiom1_geometry(model, tokenizer) -> Dict:
    """第一公理：高维几何公理——相邻层旋转角≈90°"""
    text = "The river bank was muddy and the water was rising fast."
    hiddens = extract_all_layers(model, tokenizer, text)

    # 跳过embedding层(layer 0)
    angles = []
    cos_vals = []
    for i in range(1, len(hiddens) - 1):
        cos_val = F.cosine_similarity(hiddens[i].unsqueeze(0), hiddens[i+1].unsqueeze(0)).item()
        angle = math.degrees(math.acos(max(-1, min(1, cos_val))))
        angles.append(angle)
        cos_vals.append(abs(cos_val))

    avg_angle = statistics.mean(angles) if angles else 0
    avg_cos = statistics.mean(cos_vals) if cos_vals else 0

    # 正交性：cos接近0 → 正交
    orthogonality = 1.0 - avg_cos  # 越接近1越正交

    return {
        "avg_angle_deg": avg_angle,
        "avg_cos": avg_cos,
        "orthogonality": orthogonality,
        "n_layers": len(hiddens) - 2,
        "angles": angles,
    }


def measure_axiom2_separated(model, tokenizer) -> Dict:
    """第二公理：信息域公理——单域模型→SEPARATED（方向低纠缠）"""
    cos_matrix = []
    caps = []

    for case in TEST_CASES:
        ha = extract_hidden_at_layer(model, tokenizer, case.prompt_a, -1)
        hb = extract_hidden_at_layer(model, tokenizer, case.prompt_b, -1)
        d = ha - hb
        cos_vals = []
        for other_case in TEST_CASES:
            if other_case.capability == case.capability:
                cos_vals.append(0.0)
                continue
            ha2 = extract_hidden_at_layer(model, tokenizer, other_case.prompt_a, -1)
            hb2 = extract_hidden_at_layer(model, tokenizer, other_case.prompt_b, -1)
            d2 = ha2 - hb2
            cos_val = F.cosine_similarity(d.unsqueeze(0), d2.unsqueeze(0)).item()
            cos_vals.append(cos_val)
        cos_matrix.append(cos_vals)
        caps.append(case.capability)

    # 计算纠缠度（排除对角线）
    all_cos = []
    for i, row in enumerate(cos_matrix):
        for j, val in enumerate(row):
            if i != j:
                all_cos.append(abs(val))

    avg_entanglement = statistics.mean(all_cos) if all_cos else 0
    max_entanglement = max(all_cos) if all_cos else 0

    return {
        "avg_entanglement": avg_entanglement,
        "max_entanglement": max_entanglement,
        "cos_matrix": cos_matrix,
        "capabilities": caps,
        "is_separated": avg_entanglement < 0.3,
    }


def measure_axiom3_isolation(model, tokenizer, config) -> Dict:
    """第三公理：归一化隔离公理——norm密度和隔离效果"""
    # 统计norm数量
    norm_count = 0
    for name, module in model.named_modules():
        if 'ln' in name.lower() or 'norm' in name.lower() or 'layer_norm' in name.lower():
            norm_count += 1

    n_layers = config.n_layer
    norms_per_layer = norm_count / max(n_layers, 1)

    # 测量隔离效果：注入信号经过模型后的残存率
    text = "The river bank was muddy and the water was rising fast."
    h_first = extract_hidden_at_layer(model, tokenizer, text, 1)
    h_last = extract_hidden_at_layer(model, tokenizer, text, -1)

    # 模拟注入信号：在第一层加一个随机方向
    torch.manual_seed(42)
    injection_dir = torch.randn_like(h_first)
    injection_dir = injection_dir / injection_dir.norm()
    injection_amp = 5.0

    # 注入后hidden state
    h_injected_first = h_first + injection_amp * injection_dir

    # 追踪注入信号在各层的残存
    # 简化：直接用layer-by-layer cos来估计
    hiddens = extract_all_layers(model, tokenizer, text)
    cos_to_injection = []
    for h in hiddens[1:]:
        cos_val = F.cosine_similarity(h.unsqueeze(0), injection_dir.unsqueeze(0)).item()
        cos_to_injection.append(cos_val)

    # 隔离指标：后期层与注入方向的cos
    if cos_to_injection:
        late_cos = statistics.mean([abs(c) for c in cos_to_injection[-3:]])
        early_cos = statistics.mean([abs(c) for c in cos_to_injection[:3]])
    else:
        late_cos = early_cos = 0

    return {
        "norm_count": norm_count,
        "norms_per_layer": norms_per_layer,
        "n_layers": n_layers,
        "late_cos_to_injection": late_cos,
        "early_cos_to_injection": early_cos,
        "isolation_ratio": early_cos / max(late_cos, 1e-10),
    }


def measure_axiom4_focus(model, tokenizer, config) -> Dict:
    """第四公理：信号聚焦公理——前期层cos低，后期层cos高"""
    text_a = "The river bank was muddy."
    text_b = "The bank approved the loan."

    hiddens_a = extract_all_layers(model, tokenizer, text_a)
    hiddens_b = extract_all_layers(model, tokenizer, text_b)

    n_layers = len(hiddens_a)
    layer_cos = []

    for i in range(n_layers):
        da = hiddens_a[i]
        db = hiddens_b[i]
        diff = da - db
        cos_val = F.cosine_similarity(diff.unsqueeze(0), (hiddens_a[-1] - hiddens_b[-1]).unsqueeze(0)).item()
        layer_cos.append(cos_val)

    # 分为前半和后半
    mid = n_layers // 2
    early_cos = statistics.mean([abs(c) for c in layer_cos[:mid]]) if mid > 0 else 0
    late_cos = statistics.mean([abs(c) for c in layer_cos[mid:]]) if n_layers - mid > 0 else 0

    # 聚焦度 = 后期/前期
    focus_ratio = late_cos / max(early_cos, 1e-10)

    return {
        "early_cos": early_cos,
        "late_cos": late_cos,
        "focus_ratio": focus_ratio,
        "layer_cos": layer_cos,
        "has_focus": focus_ratio > 1.5,
    }


def measure_axiom5_logit(model, tokenizer) -> Dict:
    """第五公理：Logit精确公理——margin = cos(h,u) × ||h|| × ||u||"""
    text = "The river bank was"
    correct_token = "muddy"
    incorrect_token = "approved"

    correct_ids = tokenizer.encode(correct_token, add_special_tokens=False)
    incorrect_ids = tokenizer.encode(incorrect_token, add_special_tokens=False)

    if not correct_ids or not incorrect_ids:
        return {"status": "skip", "reason": "token编码失败"}

    correct_id = correct_ids[0]
    incorrect_id = incorrect_ids[0]
    vocab_size = model.config.vocab_size

    if correct_id >= vocab_size or incorrect_id >= vocab_size:
        return {"status": "skip", "reason": f"token id超出范围"}

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        logits = outputs.logits[0, -1, :].float().cpu()
        h_last = outputs.hidden_states[-1][0, -1, :].float().cpu()

    # 获取unembed矩阵
    # GPT-2的lm_head就是transformer.wte的转置
    wte = model.get_input_embeddings().weight.data.float().cpu()  # [vocab_size, hidden_dim]

    u_correct = wte[correct_id]  # [hidden_dim]
    u_incorrect = wte[incorrect_id]
    u_diff = u_correct - u_incorrect

    # Logit方程
    actual_margin = float(logits[correct_id] - logits[incorrect_id])
    predicted_margin = float(F.cosine_similarity(h_last.unsqueeze(0), u_diff.unsqueeze(0))) * float(h_last.norm()) * float(u_diff.norm())

    ratio = predicted_margin / max(abs(actual_margin), 1e-10)

    return {
        "status": "ok",
        "actual_margin": actual_margin,
        "predicted_margin": predicted_margin,
        "ratio": ratio,
        "cos_h_u": float(F.cosine_similarity(h_last.unsqueeze(0), u_diff.unsqueeze(0))),
        "h_norm": float(h_last.norm()),
        "u_diff_norm": float(u_diff.norm()),
        "is_accurate": abs(ratio - 1.0) < 0.05,  # 5%误差内
    }


# ============================================================
# 主训练循环
# ============================================================

def create_tiny_model():
    """创建一个tiny GPT-2模型用于训练动态实验"""
    config = GPT2Config(
        vocab_size=50257,
        n_positions=128,
        n_embd=128,       # 超小hidden dim，便于快速训练
        n_layer=6,        # 6层
        n_head=4,         # 4头
        n_inner=512,      # FFN维度
        activation_function="gelu_new",
    )
    model = GPT2LMHeadModel(config)
    return model, config


def evaluate_checkpoints(model, tokenizer, config, step: int) -> Dict:
    """在当前checkpoint上测量所有五公理"""
    print(f"\n{'='*60}")
    print(f"  评估 Step {step}")
    print(f"{'='*60}")

    model.eval()
    results = {"step": step}

    # A1: 高维几何
    try:
        a1 = measure_axiom1_geometry(model, tokenizer)
        results["axiom1"] = a1
        print(f"  A1(几何): 角度={a1['avg_angle_deg']:.1f}°, cos={a1['avg_cos']:.4f}, 正交性={a1['orthogonality']:.4f}")
    except Exception as e:
        print(f"  A1(几何): 失败 - {e}")
        results["axiom1"] = {"error": str(e)}

    # A2: 信息域/SEPARATED
    try:
        a2 = measure_axiom2_separated(model, tokenizer)
        results["axiom2"] = a2
        print(f"  A2(信息域): 平均纠缠={a2['avg_entanglement']:.4f}, SEPARATED={a2['is_separated']}")
    except Exception as e:
        print(f"  A2(信息域): 失败 - {e}")
        results["axiom2"] = {"error": str(e)}

    # A3: 归一化隔离
    try:
        a3 = measure_axiom3_isolation(model, tokenizer, config)
        results["axiom3"] = a3
        print(f"  A3(隔离): norm数={a3['norm_count']}, 每层={a3['norms_per_layer']:.1f}, 隔离比={a3['isolation_ratio']:.2f}")
    except Exception as e:
        print(f"  A3(隔离): 失败 - {e}")
        results["axiom3"] = {"error": str(e)}

    # A4: 信号聚焦
    try:
        a4 = measure_axiom4_focus(model, tokenizer, config)
        results["axiom4"] = a4
        print(f"  A4(聚焦): 早期cos={a4['early_cos']:.4f}, 后期cos={a4['late_cos']:.4f}, 聚焦比={a4['focus_ratio']:.2f}, has_focus={a4['has_focus']}")
    except Exception as e:
        print(f"  A4(聚焦): 失败 - {e}")
        results["axiom4"] = {"error": str(e)}

    # A5: Logit精确
    try:
        a5 = measure_axiom5_logit(model, tokenizer)
        results["axiom5"] = a5
        if a5.get("status") == "ok":
            print(f"  A5(Logit): 预测/实际={a5['ratio']:.4f}, 精确={a5['is_accurate']}, cos(h,u)={a5['cos_h_u']:.4f}")
        else:
            print(f"  A5(Logit): {a5.get('reason', 'unknown')}")
    except Exception as e:
        print(f"  A5(Logit): 失败 - {e}")
        results["axiom5"] = {"error": str(e)}

    # 汇总
    n_ok = sum(1 for k in ["axiom1","axiom2","axiom3","axiom4","axiom5"]
               if k in results and "error" not in results[k])
    print(f"\n  汇总: {n_ok}/5 公理测量成功")

    return results


def train_and_measure():
    """主函数：训练+定期测量"""
    # 设置离线模式避免网络请求
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("="*60)
    print("  P28 训练动态实验——五公理从随机初始化的涌现")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 1. 初始化
    print("\n[1] 初始化模型和数据...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    model, config = create_tiny_model()
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  配置: n_layer={config.n_layer}, n_embd={config.n_embd}, n_head={config.n_head}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  设备: {device}")

    # 2. 准备数据
    dataset = SimpleLanguageDataset(TRAINING_SENTENCES, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=tokenizer.pad_token_id))
    print(f"  训练样本数: {len(dataset)}")

    # 3. 优化器
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    num_training_steps = 500
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=num_training_steps)

    # 4. 评估点
    eval_steps = [0, 10, 25, 50, 100, 200, 350, 500]
    all_results = []

    # 5. 训练循环
    print(f"\n[2] 开始训练 (总步数: {num_training_steps})...")
    global_step = 0

    for epoch in range(10):  # 最多10个epoch
        if global_step >= num_training_steps:
            break

        model.train()
        epoch_loss = 0
        epoch_steps = 0

        for batch in dataloader:
            if global_step >= num_training_steps:
                break

            batch = batch.to(device)
            attention_mask = (batch != tokenizer.pad_token_id).float()

            outputs = model(batch, labels=batch, attention_mask=attention_mask)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # 在指定步数评估
            if global_step in eval_steps:
                eval_result = evaluate_checkpoints(model, tokenizer, config, global_step)
                eval_result["train_loss"] = epoch_loss / max(epoch_steps, 1)
                all_results.append(eval_result)

                # 打印损失
                print(f"  训练损失(当前epoch): {epoch_loss/max(epoch_steps,1):.4f}")

        if epoch_steps > 0:
            print(f"\n  Epoch {epoch+1} 完成, 平均损失: {epoch_loss/epoch_steps:.4f}, 全局步数: {global_step}")

    # 6. 最终评估（如果500步还没有评估过）
    if global_step not in eval_steps:
        eval_result = evaluate_checkpoints(model, tokenizer, config, global_step)
        all_results.append(eval_result)

    # 7. 汇总分析
    print("\n" + "="*60)
    print("  训练动态汇总")
    print("="*60)

    print(f"\n  {'Step':>5} | {'A1角度':>7} | {'A2纠缠':>7} | {'A3隔离':>7} | {'A4聚焦':>7} | {'A5精确':>7} | {'损失':>7}")
    print(f"  {'-'*65}")

    for r in all_results:
        step = r.get("step", "?")
        a1_angle = r.get("axiom1", {}).get("avg_angle_deg", float('nan'))
        a2_ent = r.get("axiom2", {}).get("avg_entanglement", float('nan'))
        a3_iso = r.get("axiom3", {}).get("isolation_ratio", float('nan'))
        a4_focus = r.get("axiom4", {}).get("focus_ratio", float('nan'))
        a5_ratio = r.get("axiom5", {}).get("ratio", float('nan'))
        loss = r.get("train_loss", float('nan'))

        def fmt(v):
            if v != v or v == float('nan'):  # nan check
                return "  N/A  "
            return f"{v:>7.3f}"

        print(f"  {step:>5} | {fmt(a1_angle)} | {fmt(a2_ent)} | {fmt(a3_iso)} | {fmt(a4_focus)} | {fmt(a5_ratio)} | {fmt(loss)}")

    # 8. INV-316验证
    print("\n" + "="*60)
    print("  INV-316 训练涌现假说验证")
    print("="*60)

    if len(all_results) >= 2:
        first = all_results[0]
        last = all_results[-1]

        print(f"\n  随机初始化 (step={first.get('step',0)}):")
        a1_f = first.get("axiom1", {})
        a2_f = first.get("axiom2", {})
        print(f"    A1 角度: {a1_f.get('avg_angle_deg', 'N/A'):.1f}° (目标≈90°)")
        print(f"    A2 纠缠: {a2_f.get('avg_entanglement', 'N/A'):.4f} (目标<0.3)")

        print(f"\n  训练后 (step={last.get('step',0)}):")
        a1_l = last.get("axiom1", {})
        a2_l = last.get("axiom2", {})
        a5_l = last.get("axiom5", {})
        print(f"    A1 角度: {a1_l.get('avg_angle_deg', 'N/A'):.1f}° (目标≈90°)")
        print(f"    A2 纠缠: {a2_l.get('avg_entanglement', 'N/A'):.4f} (目标<0.3)")
        if a5_l.get("status") == "ok":
            print(f"    A5 精确: ratio={a5_l.get('ratio', 'N/A'):.4f} (目标≈1.0)")

        # 验证各公理的涌现趋势
        print(f"\n  各公理涌现趋势:")

        for axiom_key, axiom_name, target_direction in [
            ("axiom1", "高维几何", "角度→90°"),
            ("axiom2", "信息域SEPARATED", "纠缠→<0.3"),
            ("axiom4", "信号聚焦", "聚焦比→>1.5"),
        ]:
            values = [r.get(axiom_key, {}).get(
                "avg_angle_deg" if axiom_key == "axiom1" else
                "avg_entanglement" if axiom_key == "axiom2" else
                "focus_ratio", float('nan'))
                for r in all_results]
            values = [v for v in values if v == v and v != float('nan')]

            if len(values) >= 2:
                # 计算趋势（线性回归斜率）
                n = len(values)
                x = list(range(n))
                x_mean = sum(x) / n
                y_mean = sum(values) / n
                slope = sum((x[i]-x_mean)*(values[i]-y_mean) for i in range(n)) / max(sum((x[i]-x_mean)**2 for i in range(n)), 1e-10)

                trend = "上升↑" if slope > 0.001 else ("下降↓" if slope < -0.001 else "平稳→")
                print(f"    {axiom_name}({target_direction}): 斜率={slope:.4f}, 趋势={trend}")

    # 9. 保存结果
    output_file = OUTPUT_DIR / f"stage674_training_dynamics_{TIMESTAMP}.json"
    # 去掉不能序列化的数据
    serializable_results = []
    for r in all_results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, dict):
                sv = {}
                for kk, vv in v.items():
                    if isinstance(vv, (int, float, str, bool, type(None))):
                        sv[kk] = vv
                    elif isinstance(vv, list) and all(isinstance(x, (int, float)) for x in vv):
                        sv[kk] = vv
                sr[k] = sv
            elif isinstance(v, (int, float, str, bool, type(None))):
                sr[k] = v
        serializable_results.append(sr)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {output_file}")

    # 10. 释放GPU
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_results


if __name__ == "__main__":
    train_and_measure()
