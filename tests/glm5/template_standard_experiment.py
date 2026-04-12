"""
标准多模型实验脚本模板 (CUDA优化版)
==================================================
用途: 所有新实验必须基于此模板编写，确保三模型(Qwen3/GLM4/DeepSeek7B)兼容

使用方法:
1. 复制此模板到 tests/glm5/phase_xxx_xxx.py
2. 修改 PHASE_NAME, EXPERIMENT_DESC
3. 实现 run_pXXX() 实验函数
4. 在 main() 中注册实验
5. 运行: python phase_xxx_xxx.py --model qwen3 --experiment all

关键规则:
- 必须使用 model_utils.py 中的标准函数加载模型和提取权重
- 模型路径统一使用正斜杠 "/" (不用反斜杠 "\\")
- GPU内存管理: 每个模型测试完必须释放, 再加载下一个
- Hook必须及时remove, 避免内存泄漏
- 结果统一保存到 tests/glm5_temp/ 目录
- 使用 argparse 支持 --model 和 --experiment 参数

CUDA优化策略:
- load_model() 先CPU加载再整体.to("cuda"), 避免device_map="auto"设备分散
- 前向传播用 torch.no_grad() + CUDA tensor
- 权重提取: 仅在需要numpy分析时 .detach().cpu().float().numpy()
- 大规模计算(如随机旋转)尽量在CUDA上做, 仅最后结果转numpy
- hook收集: 直接存CPU tensor避免GPU占满
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ===== 导入标准模型工具 =====
sys.path.insert(0, str(Path(__file__).parent))
from model_utils import (
    MODEL_CONFIGS,
    load_model,
    get_layers,
    get_model_info,
    get_layer_weights,
    get_W_U,
    release_model,
    get_sample_layers,
    get_attr_direction,
    inject_at_embed,
    collect_layer_outputs,
    compute_cos,
    compute_recoding_ratio,
    LayerWeights,
    ModelInfo,
)

# ===== 实验配置 =====
PHASE_NAME = "Phase XCIX"  # 修改为当前Phase编号
EXPERIMENT_DESC = "实验描述"  # 修改为实验描述

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 常用测试数据
COLOR_ATTRS = ["red", "green", "blue", "yellow", "brown", "white", "black", "pink", "purple", "gray"]
TASTE_ATTRS = ["sweet", "sour", "bitter", "salty", "soft", "spicy", "fresh", "tart"]
SIZE_ATTRS = ["big", "small", "tall", "short", "long", "wide", "thin", "thick", "heavy", "light"]
ALL_ATTRS = COLOR_ATTRS + TASTE_ATTRS + SIZE_ATTRS

NOUNS = ["apple", "banana", "cat", "dog", "car", "bus", "chair", "table",
         "hammer", "wrench", "pear", "grape", "horse", "lion", "train", "plane"]


# ================================================================
# 实验函数 (在此实现具体实验)
# ================================================================

def run_pXXX(model, tokenizer, device, model_info):
    """
    PXXX: 实验标题

    原理:
      [在此描述实验原理]

    方法:
      [在此描述具体方法]

    CUDA注意事项:
      - model已经在CUDA上, 输入tensor也需要在CUDA上
      - inject_at_embed()自动处理CUDA tensor
      - collect_layer_outputs()收集的tensor在CPU上(避免GPU占满)
      - 权重分析: model_utils.get_layer_weights()返回numpy(CPU)
      - 大规模矩阵运算在CUDA上: torch.matmul > numpy.dot
    """
    print(f"\n{'='*60}")
    print(f"PXXX: 实验标题 - {model_info.name}")
    print(f"{'='*60}")

    results = {}

    # 获取基本信息
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model)  # [vocab_size, d_model] numpy on CPU

    # 示例: 方向流追踪 (CUDA优化版)
    test_attrs = ALL_ATTRS[:6]
    sample_layers = get_sample_layers(n_layers, n_samples=10)

    for attr in test_attrs:
        direction, attr_tok_id = get_attr_direction(model, tokenizer, attr)
        if direction is None:
            continue

        print(f"  Testing {attr}...")

        prompt = "The apple is"
        # inject_at_embed 自动处理CUDA tensor
        inputs_base, inputs_interv, input_ids, pos_ids = inject_at_embed(
            model, tokenizer, device, prompt, direction, beta=8.0
        )

        # 收集base和intervened的层输出 (结果在CPU上)
        base_out = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
        interv_out = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)

        # 计算方向保持度 (numpy计算, 数据已在CPU)
        attr_flow = []
        for li in sample_layers:
            key = f"L{li}"
            if key not in base_out or key not in interv_out:
                continue

            h_base = base_out[key][0, -1, :].numpy()
            h_interv = interv_out[key][0, -1, :].numpy()
            delta_h = h_interv - h_base

            cos_val = compute_cos(delta_h, direction)
            delta_norm = float(np.linalg.norm(delta_h))
            proj_val = float(np.dot(delta_h, direction))

            attr_flow.append({
                "layer": li,
                "cos_wlm": round(cos_val, 4),
                "delta_norm": round(delta_norm, 4),
                "proj_wlm": round(proj_val, 4),
            })

        results[attr] = attr_flow

    return results


# ================================================================
# 主函数 (标准格式，不需要修改)
# ================================================================

def run_single_model(model_name, experiments=None):
    """运行单个模型的所有实验"""
    print(f"\n{'#'*60}")
    print(f"# 模型: {model_name}")
    print(f"{'#'*60}")

    # load_model 自动: CPU加载 → CUDA转移
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  类: {model_info.model_class}")
    print(f"  层数: {model_info.n_layers}, 维度: {model_info.d_model}")
    print(f"  MLP类型: {model_info.mlp_type}")
    print(f"  设备: {device}")

    all_results = {
        "model": model_name,
        "model_class": model_info.model_class,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
        "mlp_type": model_info.mlp_type,
        "device": str(device),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    timestamp = time.strftime("%Y%m%d_%H%M")

    try:
        if experiments is None or "all" in experiments:
            exp_fns = [
                # ("pXXX", run_pXXX),  # 取消注释并修改
            ]
        else:
            exp_fns = [(e, globals()[f"run_{e}"]) for e in experiments
                       if f"run_{e}" in globals()]

        for exp_name, exp_fn in exp_fns:
            print(f"\n--- Running {exp_name} ---")
            t0 = time.time()
            result = exp_fn(model, tokenizer, device, model_info)
            dt = time.time() - t0
            print(f"  {exp_name} completed in {dt:.1f}s")
            all_results[exp_name] = result

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        out_file = OUT_DIR / f"{PHASE_NAME.lower().replace(' ', '_')}_{model_name}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to {out_file}")

        # 释放GPU (重要: 必须在finally中释放)
        release_model(model)


def main():
    parser = argparse.ArgumentParser(description=f"{PHASE_NAME}: {EXPERIMENT_DESC}")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="测试模型 (每次只测一个, 避免GPU溢出)")
    parser.add_argument("--experiment", type=str, default="all",
                       help="实验编号 (如 p439), 或 all")
    args = parser.parse_args()

    experiments = [args.experiment] if args.experiment != "all" else None
    run_single_model(args.model, experiments)


if __name__ == "__main__":
    main()
