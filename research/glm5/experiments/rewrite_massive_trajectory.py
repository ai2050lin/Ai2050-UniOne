# -*- coding: utf-8 -*-
"""
Qwen3 编码结构四维度提取器
=========================
从 Qwen3-4B 中提取编码，验证四个关键数学特性：
  1. 高维抽象 — 语义收敛能力
  2. 低维精确 — 细粒度区分能力
  3. 特异性 — 概念子空间正交性
  4. 系统性 — 类比关系一致性

输出: tempdata/qwen3_structure_report.json + 4 张可视化图
"""

import json
import os
import time
import torch
import numpy as np
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # 无头模式，兼容服务器

# ============================================================
# 第零部分：模型加载（复用已验证的 import_trace.py 逻辑）
# ============================================================

SNAPSHOT_PATH = r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"

# 环境变量
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"


def load_qwen3():
    """加载 Qwen3-4B 为 HookedTransformer"""
    import transformers.configuration_utils as config_utils
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from transformer_lens import HookedTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 加载 Qwen3-4B，设备: {device}")
    print(f"    路径: {SNAPSHOT_PATH}")

    t0 = time.time()

    # 步骤 1: 在 CPU 上加载 HF 模型 (HookedTransformer 会自行处理设备迁移)
    hf_model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT_PATH, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT_PATH, local_files_only=True, add_bos_token=False
    )

    # 修复1: Qwen3 tokenizer 缺少 bos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id
        print(f"    [fix] 设置 bos_token = eos_token ({tokenizer.bos_token})")

    # 修复2: Monkey-patch PretrainedConfig 以修复 rope_theta
    _orig_getattr = config_utils.PretrainedConfig.__getattribute__

    def _patched_getattr(self, key):
        if key == "rope_theta":
            try:
                return _orig_getattr(self, key)
            except AttributeError:
                try:
                    rs = _orig_getattr(self, "rope_scaling")
                    if isinstance(rs, dict) and "rope_theta" in rs:
                        return rs["rope_theta"]
                except (AttributeError, TypeError):
                    pass
                return 1000000
        return _orig_getattr(self, key)

    config_utils.PretrainedConfig.__getattribute__ = _patched_getattr

    # 修复3: Monkey-patch get_tokenizer_with_bos 避免重新加载 tokenizer
    import transformer_lens.utils as tl_utils
    _orig_get_tok_bos = tl_utils.get_tokenizer_with_bos

    def _patched_get_tok_bos(tok):
        # 直接返回已修复的 tokenizer，避免重新 from_pretrained
        return tok

    tl_utils.get_tokenizer_with_bos = _patched_get_tok_bos
    print("    [fix] 已 monkey-patch rope_theta + get_tokenizer_with_bos")

    try:
        model = HookedTransformer.from_pretrained(
            "Qwen/Qwen3-4B", hf_model=hf_model, device=device, tokenizer=tokenizer,
            fold_ln=False, center_writing_weights=False, center_unembed=False,
            dtype=torch.float16, default_prepend_bos=False
        )
    finally:
        config_utils.PretrainedConfig.__getattribute__ = _orig_getattr
        tl_utils.get_tokenizer_with_bos = _orig_get_tok_bos
        print("    [fix] 已恢复所有 monkey-patch")

    model.eval()
    print(f"[+] 模型加载完成 ({time.time() - t0:.1f}s)")
    print(f"    层数: {model.cfg.n_layers}, 维度: {model.cfg.d_model}")
    return model

# -------------------------------------------------------------------------
# 海量知识域语料库 (Massive Concept Dictionary) (8 个极其分明的类目)
# -------------------------------------------------------------------------
MASSIVE_UNIVERSE = {
    "Biology": ["cat", "dog", "lion", "tiger", "bear", "elephant", "monkey", "rabbit", "deer", "fox", "wolf", "zebra", "giraffe", "horse", "cow", "pig", "sheep", "goat", "kangaroo", "whale"],
    "Nations": ["USA", "China", "Japan", "Germany", "UK", "India", "France", "Italy", "Brazil", "Canada", "Russia", "South Korea", "Australia", "Spain", "Mexico", "Indonesia", "Netherlands", "Turkey", "Switzerland", "Sweden"],
    "Science": ["physics", "chemistry", "biology", "astronomy", "geology", "mathematics", "computer science", "psychology", "sociology", "economics", "medicine", "engineering", "quantum mechanics", "relativity", "thermodynamics", "genetics", "evolution"],
    "Emotions": ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "joyful", "anxious", "excited", "bored", "confused", "proud", "ashamed", "guilty", "jealous", "envious", "hopeful", "desperate", "loving", "hateful"],
    "Tools": ["hammer", "screwdriver", "wrench", "pliers", "saw", "drill", "tape measure", "level", "utility knife", "chisel", "file", "mallet", "vice", "clamp", "crowbar", "awl", "planes", "rasp", "spatula", "trowel"],
    "Elements": ["Hydrogen", "Helium", "Lithium", "Beryllium", "Boron", "Carbon", "Nitrogen", "Oxygen", "Fluorine", "Neon", "Sodium", "Magnesium", "Aluminum", "Silicon", "Phosphorus", "Sulfur", "Chlorine", "Argon", "Potassium", "Calcium"],
    "Time": ["second", "minute", "hour", "day", "week", "month", "year", "decade", "century", "millennium", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march"],
    "Vehicles": ["car", "truck", "bus", "van", "motorcycle", "bicycle", "scooter", "train", "subway", "tram", "airplane", "helicopter", "jet", "glider", "boat", "ship", "yacht", "ferry", "submarine", "hovercraft"]
}

TOP_K_CAPTURE = 30  # 每个概念在每层抽取的最顶尖孤立特征神经元数
TOP_K_UNIVERSAL = 5 # 重构步骤：统计出来的领域跨源核心索引中，取出交集最大的前 5 个绝对控制核心

def run_massive_reconstruct():
    print("\n🌌 [1/2] 启动指纹海网矩阵全态扫描...")
    model = load_qwen3()
    total_layers = model.cfg.n_layers
    
    # 打包扁平化语料
    prompts_data = []
    for cat, concepts in MASSIVE_UNIVERSE.items():
        for c in concepts:
            prompts_data.append({
                "concept": c,
                "category": cat,
                "prompt": f"The semantic concept of '{c}' is"
            })
            
    print(f"[*] 已装填 {len(MASSIVE_UNIVERSE)} 个超级维度下共计 {len(prompts_data)} 只探针...")
    
    trajectory_database = []
    
    with torch.no_grad():
        t0 = time.time()
        for i, item in enumerate(prompts_data):
            if i > 0 and i % 20 == 0:
                print(f"    --> 已在流形深度下萃取 {i} 枚概念指纹核心 ({time.time()-t0:.1f}s)")
            
            _, cache = model.run_with_cache(item["prompt"])
            
            concept_trace = {
                "concept": item["concept"],
                "category": item["category"],
                "layer_traces": []
            }
            
            for L in range(total_layers):
                vector = cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().float()
                top_values, top_indices = torch.topk(vector, TOP_K_CAPTURE)
                concept_trace["layer_traces"].append({
                    "layer": L,
                    "top_indices": top_indices.tolist()
                })
            trajectory_database.append(concept_trace)

    print("\n🌌 [2/2] 指纹图谱采信完毕，启动逆向绝对主因基底提取 (Universal Base Kernels) ...")
    
    # 结构: category_kernels[cat][layer] = [最有共性的神经元 1, 有共性的神经元 2]
    category_kernels = {cat: {L: [] for L in range(total_layers)} for cat in MASSIVE_UNIVERSE}
    
    for cat in MASSIVE_UNIVERSE.keys():
        print(f"\n    [解码] 正在分析星云 {cat} 的层级交响共振主轴...")
        # 取出此类别下所有的指纹记录
        cat_traces = [t for t in trajectory_database if t["category"] == cat]
        
        for L in range(total_layers):
            layer_all_indices = []
            for t in cat_traces:
                layer_all_indices.extend(t["layer_traces"][L]["top_indices"])
                
            # 统计这一层中，哪些极点神经元被该领域概念【击中最为频繁】
            counter = Counter(layer_all_indices)
            most_common = counter.most_common(TOP_K_UNIVERSAL)
            
            # format: [{"index": 82, "hit_rate": 0.95}, ...]
            kernels = [{"index": k, "hit_rate": round(v / len(cat_traces), 2)} for k, v in most_common]
            category_kernels[cat][L] = kernels
            
            if L % 10 == 0 or L == total_layers - 1:
                print(f"       Layer {L:02d} >> 该族群共振率最高的主祭神经突触: {kernels[0]['index']} (唤醒率 {kernels[0]['hit_rate']*100:.0f}%) | {kernels[1]['index']} (唤醒率 {kernels[1]['hit_rate']*100:.0f}%)")

    # ----- 数据最终定档 -----
    final_report = {
        "metadata": {
            "categories": len(MASSIVE_UNIVERSE),
            "concepts_total": len(prompts_data),
            "top_k_capture_per_sample": TOP_K_CAPTURE,
            "top_k_universal_kernels": TOP_K_UNIVERSAL,
            "layers": total_layers
        },
        "raw_trajectories": trajectory_database,
        "universal_base_kernels": category_kernels
    }

    output_dir = os.path.join(os.path.dirname('d:/develop/TransformerLens-main/research/glm5/experiments/rewrite_massive_trajectory.py'), '..', '..', '..', 'tempdata', 'glm5_emergence')
    out_file = os.path.join(output_dir, "qwen3_massive_trajectory_reconstructed.json")
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 所有交叉索引重构密码本均已落盘: {out_file}")

if __name__ == '__main__':
    run_massive_reconstruct()
