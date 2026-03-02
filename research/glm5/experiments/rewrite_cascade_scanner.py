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
import sys
import time

import matplotlib

matplotlib.use("Agg")  # 无头模式，兼容服务器
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

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
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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

# 定义 3 个维度的巨大探测词库
CASCADE_LEXICON = {
    "MICRO_PROPERTIES": {
        "colors": ["red", "blue", "green", "yellow", "black"],
        "physics": ["heavy", "light", "large", "small", "hot", "cold"],
        "taste": ["sweet", "sour", "bitter", "salty", "spicy"]
    },
    "MESO_ENTITIES": {
        "fruits": ["apple", "banana", "orange", "grape", "cherry"],
        "animals": ["cat", "dog", "tiger", "elephant", "rabbit"],
        "vehicles": ["car", "train", "airplane", "ship", "bicycle"],
        "occupations": ["doctor", "teacher", "engineer", "farmer", "artist"]
    },
    "MACRO_SUPERCLASSES": {
        "action_verbs": ["run", "jump", "swim", "fly", "sleep"],
        "abstract_ethics": ["justice", "truth", "love", "freedom", "peace"],
        "math_logic": ["equation", "geometry", "calculus", "infinity", "algorithm"]
    }
}

LAYERS = list(range(0, 36, 3)) + [35]  # 跳步抽取，降低开销，但精细掌握头尾尾
TOP_K = 15  # 每层提取核心表征轴

def extract_domain_genes(model, words, layer_idx):
    """提取特定单词组在某一层中的物理公用基因 (Intersection/Majority)"""
    layer_name = f"blocks.{layer_idx}.hook_resid_post"
    all_indices = []
    
    with torch.no_grad():
        for word in words:
            # 去掉 prompt 以获取极其纯粹的概念静息态张量
            # 这种纯概念输入，可以更好地看清名词和动词的绝缘态
            _, cache = model.run_with_cache(f"{word}")
            vector = cache[layer_name][0, -1, :].cpu().float()
            _, indices = torch.topk(vector, TOP_K)
            all_indices.append(set(indices.tolist()))
            
    # 共识基因：一半以上词汇共享的维度即视为组内绝对物理轴
    from collections import Counter
    flat_list = [item for sublist in all_indices for item in sublist]
    counter = Counter(flat_list)
    consensus_genes = set([k for k, v in counter.items() if v >= len(words) * 0.5])
    return consensus_genes

def calculate_overlap_ratio(set_a, set_b):
    if not set_a or not set_b: return 0.0
    return len(set_a.intersection(set_b)) / min(len(set_a), len(set_b)) # 计算子集包含率

def run_cascade_scanner():
    print("\n🌌 正在启动全维度知识级联大扫描 (Cascade Architecture)...")
    model = load_qwen3()
    
    report_data = {
        "layers": LAYERS,
        "micro_cohesion": [],      # 微观形容词的类内聚拢度
        "meso_cohesion": [],       # 中观实体的类内聚拢度
        "macro_cohesion": [],      # 宏观抽象类的聚拢度
        
        # 核心跨维度纠缠：微观属性与实体的绑定率 (例如 颜色到底是否属于水果的一部分)
        "micro_to_meso_entanglement": [],
        
        # 终极维度隔离：实体名词 vs 宏观抽象动词/规则的 重合率
        "meso_to_macro_isolation": []
    }
    
    for layer in LAYERS:
        print(f"  [+] 探针进入深渊 L{layer}...")
        
        # 1. 抽取 3 大层次的所有子界别特征轴
        # --- 微观 ---
        color_g = extract_domain_genes(model, CASCADE_LEXICON["MICRO_PROPERTIES"]["colors"], layer)
        physic_g = extract_domain_genes(model, CASCADE_LEXICON["MICRO_PROPERTIES"]["physics"], layer)
        taste_g = extract_domain_genes(model, CASCADE_LEXICON["MICRO_PROPERTIES"]["taste"], layer)
        
        # --- 中观实体 ---
        fruit_g = extract_domain_genes(model, CASCADE_LEXICON["MESO_ENTITIES"]["fruits"], layer)
        animal_g = extract_domain_genes(model, CASCADE_LEXICON["MESO_ENTITIES"]["animals"], layer)
        vehicle_g = extract_domain_genes(model, CASCADE_LEXICON["MESO_ENTITIES"]["vehicles"], layer)
        prof_g = extract_domain_genes(model, CASCADE_LEXICON["MESO_ENTITIES"]["occupations"], layer)
        
        # --- 宏观超类 ---
        verb_g = extract_domain_genes(model, CASCADE_LEXICON["MACRO_SUPERCLASSES"]["action_verbs"], layer)
        ethic_g = extract_domain_genes(model, CASCADE_LEXICON["MACRO_SUPERCLASSES"]["abstract_ethics"], layer)
        math_g = extract_domain_genes(model, CASCADE_LEXICON["MACRO_SUPERCLASSES"]["math_logic"], layer)
        
        # 2. 计算极度微观的同级聚拢度 (类内互相不交集，分别掌管自己的领地)
        # 如果颜色和物理尺寸有交集，说明形容词维度重叠
        mic_cohe = calculate_overlap_ratio(color_g, physic_g)
        mes_cohe = calculate_overlap_ratio(fruit_g, vehicle_g)
        mac_cohe = calculate_overlap_ratio(ethic_g, verb_g)
        
        # 3. 跨阶层引力计算
        # 微观属性 (如甜味/颜色) 有没有物理重组绑定到 中观实体 (如水果/动物) 上？
        micro_meso_tie = calculate_overlap_ratio(taste_g.union(color_g), fruit_g)
        
        # 实体名词 (水果/汽车) 与 抽象概念/动词 之间是否存在那该死的 "地球实体基础基座" 共用？
        entity_union = fruit_g.union(animal_g).union(vehicle_g)
        abstract_union = verb_g.union(ethic_g).union(math_g)
        meso_macro_iso = calculate_overlap_ratio(entity_union, abstract_union)
        
        report_data["micro_cohesion"].append(float(mic_cohe))
        report_data["meso_cohesion"].append(float(mes_cohe))
        report_data["macro_cohesion"].append(float(mac_cohe))
        report_data["micro_to_meso_entanglement"].append(float(micro_meso_tie))
        report_data["meso_to_macro_isolation"].append(float(meso_macro_iso))

    output_dir = os.path.join(os.path.dirname('d:/develop/TransformerLens-main/research/glm5/experiments/rewrite_cascade.py'), '..', '..', '..', 'tempdata', 'glm5_emergence')
    out_file = os.path.join(output_dir, "qwen3_knowledge_cascade_tree.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 全维级联宇宙知识树测绘完毕，落盘路径: {out_file}")

if __name__ == '__main__':
    run_cascade_scanner()
