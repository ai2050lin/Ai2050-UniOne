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

import matplotlib

matplotlib.use("Agg")  # 无头模式，兼容服务器
import torch
import torch.nn.functional as F
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

# 【实验观察坐标系层级】: 全面抽查浅、中、深渊三层
OBSERVE_LAYERS = [6, 20, 31, 35]

def get_concept_vector(model, word, layer):
    """提取一个概念在指定层级的绝对完整连续坐标张量 (Shape: [d_model])"""
    with torch.no_grad():
        _, cache = model.run_with_cache(f"{word}")
        layer_name = f"blocks.{layer}.hook_resid_post"
        # 抓取整个完整维度 (不限制前 K 个) 的真实浮点坐标
        return cache[layer_name][0, -1, :].cpu().float()

def cosine_similarity(v1, v2):
    """计算两条空间特征轴心在 N 维超体空间中的夹角重合率"""
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

def run_pure_mathematical_algebra():
    print("\n📐 启动高维欧几里得空间线性代数破解仪 (Tensor Algebra)...")
    model = load_qwen3()
    
    report = {
        "experiment_layers": OBSERVE_LAYERS,
        "gender_parallelism": [],     # 验证性别特征向量的平行可加减性
        "capital_parallelism": [],    # 验证主权首都方位向量的可平移性
        "orthogonality_isolation": [] # 验证颜色向量与性别向量是否互相垂直正交
    }
    
    for layer in OBSERVE_LAYERS:
        print(f"\n  [+] 下潜解算网格：L{layer}")
        
        # --- 方程组 1: 性别极化算子 (Gender Vector) ---
        # 提取女性与男性的空间点位
        v_woman = get_concept_vector(model, "woman", layer)
        v_man = get_concept_vector(model, "man", layer)
        v_girl = get_concept_vector(model, "girl", layer)
        v_boy = get_concept_vector(model, "boy", layer)
        v_queen = get_concept_vector(model, "queen", layer)
        v_king = get_concept_vector(model, "king", layer)
        
        # 算术生成纯粹的“性别算子向量”
        delta_gender_adult = v_woman - v_man
        delta_gender_child = v_girl - v_boy
        delta_gender_royal = v_queen - v_king
        
        # 测定三条由不同物种/年龄抽离出的属性漂移轴是否在空间中绝对平行！
        sim_ag = cosine_similarity(delta_gender_adult, delta_gender_child)
        sim_ar = cosine_similarity(delta_gender_adult, delta_gender_royal)
        gender_alignment = (sim_ag + sim_ar) / 2
        print(f"      [ Gender Arithmetic ]  Woman-Man 轴 与 Girl-Boy 轴 的平行重合率: {sim_ag*100:.2f}%")
        
        # --- 方程组 2: 空间地理坐标系平移算子 (Capital Vector) ---
        v_paris = get_concept_vector(model, "Paris", layer)
        v_france = get_concept_vector(model, "France", layer)
        v_beijing = get_concept_vector(model, "Beijing", layer)
        v_china = get_concept_vector(model, "China", layer)
        v_rome = get_concept_vector(model, "Rome", layer)
        v_italy = get_concept_vector(model, "Italy", layer)
        
        delta_cap_fr = v_paris - v_france
        delta_cap_cn = v_beijing - v_china
        delta_cap_it = v_rome - v_italy
        
        sim_fr_cn = cosine_similarity(delta_cap_fr, delta_cap_cn)
        sim_fr_it = cosine_similarity(delta_cap_fr, delta_cap_it)
        capital_alignment = (sim_fr_cn + sim_fr_it) / 2
        print(f"      [ Capital Arithmetic ] Paris-France 轴 与 Beijing-China 轴平行度: {sim_fr_cn*100:.2f}%")
        
        # --- 方程组 3: 绝对正交的暗能量跨维屏障 (Orthogonal Projection) ---
        # 提取颜色算子
        v_red = get_concept_vector(model, "red", layer)
        v_blue = get_concept_vector(model, "blue", layer)
        delta_color = v_red - v_blue
        
        # 理论上，“颜色”和“国家/首都”是两个不同维度的知识。
        # 它们的数学向位角应该逼近 0 (Cos(90度) = 0)，这意味着大模型通过多维垂直来防止知识串联！
        ortho_color_gender = cosine_similarity(delta_color, delta_gender_adult)
        ortho_color_cap = cosine_similarity(delta_color, delta_cap_fr)
        avg_ortho = (abs(ortho_color_gender) + abs(ortho_color_cap)) / 2
        print(f"      [ Cross-Orthogonality ] 颜色算子坐标 与 性别/首都坐标 的交叉干涉率: {avg_ortho*100:.2f}%  (越接近0，维度越隔离)")
        
        report["gender_parallelism"].append(float(gender_alignment))
        report["capital_parallelism"].append(float(capital_alignment))
        report["orthogonality_isolation"].append(float(avg_ortho))

    output_dir = os.path.join(os.path.dirname('d:/develop/TransformerLens-main/research/glm5/experiments/rewrite_vector_algebra.py'), '..', '..', '..', 'tempdata', 'glm5_emergence')
    out_file = os.path.join(output_dir, "qwen3_mathematical_vector_algebra.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 数学密码级向量代数计算完毕。落盘地址: {out_file}")

if __name__ == '__main__':
    run_pure_mathematical_algebra()
