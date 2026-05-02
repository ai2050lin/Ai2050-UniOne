"""
CCXIX(369): 范数控制下的暗物质转导验证
=========================================

解决CCXVII/VIII的五个核心硬伤:
1. 范数混淆: delta_dark范数是delta_wu的3倍, steering效果差异可能来自能量而非机制
2. alpha=1.0崩溃: 负效率说明非线性/补偿机制, 需要细粒度扫描
3. 暗物质维度7: 8个概念=8个样本, 有效秩不可能超过8, 需要更多概念
4. 线性假设: lambda_wu和tau_dark可能是alpha的函数
5. 概念核=7维解读过激: 不能从8个概念推广到所有概念

五个实验:
  Exp1: 范数等价steering — 用补偿alpha使W_U注入范数=暗物质注入范数
  Exp2: 细粒度alpha扫描 — alpha=0.1到1.0, 步长0.1, 找非线性临界点
  Exp3: 扩展概念集 — 20个概念重做暗物质PCA, 验证有效维度
  Exp4: 小alpha线性区验证 — alpha=0.1,0.25,0.5对比lambda和tau
  Exp5: 概念类型分组 — 具体/抽象/关系概念分别测暗物质维度

用法:
  python ccxix_norm_controlled_transduction.py --model qwen3 --exp 1
  python ccxix_norm_controlled_transduction.py --model qwen3 --exp 2
  python ccxix_norm_controlled_transduction.py --model qwen3 --exp 3
  python ccxix_norm_controlled_transduction.py --model qwen3 --exp 4
  python ccxix_norm_controlled_transduction.py --model qwen3 --exp 5
  python ccxix_norm_controlled_transduction.py --model qwen3 --exp all
"""

import argparse, os, sys, json, gc, warnings, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANS_TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS, get_W_U
)

TEMP = Path("tests/glm5_temp")

# ================================================================
# 概念定义 — 8概念集(原版) + 20概念扩展集
# ================================================================
CONCEPTS_8 = {
    "apple": {
        "templates": ["The word is apple", "I ate an apple", "A red apple", "The apple fell", "Apple is a fruit"],
        "probe_words": ["fruit", "red", "eat", "sweet", "tree", "banana", "orange", "pear"],
    },
    "dog": {
        "templates": ["The word is dog", "A big dog", "The dog barked", "My pet dog", "Dog is an animal"],
        "probe_words": ["animal", "pet", "bark", "fur", "puppy", "cat", "wolf", "horse"],
    },
    "king": {
        "templates": ["The word is king", "The king ruled", "A wise king", "The king and queen", "King is a ruler"],
        "probe_words": ["queen", "ruler", "royal", "throne", "crown", "prince", "emperor", "lord"],
    },
    "doctor": {
        "templates": ["The word is doctor", "The doctor helped", "A good doctor", "Visit the doctor", "Doctor treats patients"],
        "probe_words": ["hospital", "patient", "medicine", "nurse", "health", "surgeon", "clinic", "cure"],
    },
    "mountain": {
        "templates": ["The word is mountain", "A tall mountain", "The mountain peak", "Climb the mountain", "Mountain is high"],
        "probe_words": ["peak", "high", "climb", "snow", "valley", "hill", "summit", "rock"],
    },
    "ocean": {
        "templates": ["The word is ocean", "The deep ocean", "Ocean waves", "Swim in the ocean", "Ocean is vast"],
        "probe_words": ["sea", "deep", "wave", "water", "fish", "beach", "coast", "blue"],
    },
    "love": {
        "templates": ["The word is love", "Feel the love", "Love is strong", "Show your love", "Love and peace"],
        "probe_words": ["heart", "feel", "care", "passion", "emotion", "hate", "romance", "affection"],
    },
    "science": {
        "templates": ["The word is science", "Study of science", "Science advances", "Modern science", "Science is knowledge"],
        "probe_words": ["research", "study", "theory", "experiment", "physics", "art", "biology", "data"],
    },
}

# 20概念扩展集: 具体(7) + 抽象(7) + 关系(6) = 20
CONCEPTS_20 = {
    # === 具体概念 (Concrete) ===
    "apple": {
        "templates": ["The word is apple", "I ate an apple", "A red apple", "The apple fell", "Apple is a fruit"],
        "probe_words": ["fruit", "red", "eat", "sweet", "tree"],
    },
    "dog": {
        "templates": ["The word is dog", "A big dog", "The dog barked", "My pet dog", "Dog is an animal"],
        "probe_words": ["animal", "pet", "bark", "fur", "puppy"],
    },
    "mountain": {
        "templates": ["The word is mountain", "A tall mountain", "The mountain peak", "Climb the mountain", "Mountain is high"],
        "probe_words": ["peak", "high", "climb", "snow", "valley"],
    },
    "ocean": {
        "templates": ["The word is ocean", "The deep ocean", "Ocean waves", "Swim in the ocean", "Ocean is vast"],
        "probe_words": ["sea", "deep", "wave", "water", "fish"],
    },
    "house": {
        "templates": ["The word is house", "A big house", "The house stands", "My new house", "House is a building"],
        "probe_words": ["building", "home", "roof", "room", "door"],
    },
    "car": {
        "templates": ["The word is car", "A fast car", "The car drove", "My new car", "Car is a vehicle"],
        "probe_words": ["vehicle", "drive", "road", "engine", "speed"],
    },
    "tree": {
        "templates": ["The word is tree", "A tall tree", "The tree grew", "Under the tree", "Tree is a plant"],
        "probe_words": ["plant", "leaf", "branch", "root", "forest"],
    },
    # === 抽象概念 (Abstract) ===
    "love": {
        "templates": ["The word is love", "Feel the love", "Love is strong", "Show your love", "Love and peace"],
        "probe_words": ["heart", "feel", "care", "passion", "emotion"],
    },
    "science": {
        "templates": ["The word is science", "Study of science", "Science advances", "Modern science", "Science is knowledge"],
        "probe_words": ["research", "study", "theory", "experiment", "physics"],
    },
    "freedom": {
        "templates": ["The word is freedom", "Fight for freedom", "Freedom is precious", "True freedom", "Freedom and rights"],
        "probe_words": ["liberty", "right", "free", "choice", "independence"],
    },
    "justice": {
        "templates": ["The word is justice", "Seek justice", "Justice is fair", "Social justice", "Justice and law"],
        "probe_words": ["fair", "law", "right", "court", "equality"],
    },
    "beauty": {
        "templates": ["The word is beauty", "See the beauty", "Beauty is deep", "Inner beauty", "Beauty and art"],
        "probe_words": ["art", "pretty", "lovely", "grace", "aesthetic"],
    },
    "truth": {
        "templates": ["The word is truth", "Seek the truth", "Truth is clear", "Hidden truth", "Truth and facts"],
        "probe_words": ["fact", "real", "honest", "correct", "belief"],
    },
    "wisdom": {
        "templates": ["The word is wisdom", "Gain wisdom", "Wisdom is deep", "Ancient wisdom", "Wisdom and knowledge"],
        "probe_words": ["smart", "wise", "knowledge", "insight", "sage"],
    },
    # === 关系概念 (Relational) ===
    "king": {
        "templates": ["The word is king", "The king ruled", "A wise king", "The king and queen", "King is a ruler"],
        "probe_words": ["queen", "ruler", "royal", "throne", "crown"],
    },
    "doctor": {
        "templates": ["The word is doctor", "The doctor helped", "A good doctor", "Visit the doctor", "Doctor treats patients"],
        "probe_words": ["hospital", "patient", "medicine", "nurse", "health"],
    },
    "teacher": {
        "templates": ["The word is teacher", "A good teacher", "The teacher taught", "My best teacher", "Teacher guides students"],
        "probe_words": ["school", "student", "class", "learn", "education"],
    },
    "mother": {
        "templates": ["The word is mother", "A loving mother", "The mother cared", "My dear mother", "Mother and child"],
        "probe_words": ["parent", "child", "care", "family", "love"],
    },
    "enemy": {
        "templates": ["The word is enemy", "A fierce enemy", "The enemy attacked", "My old enemy", "Enemy is a foe"],
        "probe_words": ["foe", "fight", "opponent", "war", "hostile"],
    },
    "friend": {
        "templates": ["The word is friend", "A good friend", "The friend helped", "My best friend", "Friend is loyal"],
        "probe_words": ["pal", "buddy", "trust", "loyal", "companion"],
    },
}

BASELINE_TEXT = "The word is"


def collect_states_at_layers(model, tokenizer, device, text, capture_layers):
    """用hooks收集指定层的残差流状态"""
    captured = {}
    all_layers = get_layers(model)
    def make_hook(li):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured[li] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                captured[li] = output[0, -1, :].detach().float().cpu().numpy()
        return hook
    hooks = []
    for li in capture_layers:
        if li < len(all_layers):
            hooks.append(all_layers[li].register_forward_hook(make_hook(li)))
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
        except Exception as e:
            print(f"  Forward failed: {e}")
            for h in hooks: h.remove()
            return {}, None
    for h in hooks: h.remove()
    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    gc.collect()
    return captured, logits


def inject_and_collect(model, tokenizer, device, inject_layer, direction, alpha,
                       capture_layers, baseline_text=BASELINE_TEXT):
    """在inject_layer注入direction, 收集capture_layers的状态"""
    all_layers = get_layers(model)
    captured = {}
    injected = [False]

    def make_inject_hook():
        def hook(module, inp, output):
            if not injected[0]:
                direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
                if isinstance(output, tuple):
                    modified = output[0].clone()
                    modified[0, -1, :] += alpha * direction_t.to(modified.dtype)
                    return (modified,) + output[1:]
                else:
                    modified = output.clone()
                    modified[0, -1, :] += alpha * direction_t.to(modified.dtype)
                    return modified
                injected[0] = True
            return output
        return hook

    def make_capture_hook(li):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured[li] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                captured[li] = output[0, -1, :].detach().float().cpu().numpy()
        return hook

    hooks = []
    if inject_layer < len(all_layers):
        hooks.append(all_layers[inject_layer].register_forward_hook(make_inject_hook()))
    for li in capture_layers:
        if li < len(all_layers):
            hooks.append(all_layers[li].register_forward_hook(make_capture_hook(li)))

    input_ids = tokenizer.encode(baseline_text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            model(input_ids=input_ids)
        except Exception as e:
            print(f"  Forward failed: {e}")

    for h in hooks: h.remove()
    gc.collect()
    return captured


def compute_steering_effect(model, tokenizer, device, probe_words,
                            inject_layer, direction, alpha=0.5):
    """在指定层注入方向, 测量对probe_words的logit变化"""
    all_layers = get_layers(model)
    probe_ids = {}
    for w in probe_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            probe_ids[w] = ids[0]

    input_ids = tokenizer.encode(BASELINE_TEXT, add_special_tokens=True, return_tensors="pt").to(device)

    # Baseline
    with torch.no_grad():
        try:
            base_out = model(input_ids=input_ids)
            base_logits = base_out.logits[0, -1, :].detach().float().cpu().numpy()
        except:
            return {}

    # Injected
    direction_tensor = torch.tensor(direction, dtype=torch.float32, device=device)
    injected_flag = [False]

    def make_hook():
        def hook(module, inp, output):
            if not injected_flag[0]:
                if isinstance(output, tuple):
                    modified = output[0].clone()
                    modified[0, -1, :] += alpha * direction_tensor.to(modified.dtype)
                    return (modified,) + output[1:]
                else:
                    modified = output.clone()
                    modified[0, -1, :] += alpha * direction_tensor.to(modified.dtype)
                    return modified
                injected_flag[0] = True
            return output
        return hook

    hook = all_layers[inject_layer].register_forward_hook(make_hook())
    with torch.no_grad():
        try:
            inj_out = model(input_ids=input_ids)
            inj_logits = inj_out.logits[0, -1, :].detach().float().cpu().numpy()
        except:
            hook.remove()
            return {}

    hook.remove()
    gc.collect()

    results = {}
    for w, tid in probe_ids.items():
        results[w] = float(inj_logits[tid] - base_logits[tid])
    return results


def compute_wu_basis(model):
    """计算W_U行空间基"""
    W_U = get_W_U(model)
    from scipy.sparse.linalg import svds
    k_wu = min(200, min(W_U.shape) - 2)
    U_wu, s_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = np.asarray(U_wu, dtype=np.float64)
    del W_U
    gc.collect()
    return U_wu


def decompose_delta(delta, U_wu):
    """将delta分解为W_U分量和暗物质"""
    proj_wu = U_wu.T @ delta
    delta_wu = U_wu @ proj_wu
    delta_dark = delta - delta_wu
    return delta_wu, delta_dark, proj_wu


# ================================================================
# Exp1: 范数等价steering — 核心硬伤修复
# ================================================================
def run_exp1(model, tokenizer, device, model_info, concepts):
    """
    解决问题一: 范数混淆

    核心思路:
    - 原实验: alpha_wu = alpha_dark = 0.5
      但 ||0.5 * delta_dark|| ≈ 46.5, ||0.5 * delta_wu|| ≈ 15.0
      暗物质注入了3倍能量, 3倍效果不能证明转导更好

    - 修复: 对W_U分量使用补偿alpha, 使注入范数相等
      alpha_wu_comp = alpha * (||delta_dark|| / ||delta_wu||)
      这样 ||alpha_wu_comp * delta_wu|| = ||alpha * delta_dark||

    - 如果补偿后W_U-only效果仍然接近0 → 转导机制成立
    - 如果补偿后W_U-only效果与Dark-only相当 → 范数混淆, 之前结论推翻

    同时测试:
    A. 等范数steering (W_U补偿alpha vs Dark标准alpha)
    B. 等范数转导追踪 (注入等范数, 追踪后续层W_U投影比)
    """
    print(f"\n{'='*60}")
    print(f"  Exp1: Norm-Equivalent Steering (Fix Confound #1)")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers

    # W_U基
    U_wu = compute_wu_basis(model)

    # 收集baseline
    all_capture = list(range(n_layers))
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_capture)

    # 收集概念delta
    all_deltas = {}
    for cname, cdata in concepts.items():
        concept_states = {l: [] for l in all_capture}
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, all_capture)
            for l in all_capture:
                if l in states and l in bl_all:
                    concept_states[l].append(states[l])
        deltas = {}
        for l in all_capture:
            if concept_states[l]:
                deltas[l] = np.mean(concept_states[l], axis=0) - bl_all[l]
        all_deltas[cname] = deltas
        del concept_states
        gc.collect()

    key_layers = [l for l in [12, 18, 24] if l < n_layers]
    alpha_base = 0.5

    results = {}

    # Part A: 等范数steering
    print(f"\n  --- Part A: Norm-Equivalent Steering ---")

    for l in key_layers:
        print(f"\n  Layer {l}:")
        layer_results = {}

        for cname, cdata in concepts.items():
            if l not in all_deltas.get(cname, {}):
                continue

            delta = all_deltas[cname][l]
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-8:
                continue

            delta_wu, delta_dark, _ = decompose_delta(delta, U_wu)
            norm_wu = np.linalg.norm(delta_wu)
            norm_dark = np.linalg.norm(delta_dark)

            if norm_wu < 1e-8 or norm_dark < 1e-8:
                continue

            # 补偿alpha: 使W_U注入范数 = Dark注入范数
            # alpha_dark * ||delta_dark|| = alpha_wu_comp * ||delta_wu||
            # alpha_wu_comp = alpha_base * (||delta_dark|| / ||delta_wu||)
            alpha_wu_comp = alpha_base * (norm_dark / norm_wu)

            probe_words = cdata["probe_words"]

            # 四种steering:
            # 1. Full delta, alpha=0.5
            full_effects = compute_steering_effect(
                model, tokenizer, device, probe_words, l, delta, alpha_base
            )
            # 2. Dark-only, alpha=0.5 (原版)
            dark_effects = compute_steering_effect(
                model, tokenizer, device, probe_words, l, delta_dark, alpha_base
            )
            # 3. W_U-only, alpha=0.5 (原版, 范数不等)
            wu_effects_orig = compute_steering_effect(
                model, tokenizer, device, probe_words, l, delta_wu, alpha_base
            )
            # 4. W_U-only, alpha补偿 (等范数!)
            wu_effects_comp = compute_steering_effect(
                model, tokenizer, device, probe_words, l, delta_wu, alpha_wu_comp
            )

            # 汇总
            full_mean = np.mean(list(full_effects.values())) if full_effects else 0
            dark_mean = np.mean(list(dark_effects.values())) if dark_effects else 0
            wu_orig_mean = np.mean(list(wu_effects_orig.values())) if wu_effects_orig else 0
            wu_comp_mean = np.mean(list(wu_effects_comp.values())) if wu_effects_comp else 0

            # 注入范数
            inj_norm_dark = alpha_base * norm_dark
            inj_norm_wu_orig = alpha_base * norm_wu
            inj_norm_wu_comp = alpha_wu_comp * norm_wu

            layer_results[cname] = {
                "norm_full": float(delta_norm),
                "norm_wu": float(norm_wu),
                "norm_dark": float(norm_dark),
                "alpha_base": float(alpha_base),
                "alpha_wu_comp": float(alpha_wu_comp),
                "inj_norm_dark": float(inj_norm_dark),
                "inj_norm_wu_orig": float(inj_norm_wu_orig),
                "inj_norm_wu_comp": float(inj_norm_wu_comp),
                "full_mean": float(full_mean),
                "dark_mean": float(dark_mean),
                "wu_orig_mean": float(wu_orig_mean),
                "wu_comp_mean": float(wu_comp_mean),
                "dark_eff_orig": float(dark_mean / full_mean) if abs(full_mean) > 1e-6 else 0,
                "wu_eff_orig": float(wu_orig_mean / full_mean) if abs(full_mean) > 1e-6 else 0,
                "wu_eff_comp": float(wu_comp_mean / full_mean) if abs(full_mean) > 1e-6 else 0,
            }

            print(f"    {cname}: inj_norm dark={inj_norm_dark:.1f} wu_orig={inj_norm_wu_orig:.1f} wu_comp={inj_norm_wu_comp:.1f}")
            print(f"      steering: full={full_mean:.3f} dark={dark_mean:.3f}(eff={layer_results[cname]['dark_eff_orig']:.3f}) "
                  f"wu_orig={wu_orig_mean:.3f}(eff={layer_results[cname]['wu_eff_orig']:.3f}) "
                  f"wu_comp={wu_comp_mean:.3f}(eff={layer_results[cname]['wu_eff_comp']:.3f})")

        results[f"steering_L{l}"] = layer_results

    # Part B: 等范数转导追踪
    print(f"\n  --- Part B: Norm-Equivalent Transduction Tracking ---")

    inject_layers = [l for l in [12, 18] if l < n_layers]
    track_range = 6

    for inject_l in inject_layers:
        print(f"\n  Inject at Layer {inject_l}:")
        trans_results = {}

        for cname, cdata in concepts.items():
            if inject_l not in all_deltas.get(cname, {}):
                continue

            delta = all_deltas[cname][inject_l]
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-8:
                continue

            delta_wu, delta_dark, _ = decompose_delta(delta, U_wu)
            norm_wu = np.linalg.norm(delta_wu)
            norm_dark = np.linalg.norm(delta_dark)

            if norm_wu < 1e-8 or norm_dark < 1e-8:
                continue

            alpha_wu_comp = alpha_base * (norm_dark / norm_wu)

            track_layers = [inject_l + k for k in range(1, track_range + 1) if inject_l + k < n_layers]
            capture_layers = [inject_l] + track_layers

            # 三种注入: dark(标准alpha), wu(标准alpha), wu(补偿alpha)
            inject_configs = [
                ("dark", delta_dark, alpha_base),
                ("wu_orig", delta_wu, alpha_base),
                ("wu_comp", delta_wu, alpha_wu_comp),
            ]

            concept_trans = {}
            for itype, idir, ialpha in inject_configs:
                injected_states = inject_and_collect(
                    model, tokenizer, device, inject_l, idir, ialpha, capture_layers
                )

                trans_data = {}
                for tl in track_layers:
                    if tl not in injected_states or tl not in bl_all:
                        continue

                    delta_tl = injected_states[tl] - bl_all[tl]
                    delta_tl_norm = np.linalg.norm(delta_tl)
                    if delta_tl_norm < 1e-8:
                        continue

                    # W_U投影比
                    proj_wu_tl = U_wu.T @ delta_tl
                    wu_ratio_tl = np.sum(proj_wu_tl ** 2) / (delta_tl_norm ** 2)

                    # 方向保持性
                    cos_with_inject = float(np.dot(delta_tl, idir) / (delta_tl_norm * np.linalg.norm(idir)))

                    trans_data[str(tl)] = {
                        "delta_norm": float(delta_tl_norm),
                        "wu_ratio": float(wu_ratio_tl),
                        "cos_with_inject": cos_with_inject,
                    }

                concept_trans[itype] = trans_data

            trans_results[cname] = concept_trans

            # 打印
            for itype in ["dark", "wu_orig", "wu_comp"]:
                if itype in concept_trans and concept_trans[itype]:
                    wu_str = " ".join([f"L{k}:wu={v['wu_ratio']:.3f}" for k, v in sorted(concept_trans[itype].items())])
                    print(f"    {cname}({itype}): {wu_str}")

        results[f"transduction_L{inject_l}"] = trans_results

    # 关键对比汇总
    print(f"\n  === CRITICAL COMPARISON ===")
    for l_key in [k for k in results.keys() if k.startswith("steering_")]:
        lr = results[l_key]
        wu_orig_effs = [v["wu_eff_orig"] for v in lr.values() if abs(v["full_mean"]) > 1e-6]
        wu_comp_effs = [v["wu_eff_comp"] for v in lr.values() if abs(v["full_mean"]) > 1e-6]
        dark_effs = [v["dark_eff_orig"] for v in lr.values() if abs(v["full_mean"]) > 1e-6]

        print(f"  {l_key}:")
        print(f"    WU_orig(范数不等): eff = {np.mean(wu_orig_effs):.3f} ± {np.std(wu_orig_effs):.3f}")
        print(f"    WU_comp(范数相等): eff = {np.mean(wu_comp_effs):.3f} ± {np.std(wu_comp_effs):.3f}")
        print(f"    Dark(标准alpha):   eff = {np.mean(dark_effs):.3f} ± {np.std(dark_effs):.3f}")

        # 判决
        if np.mean(wu_comp_effs) < 0.2:
            print(f"    ★★★ 范数补偿后W_U仍然无效 → 转导机制成立!")
        else:
            print(f"    ★★★ 范数补偿后W_U变有效 → 范数混淆是主因, 转导解释有误!")

    return results


# ================================================================
# Exp2: 细粒度alpha扫描
# ================================================================
def run_exp2(model, tokenizer, device, model_info, concepts):
    """
    解决问题二: alpha=1.0崩溃

    在alpha=0.1到1.5之间做细粒度扫描(步长0.1), 找非线性临界点
    同时做等范数对比: W_U补偿alpha vs Dark标准alpha
    """
    print(f"\n{'='*60}")
    print(f"  Exp2: Fine-Grained Alpha Sweep")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers

    U_wu = compute_wu_basis(model)

    all_capture = list(range(n_layers))
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_capture)

    all_deltas = {}
    for cname, cdata in concepts.items():
        concept_states = {l: [] for l in all_capture}
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, all_capture)
            for l in all_capture:
                if l in states and l in bl_all:
                    concept_states[l].append(states[l])
        deltas = {}
        for l in all_capture:
            if concept_states[l]:
                deltas[l] = np.mean(concept_states[l], axis=0) - bl_all[l]
        all_deltas[cname] = deltas
        del concept_states
        gc.collect()

    inject_layer = 18 if 18 < n_layers else n_layers // 2
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]

    results = {}

    for alpha in alpha_values:
        print(f"\n  alpha={alpha:.1f}:")
        alpha_data = {}

        for cname, cdata in concepts.items():
            if inject_layer not in all_deltas.get(cname, {}):
                continue

            delta = all_deltas[cname][inject_layer]
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-8:
                continue

            delta_wu, delta_dark, _ = decompose_delta(delta, U_wu)
            norm_wu = np.linalg.norm(delta_wu)
            norm_dark = np.linalg.norm(delta_dark)

            if norm_wu < 1e-8 or norm_dark < 1e-8:
                continue

            probe_words = cdata["probe_words"]

            # 三种steering: full, dark, wu_comp(等范数)
            effects = {}
            for stype, sdir in [("full", delta), ("dark", delta_dark)]:
                eff = compute_steering_effect(model, tokenizer, device, probe_words, inject_layer, sdir, alpha)
                effects[stype] = np.mean(list(eff.values())) if eff else 0

            # W_U补偿alpha
            alpha_wu_comp = alpha * (norm_dark / norm_wu)
            wu_comp_eff = compute_steering_effect(model, tokenizer, device, probe_words, inject_layer, delta_wu, alpha_wu_comp)
            effects["wu_comp"] = np.mean(list(wu_comp_eff.values())) if wu_comp_eff else 0

            # 原版W_U
            wu_orig_eff = compute_steering_effect(model, tokenizer, device, probe_words, inject_layer, delta_wu, alpha)
            effects["wu_orig"] = np.mean(list(wu_orig_eff.values())) if wu_orig_eff else 0

            alpha_data[cname] = {
                "full": effects["full"],
                "dark": effects["dark"],
                "wu_orig": effects["wu_orig"],
                "wu_comp": effects["wu_comp"],
                "inj_norm_dark": alpha * norm_dark,
                "inj_norm_wu_comp": alpha_wu_comp * norm_wu,
            }

        # 汇总
        full_vals = [v["full"] for v in alpha_data.values()]
        dark_vals = [v["dark"] for v in alpha_data.values()]
        wu_orig_vals = [v["wu_orig"] for v in alpha_data.values()]
        wu_comp_vals = [v["wu_comp"] for v in alpha_data.values()]

        def safe_eff(a, b):
            return np.mean([x/y if abs(y) > 1e-6 else 0 for x, y in zip(a, b)])

        dark_eff = safe_eff(dark_vals, full_vals)
        wu_orig_eff = safe_eff(wu_orig_vals, full_vals)
        wu_comp_eff = safe_eff(wu_comp_vals, full_vals)

        results[str(alpha)] = {
            "alpha": alpha,
            "full_mean": float(np.mean(full_vals)),
            "dark_mean": float(np.mean(dark_vals)),
            "wu_orig_mean": float(np.mean(wu_orig_vals)),
            "wu_comp_mean": float(np.mean(wu_comp_vals)),
            "dark_eff": float(dark_eff),
            "wu_orig_eff": float(wu_orig_eff),
            "wu_comp_eff": float(wu_comp_eff),
        }

        print(f"    full={np.mean(full_vals):.3f}, dark={np.mean(dark_vals):.3f}(eff={dark_eff:.3f}), "
              f"wu_orig={np.mean(wu_orig_vals):.3f}(eff={wu_orig_eff:.3f}), "
              f"wu_comp={np.mean(wu_comp_vals):.3f}(eff={wu_comp_eff:.3f})")

    # 找非线性临界点
    print(f"\n  === Nonlinearity Threshold Analysis ===")
    # W_U_comp的效果突然变负的alpha
    for a_str, a_data in results.items():
        if a_data["wu_comp_mean"] < 0 and a_data["dark_mean"] > 0:
            print(f"  alpha={a_data['alpha']}: W_U_comp变负({a_data['wu_comp_mean']:.3f}) 但Dark仍正({a_data['dark_mean']:.3f})")
        if a_data["dark_mean"] < 0:
            print(f"  alpha={a_data['alpha']}: Dark也变负({a_data['dark_mean']:.3f}) — 全面崩溃")
            break

    return results


# ================================================================
# Exp3: 扩展概念集 — 暗物质有效维度的统计可靠性
# ================================================================
def run_exp3(model, tokenizer, device, model_info, concepts_20):
    """
    解决问题三: 暗物质维度7的统计可靠性

    用20个概念重做暗物质PCA
    如果有效维度仍~7 → 7是真实的
    如果有效维度随概念数增长 → 7是低估
    """
    print(f"\n{'='*60}")
    print(f"  Exp3: Extended Concept Set Dark Matter PCA")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers

    U_wu = compute_wu_basis(model)

    all_capture = list(range(n_layers))
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_capture)

    # 收集所有概念的暗物质
    # 分三组: concrete, abstract, relational
    concept_groups = {
        "concrete": ["apple", "dog", "mountain", "ocean", "house", "car", "tree"],
        "abstract": ["love", "science", "freedom", "justice", "beauty", "truth", "wisdom"],
        "relational": ["king", "doctor", "teacher", "mother", "enemy", "friend"],
    }

    all_dark = {}  # {layer: {group: [dark_vectors]}}
    all_dark_full = {}  # {layer: [all dark vectors]}

    for cname, cdata in concepts_20.items():
        concept_states = {l: [] for l in all_capture}
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, all_capture)
            for l in all_capture:
                if l in states and l in bl_all:
                    concept_states[l].append(states[l])

        for l in all_capture:
            if concept_states[l]:
                delta = np.mean(concept_states[l], axis=0) - bl_all[l]
                _, delta_dark, _ = decompose_delta(delta, U_wu)

                if np.linalg.norm(delta_dark) > 1e-8:
                    # 全部
                    if l not in all_dark_full:
                        all_dark_full[l] = []
                    all_dark_full[l].append(delta_dark)

                    # 分组
                    group = None
                    for g, glist in concept_groups.items():
                        if cname in glist:
                            group = g
                            break

                    if group:
                        if l not in all_dark:
                            all_dark[l] = {}
                        if group not in all_dark[l]:
                            all_dark[l][group] = []
                        all_dark[l][group].append(delta_dark)

        del concept_states
        gc.collect()

    key_layers = [l for l in [6, 12, 18, 24, 30] if l < n_layers]

    results = {}

    # Part 1: 全部20概念一起做PCA
    print(f"\n  --- Part 1: All 20 Concepts Together ---")
    for l in key_layers:
        if l not in all_dark_full or len(all_dark_full[l]) < 3:
            continue

        X = np.array(all_dark_full[l])
        n_samples = X.shape[0]
        X_c = X - X.mean(axis=0, keepdims=True)

        # PCA
        if n_samples < d_model:
            XXt = X_c @ X_c.T
            eigenvalues, _ = np.linalg.eigh(XXt)
            eigenvalues = eigenvalues[::-1]
            s = np.sqrt(np.maximum(eigenvalues, 0))
        else:
            _, s, _ = np.linalg.svd(X_c, full_matrices=False)

        total_var = np.sum(s ** 2)
        if total_var < 1e-20:
            continue

        var_ratio = s ** 2 / total_var
        cum_var = np.cumsum(var_ratio)
        p = var_ratio[var_ratio > 1e-10]
        entropy = -np.sum(p * np.log2(p))
        eff_rank = 2 ** entropy

        n_90 = int(np.searchsorted(cum_var, 0.90)) + 1
        n_95 = int(np.searchsorted(cum_var, 0.95)) + 1

        results[f"all_L{l}"] = {
            "n_samples": n_samples,
            "effective_rank": float(eff_rank),
            "n_for_90": n_90,
            "n_for_95": n_95,
            "cum_var_at_1": float(cum_var[0]),
            "cum_var_at_5": float(cum_var[min(4, len(cum_var)-1)]),
            "cum_var_at_10": float(cum_var[min(9, len(cum_var)-1)]),
            "cum_var_at_15": float(cum_var[min(14, len(cum_var)-1)]),
        }

        print(f"  L{l}: n={n_samples}, eff_rank={eff_rank:.1f}, n_90={n_90}, n_95={n_95}, "
              f"cum@1={cum_var[0]:.3f}, cum@5={cum_var[min(4,len(cum_var)-1)]:.3f}, "
              f"cum@10={cum_var[min(9,len(cum_var)-1)]:.3f}")

    # Part 2: 分组PCA
    print(f"\n  --- Part 2: By Concept Group ---")
    for l in key_layers:
        if l not in all_dark:
            continue

        for group, dark_vecs in all_dark[l].items():
            if len(dark_vecs) < 3:
                continue

            X = np.array(dark_vecs)
            n_samples = X.shape[0]
            X_c = X - X.mean(axis=0, keepdims=True)

            if n_samples < d_model:
                XXt = X_c @ X_c.T
                eigenvalues, _ = np.linalg.eigh(XXt)
                eigenvalues = eigenvalues[::-1]
                s = np.sqrt(np.maximum(eigenvalues, 0))
            else:
                _, s, _ = np.linalg.svd(X_c, full_matrices=False)

            total_var = np.sum(s ** 2)
            if total_var < 1e-20:
                continue

            var_ratio = s ** 2 / total_var
            cum_var = np.cumsum(var_ratio)
            p = var_ratio[var_ratio > 1e-10]
            entropy = -np.sum(p * np.log2(p))
            eff_rank = 2 ** entropy

            n_90 = int(np.searchsorted(cum_var, 0.90)) + 1

            results[f"{group}_L{l}"] = {
                "n_samples": n_samples,
                "effective_rank": float(eff_rank),
                "n_for_90": n_90,
                "cum_var_at_1": float(cum_var[0]),
                "cum_var_at_5": float(cum_var[min(4, len(cum_var)-1)]),
            }

            print(f"  L{l} ({group}, n={n_samples}): eff_rank={eff_rank:.1f}, n_90={n_90}, "
                  f"cum@1={cum_var[0]:.3f}, cum@5={cum_var[min(4,len(cum_var)-1)]:.3f}")

    # Part 3: 增量分析 — 概念数 vs 有效维度
    print(f"\n  --- Part 3: Scaling Analysis (n_concepts vs eff_rank) ---")
    for l in key_layers:
        if l not in all_dark_full:
            continue

        dark_vecs = all_dark_full[l]
        n_total = len(dark_vecs)

        scaling_data = []
        # 从3个概念开始, 逐步增加
        np.random.seed(42)
        indices = list(range(n_total))
        np.random.shuffle(indices)

        for n_concepts in [3, 5, 8, 10, 12, 15, 20]:
            if n_concepts > n_total:
                continue

            subset = [dark_vecs[i] for i in indices[:n_concepts]]
            X = np.array(subset)
            X_c = X - X.mean(axis=0, keepdims=True)

            if n_concepts < d_model:
                XXt = X_c @ X_c.T
                eigenvalues, _ = np.linalg.eigh(XXt)
                eigenvalues = eigenvalues[::-1]
                s = np.sqrt(np.maximum(eigenvalues, 0))
            else:
                _, s, _ = np.linalg.svd(X_c, full_matrices=False)

            total_var = np.sum(s ** 2)
            if total_var < 1e-20:
                continue

            var_ratio = s ** 2 / total_var
            p = var_ratio[var_ratio > 1e-10]
            entropy = -np.sum(p * np.log2(p))
            eff_rank = 2 ** entropy

            scaling_data.append({"n_concepts": n_concepts, "eff_rank": float(eff_rank)})
            print(f"  L{l}: n_concepts={n_concepts}, eff_rank={eff_rank:.1f}")

        results[f"scaling_L{l}"] = scaling_data

    # 判决
    print(f"\n  === VERDICT ===")
    for l in key_layers:
        if f"all_L{l}" in results:
            eff = results[f"all_L{l}"]["effective_rank"]
            n_samp = results[f"all_L{l}"]["n_samples"]
            if eff < 0.7 * n_samp:
                print(f"  L{l}: eff_rank={eff:.1f} << n_samples={n_samp} → 维度7-12可能是真实的!")
            else:
                print(f"  L{l}: eff_rank={eff:.1f} ≈ n_samples={n_samp} → 维度7是低估, 随概念数增长!")

    return results


# ================================================================
# Exp4: 小alpha线性区验证
# ================================================================
def run_exp4(model, tokenizer, device, model_info, concepts):
    """
    解决问题四: 传播方程的线性假设

    在alpha=0.1, 0.25, 0.5下追踪转导参数lambda_wu和tau_dark
    如果这些参数不随alpha变化 → 线性区成立
    如果变化 → alpha=0.5已经在非线性区
    """
    print(f"\n{'='*60}")
    print(f"  Exp4: Linear Regime Verification")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers

    U_wu = compute_wu_basis(model)

    all_capture = list(range(n_layers))
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_capture)

    all_deltas = {}
    for cname, cdata in concepts.items():
        concept_states = {l: [] for l in all_capture}
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, all_capture)
            for l in all_capture:
                if l in states and l in bl_all:
                    concept_states[l].append(states[l])
        deltas = {}
        for l in all_capture:
            if concept_states[l]:
                deltas[l] = np.mean(concept_states[l], axis=0) - bl_all[l]
        all_deltas[cname] = deltas
        del concept_states
        gc.collect()

    inject_layer = 12 if 12 < n_layers else n_layers // 3
    alpha_values = [0.1, 0.25, 0.5]
    track_range = 6

    results = {}

    for alpha in alpha_values:
        print(f"\n  alpha={alpha}:")
        alpha_trans = {}

        for cname, cdata in concepts.items():
            if inject_layer not in all_deltas.get(cname, {}):
                continue

            delta = all_deltas[cname][inject_layer]
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-8:
                continue

            delta_wu, delta_dark, _ = decompose_delta(delta, U_wu)
            norm_wu = np.linalg.norm(delta_wu)
            norm_dark = np.linalg.norm(delta_dark)

            if norm_wu < 1e-8 or norm_dark < 1e-8:
                continue

            track_layers = [inject_layer + k for k in range(1, track_range + 1) if inject_layer + k < n_layers]
            capture_layers = [inject_layer] + track_layers

            # 两种注入: wu_only 和 dark_only
            for itype, idir in [("wu", delta_wu), ("dark", delta_dark)]:
                injected_states = inject_and_collect(
                    model, tokenizer, device, inject_layer, idir, alpha, capture_layers
                )

                # 追踪每层的W_U投影比变化
                trans_data = []
                for tl in track_layers:
                    if tl not in injected_states or tl not in bl_all:
                        continue

                    delta_tl = injected_states[tl] - bl_all[tl]
                    delta_tl_norm = np.linalg.norm(delta_tl)
                    if delta_tl_norm < 1e-8:
                        continue

                    proj_wu_tl = U_wu.T @ delta_tl
                    wu_ratio = np.sum(proj_wu_tl ** 2) / (delta_tl_norm ** 2)

                    trans_data.append({
                        "layer": tl,
                        "k": tl - inject_layer,
                        "delta_norm": float(delta_tl_norm),
                        "wu_ratio": float(wu_ratio),
                    })

                if itype not in alpha_trans:
                    alpha_trans[itype] = {}
                alpha_trans[itype][cname] = trans_data

        # 估计lambda_wu和tau_dark
        # W_U注入: wu_ratio(k) ≈ lambda_wu^k (如果线性)
        # Dark注入: wu_ratio(k) ≈ tau_dark * k (如果线性, 一阶近似)
        print(f"  Estimating lambda_wu and tau_dark:")

        # W_U注入的W_U投影比随层数衰减
        wu_ratios_by_k = {}
        for cname, trans_list in alpha_trans.get("wu", {}).items():
            for td in trans_list:
                k = td["k"]
                if k not in wu_ratios_by_k:
                    wu_ratios_by_k[k] = []
                wu_ratios_by_k[k].append(td["wu_ratio"])

        # 拟合lambda_wu: wu_ratio(k) = lambda_wu^k
        # log(wu_ratio) = k * log(lambda_wu)
        if wu_ratios_by_k:
            ks = sorted(wu_ratios_by_k.keys())
            mean_ratios = [np.mean(wu_ratios_by_k[k]) for k in ks if k > 0]

            if len(ks) >= 2 and all(r > 0 for r in mean_ratios):
                # 线性拟合log(ratio) vs k
                log_ratios = [np.log(r) for r in mean_ratios]
                ks_valid = [k for k, r in zip(ks, mean_ratios) if r > 0]
                if len(ks_valid) >= 2:
                    log_ratios_valid = [np.log(np.mean(wu_ratios_by_k[k])) for k in ks_valid]
                    # 最小二乘
                    A = np.array(ks_valid).reshape(-1, 1)
                    b = np.array(log_ratios_valid)
                    lambda_wu_est = np.exp(np.linalg.lstsq(A, b, rcond=None)[0][0])
                else:
                    lambda_wu_est = None
            else:
                lambda_wu_est = None
        else:
            lambda_wu_est = None

        # Dark注入: W_U投影比随层数增长
        dark_ratios_by_k = {}
        for cname, trans_list in alpha_trans.get("dark", {}).items():
            for td in trans_list:
                k = td["k"]
                if k not in dark_ratios_by_k:
                    dark_ratios_by_k[k] = []
                dark_ratios_by_k[k].append(td["wu_ratio"])

        # 估计tau_dark: wu_ratio(k) ≈ tau_dark * k (一阶)
        tau_dark_est = None
        if dark_ratios_by_k:
            ks = sorted(dark_ratios_by_k.keys())
            if len(ks) >= 2:
                # 用k=1的W_U投影比作为tau_dark的估计
                if 1 in dark_ratios_by_k:
                    tau_dark_est = np.mean(dark_ratios_by_k[1])

        results[str(alpha)] = {
            "lambda_wu_est": float(lambda_wu_est) if lambda_wu_est else None,
            "tau_dark_est": float(tau_dark_est) if tau_dark_est else None,
            "wu_ratios_by_k": {str(k): [float(x) for x in v] for k, v in wu_ratios_by_k.items()},
            "dark_ratios_by_k": {str(k): [float(x) for x in v] for k, v in dark_ratios_by_k.items()},
        }

        if lambda_wu_est:
            print(f"    lambda_wu ≈ {lambda_wu_est:.4f}")
        if tau_dark_est is not None:
            print(f"    tau_dark ≈ {tau_dark_est:.4f}")

        # 打印平均W_U投影比
        for itype in ["wu", "dark"]:
            ratios_by_k = wu_ratios_by_k if itype == "wu" else dark_ratios_by_k
            if ratios_by_k:
                ratio_str = ", ".join([f"k={k}:{np.mean(v):.4f}" for k, v in sorted(ratios_by_k.items())])
                print(f"    {itype}: {ratio_str}")

    # 判决
    print(f"\n  === LINEARITY VERDICT ===")
    lambdas = [results[str(a)]["lambda_wu_est"] for a in alpha_values if results[str(a)]["lambda_wu_est"]]
    taus = [results[str(a)]["tau_dark_est"] for a in alpha_values if results[str(a)]["tau_dark_est"] is not None]

    if len(lambdas) >= 2:
        lambda_range = max(lambdas) - min(lambdas)
        lambda_mean = np.mean(lambdas)
        if lambda_range < 0.05 * lambda_mean:
            print(f"  lambda_wu stable across alpha: {lambdas} → LINEAR REGIME CONFIRMED")
        else:
            print(f"  lambda_wu varies with alpha: {lambdas} → NONLINEAR at alpha=0.5")

    if len(taus) >= 2:
        tau_range = max(taus) - min(taus)
        tau_mean = np.mean(taus)
        if tau_range < 0.05 * max(tau_mean, 1e-6):
            print(f"  tau_dark stable across alpha: {taus} → LINEAR REGIME CONFIRMED")
        else:
            print(f"  tau_dark varies with alpha: {taus} → NONLINEAR at alpha=0.5")

    return results


# ================================================================
# Exp5: 概念类型分组 — 暗物质维度的语义异质性
# ================================================================
def run_exp5(model, tokenizer, device, model_info, concepts_20):
    """
    解决问题五: "7维概念核"的解读过于激进

    测试不同类型概念的暗物质是否共享子空间
    如果concrete/abstract/relational的暗物质维度不同 → 没有统一的"概念核"
    如果三组暗物质共享主要方向 → 概念核假设更可信

    同时测试: 暗物质是否可线性组合?
    concept_A_dark + concept_B_dark ≈ concept_AB_dark?
    """
    print(f"\n{'='*60}")
    print(f"  Exp5: Concept Type Heterogeneity")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers

    U_wu = compute_wu_basis(model)

    all_capture = list(range(n_layers))
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_capture)

    # 收集暗物质
    concept_groups = {
        "concrete": ["apple", "dog", "mountain", "ocean", "house", "car", "tree"],
        "abstract": ["love", "science", "freedom", "justice", "beauty", "truth", "wisdom"],
        "relational": ["king", "doctor", "teacher", "mother", "enemy", "friend"],
    }

    all_dark = {}  # {layer: {cname: dark_vec}}
    for cname, cdata in concepts_20.items():
        concept_states = {l: [] for l in all_capture}
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, all_capture)
            for l in all_capture:
                if l in states and l in bl_all:
                    concept_states[l].append(states[l])

        for l in all_capture:
            if concept_states[l]:
                delta = np.mean(concept_states[l], axis=0) - bl_all[l]
                _, delta_dark, _ = decompose_delta(delta, U_wu)
                if np.linalg.norm(delta_dark) > 1e-8:
                    if l not in all_dark:
                        all_dark[l] = {}
                    all_dark[l][cname] = delta_dark

        del concept_states
        gc.collect()

    key_layers = [l for l in [12, 18, 24] if l < n_layers]
    results = {}

    for l in key_layers:
        if l not in all_dark:
            continue

        print(f"\n  --- Layer {l} ---")
        layer_results = {}

        # Part A: 各组分别做PCA, 比较有效维度
        print(f"  Part A: Per-Group PCA")
        group_pcas = {}
        for group, gnames in concept_groups.items():
            dark_vecs = [all_dark[l][n] for n in gnames if n in all_dark[l]]
            if len(dark_vecs) < 3:
                continue

            X = np.array(dark_vecs)
            X_c = X - X.mean(axis=0, keepdims=True)
            n_samp = X.shape[0]

            if n_samp < d_model:
                XXt = X_c @ X_c.T
                eigenvalues, _ = np.linalg.eigh(XXt)
                eigenvalues = eigenvalues[::-1]
                s = np.sqrt(np.maximum(eigenvalues, 0))
            else:
                _, s, _ = np.linalg.svd(X_c, full_matrices=False)

            total_var = np.sum(s ** 2)
            if total_var < 1e-20:
                continue

            var_ratio = s ** 2 / total_var
            p = var_ratio[var_ratio > 1e-10]
            entropy = -np.sum(p * np.log2(p))
            eff_rank = 2 ** entropy

            group_pcas[group] = {
                "n_samples": n_samp,
                "effective_rank": float(eff_rank),
                "cum_var_at_1": float(var_ratio[0]),
                "cum_var_at_3": float(np.cumsum(var_ratio)[min(2, len(var_ratio)-1)]),
            }

            print(f"    {group}(n={n_samp}): eff_rank={eff_rank:.1f}, cum@1={var_ratio[0]:.3f}")

        layer_results["group_pca"] = group_pcas

        # Part B: 跨组暗物质的主方向是否一致?
        print(f"  Part B: Cross-Group Alignment")
        group_pc1s = {}
        for group, gnames in concept_groups.items():
            dark_vecs = [all_dark[l][n] for n in gnames if n in all_dark[l]]
            if len(dark_vecs) < 3:
                continue

            X = np.array(dark_vecs)
            X_c = X - X.mean(axis=0, keepdims=True)

            # 第1个PC方向
            if X_c.shape[0] < d_model:
                XXt = X_c @ X_c.T
                eigenvalues, U_small = np.linalg.eigh(XXt)
                idx = np.argsort(eigenvalues)[::-1]
                U_small = U_small[:, idx]
                s = np.sqrt(np.maximum(eigenvalues[idx], 0))
                valid = s > 1e-10
                pc1 = X_c.T @ U_small[:, valid][:, 0] / s[valid][0]
                pc1 = pc1 / np.linalg.norm(pc1)
            else:
                _, s, Vt = np.linalg.svd(X_c, full_matrices=False)
                pc1 = Vt[0]

            group_pc1s[group] = pc1

        # 计算PC1间的余弦相似度
        cross_alignment = {}
        group_names = list(group_pc1s.keys())
        for i in range(len(group_names)):
            for j in range(i+1, len(group_names)):
                g1, g2 = group_names[i], group_names[j]
                cos_val = float(np.abs(np.dot(group_pc1s[g1], group_pc1s[g2])))
                cross_alignment[f"{g1}_vs_{g2}"] = cos_val
                print(f"    PC1 cos({g1}, {g2}) = {cos_val:.3f}")

        layer_results["cross_alignment"] = cross_alignment

        # Part C: 暗物质可加性测试
        # concept_A_dark + concept_B_dark 的方向 vs concept_AB的暗物质?
        # 这里用"king"和"queen"的对偶性: king暗物质 - queen暗物质 ≈ gender方向?
        print(f"  Part C: Additivity Test")
        additivity_results = {}

        # 选取几对概念测试可加性
        pairs = [
            ("king", "doctor"),  # 两个关系概念
            ("apple", "dog"),    # 两个具体概念
            ("love", "science"), # 两个抽象概念
            ("apple", "love"),   # 具体+抽象
        ]

        for c1, c2 in pairs:
            if c1 not in all_dark[l] or c2 not in all_dark[l]:
                continue

            d1 = all_dark[l][c1]
            d2 = all_dark[l][c2]

            # 暗物质之和
            d_sum = d1 + d2
            sum_norm = np.linalg.norm(d_sum)

            # 暗物质之差
            d_diff = d1 - d2
            diff_norm = np.linalg.norm(d_diff)

            # 各自范数
            n1 = np.linalg.norm(d1)
            n2 = np.linalg.norm(d2)

            # 可加性: ||d1+d2|| vs ||d1||+||d2||
            additivity_ratio = sum_norm / (n1 + n2) if (n1 + n2) > 1e-8 else 0

            # 差异性: ||d1-d2|| / ||d1+d2||
            diff_ratio = diff_norm / sum_norm if sum_norm > 1e-8 else 0

            # 余弦
            cos_12 = float(np.dot(d1, d2) / (n1 * n2)) if n1 > 1e-8 and n2 > 1e-8 else 0

            additivity_results[f"{c1}_{c2}"] = {
                "additivity_ratio": float(additivity_ratio),
                "diff_ratio": float(diff_ratio),
                "cos_between": cos_12,
            }

            print(f"    {c1}+{c2}: add_ratio={additivity_ratio:.3f}, diff_ratio={diff_ratio:.3f}, cos={cos_12:.3f}")

        layer_results["additivity"] = additivity_results
        results[f"L{l}"] = layer_results

    # 判决
    print(f"\n  === CONCEPT NUCLEUS VERDICT ===")
    for l_key, lr in results.items():
        group_pca = lr.get("group_pca", {})
        cross_align = lr.get("cross_alignment", {})

        # 各组有效维度
        ranks = [v["effective_rank"] for v in group_pca.values()]
        if ranks:
            rank_range = max(ranks) - min(ranks)
            rank_mean = np.mean(ranks)
            print(f"  {l_key}: Group eff_ranks = {[f'{v:.1f}' for v in ranks]}")

            if rank_range < 2:
                print(f"    → 各组维度相近, 支持'统一概念核'")
            else:
                print(f"    → 各组维度差异大, 不支持'统一7维概念核'")

        # 跨组对齐
        aligns = list(cross_align.values())
        if aligns:
            print(f"    跨组PC1对齐: mean={np.mean(aligns):.3f}")
            if np.mean(aligns) > 0.5:
                print(f"    → 跨组PC1高度对齐, 暗物质有统一主方向")
            else:
                print(f"    → 跨组PC1低对齐, 不同类型概念使用不同暗物质方向")

    return results


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["1", "2", "3", "4", "5", "all"])
    args = parser.parse_args()
    model_name = args.model

    print(f"\n{'#'*70}")
    print(f"CCXIX: Norm-Controlled Transduction Verification — {model_name}")
    print(f"{'#'*70}")

    model, tokenizer, device = load_model(model_name)
    if hasattr(model, 'config'):
        model.config.output_hidden_states = True

    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    print(f"  d_model={d_model}, n_layers={n_layers}")

    all_results = {}

    if args.exp in ["1", "all"]:
        exp1_results = run_exp1(model, tokenizer, device, model_info, CONCEPTS_8)
        all_results["exp1_norm_equivalent"] = exp1_results

    if args.exp in ["2", "all"]:
        exp2_results = run_exp2(model, tokenizer, device, model_info, CONCEPTS_8)
        all_results["exp2_alpha_sweep"] = exp2_results

    if args.exp in ["3", "all"]:
        exp3_results = run_exp3(model, tokenizer, device, model_info, CONCEPTS_20)
        all_results["exp3_extended_pca"] = exp3_results

    if args.exp in ["4", "all"]:
        exp4_results = run_exp4(model, tokenizer, device, model_info, CONCEPTS_8)
        all_results["exp4_linear_regime"] = exp4_results

    if args.exp in ["5", "all"]:
        exp5_results = run_exp5(model, tokenizer, device, model_info, CONCEPTS_20)
        all_results["exp5_concept_type"] = exp5_results

    # 保存
    output_path = TEMP / f"ccxix_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_path}")

    release_model(model)
    print(f"\nCCXIX {model_name} 完成!")


if __name__ == "__main__":
    main()
