"""
CCXXV(325): 跨模型语义几何对齐
======================================================================
同一语义关系在不同模型的子空间对齐度如何?
如果语义空间有通用数学结构, 不同模型的同关系子空间应该对齐。

实验设计:
  1. 收集三模型在相同词汇上的残差
  2. 对每种关系(habitat/food/material/color/size), 计算子空间
  3. 跨模型子空间对齐: CCA或Grassmann距离
  4. 语义方向对齐: 同关系在不同模型的均值方向cos
  5. 检查: 对齐是否来自语义, 还是来自词嵌入的统计特性?

关键: 三模型d_model不同(2560/4096/3584), 需要降维后比较

用法:
  python ccxxv_cross_model_alignment.py --models qwen3 glm4 deepseek7b
  (一次运行所有模型, 保存中间结果, 最后计算跨模型对齐)
"""
import argparse, os, sys, json, gc
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from scipy.sparse.linalg import svds

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxxv_cross_model_alignment_log.txt"

# 通用词汇(确保三模型都能tokenize)
RELATION_WORDS = {
    "habitat": {
        "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "fox", "deer"],
        "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "crab", "seal", "squid"],
        "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "falcon", "swallow"],
    },
    "food": {
        "meat": ["beef", "pork", "chicken", "lamb", "bacon", "ham", "steak", "veal"],
        "fruit": ["apple", "banana", "orange", "grape", "mango", "peach", "pear", "cherry"],
        "grain": ["rice", "wheat", "corn", "oats", "barley", "rye", "millet", "quinoa"],
    },
    "material": {
        "metal": ["iron", "steel", "copper", "gold", "silver", "bronze", "tin", "lead"],
        "wood": ["oak", "pine", "cedar", "maple", "birch", "elm", "ash", "walnut"],
        "stone": ["granite", "marble", "slate", "limestone", "basalt", "quartz", "obsidian", "sandstone"],
    },
}

TEMPLATE = "The {}"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def collect_model_resids(model_name):
    """收集单个模型的残差, 保存中间结果"""
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    log(f"\n  收集 {model_name} 残差 (d_model={d_model}, n_layers={n_layers})")
    
    # 测试层: 中间层(单纯形最精确的层)
    if model_name == "qwen3":
        mid_layer = 9
    elif model_name == "glm4":
        mid_layer = 10
    elif model_name == "deepseek7b":
        mid_layer = 7
    else:
        mid_layer = n_layers // 4
    
    test_layers = [0, mid_layer, n_layers // 2, n_layers - 1]
    test_layers = sorted(set(test_layers))
    
    # 收集残差
    all_resids = {}  # {li: {relation_group: {sub_group: [vec1, vec2, ...]}}}
    
    for rel_group, sub_groups in RELATION_WORDS.items():
        for sub_name, words in sub_groups.items():
            for word in words:
                prompt = TEMPLATE.format(word)
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                seq_len = toks.input_ids.shape[1]
                last_pos = seq_len - 1
                
                captured = {}
                def mk_hook(k):
                    def hook(m, inp, out):
                        o = out[0] if isinstance(out, tuple) else out
                        captured[k] = o[0, last_pos, :].detach().float().cpu().numpy()
                    return hook
                
                hooks = [layers[li].register_forward_hook(mk_hook(f"L{li}")) for li in test_layers]
                with torch.no_grad():
                    _ = model(**toks)
                for h in hooks:
                    h.remove()
                
                for li in test_layers:
                    if li not in all_resids:
                        all_resids[li] = {}
                    key = f"{rel_group}_{sub_name}"
                    if key not in all_resids[li]:
                        all_resids[li][key] = []
                    if f"L{li}" in captured:
                        all_resids[li][key].append(captured[f"L{li}"])
    
    # 保存中间结果
    # 由于numpy数组不能直接json化, 转为list
    save_data = {
        "model": model_name,
        "d_model": d_model,
        "n_layers": n_layers,
        "test_layers": test_layers,
        "mid_layer": mid_layer,
    }
    
    # 保存为numpy格式(更紧凑)
    out_path = TEMP / f"ccxxv_resids_{model_name}.npz"
    save_dict = {}
    for li in all_resids:
        for key in all_resids[li]:
            arr = np.array(all_resids[li][key])
            save_dict[f"L{li}_{key}"] = arr
    np.savez_compressed(out_path, **save_dict)
    
    # 也计算并保存每层的子空间
    subspaces = {}
    for li in test_layers:
        for rel_group, sub_groups in RELATION_WORDS.items():
            # 整个关系组的子空间
            all_vecs = []
            for sub_name, words in sub_groups.items():
                key = f"{rel_group}_{sub_name}"
                if key in all_resids[li]:
                    all_vecs.extend(all_resids[li][key])
            
            if len(all_vecs) < 5:
                continue
            
            X = np.array(all_vecs)
            mean = X.mean(axis=0)
            Xc = X - mean
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            
            subspaces[f"L{li}_{rel_group}"] = {
                "mean": mean,
                "Vt": Vt[:min(10, Vt.shape[0])],  # 前10个PC
                "S": S[:min(10, len(S))],
            }
            
            # 每个子组的方向
            for sub_name in sub_groups:
                key = f"{rel_group}_{sub_name}"
                if key in all_resids[li] and len(all_resids[li][key]) >= 3:
                    sub_vecs = np.array(all_resids[li][key])
                    sub_mean = sub_vecs.mean(axis=0) - mean
                    norm = np.linalg.norm(sub_mean)
                    if norm > 1e-10:
                        subspaces[f"L{li}_{key}_dir"] = {
                            "direction": sub_mean / norm,
                            "norm": norm,
                        }
    
    # 保存子空间
    subspace_path = TEMP / f"ccxxv_subspaces_{model_name}.npz"
    subspace_dict = {}
    for k, v in subspaces.items():
        for vk, vv in v.items():
            subspace_dict[f"{k}_{vk}"] = vv
    np.savez_compressed(subspace_path, **subspace_dict)
    
    log(f"  {model_name}: 保存了 {len(save_dict)} 个残差数组, {len(subspace_dict)} 个子空间")
    
    release_model(model)
    return all_resids, subspaces, d_model, n_layers, test_layers


def compute_cross_model_alignment(model_names):
    """计算跨模型对齐"""
    log(f"\n{'='*70}")
    log(f"跨模型对齐分析")
    log(f"{'='*70}")
    
    results = {}
    
    # 加载各模型数据
    model_data = {}
    for mn in model_names:
        resid_path = TEMP / f"ccxxv_resids_{mn}.npz"
        if not resid_path.exists():
            log(f"  跳过 {mn}: 残差文件不存在")
            continue
        data = np.load(resid_path, allow_pickle=True)
        d_model = int(str(data.get("d_model", 2560))) if "d_model" in data else None
        model_data[mn] = data
    
    if len(model_data) < 2:
        log("  需要至少2个模型的数据!")
        return {}
    
    # 获取共同的关系组
    relation_groups = list(RELATION_WORDS.keys())
    
    # 对每对模型计算对齐
    model_list = list(model_data.keys())
    for i, m1 in enumerate(model_list):
        for j, m2 in enumerate(model_list):
            if i >= j:
                continue
            
            log(f"\n--- {m1} vs {m2} ---")
            
            d1 = model_data[m1]
            d2 = model_data[m2]
            
            # 获取各模型使用的中间层
            mid1 = 9 if m1 == "qwen3" else (10 if m1 == "glm4" else 7)
            mid2 = 9 if m2 == "qwen3" else (10 if m2 == "glm4" else 7)
            
            # 层列表
            n1 = 36 if m1 == "qwen3" else (40 if m1 == "glm4" else 28)
            n2 = 36 if m2 == "qwen3" else (40 if m2 == "glm4" else 28)
            
            # 对比层: L0, 中间层, L_last
            layer_pairs = [
                (0, 0, "L0"),
                (mid1, mid2, f"Lmid({mid1}vs{mid2})"),
                (n1-1, n2-1, f"Llast({n1-1}vs{n2-1})"),
            ]
            
            for li1, li2, layer_label in layer_pairs:
                for rel_group in relation_groups:
                    # 获取两个模型在该关系组的残差
                    keys1 = [k for k in d1.files if k.startswith(f"L{li1}_{rel_group}_") and k.endswith("_dir")]
                    keys2 = [k for k in d2.files if k.startswith(f"L{li2}_{rel_group}_") and k.endswith("_dir")]
                    
                    # 子空间对齐(CCA)
                    # 获取各子组的向量
                    sub_keys1 = [k for k in d1.files if k.startswith(f"L{li1}_{rel_group}_") and not k.endswith("_dir") and "mean" not in k and "Vt" not in k and "S" not in k]
                    sub_keys2 = [k for k in d2.files if k.startswith(f"L{li2}_{rel_group}_") and not k.endswith("_dir") and "mean" not in k and "Vt" not in k and "S" not in k]
                    
                    # 简化: 使用子空间(前5个PC)的Grassmann距离
                    Vt1_key = f"L{li1}_{rel_group}_Vt"
                    Vt2_key = f"L{li2}_{rel_group}_Vt"
                    
                    if Vt1_key not in d1 or Vt2_key not in d2:
                        continue
                    
                    Vt1 = d1[Vt1_key][:5]  # [5, d1]
                    Vt2 = d2[Vt2_key][:5]  # [5, d2]
                    
                    # CCA: 找两个子空间的最大相关方向
                    # 简化方法: 投影到共享的低维空间
                    # 方法: 子空间重叠 = ||P1 @ P2||_F / sqrt(k)
                    # 但d_model不同, 不能直接计算
                    
                    # 替代方法: 使用方向对齐
                    # 对每个子组的方向, 在另一个模型的子空间中的投影
                    
                    # 先用子组方向对齐
                    dir_cosines = []
                    for sub_name in RELATION_WORDS[rel_group]:
                        key1 = f"L{li1}_{rel_group}_{sub_name}_dir_direction"
                        key2 = f"L{li2}_{rel_group}_{sub_name}_dir_direction"
                        
                        if key1 not in d1 or key2 not in d2:
                            continue
                        
                        dir1 = d1[key1]  # [d1]
                        dir2 = d2[key2]  # [d2]
                        
                        # 不同d_model, 不能直接计算cos!
                        # 替代: 计算dir1在Vt1子空间中的系数, 
                        #        然后看Vt2子空间是否也有类似结构
                        
                        # dir1在Vt1中的系数
                        coeff1 = Vt1 @ dir1  # [5]
                        # dir2在Vt2中的系数
                        coeff2 = Vt2 @ dir2  # [5]
                        
                        # 归一化系数
                        c1_norm = np.linalg.norm(coeff1)
                        c2_norm = np.linalg.norm(coeff2)
                        
                        if c1_norm > 1e-10 and c2_norm > 1e-10:
                            coeff1_n = coeff1 / c1_norm
                            coeff2_n = coeff2 / c2_norm
                            cos_sub = float(np.abs(np.dot(coeff1_n, coeff2_n)))
                            dir_cosines.append(cos_sub)
                    
                    if dir_cosines:
                        mean_cos = float(np.mean(dir_cosines))
                        
                        pair_key = f"{m1}_vs_{m2}_{layer_label}_{rel_group}"
                        results[pair_key] = {
                            "model1": m1,
                            "model2": m2,
                            "layer_label": layer_label,
                            "relation": rel_group,
                            "sub_dir_cosines": [round(c, 4) for c in dir_cosines],
                            "mean_sub_dir_cos": round(mean_cos, 4),
                        }
                        
                        log(f"  {layer_label} {rel_group}: mean_sub_dir_cos={mean_cos:.4f}")
    
    return results


def run_single(model_name):
    """收集单个模型的数据"""
    log(f"\n{'='*70}\nCCXXV(325): 跨模型语义几何对齐 - 收集 {model_name}")
    log(f"{'='*70}")
    
    all_resids, subspaces, d_model, n_layers, test_layers = collect_model_resids(model_name)
    
    # 保存元信息
    meta = {
        "model": model_name,
        "d_model": d_model,
        "n_layers": n_layers,
        "test_layers": test_layers,
    }
    meta_path = TEMP / f"ccxxv_meta_{model_name}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    
    log(f"  {model_name} 数据收集完成")
    return meta


def run_alignment():
    """运行跨模型对齐分析"""
    log(f"\n{'='*70}\nCCXXV(325): 跨模型语义几何对齐 - 对齐分析")
    log(f"{'='*70}")
    
    # 检查哪些模型有数据
    available = []
    for mn in ["qwen3", "glm4", "deepseek7b"]:
        meta_path = TEMP / f"ccxxv_meta_{mn}.json"
        if meta_path.exists():
            available.append(mn)
    
    if len(available) < 2:
        log(f"  只有 {len(available)} 个模型的数据, 需要至少2个!")
        return {}
    
    # 加载所有子空间数据
    model_subspaces = {}
    model_dmodels = {}
    for mn in available:
        subspace_path = TEMP / f"ccxxv_subspaces_{mn}.npz"
        if subspace_path.exists():
            model_subspaces[mn] = dict(np.load(subspace_path, allow_pickle=True))
        meta_path = TEMP / f"ccxxv_meta_{mn}.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
            model_dmodels[mn] = meta["d_model"]
    
    results = {}
    
    # 重新收集残差并计算对齐
    # 由于d_model不同, 使用方向在PC子空间中的系数来比较
    
    relation_groups = list(RELATION_WORDS.keys())
    
    for i, m1 in enumerate(available):
        for j, m2 in enumerate(available):
            if i >= j:
                continue
            
            log(f"\n--- {m1}(d={model_dmodels[m1]}) vs {m2}(d={model_dmodels[m2]}) ---")
            
            # 层对应
            n1 = 36 if m1 == "qwen3" else (40 if m1 == "glm4" else 28)
            n2 = 36 if m2 == "qwen3" else (40 if m2 == "glm4" else 28)
            mid1 = 9 if m1 == "qwen3" else (10 if m1 == "glm4" else 7)
            mid2 = 9 if m2 == "qwen3" else (10 if m2 == "glm4" else 7)
            
            layer_pairs = [
                (0, 0, "L0"),
                (mid1, mid2, f"Lmid"),
                (n1-1, n2-1, f"Llast"),
            ]
            
            d1 = model_subspaces[m1]
            d2 = model_subspaces[m2]
            
            for li1, li2, layer_label in layer_pairs:
                for rel_group in relation_groups:
                    # 获取子空间的Vt
                    Vt1_key = f"L{li1}_{rel_group}_Vt"
                    Vt2_key = f"L{li2}_{rel_group}_Vt"
                    S1_key = f"L{li1}_{rel_group}_S"
                    S2_key = f"L{li2}_{rel_group}_S"
                    
                    if Vt1_key not in d1 or Vt2_key not in d2:
                        continue
                    
                    Vt1 = d1[Vt1_key][:5]  # [5, d_model1]
                    Vt2 = d2[Vt2_key][:5]  # [5, d_model2]
                    S1 = d1[S1_key][:5] if S1_key in d1 else None
                    S2 = d2[S2_key][:5] if S2_key in d2 else None
                    
                    # 子组方向对齐(系数空间)
                    dir_cosines = []
                    for sub_name in RELATION_WORDS[rel_group]:
                        key1 = f"L{li1}_{rel_group}_{sub_name}_dir_direction"
                        key2 = f"L{li2}_{rel_group}_{sub_name}_dir_direction"
                        
                        if key1 not in d1 or key2 not in d2:
                            continue
                        
                        dir1 = d1[key1]  # [d1]
                        dir2 = d2[key2]  # [d2]
                        
                        # 在各自PC子空间中的系数
                        coeff1 = Vt1 @ dir1  # [5]
                        coeff2 = Vt2 @ dir2  # [5]
                        
                        c1n = np.linalg.norm(coeff1)
                        c2n = np.linalg.norm(coeff2)
                        
                        if c1n > 1e-10 and c2n > 1e-10:
                            cos_val = float(np.abs(np.dot(coeff1/c1n, coeff2/c2n)))
                            dir_cosines.append(round(cos_val, 4))
                    
                    # 奇异值分布对比
                    sv_corr = 0
                    if S1 is not None and S2 is not None:
                        s1_top = S1[:5].astype(float)
                        s2_top = S2[:5].astype(float)
                        # 归一化后计算相关
                        s1_n = s1_top / (np.sum(s1_top) + 1e-10)
                        s2_n = s2_top / (np.sum(s2_top) + 1e-10)
                        if np.std(s1_n) > 1e-10 and np.std(s2_n) > 1e-10:
                            sv_corr = float(np.corrcoef(s1_n, s2_n)[0, 1])
                    
                    pair_key = f"{m1}_vs_{m2}_{layer_label}_{rel_group}"
                    results[pair_key] = {
                        "model1": m1,
                        "model2": m2,
                        "d1": model_dmodels[m1],
                        "d2": model_dmodels[m2],
                        "layer_label": layer_label,
                        "relation": rel_group,
                        "sub_dir_cosines": dir_cosines,
                        "mean_sub_dir_cos": round(float(np.mean(dir_cosines)), 4) if dir_cosines else 0,
                        "sv_correlation": round(sv_corr, 4),
                    }
                    
                    log(f"  {layer_label} {rel_group}: mean_cos={np.mean(dir_cosines):.4f}, sv_corr={sv_corr:.4f}")
    
    # 保存对齐结果
    out_path = TEMP / "ccxxv_cross_model_alignment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    log(f"\n对齐结果保存到: {out_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen3", "glm4", "deepseek7b"],
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--phase", choices=["collect", "align", "all"], default="all")
    args = parser.parse_args()
    
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("")
    
    if args.phase in ["collect", "all"]:
        for mn in args.models:
            run_single(mn)
            gc.collect()
            torch.cuda.empty_cache()
    
    if args.phase in ["align", "all"]:
        run_alignment()
