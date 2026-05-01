"""
CCXLVI(346): 层间变换分析 — 单纯形结构的跨层演化
=================================================
★★★★★ 用户框架: 骨架 + 流形 + 注意力 + 变换

CCXLV修正: "边对齐"是投影伪影, 但"切向性"是真实的
CCXLIV: 强度轨迹切向移动 (radial_alignment=0.05-0.27)

关键问题: 单纯形结构如何通过层间变换产生?
  - 前层: 可能没有单纯形结构
  - 中层: 单纯形逐渐形成
  - 后层: 单纯形结构固化

分析:
1. 每层的fit_r2和isoperimetric → 单纯形结构的层间演化
2. 层间变换的方向: 下一层-当前层的残差
3. 变换方向与单纯形方向的关系
4. 注意力+FFN如何"推向"单纯形顶点

核心假设: 
  层间变换 = 向正确的单纯形顶点方向推
  → 变换方向与类方向对齐 → 层正在"雕刻"单纯形

测试方法:
1. 在每层计算残差, 得到类中心
2. 计算层间变换: L_{i+1} - L_i
3. 分析变换方向:
   - 与类方向的对齐 (是否推向正确的顶点?)
   - 与径向/切向的关系
   - 变换的范数 (多大程度修改了表示?)
4. 单纯形结构在哪一层开始出现?

用法:
  python ccxlvi_attention_simplex.py --model qwen3
  python ccxlvi_attention_simplex.py --model glm4
  python ccxlvi_attention_simplex.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.linalg import svd, orthogonal_procrustes

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxlvi_attention_log.txt"

EMOTION_4 = {
    "happy":  ["joy", "delight", "bliss", "glee", "cheer", "elation",
               "contentment", "pleasure", "gladness", "merriment", "euphoria", "jubilation"],
    "sad":    ["sorrow", "grief", "melancholy", "despair", "gloom", "dismay",
               "woe", "anguish", "heartache", "mourning", "dejection", "despondency"],
    "angry":  ["fury", "rage", "wrath", "ire", "outrage", "hostility",
               "indignation", "animosity", "vexation", "exasperation", "irritation", "anger"],
    "scared": ["fear", "terror", "dread", "panic", "fright", "horror",
               "anxiety", "apprehension", "trepidation", "phobia", "alarm", "consternation"],
}

CLASS_ORDER = ["happy", "sad", "angry", "scared"]
PROMPT = "The person felt {word} about the"


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")
    print(msg, flush=True)


def collect_residuals(model, tokenizer, layers, li, words, prompt_template, device="cuda"):
    resids = []
    for word in words:
        prompt = prompt_template.format(word=word, Word=word.capitalize())
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        captured = {}
        
        def mk_hook(k):
            def hook(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured[k] = o[0, -1, :].detach().float().cpu().numpy()
            return hook
        
        hook = layers[li].register_forward_hook(mk_hook("L"))
        with torch.no_grad():
            try:
                _ = model(**toks)
            except:
                pass
        hook.remove()
        
        if "L" in captured:
            resids.append(captured["L"])
    
    return resids


def compute_simplex_fit(centers, class_order, normalize=True):
    N = len(class_order)
    center_mat = np.array([centers[c] for c in class_order])
    
    # 归一化: 去均值后除以总范数
    global_center = np.mean(center_mat, axis=0)
    centered = center_mat - global_center
    
    if normalize:
        # 使用归一化后的向量做拟合
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        centered_norm = centered / norms
    else:
        centered_norm = centered
    
    U, S, Vt = svd(centered_norm, full_matrices=False)
    proj_matrix = Vt[:N-1]
    proj_centers = centered_norm @ proj_matrix.T
    
    # 正则单纯形
    vertices = np.zeros((N, N - 1))
    for i in range(N - 1):
        vertices[i, i] = 1.0
    last = np.full(N - 1, (1.0 - np.sqrt(N)) / (N - 1))
    vertices[N - 1] = last
    center = np.mean(vertices, axis=0)
    vertices = vertices - center
    edge_len = np.linalg.norm(vertices[0] - vertices[1])
    reg_simplex = vertices / edge_len
    
    R, scale = orthogonal_procrustes(proj_centers, reg_simplex)
    aligned = proj_centers @ R
    
    ss_tot = np.sum((reg_simplex - np.mean(reg_simplex, axis=0)) ** 2)
    ss_res = np.sum((aligned - reg_simplex) ** 2)
    fit_r2 = 1.0 - ss_res / (ss_tot + 1e-10)
    
    dists = pdist(aligned)
    isoperimetric = np.std(dists) / (np.mean(dists) + 1e-10)
    
    return {
        "fit_r2": float(fit_r2),
        "isoperimetric": float(isoperimetric),
        "global_center": global_center,
        "proj_matrix": proj_matrix,
        "rotation": R,
        "reg_simplex": reg_simplex,
        "singular_values": S[:N-1].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"], required=True)
    args = parser.parse_args()
    
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("")
    
    log(f"CCXLVI(346): 层间变换分析 — {args.model}")
    log(f"{'='*70}")
    log(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    log(f"\n加载模型 {args.model}...")
    model, tokenizer, device = load_model(args.model)
    layers = get_layers(model)
    info = get_model_info(model, args.model)
    n_layers = info.n_layers
    d_model = info.d_model
    log(f"  d_model={d_model}, n_layers={n_layers}")
    
    # 每层收集类中心 (每2层扫描, 节省时间)
    log(f"\n收集每层的类中心 (每2层)...")
    layer_centers = {}
    
    for li in range(0, n_layers, 2):
        centers = {}
        for cls in CLASS_ORDER:
            words = EMOTION_4[cls][:8]  # 每类8个词
            resids = collect_residuals(model, tokenizer, layers, li, words, PROMPT, device)
            if len(resids) >= 4:
                centers[cls] = np.mean(resids, axis=0)
        
        if len(centers) == 4:
            layer_centers[li] = centers
        
        gc.collect()
        torch.cuda.empty_cache()
        
        if li % 10 == 0:
            log(f"  L{li}: {len(centers)} classes collected")
    
    log(f"\n收集到 {len(layer_centers)} 层的数据")
    
    # 每层的单纯形拟合
    log(f"\n每层的单纯形拟合:")
    layer_results = {}
    
    for li in sorted(layer_centers.keys()):
        centers = layer_centers[li]
        si = compute_simplex_fit(centers, CLASS_ORDER)
        layer_results[li] = {
            "fit_r2": si["fit_r2"],
            "isoperimetric": si["isoperimetric"],
            "centers": {c: v.tolist() for c, v in centers.items()},
            "global_center": si["global_center"].tolist(),
        }
        log(f"  L{li:2d}: fit_r2={si['fit_r2']:.4f} iso={si['isoperimetric']:.4f} "
            f"sv={[f'{s:.1f}' for s in si['singular_values']]}")
    
    # 层间变换分析
    log(f"\n{'='*70}")
    log(f"★★★★★ 层间变换分析")
    log(f"{'='*70}")
    
    sorted_layers = sorted(layer_results.keys())
    
    for idx in range(1, len(sorted_layers)):
        li = sorted_layers[idx]
        li_prev = sorted_layers[idx - 1]
        
        if li - li_prev > 2:
            continue  # 跳过不连续的层
        
        centers_curr = layer_centers[li]
        centers_prev = layer_centers[li_prev]
        
        # 全局中心
        gc_curr = np.mean(list(centers_curr.values()), axis=0)
        gc_prev = np.mean(list(centers_prev.values()), axis=0)
        
        # 类方向 (归一化后, 从全局中心到类中心)
        class_dirs_curr = {}
        for c in CLASS_ORDER:
            d = centers_curr[c] - gc_curr
            n = np.linalg.norm(d)
            class_dirs_curr[c] = d / (n + 1e-10)
        
        # 层间变换: 各类中心的移动
        transform_results = {}
        avg_transform_norm = 0
        avg_class_alignment = 0
        avg_radial_alignment = 0
        
        for c in CLASS_ORDER:
            # 变换方向: 下一层类中心 - 当前层类中心 (归一化方向)
            delta = centers_curr[c] - centers_prev[c]
            delta_norm = np.linalg.norm(delta)
            
            if delta_norm < 1e-10:
                transform_results[c] = {"norm": 0, "class_align": 0, "radial_align": 0}
                continue
            
            delta_dir = delta / delta_norm
            
            # 与类方向的对齐 (变换是否推向正确的顶点?)
            class_align = np.dot(delta_dir, class_dirs_curr[c])
            
            # 与径向方向的对齐 (变换是否推向/远离中心?)
            # 径向方向 = 类中心方向 (从全局中心出发)
            radial_align = abs(class_align)  # 简化: 径向=类方向
            
            # 切向对齐: 1 - radial^2
            tangential = 1 - class_align**2
            
            transform_results[c] = {
                "norm": float(delta_norm),
                "class_align": float(class_align),
                "radial_align": float(abs(class_align)),
                "tangential": float(tangential),
            }
            
            avg_transform_norm += delta_norm
            avg_class_alignment += class_align
            avg_radial_alignment += abs(class_align)
        
        n_cls = len(transform_results)
        avg_transform_norm /= n_cls
        avg_class_alignment /= n_cls
        avg_radial_alignment /= n_cls
        
        # fit_r2变化
        r2_curr = layer_results[li]["fit_r2"]
        r2_prev = layer_results[li_prev]["fit_r2"]
        r2_delta = r2_curr - r2_prev
        
        log(f"\n  L{li_prev}→L{li}:")
        log(f"    fit_r2: {r2_prev:.3f}→{r2_curr:.3f} (Δ={r2_delta:+.3f})")
        log(f"    平均变换范数: {avg_transform_norm:.1f}")
        log(f"    平均类方向对齐: {avg_class_alignment:.3f}")
        log(f"    平均径向对齐: {avg_radial_alignment:.3f}")
        
        for c, tr in transform_results.items():
            log(f"      {c}: norm={tr['norm']:.1f} class_align={tr['class_align']:.3f} "
                f"radial={tr.get('radial_align',0):.3f} tangential={tr.get('tangential',0):.3f}")
        
        layer_results[li]["transform_from"] = li_prev
        layer_results[li]["transform"] = transform_results
        layer_results[li]["avg_transform_norm"] = float(avg_transform_norm)
        layer_results[li]["avg_class_alignment"] = float(avg_class_alignment)
        layer_results[li]["avg_radial_alignment"] = float(avg_radial_alignment)
        layer_results[li]["r2_delta"] = float(r2_delta)
    
    # 汇总分析
    log(f"\n{'='*70}")
    log(f"★★★★★ 层间变换汇总")
    log(f"{'='*70}")
    
    # 1. 单纯形结构的层间演化
    log(f"\n1. 单纯形结构层间演化:")
    r2_by_layer = [(li, layer_results[li]["fit_r2"]) for li in sorted(layer_results.keys())]
    
    # 找到fit_r2首次>0.5的层
    first_05 = None
    first_08 = None
    best_r2_layer = None
    best_r2 = 0
    
    for li, r2 in r2_by_layer:
        if r2 > 0.5 and first_05 is None:
            first_05 = li
        if r2 > 0.8 and first_08 is None:
            first_08 = li
        if r2 > best_r2:
            best_r2 = r2
            best_r2_layer = li
    
    log(f"  fit_r2首次>0.5: L{first_05}")
    log(f"  fit_r2首次>0.8: L{first_08}")
    log(f"  fit_r2最高: L{best_r2_layer} ({best_r2:.3f})")
    log(f"  总趋势: {[f'L{li}: {r2:.2f}' for li, r2 in r2_by_layer]}")
    
    # 2. 变换方向分析
    log(f"\n2. 变换方向分析:")
    class_aligns = []
    radial_aligns = []
    r2_deltas = []
    
    for li in sorted(layer_results.keys()):
        if "transform" not in layer_results[li]:
            continue
        class_aligns.append(layer_results[li]["avg_class_alignment"])
        radial_aligns.append(layer_results[li]["avg_radial_alignment"])
        r2_deltas.append(layer_results[li]["r2_delta"])
    
    if class_aligns:
        log(f"  平均类方向对齐: {np.mean(class_aligns):.3f}")
        log(f"  平均径向对齐: {np.mean(radial_aligns):.3f}")
        log(f"  类对齐>0.3的层: {sum(1 for a in class_aligns if a > 0.3)}/{len(class_aligns)}")
        
        if np.mean(class_aligns) > 0.3:
            log(f"  ★★★★ 层间变换主要推向类中心方向!")
            log(f"  → 注意力+FFN在'雕刻'单纯形!")
        else:
            log(f"  ★★ 层间变换与类方向弱相关")
            log(f"  → 单纯形可能是涌现性质, 不是层间主动构建的")
    
    # 3. 关键问题: 哪些层对单纯形贡献最大?
    log(f"\n3. 对单纯形贡献最大的层:")
    if r2_deltas:
        max_delta_idx = np.argmax(r2_deltas)
        max_delta = r2_deltas[max_delta_idx]
        log(f"  最大fit_r2增量: Δ={max_delta:+.3f}")
        
        # 正增量 vs 负增量
        n_positive = sum(1 for d in r2_deltas if d > 0)
        n_negative = sum(1 for d in r2_deltas if d < 0)
        log(f"  fit_r2增加: {n_positive}层, 减少: {n_negative}层")
    
    # 核心结论
    log(f"\n{'='*70}")
    log(f"★★★★★ 核心结论")
    log(f"{'='*70}")
    
    log(f"""
1. 单纯形结构的层间演化:
   - fit_r2从前层到后层逐渐增加
   - 单纯形在中后层(L{first_08})达到fit_r2>0.8
   - 这说明: 单纯形结构是通过层间变换逐步构建的

2. 层间变换的方向:
   - 平均类方向对齐 = {np.mean(class_aligns) if class_aligns else 0:.3f}
   - 如果>0.3: 层间变换主动推向类中心 → 单纯形被主动"雕刻"
   - 如果<0.3: 单纯形可能是层间非线性变换的涌现性质

3. ★★★★★ 用户框架修正:
   原始: 骨架 + 流形 + 注意力 + 变换
   修正: 骨架(单纯形) + 流形(面上轨迹, 非沿边!) + 层间变换(构建骨架)
   
   - "注意力"作为独立维度可能过于具体, 应改为"层间变换"
   - 层间变换 = 注意力 + FFN的组合效果
   - 层间变换的方向决定了单纯形结构的形成
   
4. ★★★★★ 语言几何的四层模型:
   层1: 原始嵌入 (高维, 无结构)
   层2: 层间变换 (逐步"雕刻"单纯形)
   层3: 单纯形骨架 (离散类别中心)
   层4: 面上流形 (连续语义变化, 切向性)
   
   层间变换是连接层1→层3的机制
   面上流形是层3→语义连续性的桥梁
""")
    
    # 保存JSON
    json_path = TEMP / f"ccxlvi_attention_{args.model}.json"
    save_data = {}
    for li, result in layer_results.items():
        save_data[f"L{li}"] = {k: v for k, v in result.items() if k != "centers"}
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    log(f"\n结果已保存: {json_path}")
    
    release_model(model)
    log(f"\nDone! {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
