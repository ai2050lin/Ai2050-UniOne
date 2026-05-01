"""
CCXLIV(344): 切向对齐验证 — 强度方向与单纯形面的关系
=====================================================
★★★★★ CCXLIII核心发现: 强度方向是切向的!
  - radial_alignment = 0.07-0.23
  - 强度变化垂直于类中心方向

关键问题: 切向方向具体是什么?
  假设A: 沿单纯形的边 (从一个顶点向另一个顶点移动)
  假设B: 沿单纯形的面 (在面内移动)
  假设C: 随机切向 (与单纯形无关)

如果假设A成立:
  "glad→happy→ecstatic" 可能是向另一个emotion类别移动
  → 情感强度增加 = 情感纯度增加 = 更接近该情感的原型

如果假设B成立:
  强度变化在单纯形面内, 但不沿特定边
  → 有独立的强度维度嵌入在单纯形面内

测试方法:
1. 对每个强度梯度(mild→strong), 计算轨迹方向
2. 计算轨迹方向与所有单纯形边的对齐度
3. 如果某条边的对齐度>>其他边 → 假设A
4. 如果所有边都有中等对齐度 → 假设B
5. 如果所有边对齐度都低 → 假设C

附加验证:
- 不同强度水平的点投影到单纯形面后的位置
- 强度增加 = 远离中心 = 更"纯"的类别成员?

用法:
  python ccxliv_tangential_alignment.py --model qwen3
  python ccxliv_tangential_alignment.py --model glm4
  python ccxliv_tangential_alignment.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd, orthogonal_procrustes

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxliv_tangential_log.txt"

# 4类emotion + 强度梯度
DOMAINS = {
    "emotion_4": {
        "classes": {
            "happy": ["joy", "delight", "bliss", "glee", "cheer", "elation",
                      "contentment", "pleasure", "gladness", "merriment", "euphoria", "jubilation"],
            "sad": ["sorrow", "grief", "melancholy", "despair", "gloom", "dismay",
                    "woe", "anguish", "heartache", "mourning", "dejection", "despondency"],
            "angry": ["fury", "rage", "wrath", "ire", "outrage", "hostility",
                      "indignation", "animosity", "vexation", "exasperation", "irritation", "anger"],
            "scared": ["fear", "terror", "dread", "panic", "fright", "horror",
                       "anxiety", "apprehension", "trepidation", "phobia", "alarm", "dismay"],
        },
        "order": ["happy", "sad", "angry", "scared"],
        "prompt": "The person felt {word} about the",
    },
}

INTENSITY_DATA = {
    "happy": {
        "mild":   ["content", "pleased", "satisfied", "comfortable"],
        "medium": ["happy", "glad", "cheerful", "joyful"],
        "strong": ["ecstatic", "elated", "euphoric", "jubilant"],
    },
    "sad": {
        "mild":   ["down", "unhappy", "disappointed", "blue"],
        "medium": ["sad", "sorrowful", "melancholy", "gloomy"],
        "strong": ["despairing", "devastated", "anguished", "heartbroken"],
    },
    "angry": {
        "mild":   ["annoyed", "irritated", "bothered", "irked"],
        "medium": ["angry", "mad", "furious", "enraged"],
        "strong": ["livid", "infuriated", "incensed", "rage-filled"],
    },
    "scared": {
        "mild":   ["uneasy", "wary", "cautious", "concerned"],
        "medium": ["scared", "afraid", "frightened", "alarmed"],
        "strong": ["terrified", "horrified", "panicked", "petrified"],
    },
}

PROMPT = "The person felt {word} about the"


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")
    print(msg, flush=True)


def collect_residuals(model, tokenizer, layers, li, words, prompt_template, device="cuda"):
    """收集一组词在某层的残差"""
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


def compute_regular_simplex(N):
    """生成N维正则单纯形的顶点 (在N-1维空间中)"""
    # 使用标准构造: 在N-1维空间中, N个等距等角的点
    vertices = np.zeros((N, N - 1))
    for i in range(N - 1):
        vertices[i, i] = 1.0
    # 最后一个顶点使得所有边等长
    last = np.full(N - 1, (1.0 - np.sqrt(N)) / (N - 1))
    vertices[N - 1] = last
    
    # 中心化
    center = np.mean(vertices, axis=0)
    vertices = vertices - center
    
    # 归一化边长
    edge_len = np.linalg.norm(vertices[0] - vertices[1])
    vertices = vertices / edge_len
    
    return vertices


def analyze_tangential_alignment(class_centers_dict, intensity_resids, class_order):
    """
    核心分析: 强度轨迹方向与单纯形边/面的对齐
    
    class_centers_dict: {class_name: center_vector}
    intensity_resids: {class_name: {level: [vectors]}}
    class_order: [class_names]
    """
    N = len(class_order)
    
    # 1. 计算类中心矩阵
    centers = np.array([class_centers_dict[c] for c in class_order])
    global_center = np.mean(centers, axis=0)
    
    # 2. PCA投影到N-1维子空间
    centered = centers - global_center
    U, S, Vt = svd(centered, full_matrices=False)
    # 投影矩阵: 原始空间 → N-1维子空间
    proj_matrix = Vt[:N-1]  # (N-1, d_model)
    
    # 投影类中心
    proj_centers = (centers - global_center) @ proj_matrix.T  # (N, N-1)
    
    # 3. 构造正则单纯形并对齐
    reg_simplex = compute_regular_simplex(N)
    
    # Procrustes对齐: proj_centers → reg_simplex
    R, scale = orthogonal_procrustes(proj_centers, reg_simplex)
    aligned_centers = proj_centers @ R
    
    # 4. 计算单纯形的边方向 (在N-1维空间中)
    simplex_edges = {}
    for i in range(N):
        for j in range(i + 1, N):
            edge_dir = reg_simplex[j] - reg_simplex[i]
            edge_dir = edge_dir / (np.linalg.norm(edge_dir) + 1e-10)
            simplex_edges[(i, j)] = edge_dir
    
    results = {}
    
    for cls_name, cls_intensities in intensity_resids.items():
        if cls_name not in class_centers_dict:
            continue
        
        cls_idx = class_order.index(cls_name)
        
        levels = sorted(cls_intensities.keys())
        if len(levels) < 2:
            continue
        
        # 计算每个强度水平的中心
        int_centers = {}
        for level in levels:
            vecs = cls_intensities[level]
            if len(vecs) > 0:
                int_centers[level] = np.mean(vecs, axis=0)
        
        if len(int_centers) < 2:
            continue
        
        # 5. 强度轨迹方向 (弱→强)
        trajectory = int_centers[levels[-1]] - int_centers[levels[0]]
        traj_norm = np.linalg.norm(trajectory)
        
        if traj_norm < 1e-10:
            results[cls_name] = {"error": "trajectory too small"}
            continue
        
        trajectory_dir = trajectory / traj_norm
        
        # 6. 径向方向 (类中心方向)
        radial_dir = class_centers_dict[cls_name] - global_center
        radial_norm = np.linalg.norm(radial_dir)
        if radial_norm > 1e-10:
            radial_dir = radial_dir / radial_norm
        
        radial_alignment = abs(np.dot(trajectory_dir, radial_dir))
        
        # 7. 投影轨迹到N-1维子空间
        proj_trajectory = trajectory @ proj_matrix.T @ R  # 在对齐后的子空间中
        proj_traj_dir = proj_trajectory / (np.linalg.norm(proj_trajectory) + 1e-10)
        
        # 8. 计算与每条单纯形边的对齐度
        edge_alignments = {}
        for (i, j), edge_dir in simplex_edges.items():
            alignment = abs(np.dot(proj_traj_dir, edge_dir))
            edge_alignments[(i, j)] = float(alignment)
        
        # 找最对齐的边
        best_edge = max(edge_alignments, key=edge_alignments.get)
        best_alignment = edge_alignments[best_edge]
        
        # 9. 计算与单纯形面的对齐度
        # 面由N-2个边张成, 轨迹在面内 = 轨迹可以被面的边表示
        # 简化: 计算轨迹与包含该顶点的所有面的对齐度
        
        # 包含顶点cls_idx的面: 所有不包含顶点cls_idx的顶点集的补集
        # 更直接: 轨迹与从顶点cls_idx出发的所有边张成的子空间的对齐度
        edges_from_cls = []
        for j in range(N):
            if j != cls_idx:
                edge = reg_simplex[j] - reg_simplex[cls_idx]
                edge = edge / (np.linalg.norm(edge) + 1e-10)
                edges_from_cls.append(edge)
        
        if edges_from_cls:
            # 子空间投影
            edge_matrix = np.array(edges_from_cls).T  # (N-1, N-1)
            proj_onto_edges = edge_matrix @ edge_matrix.T  # 投影矩阵
            proj_traj = proj_onto_edges @ proj_traj_dir
            face_alignment = np.linalg.norm(proj_traj) / (np.linalg.norm(proj_traj_dir) + 1e-10)
        else:
            face_alignment = 0.0
        
        # 10. 强度水平在单纯形面上的位置
        # 投影每个强度点到子空间
        proj_positions = {}
        for level in levels:
            vecs = cls_intensities[level]
            if len(vecs) > 0:
                mean_vec = np.mean(vecs, axis=0)
                proj_vec = (mean_vec - global_center) @ proj_matrix.T @ R
                proj_positions[level] = proj_vec.tolist()
        
        # 到各类中心的距离 (在对齐子空间中)
        dists_to_centers = {}
        for level in levels:
            vecs = cls_intensities[level]
            if len(vecs) > 0:
                mean_vec = np.mean(vecs, axis=0)
                proj_mean = (mean_vec - global_center) @ proj_matrix.T @ R
                d = [np.linalg.norm(proj_mean - aligned_centers[k]) for k in range(N)]
                dists_to_centers[level] = {
                    class_order[k]: float(d[k]) for k in range(N)
                }
        
        results[cls_name] = {
            "trajectory_norm": float(traj_norm),
            "radial_alignment": float(radial_alignment),
            "best_edge": f"{class_order[best_edge[0]]}-{class_order[best_edge[1]]}",
            "best_edge_alignment": float(best_alignment),
            "face_alignment": float(face_alignment),
            "edge_alignments": {f"{class_order[i]}-{class_order[j]}": v for (i,j), v in edge_alignments.items()},
            "proj_positions": proj_positions,
            "dists_to_centers": dists_to_centers,
        }
        
        # 判定
        if best_alignment > 0.7:
            verdict = f"EDGE-ALIGNED (沿{class_order[best_edge[0]]}-{class_order[best_edge[1]]}边)"
        elif face_alignment > 0.7:
            verdict = "FACE-ALIGNED (沿单纯形面)"
        elif radial_alignment > 0.5:
            verdict = "RADIAL (沿径向)"
        else:
            verdict = "OTHER (其他方向)"
        
        results[cls_name]["verdict"] = verdict
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("")
    
    log(f"\n{'='*80}")
    log(f"CCXLIV: 切向对齐验证 — 强度方向与单纯形面 — {model_name}")
    log(f"{'='*80}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    d_model = info.d_model
    
    log(f"  d_model={d_model}, n_layers={n_layers}")
    
    # 选择中间层
    best_li = n_layers // 2
    log(f"  使用层: L{best_li}")
    
    all_results = {
        "model": model_name,
        "d_model": d_model,
        "n_layers": n_layers,
        "best_layer": best_li,
        "results": {},
    }
    
    # 1. 收集4类emotion的中心
    domain = DOMAINS["emotion_4"]
    class_order = domain["order"]
    
    log(f"\n收集类中心...")
    class_centers = {}
    class_all_resids = {}
    for cls in class_order:
        words = domain["classes"][cls][:12]
        resids = collect_residuals(model, tokenizer, layers, best_li, words, domain["prompt"], device)
        if len(resids) >= 5:
            class_centers[cls] = np.mean(resids, axis=0)
            class_all_resids[cls] = resids
    
    log(f"  有效类: {list(class_centers.keys())}")
    
    # 2. 收集强度数据
    log(f"\n收集强度数据...")
    intensity_resids = {}
    for cls in class_order:
        if cls not in INTENSITY_DATA:
            continue
        cls_intensities = {}
        for level, words in INTENSITY_DATA[cls].items():
            resids = collect_residuals(model, tokenizer, layers, best_li, words, PROMPT, device)
            if len(resids) >= 3:
                cls_intensities[level] = resids
        if cls_intensities:
            intensity_resids[cls] = cls_intensities
            log(f"  {cls}: levels={list(cls_intensities.keys())}")
    
    # 3. 核心分析
    log(f"\n{'='*60}")
    log(f"核心分析: 强度轨迹与单纯形几何的对齐")
    log(f"{'='*60}")
    
    alignment_results = analyze_tangential_alignment(class_centers, intensity_resids, class_order)
    
    for cls, ar in alignment_results.items():
        if "error" in ar:
            log(f"\n  {cls}: {ar['error']}")
            continue
        
        log(f"\n  {cls}:")
        log(f"    径向对齐: {ar['radial_alignment']:.4f}")
        log(f"    最佳边: {ar['best_edge']} (align={ar['best_edge_alignment']:.4f})")
        log(f"    面对齐: {ar['face_alignment']:.4f}")
        log(f"    判定: {ar['verdict']}")
        
        # 边对齐详情
        log(f"    各边对齐度:")
        for edge_name, align in sorted(ar["edge_alignments"].items(), key=lambda x: -x[1]):
            bar = "█" * int(align * 20)
            log(f"      {edge_name:>15}: {align:.4f} {bar}")
        
        # 到各类中心距离
        if ar.get("dists_to_centers"):
            log(f"    各强度水平到类中心距离:")
            for level, dists in ar["dists_to_centers"].items():
                closest = min(dists, key=dists.get)
                log(f"      {level:>8}: closest={closest}({dists[closest]:.2f}), "
                    f"own={dists.get(cls, 0):.2f}")
    
    all_results["results"] = alignment_results
    
    # 4. 多层验证
    log(f"\n{'='*60}")
    log(f"多层验证")
    log(f"{'='*60}")
    
    sample_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    sample_layers = sorted(set([max(0, l) for l in sample_layers]))
    
    layer_results = {}
    for li in sample_layers:
        log(f"\n  Layer {li}:")
        
        # 收集类中心
        cls_c = {}
        for cls in class_order:
            words = domain["classes"][cls][:8]
            resids = collect_residuals(model, tokenizer, layers, li, words, domain["prompt"], device)
            if len(resids) >= 3:
                cls_c[cls] = np.mean(resids, axis=0)
        
        # 收集强度数据
        int_r = {}
        for cls in class_order:
            if cls not in INTENSITY_DATA:
                continue
            cls_int = {}
            for level, words in INTENSITY_DATA[cls].items():
                resids = collect_residuals(model, tokenizer, layers, li, words, PROMPT, device)
                if len(resids) >= 2:
                    cls_int[level] = resids
            if cls_int:
                int_r[cls] = cls_int
        
        if len(cls_c) >= 3 and int_r:
            ar = analyze_tangential_alignment(cls_c, int_r, class_order)
            for cls, data in ar.items():
                if "error" not in data:
                    log(f"    {cls}: radial={data['radial_alignment']:.4f}, "
                        f"best_edge={data['best_edge']}({data['best_edge_alignment']:.4f}), "
                        f"face={data['face_alignment']:.4f}, {data['verdict']}")
            layer_results[f"L{li}"] = ar
    
    all_results["layer_results"] = layer_results
    
    release_model(model)
    
    # 保存
    out_path = TEMP / f"ccxliv_tangential_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    log(f"\n结果已保存: {out_path}")
    
    # 汇总
    log(f"\n{'='*80}")
    log(f"CCXLIV 汇总 — {model_name}")
    log(f"{'='*80}")
    
    # 主要结果
    for cls, ar in alignment_results.items():
        if "error" in ar:
            continue
        log(f"\n  {cls}:")
        log(f"    径向={ar['radial_alignment']:.4f}, 最佳边={ar['best_edge']}({ar['best_edge_alignment']:.4f}), "
            f"面={ar['face_alignment']:.4f}")
        log(f"    → {ar['verdict']}")
    
    # 统计
    radial_aligns = [ar["radial_alignment"] for ar in alignment_results.values() if "error" not in ar]
    edge_aligns = [ar["best_edge_alignment"] for ar in alignment_results.values() if "error" not in ar]
    face_aligns = [ar["face_alignment"] for ar in alignment_results.values() if "error" not in ar]
    
    if radial_aligns:
        log(f"\n  平均径向对齐: {np.mean(radial_aligns):.4f}")
        log(f"  平均最佳边对齐: {np.mean(edge_aligns):.4f}")
        log(f"  平均面对齐: {np.mean(face_aligns):.4f}")
        
        if np.mean(face_aligns) > 0.7:
            log(f"\n  ★★★★★ 结论: 强度方向沿单纯形面! 支持\"单纯形+面上轨迹\"模型")
        elif np.mean(edge_aligns) > 0.7:
            log(f"\n  ★★★★ 结论: 强度方向沿单纯形边! 支持\"强度=向另一类别移动\"模型")
        elif np.mean(radial_aligns) > 0.5:
            log(f"\n  ★★★ 结论: 强度方向以径向为主! 支持\"单纯形+径向噪声\"模型")
        else:
            log(f"\n  ★★ 结论: 强度方向与单纯形几何的对齐不明确")
            log(f"     可能需要更精细的分析 (如非线性结构)")
    
    log(f"\nDone! {model_name}")


if __name__ == "__main__":
    main()
