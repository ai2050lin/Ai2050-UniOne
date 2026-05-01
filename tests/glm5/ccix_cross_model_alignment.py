"""
CCIX(309): 跨模型语义轴对齐分析
=================================
基于CCVIII(308)收集的三模型语义方向, 分析跨模型一致性。

核心问题:
1. 不同模型的语义力线方向是否一致?
2. 语义轴是否是普适数学结构?
3. Attention和MLP的语义轴对齐度有何差异?

方法:
- 余弦相似度: 比较不同模型在同一类别对上的语义方向
- CCA (Canonical Correlation Analysis): 比较不同模型的语义子空间
- Procrustes对齐: 寻找最优线性变换使语义方向对齐

用法:
  python ccix_cross_model_alignment.py
"""
import sys, os, json, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine
from scipy.linalg import orthogonal_procrustes

TEMP_DIR = Path("tests/glm5_temp")
LOG_FILE = TEMP_DIR / "ccix_cross_model_alignment_log.txt"

MODELS = ["qwen3", "glm4", "deepseek7b"]


def log_f(msg="", end="\n"):
    print(msg, end=end, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + end)


def load_semantic_directions(model_name):
    """加载CCVIII收集的语义方向"""
    path = TEMP_DIR / f"ccviii_attn_mlp_decompose_{model_name}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def run_cross_model_alignment():
    start_time = time.time()

    log_f(f"\n{'#'*70}")
    log_f(f"CCIX(309): Cross-Model Semantic Axis Alignment")
    log_f(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_f(f"{'#'*70}")

    # 加载三模型数据
    model_data = {}
    for model_name in MODELS:
        model_data[model_name] = load_semantic_directions(model_name)
        log_f(f"  Loaded {model_name}: {model_data[model_name]['d_model']}d, {model_data[model_name]['n_layers']}L")

    # ===== Analysis 1: 跨模型语义方向余弦相似度 =====
    log_f(f"\n{'='*70}")
    log_f(f"Analysis 1: Cross-Model Semantic Direction Cosine Similarity")
    log_f(f"{'='*70}")

    # 不同模型的d_model不同, 不能直接比较方向
    # 但可以通过投影到各自W_U行空间, 比较logit空间的对齐度
    # 或者用更直接的方法: 比较语义方向在各模型W_U SVD模式上的投影分布

    # 我们用另一种方法: CCA-like比较
    # 对每个模型, 计算语义方向的内部几何(类别对之间的角度)
    # 如果不同模型的内部几何一致, 说明语义空间是普适的

    cat_pairs = [("animals", "food"), ("animals", "tools"), ("animals", "nature"),
                 ("food", "tools"), ("food", "nature"), ("tools", "nature")]

    spaces = ["attn", "mlp", "resid"]

    # Analysis 1a: 语义方向的内部几何 — 类别对之间的角度
    log_f(f"\n--- 1a: Internal Geometry of Semantic Directions ---")

    for space in spaces:
        log_f(f"\n  Space: {space}")

        for model_name in MODELS:
            data = model_data[model_name]
            sem_dirs = data.get("semantic_directions", {})
            n_layers = data["n_layers"]
            sample_layers = data["sample_layers"]

            # 取最后3层的平均
            late_layers = sample_layers[-3:]
            late_dirs = {}

            for li in late_layers:
                li_str = str(li)
                if li_str in sem_dirs and space in sem_dirs[li_str]:
                    for pair_name, direction in sem_dirs[li_str][space].items():
                        if pair_name not in late_dirs:
                            late_dirs[pair_name] = []
                        late_dirs[pair_name].append(np.array(direction))

            # 平均
            avg_dirs = {}
            for pair_name, dirs in late_dirs.items():
                avg_dirs[pair_name] = np.mean(dirs, axis=0)
                norm = np.linalg.norm(avg_dirs[pair_name])
                if norm > 1e-10:
                    avg_dirs[pair_name] = avg_dirs[pair_name] / norm

            # 计算类别对之间的余弦相似度
            pair_names = sorted(avg_dirs.keys())
            log_f(f"    {model_name} (late layers):")
            for i in range(len(pair_names)):
                for j in range(i+1, len(pair_names)):
                    cos_sim = 1 - cosine(avg_dirs[pair_names[i]], avg_dirs[pair_names[j]])
                    log_f(f"      cos({pair_names[i]}, {pair_names[j]}) = {cos_sim:.4f}")

    # ===== Analysis 2: Attn vs MLP语义贡献的时间动态 =====
    log_f(f"\n{'='*70}")
    log_f(f"Analysis 2: Attn vs MLP Semantic Contribution Over Layers")
    log_f(f"{'='*70}")

    for model_name in MODELS:
        data = model_data[model_name]
        log_f(f"\n  {model_name}:")

        decomp = data.get("decomposition", {})
        for li_str in sorted(decomp.keys(), key=int):
            if "_summary" not in decomp[li_str]:
                continue
            s = decomp[li_str]["_summary"]

            # 语义贡献比例
            total_norm = s["attn_avg_norm"] + s["mlp_avg_norm"]
            if total_norm > 0:
                attn_frac = s["attn_avg_norm"] / total_norm
                mlp_frac = s["mlp_avg_norm"] / total_norm
            else:
                attn_frac = mlp_frac = 0

            log_f(f"    L{li_str}: Attn_norm={s['attn_avg_norm']:.3f}({attn_frac:.1%}), "
                  f"MLP_norm={s['mlp_avg_norm']:.3f}({mlp_frac:.1%}), "
                  f"Attn_sig={s['attn_avg_sig']:.3f}, MLP_sig={s['mlp_avg_sig']:.3f}, "
                  f"Attn_acc={s['attn_avg_acc']:.3f}, MLP_acc={s['mlp_avg_acc']:.3f}")

    # ===== Analysis 3: W_U对齐度对比 — Attn vs MLP =====
    log_f(f"\n{'='*70}")
    log_f(f"Analysis 3: W_U Alignment — Attn vs MLP Semantic Directions")
    log_f(f"{'='*70}")

    for model_name in MODELS:
        data = model_data[model_name]
        log_f(f"\n  {model_name}:")

        wu_align = data.get("wu_alignment", {})
        for li_str in sorted(wu_align.keys(), key=int):
            li_data = wu_align[li_str]
            row = []
            for space in ["attn", "mlp", "resid"]:
                if space in li_data and "_summary" in li_data[space]:
                    s = li_data[space]["_summary"]
                    row.append(f"{space}: gain_ratio={s['sem_rnd_gain_ratio']:.2f}, "
                              f"top10={s['sem_avg_top10']:.3f}(RND={s['rnd_avg_top10']:.3f})")
            log_f(f"    L{li_str}: {' | '.join(row)}")

    # ===== Analysis 4: SVD模式语义区分力 =====
    log_f(f"\n{'='*70}")
    log_f(f"Analysis 4: SVD Mode Semantic Discrimination")
    log_f(f"{'='*70}")

    for model_name in MODELS:
        data = model_data[model_name]
        log_f(f"\n  {model_name}:")

        svd_perturb = data.get("svd_perturb", {})
        for li_str in sorted(svd_perturb.keys(), key=int):
            li_data = svd_perturb[li_str]
            log_f(f"    L{li_str}:")
            for space in ["attn", "mlp"]:
                if space in li_data and "error" not in li_data[space]:
                    top_disc = li_data[space].get("top_discriminative_modes", [])
                    if top_disc:
                        modes_str = ", ".join([f"M{m[0]}(F={m[1]:.1f})" for m in top_disc[:3]])
                        log_f(f"      {space}: {modes_str}")

    # ===== Analysis 5: 语义几何结构 — 单纯形/正交性 =====
    log_f(f"\n{'='*70}")
    log_f(f"Analysis 5: Semantic Geometry — Simplex vs Orthogonal?")
    log_f(f"{'='*70}")

    # 如果4个类别的语义方向形成正四面体(3-simplex), 
    # 则6个类别对方向之间的平均余弦相似度=-1/3≈-0.333
    # 如果正交, 平均余弦=0

    for model_name in MODELS:
        data = model_data[model_name]
        sem_dirs = data.get("semantic_directions", {})
        sample_layers = data["sample_layers"]

        for space in ["resid"]:
            # 收集所有层的语义方向
            all_cos_sims = []

            for li in sample_layers:
                li_str = str(li)
                if li_str not in sem_dirs or space not in sem_dirs[li_str]:
                    continue

                dirs = sem_dirs[li_str][space]
                pair_names = sorted(dirs.keys())

                for i in range(len(pair_names)):
                    for j in range(i+1, len(pair_names)):
                        d1 = np.array(dirs[pair_names[i]])
                        d2 = np.array(dirs[pair_names[j]])
                        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                        if n1 > 1e-10 and n2 > 1e-10:
                            cos_sim = np.dot(d1, d2) / (n1 * n2)
                            all_cos_sims.append(cos_sim)

            if all_cos_sims:
                avg_cos = np.mean(all_cos_sims)
                std_cos = np.std(all_cos_sims)
                log_f(f"  {model_name} {space}: avg_cos={avg_cos:.4f}±{std_cos:.4f} "
                      f"(simplex=-0.333, orthogonal=0.0, range=[{min(all_cos_sims):.4f}, {max(all_cos_sims):.4f}])")

    # ===== Summary =====
    log_f(f"\n{'='*70}")
    log_f(f"SUMMARY: Cross-Model Semantic Axis Alignment")
    log_f(f"{'='*70}")

    # 总结关键发现
    log_f("\n  Key Findings:")

    # 1. Attention vs MLP语义贡献
    log_f("\n  1. Attn vs MLP Semantic Contribution:")
    for model_name in MODELS:
        data = model_data[model_name]
        decomp = data.get("decomposition", {})
        # 最后层的Attn/MLP ratio
        last_li = str(data["sample_layers"][-1])
        if last_li in decomp and "_summary" in decomp[last_li]:
            s = decomp[last_li]["_summary"]
            total = s["attn_avg_norm"] + s["mlp_avg_norm"]
            attn_pct = s["attn_avg_norm"] / total * 100 if total > 0 else 0
            mlp_pct = s["mlp_avg_norm"] / total * 100 if total > 0 else 0
            log_f(f"    {model_name} L{last_li}: Attn={attn_pct:.0f}%, MLP={mlp_pct:.0f}% (norm), "
                  f"Attn_sig={s['attn_avg_sig']:.3f}, MLP_sig={s['mlp_avg_sig']:.3f}")

    # 2. W_U对齐
    log_f("\n  2. W_U Alignment (Attn vs MLP semantic directions):")
    for model_name in MODELS:
        data = model_data[model_name]
        wu_align = data.get("wu_alignment", {})
        last_li = str(data["sample_layers"][-1])
        if last_li in wu_align:
            li_data = wu_align[last_li]
            for space in ["attn", "mlp", "resid"]:
                if space in li_data and "_summary" in li_data[space]:
                    s = li_data[space]["_summary"]
                    log_f(f"    {model_name} {space}: SEM/RND gain={s['sem_rnd_gain_ratio']:.2f}, "
                          f"top10 SEM={s['sem_avg_top10']:.3f} RND={s['rnd_avg_top10']:.3f}")

    # 3. 语义几何
    log_f("\n  3. Semantic Geometry:")
    log_f("    Simplex (正四面体) → avg_cos = -0.333")
    log_f("    Orthogonal → avg_cos = 0.0")

    elapsed = time.time() - start_time
    log_f(f"\n  Total time: {elapsed:.1f}s")

    # Save summary
    summary_path = TEMP_DIR / "ccix_cross_model_alignment_summary.json"
    summary = {
        "model_data_available": {m: True for m in MODELS},
        "elapsed": elapsed,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log_f(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    run_cross_model_alignment()
