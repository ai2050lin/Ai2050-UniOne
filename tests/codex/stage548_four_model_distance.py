"""
stage548: 四模型编码距离矩阵 + 跨模型 Pearson r
目标：在四个模型(Qwen3/DS7B/GLM4/Gemma4)上计算18个名词的编码距离矩阵，
      然后计算跨模型Pearson r，升级INV-1从2模型到4模型。

逐模型运行，保存中间结果，最终统一分析。
"""
import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_language_shared import load_qwen3_model, discover_layers
from multimodel_language_shared import (
    encode_to_device, evenly_spaced_layers, free_model,
    load_deepseek_model, load_glm4_model, load_gemma4_model,
)

NOUN_FAMILIES = {
    "fruit": {"members": ["apple", "banana", "cherry"]},
    "animal": {"members": ["cat", "dog", "horse"]},
    "tool": {"members": ["hammer", "knife", "screwdriver"]},
    "org": {"members": ["university", "company", "hospital"]},
    "celestial": {"members": ["sun", "moon", "mars"]},
    "abstract": {"members": ["freedom", "justice", "truth"]},
}
ALL_WORDS = []
for fam in NOUN_FAMILIES.values():
    ALL_WORDS.extend(fam["members"])

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'glm5', 'temp')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_encoding(model, tokenizer, text, layer_idx):
    """获取某个token在某层的编码（最后一个token位置）"""
    encoded = encode_to_device(model, tokenizer, text)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    # 取最后一个token的hidden state
    hs = outputs.hidden_states[layer_idx + 1]  # [batch, seq, dim]
    return hs[0, -1, :]  # [dim]


def compute_dist_matrix_for_model(model_name):
    """加载模型，计算所有名词在所有层的编码距离矩阵"""
    print(f"\n{'='*60}")
    print(f"  处理模型: {model_name}")
    print(f"{'='*60}")

    # 加载模型
    if model_name == "Qwen3":
        model, tokenizer = load_qwen3_model()
    elif model_name == "DeepSeek7B":
        model, tokenizer = load_deepseek_model()
    elif model_name == "GLM4":
        model, tokenizer = load_glm4_model()
    elif model_name == "Gemma4":
        model, tokenizer = load_gemma4_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 发现层
    layers = discover_layers(model)
    n_layers = len(layers)
    sample_layers = evenly_spaced_layers(model, count=10)
    print(f"  总层数: {n_layers}, 采样层: {sample_layers}")
    print(f"  词汇数: {len(ALL_WORDS)}")

    # 计算编码
    print(f"\n  [1] 计算编码...")
    all_encodings = {}  # {word: {layer_idx: tensor}}
    for w in ALL_WORDS:
        all_encodings[w] = {}
        for li in sample_layers:
            enc = get_encoding(model, tokenizer, w, li)
            all_encodings[w][li] = enc
        print(f"    {w}: done", end="\r")
    print(f"    {len(ALL_WORDS)} words x {len(sample_layers)} layers done")

    # 计算每层的距离矩阵
    print(f"\n  [2] 计算距离矩阵...")
    dist_matrices = {}  # {layer_idx: 18x18 numpy array}
    for li in sample_layers:
        n = len(ALL_WORDS)
        dm = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                v1 = all_encodings[ALL_WORDS[i]][li].float().cpu()
                v2 = all_encodings[ALL_WORDS[j]][li].float().cpu()
                cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                dm[i, j] = 1 - cos_sim
                dm[j, i] = dm[i, j]
        dist_matrices[str(li)] = dm.tolist()

    # 保存结果
    result = {
        "model": model_name,
        "n_layers": n_layers,
        "words": ALL_WORDS,
        "sample_layers": sample_layers,
        "dist_matrices": dist_matrices,
    }
    out_path = os.path.join(OUTPUT_DIR, f"stage548_{model_name}_dist.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  保存到: {out_path}")

    # 释放模型
    print(f"\n  [3] 释放模型...")
    free_model(model)
    time.sleep(3)

    return result


def cross_model_analysis(all_results):
    """跨模型 Pearson r 分析，用归一化层位置对齐"""
    print(f"\n{'='*60}")
    print(f"  跨模型 Pearson r 分析（归一化层对齐）")
    print(f"{'='*60}")

    model_names = list(all_results.keys())
    # 各模型的采样层
    model_layers = {}
    for mname in model_names:
        model_layers[mname] = [int(l) for l in all_results[mname]["sample_layers"]]
        n = all_results[mname]["n_layers"]
        print(f"  {mname}: n_layers={n}, 采样={model_layers[mname]}")

    # 用归一化位置对齐: 0%, 25%, 50%, 75%, 100%
    norm_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"\n  归一化对齐位置: {norm_positions}")

    model_names = list(all_results.keys())
    pair_results = {}

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if j <= i:
                continue
            pair_key = f"{m1} vs {m2}"
            n1 = all_results[m1]["n_layers"]
            n2 = all_results[m2]["n_layers"]
            layer_rs = []

            print(f"\n  {pair_key}:")
            for pos in norm_positions:
                # 找最近的采样层
                li1 = min(model_layers[m1], key=lambda x: abs(x / max(n1 - 1, 1) - pos))
                li2 = min(model_layers[m2], key=lambda x: abs(x / max(n2 - 1, 1) - pos))
                actual_pos1 = li1 / max(n1 - 1, 1)
                actual_pos2 = li2 / max(n2 - 1, 1)

                dm1 = np.array(all_results[m1]["dist_matrices"][str(li1)])
                dm2 = np.array(all_results[m2]["dist_matrices"][str(li2)])
                triu_idx = np.triu_indices(len(ALL_WORDS), k=1)
                v1 = dm1[triu_idx]
                v2 = dm2[triu_idx]

                if np.std(v1) > 1e-10 and np.std(v2) > 1e-10:
                    r = np.corrcoef(v1, v2)[0, 1]
                else:
                    r = 0.0
                layer_rs.append((pos, li1, li2, r))
                print(f"    {int(pos*100)}%: {m1}.L{li1} vs {m2}.L{li2} => r = {r:.6f}")

            mean_r = np.mean([r for _, _, _, r in layer_rs])
            min_r = min([r for _, _, _, r in layer_rs])
            max_r = max([r for _, _, _, r in layer_rs])
            print(f"    平均: {mean_r:.6f}, 最小: {min_r:.6f}, 最大: {max_r:.6f}")
            pair_results[pair_key] = {
                "per_pos": {f"{int(p*100)}%": {"layer_m1": l1, "layer_m2": l2, "r": round(r, 6)}
                           for p, l1, l2, r in layer_rs},
                "mean": round(mean_r, 6),
                "min": round(min_r, 6),
                "max": round(max_r, 6),
            }

    # Spearman at last layer
    print(f"\n  Spearman rank correlation (末层):")
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if j <= i:
                continue
            n1 = all_results[m1]["n_layers"]
            n2 = all_results[m2]["n_layers"]
            li1 = model_layers[m1][-1]
            li2 = model_layers[m2][-1]
            dm1 = np.array(all_results[m1]["dist_matrices"][str(li1)])
            dm2 = np.array(all_results[m2]["dist_matrices"][str(li2)])
            triu_idx = np.triu_indices(len(ALL_WORDS), k=1)
            v1 = dm1[triu_idx]
            v2 = dm2[triu_idx]
            try:
                from scipy.stats import spearmanr
                rho, p = spearmanr(v1, v2)
                print(f"  {m1}.L{li1} vs {m2}.L{li2}: rho = {rho:.6f}, p = {p:.2e}")
            except ImportError:
                pass

    # 总体
    print(f"\n{'='*60}")
    print(f"  拼图总结")
    print(f"{'='*60}")
    for pk, pr in pair_results.items():
        print(f"  {pk}: mean_r={pr['mean']:.4f}, range=[{pr['min']:.4f}, {pr['max']:.4f}]")

    all_mean_rs = [pr["mean"] for pr in pair_results.values()]
    overall_mean = np.mean(all_mean_rs)
    overall_min = min(all_mean_rs)
    print(f"\n  总体: mean_of_means={overall_mean:.4f}, min_of_means={overall_min:.4f}")

    if overall_min >= 0.99:
        print(f"  结论: 编码拓扑跨模型高度一致（4/4强不变量）")
    elif overall_min >= 0.90:
        print(f"  结论: 编码拓扑跨模型较强一致（3-4/4中-强不变量）")
    elif overall_min >= 0.70:
        print(f"  结论: 编码拓扑跨模型部分一致（2-3/4中不变量）")
    else:
        print(f"  结论: 编码拓扑跨模型不一致（弱不变量或模型特有）")

    return pair_results


def intra_inter_per_model(all_results):
    """每个模型在每层的intra/inter ratio"""
    print(f"\n{'='*60}")
    print(f"  各模型 intra/inter ratio 逐层分析")
    print(f"{'='*60}")

    family_map = []
    for fk, fv in NOUN_FAMILIES.items():
        family_map.extend([(ALL_WORDS.index(w), fk) for w in fv["members"]])

    for mname, mdata in all_results.items():
        print(f"\n  {mname}:")
        for li in sorted([int(x) for x in mdata["dist_matrices"].keys()]):
            dm = np.array(mdata["dist_matrices"][str(li)])
            intra, inter = [], []
            fam_names = list(NOUN_FAMILIES.keys())
            for fi, fk1 in enumerate(fam_names):
                m1 = NOUN_FAMILIES[fk1]["members"]
                idx1 = [ALL_WORDS.index(w) for w in m1]
                for a in range(len(idx1)):
                    for b in range(a + 1, len(idx1)):
                        intra.append(dm[idx1[a], idx1[b]])
                for fj, fk2 in enumerate(fam_names):
                    if fj <= fi:
                        continue
                    m2 = NOUN_FAMILIES[fk2]["members"]
                    idx2 = [ALL_WORDS.index(w) for w in m2]
                    for a in idx1:
                        for b in idx2:
                            inter.append(dm[a, b])
            ratio = np.mean(intra) / max(np.mean(inter), 1e-10)
            print(f"    L{li}: intra={np.mean(intra):.4f}, inter={np.mean(inter):.4f}, ratio={ratio:.4f}")


if __name__ == "__main__":
    # 四模型列表
    models_to_run = ["Qwen3", "DeepSeek7B", "GLM4", "Gemma4"]

    # 尝试加载已有的结果
    all_results = {}
    for mname in models_to_run:
        p = os.path.join(OUTPUT_DIR, f"stage548_{mname}_dist.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                all_results[mname] = json.load(f)
            print(f"  已加载缓存: {mname}")

    # 逐模型计算
    for mname in models_to_run:
        if mname not in all_results:
            result = compute_dist_matrix_for_model(mname)
            all_results[mname] = result

    # 跨模型分析
    cross_results = cross_model_analysis(all_results)

    # intra/inter分析
    intra_inter_per_model(all_results)

    # 保存综合结果
    synthesis = {
        "pairwise_pearson": cross_results,
        "models": {m: {"n_layers": d["n_layers"], "sample_layers": d["sample_layers"]} for m, d in all_results.items()},
    }
    synth_path = os.path.join(OUTPUT_DIR, "stage548_synthesis.json")
    with open(synth_path, "w", encoding="utf-8") as f:
        json.dump(synthesis, f, indent=2, ensure_ascii=False)
    print(f"\n  综合结果保存到: {synth_path}")
