"""
stage549: 12类名词家族扩展
目标：从6类扩展到12类，在Qwen3上验证家族内聚性(intra/inter)和编码距离矩阵是否扩展后仍然成立。
新增6类：颜色(color)、身体部位(body)、乐器(instrument)、国家(country)、情绪(emotion)、材料(material)

每个模型逐个运行，保存中间结果。
"""
import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_language_shared import load_qwen3_model, discover_layers
from multimodel_language_shared import encode_to_device, evenly_spaced_layers, free_model

# 12类名词家族（每类3个成员）
NOUN_FAMILIES_12 = {
    "fruit": {"members": ["apple", "banana", "cherry"], "label": "水果"},
    "animal": {"members": ["cat", "dog", "horse"], "label": "动物"},
    "tool": {"members": ["hammer", "knife", "screwdriver"], "label": "工具"},
    "org": {"members": ["university", "company", "hospital"], "label": "组织"},
    "celestial": {"members": ["sun", "moon", "mars"], "label": "天体"},
    "abstract": {"members": ["freedom", "justice", "truth"], "label": "抽象"},
    "color": {"members": ["red", "blue", "green"], "label": "颜色"},
    "body": {"members": ["heart", "brain", "hand"], "label": "身体"},
    "instrument": {"members": ["piano", "guitar", "violin"], "label": "乐器"},
    "country": {"members": ["china", "france", "japan"], "label": "国家"},
    "emotion": {"members": ["anger", "joy", "fear"], "label": "情绪"},
    "material": {"members": ["gold", "iron", "wood"], "label": "材料"},
}
ALL_WORDS_12 = []
for fam in NOUN_FAMILIES_12.values():
    ALL_WORDS_12.extend(fam["members"])

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'glm5', 'temp')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_encoding(model, tokenizer, text, layer_idx):
    encoded = encode_to_device(model, tokenizer, text)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    hs = outputs.hidden_states[layer_idx + 1]
    return hs[0, -1, :]


def compute_for_model(model_name):
    print(f"\n{'='*60}")
    print(f"  模型: {model_name} - 12类名词家族分析")
    print(f"{'='*60}")

    if model_name == "Qwen3":
        model, tokenizer = load_qwen3_model()
    elif model_name == "DeepSeek7B":
        model, tokenizer = __import__('multimodel_language_shared', fromlist=['load_deepseek_model']).load_deepseek_model()
    elif model_name == "GLM4":
        model, tokenizer = __import__('multimodel_language_shared', fromlist=['load_glm4_model']).load_glm4_model()
    elif model_name == "Gemma4":
        model, tokenizer = __import__('multimodel_language_shared', fromlist=['load_gemma4_model']).load_gemma4_model()

    layers = discover_layers(model)
    n_layers = len(layers)
    sample_layers = evenly_spaced_layers(model, count=10)
    print(f"  层数: {n_layers}, 采样: {sample_layers}")
    print(f"  词汇: {len(ALL_WORDS_12)} ({len(NOUN_FAMILIES_12)}类)")

    # 计算编码
    print(f"\n  [1] 计算编码...")
    encodings = {}
    for w in ALL_WORDS_12:
        encodings[w] = {}
        for li in sample_layers:
            encodings[w][li] = get_encoding(model, tokenizer, w, li)
    print(f"  {len(ALL_WORDS_12)} words x {len(sample_layers)} layers")

    # 分析1: intra/inter per layer
    print(f"\n  [2] 家族内聚性 (intra/inter ratio) 逐层:")
    family_results = {}
    fam_names = list(NOUN_FAMILIES_12.keys())
    for li in sample_layers:
        intra, inter = [], []
        for fi, fk1 in enumerate(fam_names):
            m1 = NOUN_FAMILIES_12[fk1]["members"]
            for a in range(len(m1)):
                for b in range(a + 1, len(m1)):
                    v1 = encodings[m1[a]][li].float().cpu()
                    v2 = encodings[m1[b]][li].float().cpu()
                    d = 1 - torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                    intra.append(d)
            for fj, fk2 in enumerate(fam_names):
                if fj <= fi:
                    continue
                m2 = NOUN_FAMILIES_12[fk2]["members"]
                for w1 in m1:
                    for w2 in m2:
                        v1 = encodings[w1][li].float().cpu()
                        v2 = encodings[w2][li].float().cpu()
                        d = 1 - torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                        inter.append(d)
        ratio = np.mean(intra) / max(np.mean(inter), 1e-10)
        family_results[str(li)] = {
            "intra_mean": round(float(np.mean(intra)), 6),
            "inter_mean": round(float(np.mean(inter)), 6),
            "intra_inter_ratio": round(ratio, 6),
        }
        print(f"    L{li}: intra={np.mean(intra):.4f}, inter={np.mean(inter):.4f}, ratio={ratio:.4f}")

    # 分析2: 逐类intra距离
    print(f"\n  [3] 逐类内部距离 (末层 L{sample_layers[-1]}):")
    last_li = sample_layers[-1]
    per_family = {}
    for fk, fv in NOUN_FAMILIES_12.items():
        members = fv["members"]
        dists = []
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                v1 = encodings[members[a]][last_li].float().cpu()
                v2 = encodings[members[b]][last_li].float().cpu()
                d = 1 - torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                dists.append(d)
        mean_d = np.mean(dists)
        per_family[fk] = {"label": fv["label"], "intra_dist": round(float(mean_d), 6)}
        print(f"    {fv['label']}({fk}): intra_dist={mean_d:.4f}")

    # 分析3: 信息几何
    print(f"\n  [4] 有效维度 逐层:")
    geo_results = {}
    for li in sample_layers:
        matrix = torch.stack([encodings[w][li].float() for w in ALL_WORDS_12])
        n = matrix.shape[0]
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / max(n - 1, 1)
        eigenvalues, _ = torch.linalg.eigh(cov)
        top_eigs = eigenvalues[eigenvalues > 1e-10].flip(0).cpu().numpy()
        cumsum = np.cumsum(top_eigs) / max(top_eigs.sum(), 1e-10)
        eff_dim = int(np.searchsorted(cumsum, 0.90) + 1) if len(cumsum) > 0 else 0
        geo_results[str(li)] = {"effective_dim_90": eff_dim}
        print(f"    L{li}: eff_dim={eff_dim}")

    # 分析4: 距离矩阵
    print(f"\n  [5] 距离矩阵 (末层 L{last_li}):")
    n_words = len(ALL_WORDS_12)
    dm = np.zeros((n_words, n_words))
    for i in range(n_words):
        for j in range(i + 1, n_words):
            v1 = encodings[ALL_WORDS_12[i]][last_li].float().cpu()
            v2 = encodings[ALL_WORDS_12[j]][last_li].float().cpu()
            cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            dm[i, j] = 1 - cos_sim
            dm[j, i] = dm[i, j]
    # 打印前6个词的距离（原6类）
    print(f"    前6类距离矩阵片段:")
    header = "          " + "  ".join(f"{w:>10}" for w in ALL_WORDS_12[:12])
    print(f"    {header}")
    for i in range(min(12, n_words)):
        row = f"    {ALL_WORDS_12[i]:>10}" + "".join(f"{dm[i,j]:>10.4f}" for j in range(min(12, n_words)))
        print(row)

    # 分析5: 跨类距离 vs 类内距离排序
    print(f"\n  [6] 类间距离矩阵 (末层类中心间):")
    family_centers = {}
    for fk, fv in NOUN_FAMILIES_12.items():
        vecs = torch.stack([encodings[w][last_li].float() for w in fv["members"]])
        family_centers[fk] = vecs.mean(dim=0)

    fam_names_list = list(NOUN_FAMILIES_12.keys())
    center_dists = {}
    for fi, fk1 in enumerate(fam_names_list):
        for fj, fk2 in enumerate(fam_names_list):
            if fj <= fi:
                continue
            v1 = family_centers[fk1].cpu()
            v2 = family_centers[fk2].cpu()
            d = 1 - torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            center_dists[f"{fk1}-{fk2}"] = round(d, 4)

    # 排序：最相似和最不相似的类对
    sorted_pairs = sorted(center_dists.items(), key=lambda x: x[1])
    print(f"    最相似的5对类:")
    for pk, pd in sorted_pairs[:5]:
        print(f"      {pk}: {pd:.4f}")
    print(f"    最不相似的5对类:")
    for pk, pd in sorted_pairs[-5:]:
        print(f"      {pk}: {pd:.4f}")

    # 保存
    result = {
        "model": model_name,
        "n_layers": n_layers,
        "n_families": 12,
        "n_words": len(ALL_WORDS_12),
        "sample_layers": sample_layers,
        "family_cohesion": family_results,
        "per_family_intra": per_family,
        "info_geometry": geo_results,
        "center_distances": center_dists,
    }
    out_path = os.path.join(OUTPUT_DIR, f"stage549_{model_name}_12families.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  保存到: {out_path}")

    free_model(model)
    time.sleep(3)
    return result


if __name__ == "__main__":
    models_to_run = ["Qwen3", "DeepSeek7B", "GLM4", "Gemma4"]

    all_results = {}
    for mname in models_to_run:
        p = os.path.join(OUTPUT_DIR, f"stage549_{mname}_12families.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                all_results[mname] = json.load(f)
            print(f"  已加载缓存: {mname}")
        else:
            result = compute_for_model(mname)
            all_results[mname] = result

    # 综合分析
    print(f"\n{'='*60}")
    print(f"  四模型 12类家族 综合分析")
    print(f"{'='*60}")

    # 1. 逐模型末层intra/inter
    print(f"\n  [A] 末层 intra/inter ratio:")
    for mname, mdata in all_results.items():
        last_layer = str(mdata["sample_layers"][-1])
        if last_layer in mdata["family_cohesion"]:
            fc = mdata["family_cohesion"][last_layer]
            print(f"    {mname}: intra={fc['intra_mean']:.4f}, inter={fc['inter_mean']:.4f}, ratio={fc['intra_inter_ratio']:.4f}")

    # 2. 逐类内部距离跨模型比较
    print(f"\n  [B] 逐类内部距离 跨模型 (末层):")
    fam_names = list(all_results[models_to_run[0]]["per_family_intra"].keys())
    for fk in fam_names:
        vals = []
        for mname in models_to_run:
            if fk in all_results[mname]["per_family_intra"]:
                vals.append(all_results[mname]["per_family_intra"][fk]["intra_dist"])
        if vals:
            label = all_results[models_to_run[0]]["per_family_intra"][fk]["label"]
            print(f"    {label}({fk}): {', '.join(f'{m}={v:.4f}' for m, v in zip(models_to_run, vals))} | mean={np.mean(vals):.4f}")

    # 3. 维度坍缩检测
    print(f"\n  [C] 维度坍缩 (12词版):")
    for mname, mdata in all_results.items():
        geo = mdata["info_geometry"]
        layers = sorted([int(l) for l in geo.keys()])
        n_total = mdata["n_layers"]
        for i, li in enumerate(layers):
            dim = geo[str(li)]["effective_dim_90"]
            if dim <= 1 and (i == 0 or geo[str(layers[i-1])]["effective_dim_90"] > 1):
                print(f"    {mname}: 坍缩 L{li} (归一化 {li/max(n_total-1,1)*100:.1f}%)")
                break
        else:
            print(f"    {mname}: 未坍缩 (L0 dim={geo[str(layers[0])]['effective_dim_90']})")

    # 4. 类间距离最相似/最不相似 对跨模型
    print(f"\n  [D] 类间距离跨模型排名一致性:")
    # 收集各模型的类间距离排名
    for mname in models_to_run:
        cd = all_results[mname]["center_distances"]
        sorted_p = sorted(cd.items(), key=lambda x: x[1])
        print(f"\n    {mname} 最相似5对:")
        for pk, pd in sorted_p[:5]:
            print(f"      {pk}: {pd:.4f}")
