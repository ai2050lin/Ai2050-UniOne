"""
stage555: 上下文窗口信息分布
目标：在一个句子中，不同位置的信息量分布
- 哪些层"整合"上下文（所有token编码趋同），哪些层"区分"个体（编码分离）
- 逐层的全局方差分析
- 不同位置token的有效维度
- 中心词 vs 边缘词的信息保留

使用Qwen3快速验证。
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_language_shared import load_qwen3_model, discover_layers
from multimodel_language_shared import encode_to_device, evenly_spaced_layers, free_model

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'glm5', 'temp')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SENTENCES = [
    "The red apple is sweet",
    "A big cat sat on the mat",
    "The sun is very bright today",
    "I went to the bank near the river",
    "She likes to eat fresh fruit every morning",
]


def compute_effective_dim(vectors):
    """计算一组向量的有效维度"""
    if len(vectors) < 2:
        return 1.0
    X = np.array(vectors)
    X = X - X.mean(axis=0)
    cov = (X.T @ X) / (len(X) - 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues = eigenvalues[::-1]  # 降序
    total = eigenvalues.sum()
    if total < 1e-10:
        return 1.0
    cumsum = np.cumsum(eigenvalues) / total
    for k in range(1, len(cumsum)):
        if cumsum[k] >= 0.9:
            return float(k + 1)
    return float(len(eigenvalues))


def experiment1_global_variance(model, tokenizer, n_layers):
    """逐层分析：所有token编码的全局方差→识别整合层vs区分层"""
    print(f"\n{'='*60}")
    print(f"  实验1：逐层全局方差（整合层 vs 区分层）")
    print(f"{'='*60}")

    for sent in TEST_SENTENCES:
        encoded = encode_to_device(model, tokenizer, sent)
        token_ids = encoded["input_ids"][0].tolist()
        n_tokens = len(token_ids)
        tokens = [tokenizer.convert_ids_to_tokens(t) for t in token_ids]
        tokens_safe = [t.encode('ascii', 'replace').decode() for t in tokens]

        print(f"\n  '{sent}' ({n_tokens} tokens):")
        print(f"    tokens: {tokens_safe}")

        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True)

        all_layers = len(out.hidden_states)
        sample = list(range(0, all_layers, max(1, all_layers // 20)))
        if (all_layers - 1) not in sample:
            sample.append(all_layers - 1)

        print(f"    层    | 全局方差  | 有效维度 | Token间平均cos_d")
        print(f"    -------+-----------+----------+------------------")
        for li in sample:
            # 所有token的编码
            layer_embs = out.hidden_states[li][0]  # (n_tokens, dim)
            layer_np = layer_embs.cpu().float().numpy()

            # 全局方差：所有维度方差之和
            global_var = float(layer_np.var())

            # 有效维度
            eff_dim = compute_effective_dim(layer_np)

            # Token间平均cosine距离
            pair_dists = []
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    cos_d = 1 - F.cosine_similarity(
                        layer_embs[i], layer_embs[j], dim=0
                    ).item()
                    pair_dists.append(cos_d)
            mean_cos_d = np.mean(pair_dists)

            marker = ""
            if li == 0:
                marker = " (L0)"
            elif li == all_layers - 1:
                marker = " (末层)"
            elif mean_cos_d < 0.05:
                marker = " ← 整合!"
            print(f"    L{li:2d}{marker:6s}| {global_var:9.4f} | {eff_dim:8.1f} | {mean_cos_d:.4f}")

    return {}


def experiment2_position_information(model, tokenizer, n_layers):
    """不同位置token在各层的信息保留量"""
    print(f"\n{'='*60}")
    print(f"  实验2：位置信息保留（中心词 vs 边缘词 vs 功能词）")
    print(f"{'='*60}")

    sent = "The red apple is sweet and delicious"
    encoded = encode_to_device(model, tokenizer, sent)
    token_ids = encoded["input_ids"][0].tolist()
    n_tokens = len(token_ids)
    tokens = [tokenizer.convert_ids_to_tokens(t) for t in token_ids]
    tokens_safe = [t.encode('ascii', 'replace').decode() for t in tokens]

    print(f"  '{sent}' ({n_tokens} tokens):")
    print(f"    tokens: {tokens_safe}")

    # 标记每个token的类型
    word_types = []
    for t in tokens:
        tl = t.lower()
        if tl in ["the", "a", "an", "is", "are", "and", "or"]:
            word_types.append("func")  # 功能词
        elif any(c in tl for c in [",", "."]):
            word_types.append("punc")
        else:
            word_types.append("content")  # 内容词

    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True)

    all_layers = len(out.hidden_states)
    sample = list(range(0, all_layers, max(1, all_layers // 12)))
    if (all_layers - 1) not in sample:
        sample.append(all_layers - 1)

    # 每个token的L2 norm逐层变化
    print(f"\n    逐层Token L2 norm:")
    header = "    层    | " + " | ".join(f"{t[:8]:>8s}" for t in tokens_safe[:10])
    print(header)

    for li in sample:
        layer_embs = out.hidden_states[li][0]
        norms = [float(torch.norm(layer_embs[i])) for i in range(n_tokens)]
        norms_str = " | ".join(f"{n:8.2f}" for n in norms[:10])
        print(f"    L{li:2d}    | {norms_str}")

    # L0 vs 末层 cosine距离
    print(f"\n    L0→末层 cosine距离:")
    for pos in range(min(n_tokens, 12)):
        e0 = out.hidden_states[0][0, pos].cpu().float()
        eL = out.hidden_states[-1][0, pos].cpu().float()
        cos_d = 1 - F.cosine_similarity(e0.unsqueeze(0), eL.unsqueeze(0), dim=1).item()
        wtype = word_types[pos] if pos < len(word_types) else "?"
        print(f"    pos {pos:2d} ({tokens_safe[pos]:>8s}, {wtype:7s}): cos_d = {cos_d:.4f}")

    # 功能词 vs 内容词的平均cos_d
    func_dists = []
    content_dists = []
    for pos in range(n_tokens):
        e0 = out.hidden_states[0][0, pos].cpu().float()
        eL = out.hidden_states[-1][0, pos].cpu().float()
        cos_d = 1 - F.cosine_similarity(e0.unsqueeze(0), eL.unsqueeze(0), dim=1).item()
        if pos < len(word_types):
            if word_types[pos] == "func":
                func_dists.append(cos_d)
            else:
                content_dists.append(cos_d)

    print(f"\n    功能词平均L0→末层 cos_d: {np.mean(func_dists):.4f} (n={len(func_dists)})")
    print(f"    内容词平均L0→末层 cos_d: {np.mean(content_dists):.4f} (n={len(content_dists)})")

    return {}


def experiment3_integration_discrimination_curve(model, tokenizer, n_layers):
    """识别哪些层整合上下文，哪些层区分个体"""
    print(f"\n{'='*60}")
    print(f"  实验3：整合-区分曲线（多句子平均）")
    print(f"{'='*60}")

    all_layers = len(discover_layers(model))
    sample = list(range(0, all_layers, max(1, all_layers // 20)))
    if (all_layers - 1) not in sample:
        sample.append(all_layers - 1)

    # 对每个句子计算逐层mean_cos_d
    sentence_curves = []
    for sent in TEST_SENTENCES:
        encoded = encode_to_device(model, tokenizer, sent)
        n_tokens = encoded["input_ids"].shape[1]

        curve = []
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True)
            for li in sample:
                layer_embs = out.hidden_states[li][0]
                pair_dists = []
                for i in range(n_tokens):
                    for j in range(i + 1, n_tokens):
                        cos_d = 1 - F.cosine_similarity(
                            layer_embs[i], layer_embs[j], dim=0
                        ).item()
                        pair_dists.append(cos_d)
                curve.append(np.mean(pair_dists))
        sentence_curves.append(curve)

    # 平均曲线
    avg_curve = np.mean(sentence_curves, axis=0)

    print(f"\n    归一化层位置 | 平均token间cos_d | 趋势")
    print(f"    -----------+-----------------+------")
    for i, li in enumerate(sample):
        norm_pos = li / max(all_layers - 1, 1)
        trend = ""
        if i > 0:
            delta = avg_curve[i] - avg_curve[i - 1]
            if delta > 0.01:
                trend = "↑ 区分增强"
            elif delta < -0.01:
                trend = "↓ 整合增强"
        print(f"    {norm_pos:10.1%}  | {avg_curve[i]:15.4f} | {trend}")

    # 找整合最深的层
    min_idx = np.argmin(avg_curve)
    min_layer = sample[min_idx]
    min_val = avg_curve[min_idx]
    print(f"\n    最强整合层: L{min_layer} ({min_layer/max(all_layers-1,1):.1%}), mean_cos_d={min_val:.4f}")
    print(f"    末层区分度: L{sample[-1]}, mean_cos_d={avg_curve[-1]:.4f}")

    # 计算恢复比例
    if len(avg_curve) > 2:
        recovery = avg_curve[-1] / max(avg_curve[0], 1e-10)
        print(f"    末层 vs L0 比例: {recovery:.2f}")

    return {"avg_curve": {str(li): round(v, 4) for li, v in zip(sample, avg_curve)}}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  stage555: 上下文窗口信息分布")
    print("=" * 60)

    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    print(f"  Qwen3 n_layers={n_layers}")

    try:
        r1 = experiment1_global_variance(model, tokenizer, n_layers)
        r2 = experiment2_position_information(model, tokenizer, n_layers)
        r3 = experiment3_integration_discrimination_curve(model, tokenizer, n_layers)
    finally:
        free_model(model)

    # 保存
    results = {"integration_curve": r3}
    out_path = os.path.join(OUTPUT_DIR, "stage555_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_path}")

    print(f"\n  总耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
