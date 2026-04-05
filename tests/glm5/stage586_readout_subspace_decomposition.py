"""
Stage 586: 读出子空间功能分解

已知unembedding有效秩=113，编码信息偏低频维度(band0)。
本脚本分析113个有效方向中，哪些承载消歧信息、语法信息、语义信息、频率信息。

核心方法：
1. 对消歧/语法/语义/频率四类任务分别构建hidden state→logit映射
2. 对每类任务的映射矩阵做SVD
3. 对比不同任务的SVD方向重叠度（RV系数/子空间交角）
4. 找到每类任务的"特征方向"
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

MODEL_PATH = r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"

# ========== 测试句式（四类任务）==========
TASK_SENTENCES = {
    "disambiguation": {
        "bank-river": "The children played by the river bank",
        "bank-financial": "The financial bank approved the loan",
        "apple-fruit": "She ate a delicious red apple",
        "apple-company": "Apple released a new product today",
        "plant-tree": "The plant grew tall in the garden",
        "plant-factory": "The power plant generates electricity",
        "spring-season": "Spring is the most beautiful season",
        "spring-water": "Water flows from the natural spring",
    },
    "syntax": {
        "subject": "The cat sat on the mat",
        "object": "The dog chased the cat",
        "passive": "The ball was thrown by the boy",
        "question": "What did the cat see",
        "negative": "The cat did not eat the fish",
        "plural": "The cats are sleeping together",
    },
    "semantics": {
        "animal": "The lion roared in the savanna",
        "tool": "The hammer broke the window",
        "emotion": "She felt extremely happy today",
        "location": "The city is near the ocean",
        "time": "The meeting starts at three pm",
        "abstract": "Justice is a fundamental human right",
    },
    "frequency": {
        "high-freq-1": "The cat sat on the mat",
        "high-freq-2": "I went to the store",
        "high-freq-3": "She said hello to him",
        "low-freq-1": "The epistemological framework delineates",  # 高级词汇
        "low-freq-2": "Phenomenological analysis reveals",
        "low-freq-3": "The ontological paradigm suggests",
    },
}


def get_hidden_and_logits(model, tokenizer, sentence, device):
    """获取最后一个token的hidden state和logits"""
    enc = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states[-1][0, -1, :].float().cpu()  # 末层最后一个token
    logits = out.logits[0, -1, :].float().cpu()  # 对应的logits
    return hidden, logits


def subspace_rv_coefficient(U1, U2, k=50):
    """计算两个子空间之间的RV系数（相似度）"""
    k = min(k, U1.shape[1], U2.shape[1])
    P1 = U1[:, :k]  # [hidden_dim, k]
    P2 = U2[:, :k]  # [hidden_dim, k]
    # RV = trace(P1^T P2 P2^T P1) / sqrt(trace(P1^T P1 P1^T P1) * trace(P2^T P2 P2^T P2))
    num = torch.trace(P1.T @ P2 @ P2.T @ P1)
    den = torch.sqrt(torch.trace(P1.T @ P1 @ P1.T @ P1) * torch.trace(P2.T @ P2 @ P2.T @ P2))
    return (num / den).item()


def task_logit_matrix(model, tokenizer, sentences, device):
    """
    构建任务特定的logit映射矩阵
    每个句子的logit代表一种"任务响应"
    返回: [n_sentences, vocab_size] 矩阵
    """
    all_logits = []
    for sent in sentences:
        _, logits = get_hidden_and_logits(model, tokenizer, sent, device)
        all_logits.append(logits)
    return torch.stack(all_logits)


def task_direction_analysis(model, tokenizer, task_name, sentences, device):
    """
    分析一个任务的hidden state方向特征
    """
    all_hidden = []
    for sent in sentences:
        hidden, _ = get_hidden_and_logits(model, tokenizer, sent, device)
        all_hidden.append(hidden)
    H = torch.stack(all_hidden)  # [n_sentences, hidden_dim]
    
    # 中心化
    mean = H.mean(dim=0, keepdim=True)
    H_centered = H - mean
    
    # SVD
    try:
        U, S, Vt = torch.linalg.svd(H_centered, full_matrices=False)
    except RuntimeError:
        H_np = H_centered.numpy()
        _, S_np, Vt_np = np.linalg.svd(H_np, full_matrices=False)
        S = torch.from_numpy(S_np.astype(np.float32))
        Vt = torch.from_numpy(Vt_np.astype(np.float32))
    
    eff_rank = (S.sum() ** 2) / (S[S > 1e-8] ** 2).sum() if (S > 1e-8).any() else 1.0
    
    return {
        "task": task_name,
        "n_sentences": len(sentences),
        "singular_values": S[:20].tolist(),
        "effective_rank": round(min(eff_rank.item(), len(sentences)), 2),
        "Vt": Vt[:20],  # 保留前20个方向
    }


def frequency_sensitive_directions(model, tokenizer, high_freq, low_freq, device, top_k=100):
    """
    找到区分高频和低频词汇的hidden state方向
    """
    high_hidden = []
    for sent in high_freq:
        h, _ = get_hidden_and_logits(model, tokenizer, sent, device)
        high_hidden.append(h)
    low_hidden = []
    for sent in low_freq:
        h, _ = get_hidden_and_logits(model, tokenizer, sent, device)
        low_hidden.append(h)
    
    H_high = torch.stack(high_hidden).mean(dim=0)
    H_low = torch.stack(low_hidden).mean(dim=0)
    
    diff = H_high - H_low
    top_dims = torch.topk(diff.abs(), top_k).indices.tolist()
    
    return {
        "diff_norm": round(diff.norm().item(), 4),
        "cosine_sim": round(F.cosine_similarity(H_high.unsqueeze(0), H_low.unsqueeze(0)).item(), 4),
        "top_dims": top_dims[:20],
        "band_distribution": {
            "low_0_160": sum(1 for d in top_dims if d < 160),
            "mid_160_1120": sum(1 for d in top_dims if 160 <= d < 1120),
            "high_1120_2560": sum(1 for d in top_dims if d >= 1120),
        }
    }


def main():
    print("=" * 60)
    print("Stage 586: 读出子空间功能分解")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    print("加载模型...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codex'))
    from qwen3_language_shared import load_qwen3_model
    model, tokenizer = load_qwen3_model(prefer_cuda=True)
    
    # ========== 实验1: 四类任务的方向分析 ==========
    print("\n实验1: 四类任务的hidden state方向特征")
    task_analyses = {}
    for task_name, sentences in TASK_SENTENCES.items():
        print(f"\n  任务: {task_name}")
        result = task_direction_analysis(model, tokenizer, task_name, sentences, device)
        task_analyses[task_name] = result
        print(f"    有效秩: {result['effective_rank']}")
        print(f"    Top-5 SV: {[round(s, 2) for s in result['singular_values'][:5]]}")
    
    # ========== 实验2: 任务间子空间相似度（RV系数）==========
    print("\n实验2: 任务间子空间相似度(RV系数)")
    task_names = list(task_analyses.keys())
    rv_matrix = {}
    for i, t1 in enumerate(task_names):
        for j, t2 in enumerate(task_names):
            if i >= j:
                continue
            rv = subspace_rv_coefficient(
                task_analyses[t1]["Vt"].T,
                task_analyses[t2]["Vt"].T,
                k=5
            )
            rv_matrix[f"{t1}_vs_{t2}"] = round(rv, 4)
            print(f"  {t1:20s} vs {t2:20s}: RV = {rv:.4f}")
    
    # ========== 实验3: 消歧方向 vs 语义方向 vs 语法方向 ==========
    print("\n实验3: 消歧任务内部——消歧对 vs 非消歧对的方向差异")
    disamb_sents = TASK_SENTENCES["disambiguation"]
    # 每个歧义词有两个意义，计算两个意义hidden state的差异方向
    pairs = [("bank-river", "bank-financial"), ("apple-fruit", "apple-company"),
             ("plant-tree", "plant-factory"), ("spring-season", "spring-water")]
    for p1, p2 in pairs:
        h1, _ = get_hidden_and_logits(model, tokenizer, disamb_sents[p1], device)
        h2, _ = get_hidden_and_logits(model, tokenizer, disamb_sents[p2], device)
        diff = h1 - h2
        cos = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
        # 找差异最大的维度
        top_dims = torch.topk(diff.abs(), 20).indices.tolist()
        band_low = sum(1 for d in top_dims if d < 160)
        band_high = sum(1 for d in top_dims if d >= 1120)
        print(f"  {p1} vs {p2}: cos={cos:.4f}, diff_norm={diff.norm():.2f}, "
              f"band0={band_low}/20, band_high={band_high}/20")
    
    # ========== 实验4: 频率敏感方向 ==========
    print("\n实验4: 高频 vs 低频词汇的方向差异")
    freq_result = frequency_sensitive_directions(
        model, tokenizer,
        TASK_SENTENCES["frequency"]["high-freq-1"],
        TASK_SENTENCES["frequency"]["low-freq-1"],
        device
    )
    print(f"  差异向量norm: {freq_result['diff_norm']}")
    print(f"  高低频cosine: {freq_result['cosine_sim']}")
    print(f"  频段分布: {freq_result['band_distribution']}")
    
    # ========== 实验5: unembedding子空间的功能标注 ==========
    print("\n实验5: unembedding子空间的功能标注")
    # 获取unembedding矩阵
    W = model.lm_head.weight.detach().float().cpu()  # [vocab, hidden_dim]
    # 中心化
    W_mean = W.mean(dim=0, keepdim=True)
    W_centered = W - W_mean
    try:
        _, S, Vt = torch.linalg.svd(W_centered, full_matrices=False)
    except RuntimeError:
        W_np = W_centered.numpy()
        rng = np.random.RandomState(42)
        proj_dim = min(512, W_np.shape[0], W_np.shape[1])
        Q, _ = np.linalg.qr(W_np.T @ rng.randn(W_np.shape[1], proj_dim))
        B = Q.T @ W_np.T
        _, S_np, Vt_np = np.linalg.svd(B, full_matrices=False)
        S = torch.from_numpy(S_np.astype(np.float32))
        Vt = torch.from_numpy(Vt_np.astype(np.float32))
    
    # 前50个方向
    directions = Vt[:50]  # [50, hidden_dim]
    
    # 每个方向对消歧差异的贡献
    print("  前10个SVD方向对各任务差异的贡献:")
    for task_name in ["disambiguation", "syntax", "semantics"]:
        sents = TASK_SENTENCES[task_name]
        # 取前两个句子的差异投影到每个方向
        if len(sents) >= 2:
            sents_list = list(sents.values())
            h1, _ = get_hidden_and_logits(model, tokenizer, sents_list[0], device)
            h2, _ = get_hidden_and_logits(model, tokenizer, sents_list[1], device)
            diff = h1 - h2
            # 每个方向上的投影
            projections = [round((diff @ directions[i]).abs().item(), 2) for i in range(10)]
            print(f"    {task_name}: proj={projections}")
    
    # ========== 保存结果 ==========
    out_path = "tests/glm5_temp/stage586_readout_subspace_decomposition.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_data = {
        "task_analyses": {k: {
            "effective_rank": v["effective_rank"],
            "singular_values_top10": [round(s, 2) for s in v["singular_values"][:10]],
        } for k, v in task_analyses.items()},
        "rv_coefficients": rv_matrix,
        "frequency_analysis": freq_result,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存到: {out_path}")
    
    print("\n" + "=" * 60)
    print("关键结论")
    print("=" * 60)
    
    # 判断各任务间RV系数
    max_rv = max(rv_matrix.values())
    min_rv = min(rv_matrix.values())
    max_pair = [k for k, v in rv_matrix.items() if v == max_rv][0]
    min_pair = [k for k, v in rv_matrix.items() if v == min_rv][0]
    print(f"  最相似任务对: {max_pair} (RV={max_rv:.4f})")
    print(f"  最不相似任务对: {min_pair} (RV={min_rv:.4f})")
    print(f"  消歧信息频段分布: band0={freq_result['band_distribution']['low_0_160']}/100")


if __name__ == "__main__":
    main()
