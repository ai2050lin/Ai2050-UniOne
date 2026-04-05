"""
stage550: 完整句式验证
目标：将单token "apple" 替换为完整句式 "An apple is a fruit"，验证：
1. 距离矩阵是否仍然跨模型相关
2. 家族内聚性是否在句式中更强
3. 维度坍缩是否仍然发生
4. 单token vs 句式编码的差异

使用Qwen3快速验证，如有时间再扩展到其他模型。
"""
import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_language_shared import load_qwen3_model, discover_layers
from multimodel_language_shared import encode_to_device, evenly_spaced_layers, free_model

# 测试句子：名词在句中的编码
SENTENCE_TEMPLATES = {
    "fruit": [
        ("An apple is a fruit", "apple"),
        ("A banana is yellow", "banana"),
        ("A cherry is small and red", "cherry"),
    ],
    "animal": [
        ("A cat is a small pet", "cat"),
        ("A dog is loyal to humans", "dog"),
        ("A horse runs very fast", "horse"),
    ],
    "tool": [
        ("A hammer hits nails", "hammer"),
        ("A knife cuts food", "knife"),
        ("A screwdriver fixes screws", "screwdriver"),
    ],
    "org": [
        ("A university teaches students", "university"),
        ("A company makes products", "company"),
        ("A hospital treats patients", "hospital"),
    ],
    "celestial": [
        ("The sun gives us light", "sun"),
        ("The moon shines at night", "moon"),
        ("Mars is a red planet", "mars"),
    ],
    "abstract": [
        ("Freedom is a basic right", "freedom"),
        ("Justice must be served", "justice"),
        ("Truth is always important", "truth"),
    ],
}

# 构建词汇列表
ALL_WORDS_SENT = []
WORD_SENTENCE_MAP = {}  # word -> sentence
for fk, entries in SENTENCE_TEMPLATES.items():
    for sent, word in entries:
        ALL_WORDS_SENT.append(word)
        WORD_SENTENCE_MAP[word] = sent

# 词性分类
WORDCLASS_SENTENCES = {
    "noun": [("An apple is a fruit", "apple"), ("A cat is small", "cat"), ("A hammer hits nails", "hammer")],
    "adj": [("The sky is red", "red"), ("A big house stands there", "big"), ("She runs fast", "fast")],
    "verb": [("I run every day", "run"), ("They eat lunch", "eat"), ("He thinks deeply", "thinks")],
    "prep": [("It is in the box", "in"), ("The book is on the table", "on"), ("She came from school", "from")],
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'glm5', 'temp')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_token_encoding(model, tokenizer, sentence, target_word, layer_idx):
    """获取句中目标词在指定层的编码"""
    encoded = encode_to_device(model, tokenizer, sentence)
    input_ids = encoded["input_ids"]
    
    # 找到target_word的token位置
    target_token_ids = tokenizer.encode(target_word, add_special_tokens=False)
    # 在input_ids中找第一个匹配
    target_pos = None
    for tid in target_token_ids:
        for pos in range(input_ids.shape[1]):
            if input_ids[0, pos].item() == tid:
                target_pos = pos
                break
        if target_pos is not None:
            break
    
    if target_pos is None:
        # fallback: 取最后一个token
        target_pos = input_ids.shape[1] - 1
    
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    hs = outputs.hidden_states[layer_idx + 1]
    return hs[0, target_pos, :], target_pos


def main():
    print(f"{'='*60}")
    print(f"  stage550: 完整句式验证 (Qwen3)")
    print(f"{'='*60}")

    model, tokenizer = load_qwen3_model()
    layers = discover_layers(model)
    n_layers = len(layers)
    sample_layers = evenly_spaced_layers(model, count=10)
    print(f"  层数: {n_layers}, 采样: {sample_layers}")

    # 1. 句式中名词编码
    print(f"\n{'='*60}")
    print(f"  [1] 句式中名词编码 - 家族内聚性")
    print(f"{'='*60}")

    sent_encodings = {}
    for word in ALL_WORDS_SENT:
        sentence = WORD_SENTENCE_MAP[word]
        sent_encodings[word] = {}
        for li in sample_layers:
            enc, pos = get_token_encoding(model, tokenizer, sentence, word, li)
            sent_encodings[word][li] = enc
    print(f"  {len(ALL_WORDS_SENT)} 词句对 x {len(sample_layers)} 层")

    # intra/inter
    fam_names = list(SENTENCE_TEMPLATES.keys())
    print(f"\n  句式版 intra/inter ratio:")
    sent_intra_inter = {}
    for li in sample_layers:
        intra, inter = [], []
        for fi, fk1 in enumerate(fam_names):
            m1 = [w for s, w in SENTENCE_TEMPLATES[fk1]]
            for a in range(len(m1)):
                for b in range(a + 1, len(m1)):
                    v1 = sent_encodings[m1[a]][li].float().cpu()
                    v2 = sent_encodings[m1[b]][li].float().cpu()
                    d = 1 - torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                    intra.append(d)
            for fj, fk2 in enumerate(fam_names):
                if fj <= fi:
                    continue
                m2 = [w for s, w in SENTENCE_TEMPLATES[fk2]]
                for w1 in m1:
                    for w2 in m2:
                        v1 = sent_encodings[w1][li].float().cpu()
                        v2 = sent_encodings[w2][li].float().cpu()
                        d = 1 - torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                        inter.append(d)
        ratio = np.mean(intra) / max(np.mean(inter), 1e-10)
        sent_intra_inter[str(li)] = {
            "intra": round(float(np.mean(intra)), 6),
            "inter": round(float(np.mean(inter)), 6),
            "ratio": round(ratio, 6),
        }
        print(f"    L{li}: intra={np.mean(intra):.4f}, inter={np.mean(inter):.4f}, ratio={ratio:.4f}")

    # 2. 有效维度
    print(f"\n{'='*60}")
    print(f"  [2] 句式版 有效维度")
    print(f"{'='*60}")
    sent_geo = {}
    for li in sample_layers:
        matrix = torch.stack([sent_encodings[w][li].float() for w in ALL_WORDS_SENT])
        n = matrix.shape[0]
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / max(n - 1, 1)
        eigenvalues, _ = torch.linalg.eigh(cov)
        top_eigs = eigenvalues[eigenvalues > 1e-10].flip(0).cpu().numpy()
        cumsum = np.cumsum(top_eigs) / max(top_eigs.sum(), 1e-10)
        eff_dim = int(np.searchsorted(cumsum, 0.90) + 1) if len(cumsum) > 0 else 0
        sent_geo[str(li)] = {"effective_dim_90": eff_dim}
        print(f"    L{li}: eff_dim={eff_dim}")

    # 3. 单token vs 句式 对比
    print(f"\n{'='*60}")
    print(f"  [3] 单token vs 句式 对比 (Qwen3)")
    print(f"{'='*60}")
    
    single_encodings = {}
    for word in ALL_WORDS_SENT:
        single_encodings[word] = {}
        for li in sample_layers:
            enc, _ = get_token_encoding(model, tokenizer, word, word, li)
            single_encodings[word][li] = enc

    print(f"\n  单token vs 句式 编码L2距离 (末层):")
    last_li = sample_layers[-1]
    for fk in fam_names:
        members = [w for s, w in SENTENCE_TEMPLATES[fk]]
        for w in members:
            v_single = single_encodings[w][last_li].float().cpu()
            v_sent = sent_encodings[w][last_li].float().cpu()
            l2 = torch.norm(v_single - v_sent).item()
            cos = torch.nn.functional.cosine_similarity(v_single.unsqueeze(0), v_sent.unsqueeze(0)).item()
            print(f"    {w:>12}: L2={l2:.4f}, cos={cos:.4f}")

    # 4. 距离矩阵对比：单token vs 句式
    print(f"\n{'='*60}")
    print(f"  [4] 距离矩阵结构对比 (末层)")
    print(f"{'='*60}")

    for label, encs in [("单token", single_encodings), ("句式", sent_encodings)]:
        n = len(ALL_WORDS_SENT)
        dm = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                v1 = encs[ALL_WORDS_SENT[i]][last_li].float().cpu()
                v2 = encs[ALL_WORDS_SENT[j]][last_li].float().cpu()
                cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                dm[i, j] = 1 - cos_sim
                dm[j, i] = dm[i, j]
        mean_d = np.mean(dm[np.triu_indices(n, k=1)])
        std_d = np.std(dm[np.triu_indices(n, k=1)])
        print(f"  {label}: mean_dist={mean_d:.4f}, std_dist={std_d:.4f}")

    # 5. 词性分类验证
    print(f"\n{'='*60}")
    print(f"  [5] 词性分类 in 句式 (末层)")
    print(f"{'='*60}")

    pos_encodings = {}
    for pos, entries in WORDCLASS_SENTENCES.items():
        for sent, word in entries:
            if pos not in pos_encodings:
                pos_encodings[pos] = {}
            enc, _ = get_token_encoding(model, tokenizer, sent, word, last_li)
            pos_encodings[pos][word] = enc

    for pos1 in WORDCLASS_SENTENCES:
        for pos2 in WORDCLASS_SENTENCES:
            if pos2 <= pos1:
                continue
            dists = []
            for w1 in pos_encodings[pos1]:
                for w2 in pos_encodings[pos2]:
                    v1 = pos_encodings[pos1][w1].float().cpu()
                    v2 = pos_encodings[pos2][w2].float().cpu()
                    d = 1 - torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                    dists.append(d)
            print(f"    {pos1:>5} vs {pos2:<5}: mean={np.mean(dists):.4f}")

    # 保存
    result = {
        "sent_intra_inter": sent_intra_inter,
        "sent_info_geometry": sent_geo,
    }
    out_path = os.path.join(OUTPUT_DIR, "stage550_qwen3_sentence.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    free_model(model)
    print(f"\n  保存到: {out_path}")


if __name__ == "__main__":
    main()
