"""
stage558: 动词和句法结构——不同词性的编码模式对比
目标：验证之前的发现（基于名词）是否适用于动词、形容词等
- 名词vs动词vs形容词vs功能词的编码结构差异
- 动词家族内聚性（动作类vs状态类vs感知类）
- 句法角色：主语vs宾语vs谓语的同词编码差异
- 句法结构对编码的影响（SVO vs SOV vs OSV）

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

# 不同词性的词
POS_WORDS = {
    "noun": ["apple", "bank", "river", "sun", "moon", "cat", "dog", "tree", "house", "car"],
    "verb": ["run", "eat", "think", "see", "make", "give", "break", "write", "read", "sleep"],
    "adj":  ["red", "big", "fast", "cold", "sweet", "dark", "loud", "soft", "warm", "old"],
    "func": ["the", "a", "is", "and", "or", "but", "in", "on", "to", "for"],
}

# 动词家族
VERB_FAMILIES = {
    "action": ["run", "jump", "hit", "break", "throw", "catch", "push", "pull", "cut", "kick"],
    "state":   ["is", "has", "want", "need", "like", "know", "believe", "think", "feel", "seem"],
    "perception": ["see", "hear", "feel", "smell", "taste", "watch", "notice", "listen", "touch", "observe"],
}

# 句法角色实验
SYNTAX_ROLE = {
    "subject": ["The cat sees the dog", "The dog sees the cat", "A man eats food"],
    "object":  ["The dog sees the cat", "The cat sees the dog", "Food feeds a man"],
}

# 句式结构
SENTENCE_STRUCTURES = {
    "SVO": ["The cat chased the mouse", "The boy kicked the ball", "She reads the book"],
    "passive": ["The mouse was chased by the cat", "The ball was kicked by the boy", "The book was read by her"],
    "question": ["Did the cat chase the mouse?", "Did the boy kick the ball?", "Does she read the book?"],
}


def compute_effective_dim(vectors):
    if len(vectors) < 2:
        return 1.0
    X = np.array(vectors)
    X = X - X.mean(axis=0)
    cov = (X.T @ X) / (len(X) - 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues = eigenvalues[::-1]
    total = eigenvalues.sum()
    if total < 1e-10:
        return 1.0
    cumsum = np.cumsum(eigenvalues) / total
    for k in range(1, len(cumsum)):
        if cumsum[k] >= 0.9:
            return float(k + 1)
    return float(len(eigenvalues))


def experiment1_pos_comparison(model, tokenizer, n_layers):
    """词性对比：名词vs动词vs形容词vs功能词"""
    print(f"\n{'='*60}")
    print(f"  实验1：不同词性的编码结构对比")
    print(f"{'='*60}")

    sample_layers = evenly_spaced_layers(model, count=7)

    for pos_tag, words in POS_WORDS.items():
        # 收集各层编码
        layer_encodings = {li: [] for li in sample_layers}
        for word in words:
            encoded = encode_to_device(model, tokenizer, word)
            with torch.no_grad():
                out = model(**encoded, output_hidden_states=True)
            for li in sample_layers:
                enc = out.hidden_states[li][0, -1].cpu().float().numpy()
                layer_encodings[li].append(enc)

        print(f"\n  {pos_tag.upper()} (n={len(words)}):")
        print(f"    层  | 有效维度 | mean_cos_d | std_cos_d")
        for li in sample_layers:
            encs = np.array(layer_encodings[li])
            eff_dim = compute_effective_dim(encs)

            # 计算所有词对的cos距离
            dists = []
            for i in range(len(encs)):
                for j in range(i + 1, len(encs)):
                    cos_d = 1 - F.cosine_similarity(
                        torch.tensor(encs[i]), torch.tensor(encs[j]), dim=0
                    ).item()
                    dists.append(cos_d)

            print(f"    L{li:2d} | {eff_dim:8.1f} | {np.mean(dists):10.4f} | {np.std(dists):.4f}")

    return {}


def experiment2_verb_family(model, tokenizer, n_layers):
    """动词家族内聚性"""
    print(f"\n{'='*60}")
    print(f"  实验2：动词家族内聚性（action vs state vs perception）")
    print(f"{'='*60}")

    sample_layers = evenly_spaced_layers(model, count=7)
    last_layer = sample_layers[-1]

    family_encodings = {}
    for fam_name, fam_words in VERB_FAMILIES.items():
        encs = []
        for w in fam_words:
            encoded = encode_to_device(model, tokenizer, w)
            with torch.no_grad():
                out = model(**encoded, output_hidden_states=True)
            encs.append(out.hidden_states[last_layer][0, -1].cpu().float().numpy())
        family_encodings[fam_name] = np.array(encs)

    # 计算intra/inter
    print(f"  末层 L{last_layer}:")
    all_fams = list(family_encodings.keys())
    for fam in all_fams:
        encs = family_encodings[fam]
        intra_dists = []
        for i in range(len(encs)):
            for j in range(i + 1, len(encs)):
                d = 1 - F.cosine_similarity(torch.tensor(encs[i]), torch.tensor(encs[j]), dim=0).item()
                intra_dists.append(d)

        inter_dists = []
        for other_fam in all_fams:
            if other_fam == fam:
                continue
            other_encs = family_encodings[other_fam]
            for i in range(len(encs)):
                for j in range(len(other_encs)):
                    d = 1 - F.cosine_similarity(torch.tensor(encs[i]), torch.tensor(other_encs[j]), dim=0).item()
                    inter_dists.append(d)

        ratio = np.mean(intra_dists) / np.mean(inter_dists) if np.mean(inter_dists) > 1e-10 else 0
        print(f"    {fam:12s}: intra={np.mean(intra_dists):.4f}, inter={np.mean(inter_dists):.4f}, ratio={ratio:.4f}")

    # 有效维度
    print(f"\n  末层有效维度:")
    for fam in all_fams:
        ed = compute_effective_dim(family_encodings[fam])
        print(f"    {fam:12s}: eff_dim={ed:.1f}")

    return {}


def experiment3_syntax_role(model, tokenizer, n_layers):
    """句法角色：主语vs宾语同词编码差异"""
    print(f"\n{'='*60}")
    print(f"  实验3：句法角色对编码的影响")
    print(f"{'='*60}")

    # "cat"作主语 vs "cat"作宾语
    pairs = [
        ("The cat chased the mouse", "The mouse chased the cat", "cat", "subject", "object"),
        ("The dog bit the man", "The man bit the dog", "dog", "subject", "object"),
        ("She saw the bird", "The bird saw her", "She/bird", "subject", "object"),
    ]

    sample_layers = evenly_spaced_layers(model, count=7)

    for sent1, sent2, target, role1, role2 in pairs:
        enc1 = encode_to_device(model, tokenizer, sent1)
        enc2 = encode_to_device(model, tokenizer, sent2)

        ids1 = enc1["input_ids"][0].tolist()
        ids2 = enc2["input_ids"][0].tolist()
        tokens1 = [tokenizer.convert_ids_to_tokens(t) for t in ids1]
        tokens2 = [tokenizer.convert_ids_to_tokens(t) for t in ids2]

        # 找目标词位置
        t1_safe = [t.encode('ascii', 'replace').decode() for t in tokens1]
        t2_safe = [t.encode('ascii', 'replace').decode() for t in tokens2]

        print(f"\n  '{target}' as {role1} vs {role2}:")
        print(f"    '{sent1}' -> {t1_safe}")
        print(f"    '{sent2}' -> {t2_safe}")

        with torch.no_grad():
            out1 = model(**enc1, output_hidden_states=True)
            out2 = model(**enc2, output_hidden_states=True)

        # 取最后一个内容词的位置
        pos1 = len(ids1) - 1
        pos2 = len(ids2) - 1

        print(f"    层  | cos_d (role effect)")
        for li in sample_layers:
            e1 = out1.hidden_states[li][0, pos1].cpu().float()
            e2 = out2.hidden_states[li][0, pos2].cpu().float()
            cos_d = 1 - F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0), dim=1).item()
            print(f"    L{li:2d} | {cos_d:.4f}")

    return {}


def experiment4_sentence_structure(model, tokenizer, n_layers):
    """句式结构对编码的影响"""
    print(f"\n{'='*60}")
    print(f"  实验4：句式结构（SVO vs passive vs question）")
    print(f"{'='*60}")

    # 对比：同一语义内容，不同句式
    triplets = [
        ("The cat chased the mouse", "The mouse was chased by the cat",
         "Did the cat chase the mouse?", "cat/mouse chase"),
        ("The boy kicked the ball", "The ball was kicked by the boy",
         "Did the boy kick the ball?", "boy/ball kick"),
    ]

    sample_layers = evenly_spaced_layers(model, count=7)

    for svo, passive, question, desc in triplets:
        print(f"\n  [{desc}]:")
        sentences = {"SVO": svo, "passive": passive, "question": question}

        encodings = {}
        for label, sent in sentences.items():
            encoded = encode_to_device(model, tokenizer, sent)
            with torch.no_grad():
                out = model(**encoded, output_hidden_states=True)
            encodings[label] = out.hidden_states

        # 逐层比较：SVO vs passive, SVO vs question
        print(f"    层  | SVO vs passive | SVO vs question | passive vs question")
        n_tokens = min(
            encodings["SVO"].shape[1],
            encodings["passive"].shape[1],
            encodings["question"].shape[1]
        )

        for li in sample_layers:
            svo_h = encodings["SVO"][li][0].cpu().float()
            pas_h = encodings["passive"][li][0].cpu().float()
            que_h = encodings["question"][li][0].cpu().float()

            # 整句的平均cos距离
            cos_sp = 1 - F.cosine_similarity(svo_h.mean(dim=0, keepdim=True),
                                              pas_h.mean(dim=0, keepdim=True), dim=1).item()
            cos_sq = 1 - F.cosine_similarity(svo_h.mean(dim=0, keepdim=True),
                                              que_h.mean(dim=0, keepdim=True), dim=1).item()
            cos_pq = 1 - F.cosine_similarity(pas_h.mean(dim=0, keepdim=True),
                                              que_h.mean(dim=0, keepdim=True), dim=1).item()
            print(f"    L{li:2d} | {cos_sp:15.4f} | {cos_sq:15.4f} | {cos_pq:15.4f}")

    return {}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  stage558: 动词和句法结构——不同词性编码模式")
    print("=" * 60)

    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    print(f"  Qwen3 n_layers={n_layers}")

    try:
        r1 = experiment1_pos_comparison(model, tokenizer, n_layers)
        r2 = experiment2_verb_family(model, tokenizer, n_layers)
        r3 = experiment3_syntax_role(model, tokenizer, n_layers)
        r4 = experiment4_sentence_structure(model, tokenizer, n_layers)
    finally:
        free_model(model)

    out_path = os.path.join(OUTPUT_DIR, "stage558_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({}, f, ensure_ascii=False, indent=2)
    print(f"\n  results saved: {out_path}")
    print(f"\n  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
