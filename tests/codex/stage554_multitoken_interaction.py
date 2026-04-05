"""
stage554: 多token交互编码结构
目标：逐层追踪修饰语对目标词编码的贡献
- "red apple" vs "apple" → 逐层cosine距离变化曲线
- 不同修饰语的贡献差异（颜色/大小/状态/情感）
- Attention flow：red→apple的attention权重逐层变化
- 邻近名词 vs 远隔名词的交互差异

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

# ========== 实验1：修饰语贡献逐层追踪 ==========
ADJECTIVE_TARGET = "apple"
ADJECTIVES = {
    "red": "color",
    "green": "color",
    "blue": "color",
    "big": "size",
    "small": "size",
    "fresh": "quality",
    "rotten": "quality",
    "sweet": "taste",
    "delicious": "taste",
    "expensive": "value",
    "wild": "origin",
    "poisonous": "danger",
}

def experiment1_modifier_contribution(model, tokenizer, n_layers):
    """逐层追踪修饰语对目标词编码的贡献"""
    print(f"\n{'='*60}")
    print(f"  实验1：修饰语对 '{ADJECTIVE_TARGET}' 编码的逐层贡献")
    print(f"{'='*60}")

    # 先获取baseline: "apple"
    encoded_base = encode_to_device(model, tokenizer, "apple")
    with torch.no_grad():
        base_out = model(**encoded_base, output_hidden_states=True)
    base_layers = n_layers
    all_layers = len(base_out.hidden_states)

    sample_layers = list(range(min(all_layers, 10)))  # L0-L9细粒度
    if all_layers > 10:
        sample_layers += list(range(10, all_layers, max(1, (all_layers - 10) // 10)))
    if (all_layers - 1) not in sample_layers:
        sample_layers.append(all_layers - 1)
    sample_layers = sorted(set(sample_layers))

    # baseline: 取最后一个token("apple"应该就是最后一个)
    base_token_ids = encoded_base["input_ids"][0].tolist()
    target_pos = len(base_token_ids) - 1  # apple的位置

    # 获取各层的target编码
    base_encodings = {}
    with torch.no_grad():
        out = model(**encoded_base, output_hidden_states=True)
        for li in sample_layers:
            enc = out.hidden_states[li][0, target_pos].cpu().float().numpy()
            base_encodings[li] = enc

    # 每个修饰语
    results = {}
    for adj, category in ADJECTIVES.items():
        phrase = f"{adj} {ADJECTIVE_TARGET}"
        encoded = encode_to_device(model, tokenizer, phrase)
        token_ids = encoded["input_ids"][0].tolist()

        # 找target词的位置
        target_pos_phrase = None
        for pos in range(len(token_ids) - 1, -1, -1):
            tok_str = tokenizer.convert_ids_to_tokens(token_ids[pos]).lower()
            if "apple" in tok_str:
                target_pos_phrase = pos
                break

        if target_pos_phrase is None:
            # fallback: last token
            target_pos_phrase = len(token_ids) - 1

        # 逐层计算距离
        layer_distances = []
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True)
            for li in sample_layers:
                enc = out.hidden_states[li][0, target_pos_phrase].cpu().float().numpy()
                cos_d = 1 - F.cosine_similarity(
                    torch.tensor(enc), torch.tensor(base_encodings[li]), dim=0
                ).item()
                l2_d = float(np.linalg.norm(enc - base_encodings[li]))
                layer_distances.append({
                    "layer": li, "cos_d": round(cos_d, 4), "l2_d": round(l2_d, 2)
                })

        results[adj] = {"category": category, "distances": layer_distances}
        max_cos = max(d["cos_d"] for d in layer_distances)
        peak_layer = [d for d in layer_distances if d["cos_d"] == max_cos][0]
        print(f"  {adj:12s} ({category:8s}): peak cos_d={max_cos:.4f} @ L{peak_layer['layer']}, "
              f"末层 cos_d={layer_distances[-1]['cos_d']:.4f}")

    # 按类别汇总
    print(f"\n  按类别汇总（末层 cos_d 平均）:")
    cat_stats = {}
    for adj, data in results.items():
        cat = data["category"]
        if cat not in cat_stats:
            cat_stats[cat] = []
        cat_stats[cat].append(data["distances"][-1]["cos_d"])

    for cat, vals in sorted(cat_stats.items()):
        print(f"    {cat:10s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

    return results


# ========== 实验2：Attention Flow分析 ==========
def experiment2_attention_flow(model, tokenizer, n_layers):
    """追踪修饰语→目标词的attention权重"""
    print(f"\n{'='*60}")
    print(f"  实验2：Attention Flow（修饰语→目标词）")
    print(f"{'='*60}")

    phrases = [
        "red apple",
        "big apple",
        "the red apple",
        "the big red apple",
        "an apple",
    ]

    for phrase in phrases:
        encoded = encode_to_device(model, tokenizer, phrase)
        token_ids = encoded["input_ids"][0].tolist()
        tokens = [tokenizer.convert_ids_to_tokens(t) for t in token_ids]
        tokens_safe = [t.encode('ascii', 'replace').decode() for t in tokens]
        n_tokens = len(tokens)

        # 找target位置
        target_pos = None
        for pos in range(n_tokens - 1, -1, -1):
            if "apple" in tokens[pos].lower():
                target_pos = pos
                break
        if target_pos is None:
            target_pos = n_tokens - 1

        # 获取attention weights
        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True)

        # 逐层看: 从target_pos看向其他位置的attention
        print(f"\n  '{phrase}' (target_pos={target_pos}):")
        print(f"    tokens: {tokens_safe}")

        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            # 每层、每个head的attention
            n_attn_layers = len(outputs.attentions)
            sample = list(range(0, n_attn_layers, max(1, n_attn_layers // 8)))
            if (n_attn_layers - 1) not in sample:
                sample.append(n_attn_layers - 1)

            for li in sample:
                attn = outputs.attentions[li][0]  # (n_heads, seq_len, seq_len)
                # target_pos 行：target看向谁
                avg_attn = attn[:, target_pos, :].float().mean(dim=0).cpu().numpy()
                # 找最大的source
                sorted_idx = np.argsort(-avg_attn)
                top3 = [(tokens_safe[i], f"{avg_attn[i]:.4f}") for i in sorted_idx[:3] if i < n_tokens]
                print(f"    L{li:2d}: target attends to → {top3}")
        else:
            print(f"    (无attention输出)")

    return {}


# ========== 实验3：邻近 vs 远隔名词交互 ==========
def experiment3_proximity_interaction(model, tokenizer, n_layers):
    """邻近名词 vs 远隔名词的编码交互差异"""
    print(f"\n{'='*60}")
    print(f"  实验3：邻近 vs 远隔名词交互")
    print(f"{'='*60}")

    sentences = [
        ("apple banana", "邻近名词"),
        ("apple and banana", "conj连接"),
        ("apple is not banana", "否定连接"),
        ("apple is a fruit and banana is yellow", "长距离"),
        ("I like apple but I prefer banana", "对比长距离"),
        ("cat dog", "邻近名词-动物"),
        ("sun moon", "邻近名词-天体"),
        ("sun is bright and moon is dark", "远隔-天体"),
    ]

    sample_layers = evenly_spaced_layers(model, count=7)

    for sent, desc in sentences:
        encoded = encode_to_device(model, tokenizer, sent)
        token_ids = encoded["input_ids"][0].tolist()
        tokens = [tokenizer.convert_ids_to_tokens(t) for t in token_ids]
        tokens_safe = [t.encode('ascii', 'replace').decode() for t in tokens]
        n_tokens = len(tokens)

        # 找两个目标词的位置
        word1, word2 = sent.split()[:2]
        pos1 = pos2 = None
        for pos in range(n_tokens):
            tok = tokens[pos].lower()
            if word1 in tok and pos1 is None:
                pos1 = pos
            elif word2 in tok:
                pos2 = pos

        if pos1 is None or pos2 is None:
            print(f"  [{desc}] '{sent}': 找不到两个目标词，跳过")
            continue

        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True)

        # L0和末层的cos距离
        for li in [0, sample_layers[-1]]:
            e1 = out.hidden_states[li][0, pos1].cpu().float().numpy()
            e2 = out.hidden_states[li][0, pos2].cpu().float().numpy()
            cos_d = 1 - F.cosine_similarity(
                torch.tensor(e1), torch.tensor(e2), dim=0
            ).item()
            l2_d = float(np.linalg.norm(e1 - e2))

        # 只打印末层
        li_last = sample_layers[-1]
        e1 = out.hidden_states[li_last][0, pos1].cpu().float().numpy()
        e2 = out.hidden_states[li_last][0, pos2].cpu().float().numpy()
        cos_d = 1 - F.cosine_similarity(
            torch.tensor(e1), torch.tensor(e2), dim=0
        ).item()

        # 对比单token编码距离
        enc1 = encode_to_device(model, tokenizer, word1)
        enc2 = encode_to_device(model, tokenizer, word2)
        with torch.no_grad():
            o1 = model(**enc1, output_hidden_states=True)
            o2 = model(**enc2, output_hidden_states=True)
        solo_e1 = o1.hidden_states[li_last][0, -1].cpu().float().numpy()
        solo_e2 = o2.hidden_states[li_last][0, -1].cpu().float().numpy()
        solo_cos_d = 1 - F.cosine_similarity(
            torch.tensor(solo_e1), torch.tensor(solo_e2), dim=0
        ).item()

        diff = cos_d - solo_cos_d
        print(f"  [{desc:12s}] pos1={pos1}, pos2={pos2}: "
              f"句式cos_d={cos_d:.4f}, 单token={solo_cos_d:.4f}, "
              f"差={diff:+.4f} ({'句式更远' if diff > 0 else '句式更近'})")

    return {}


# ========== 主函数 ==========
def main():
    t0 = time.time()
    print("=" * 60)
    print("  stage554: 多token交互编码结构")
    print("=" * 60)

    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    print(f"  Qwen3 n_layers={n_layers}")

    try:
        r1 = experiment1_modifier_contribution(model, tokenizer, n_layers)
        r2 = experiment2_attention_flow(model, tokenizer, n_layers)
        r3 = experiment3_proximity_interaction(model, tokenizer, n_layers)
    finally:
        free_model(model)

    # 保存结果
    results = {
        "modifier_contribution": {k: v for k, v in r1.items()},
    }
    out_path = os.path.join(OUTPUT_DIR, "stage554_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_path}")

    print(f"\n  总耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
