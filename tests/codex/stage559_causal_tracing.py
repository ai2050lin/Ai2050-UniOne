"""
stage559: 信息流追踪——因果追踪red→apple的信息流
目标：用causal tracing方法追踪"red"的信息如何流向"apple"的编码
- 逐步替换各层的"red" token信息，观察"apple"编码的变化
- 识别信息流的关键层（信息传递的瓶颈层）
- 对比正向(red→apple) vs 反向(apple→red)信息流
- 对比不同修饰语的信息流模式

使用Qwen3。
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

INFO_FLOW_EXPERIMENTS = [
    ("red apple", "red", "apple", "color→noun"),
    ("big house", "big", "house", "size→noun"),
    ("fast car", "fast", "car", "speed→noun"),
    ("delicious food", "delicious", "food", "taste→noun"),
    ("the red apple", "red", "apple", "color→noun(longer)"),
    ("the big red apple is sweet", "red", "apple", "color→noun(sentence)"),
]


def find_token_pos(tokenizer, token_ids, word):
    tokens = [tokenizer.convert_ids_to_tokens(t) for t in token_ids]
    for pos in range(len(tokens)):
        if word in tokens[pos].lower():
            return pos
    return None


def trace_info_flow(model, tokenizer, sentence, source_word, target_word):
    """
    因果追踪：逐层比较"有source" vs "无source"时target编码的差异
    方法：
    1. 正常前向传播sentence → 得到target词各层编码(clean)
    2. 构造"remove source"的句子 → 前向传播 → 得到target词各层编码(no_source)
    3. 逐层计算 |clean - no_source| 作为信息流指标
    """
    # 正常句子
    clean_encoded = encode_to_device(model, tokenizer, sentence)
    clean_ids = clean_encoded["input_ids"][0].tolist()
    clean_tokens = [tokenizer.convert_ids_to_tokens(t) for t in clean_ids]

    source_pos = find_token_pos(tokenizer, clean_ids, source_word)
    target_pos = find_token_pos(tokenizer, clean_ids, target_word)

    if source_pos is None or target_pos is None:
        return None

    # 构造"remove source"版本：用"a"替代source词
    no_source_sentence = sentence.replace(source_word, "the")
    # 确保token数相同（简化处理）
    no_source_encoded = encode_to_device(model, tokenizer, no_source_sentence)
    no_source_ids = no_source_encoded["input_ids"][0].tolist()
    no_source_target_pos = find_token_pos(tokenizer, no_source_ids, target_word)

    if no_source_target_pos is None:
        no_source_target_pos = len(no_source_ids) - 1

    # 前向传播
    with torch.no_grad():
        clean_out = model(**clean_encoded, output_hidden_states=True)
        no_source_out = model(**no_source_encoded, output_hidden_states=True)

    all_layers = len(clean_out.hidden_states)
    flow_data = []

    for li in range(all_layers):
        clean_h = clean_out.hidden_states[li][0, target_pos].cpu().float()
        no_source_h = no_source_out.hidden_states[li][0, no_source_target_pos].cpu().float()

        cos_d = 1 - F.cosine_similarity(clean_h.unsqueeze(0), no_source_h.unsqueeze(0), dim=1).item()
        l2_d = float(torch.norm(clean_h - no_source_h))

        # 逐神经元差异
        neuron_diff = (clean_h - no_source_h).abs()
        top_diff_val = float(neuron_diff.max())
        top_diff_idx = int(neuron_diff.argmax())
        mean_diff = float(neuron_diff.mean())

        flow_data.append({
            "layer": li, "cos_d": cos_d, "l2_d": l2_d,
            "top_diff_val": top_diff_val, "top_diff_idx": top_diff_idx,
            "mean_diff": mean_diff,
        })

    return {
        "source_pos": source_pos, "target_pos": target_pos,
        "clean_tokens": [t.encode('ascii', 'replace').decode() for t in clean_tokens],
        "flow": flow_data,
    }


def experiment1_layer_flow(model, tokenizer, n_layers):
    """逐层信息流追踪"""
    print(f"\n{'='*60}")
    print(f"  实验1：逐层信息流（source→target编码差异）")
    print(f"{'='*60}")

    results = {}

    for sent, src, tgt, desc in INFO_FLOW_EXPERIMENTS:
        print(f"\n  [{desc}] '{sent}' ({src}→{tgt}):")
        data = trace_info_flow(model, tokenizer, sent, src, tgt)
        if data is None:
            print(f"    SKIP: 找不到token位置")
            continue

        print(f"    source_pos={data['source_pos']}, target_pos={data['target_pos']}")
        print(f"    tokens: {data['clean_tokens']}")

        # 打印关键层
        print(f"    层  | cos_d  | L2_diff | mean_diff | top_diff_idx")
        for fd in data["flow"]:
            li = fd["layer"]
            if li % 5 == 0 or li == n_layers - 1 or fd["cos_d"] > 0.5:
                print(f"    L{li:2d} | {fd['cos_d']:.4f} | {fd['l2_d']:7.1f} | {fd['mean_diff']:9.4f} | {fd['top_diff_idx']}")

        # 找信息流峰值层
        peak = max(data["flow"], key=lambda x: x["cos_d"])
        print(f"    信息流峰值: L{peak['layer']}, cos_d={peak['cos_d']:.4f}")

        results[f"{desc}"] = {
            "peak_layer": peak["layer"],
            "peak_cos_d": round(peak["cos_d"], 4),
        }

    return results


def experiment2_bidirectional_flow(model, tokenizer, n_layers):
    """双向信息流：red→apple vs apple→red"""
    print(f"\n{'='*60}")
    print(f"  实验2：双向信息流")
    print(f"{'='*60}")

    pairs = [
        ("red apple", "red", "apple"),
        ("big house", "big", "house"),
        ("fast car", "fast", "car"),
    ]

    for sent, w1, w2 in pairs:
        # w1→w2
        fwd = trace_info_flow(model, tokenizer, sent, w1, w2)
        # w2→w1 (反转)
        bwd = trace_info_flow(model, tokenizer, sent, w2, w1)

        if fwd is None or bwd is None:
            continue

        print(f"\n  '{sent}':")
        print(f"    层  | {w1}→{w2} cos_d | {w2}→{w1} cos_d | 差值")
        for i in range(len(fwd["flow"])):
            li = fwd["flow"][i]["layer"]
            if li % 5 == 0 or li == n_layers - 1:
                f_cos = fwd["flow"][i]["cos_d"]
                b_cos = bwd["flow"][i]["cos_d"]
                print(f"    L{li:2d} | {f_cos:.4f}          | {b_cos:.4f}          | {f_cos-b_cos:+.4f}")

    return {}


def experiment3_cumulative_flow(model, tokenizer, n_layers):
    """累积信息流：逐层累积source的信息"""
    print(f"\n{'='*60}")
    print(f"  实验3：累积信息流（逐层累积）")
    print(f"{'='*60}")

    # 方法：逐层将clean的source token信息注入no_source的hidden state
    sent = "the red apple is sweet"
    src_word = "red"
    tgt_word = "apple"

    clean_encoded = encode_to_device(model, tokenizer, sent)
    no_source_sentence = sent.replace(src_word, "the")
    no_source_encoded = encode_to_device(model, tokenizer, no_source_sentence)

    clean_ids = clean_encoded["input_ids"][0].tolist()
    target_pos = find_token_pos(tokenizer, clean_ids, tgt_word)
    source_pos = find_token_pos(tokenizer, clean_ids, src_word)

    no_source_ids = no_source_encoded["input_ids"][0].tolist()
    no_target_pos = find_token_pos(tokenizer, no_source_ids, tgt_word)

    print(f"  sentence: '{sent}'")
    print(f"  source: '{src_word}' (pos={source_pos}), target: '{tgt_word}' (pos={target_pos})")

    with torch.no_grad():
        clean_out = model(**clean_encoded, output_hidden_states=True)
        no_source_out = model(**no_source_encoded, output_hidden_states=True)

    # 逐层：用clean的source token隐藏状态替换no_source的，然后比较target
    print(f"\n  逐层替换source token hidden state:")
    print(f"  层  | target cos_d | gain")
    prev_cos_d = 0
    for li in range(min(n_layers, 36)):
        # source token在clean和no_source中的隐藏状态
        clean_src_h = clean_out.hidden_states[li][0, source_pos].cpu().float()
        no_source_tgt_h = no_source_out.hidden_states[li][0, no_target_pos].cpu().float()
        clean_tgt_h = clean_out.hidden_states[li][0, target_pos].cpu().float()

        # 比较no_source_target vs clean_target（原始差异）
        cos_d = 1 - F.cosine_similarity(no_source_tgt_h.unsqueeze(0),
                                          clean_tgt_h.unsqueeze(0), dim=1).item()

        if li % 4 == 0 or li == n_layers - 1:
            print(f"  L{li:2d} | {cos_d:.4f}       | {cos_d - prev_cos_d:+.4f}")
            prev_cos_d = cos_d

    # 信息累积曲线
    print(f"\n  信息传递指标（target差异随层变化）:")
    print(f"  层  | target_L2_diff | source_self_cos_d")
    for li in range(min(n_layers, 36)):
        no_tgt = no_source_out.hidden_states[li][0, no_target_pos].cpu().float()
        cl_tgt = clean_out.hidden_states[li][0, target_pos].cpu().float()
        l2_diff = float(torch.norm(no_tgt - cl_tgt))

        no_src = no_source_out.hidden_states[li][0, min(source_pos, no_source_ids.__len__()-1)].cpu().float()
        cl_src = clean_out.hidden_states[li][0, source_pos].cpu().float()
        src_cos = 1 - F.cosine_similarity(no_src.unsqueeze(0), cl_src.unsqueeze(0), dim=1).item()

        if li % 5 == 0 or li == n_layers - 1:
            print(f"  L{li:2d} | {l2_diff:14.1f} | {src_cos:.4f}")

    return {}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  stage559: 信息流追踪——causal tracing")
    print("=" * 60)

    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    print(f"  Qwen3 n_layers={n_layers}")

    try:
        r1 = experiment1_layer_flow(model, tokenizer, n_layers)
        r2 = experiment2_bidirectional_flow(model, tokenizer, n_layers)
        r3 = experiment3_cumulative_flow(model, tokenizer, n_layers)
    finally:
        free_model(model)

    out_path = os.path.join(OUTPUT_DIR, "stage559_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(r1, f, ensure_ascii=False, indent=2)
    print(f"\n  results saved: {out_path}")
    print(f"\n  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
