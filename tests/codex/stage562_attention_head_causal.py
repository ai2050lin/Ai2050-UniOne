"""
stage562: L8 Attention Head重要性评分——逐头消融寻找因果头
目标：L8是消歧和信息流的峰值层，但L8有多个attention head，
需要找出哪些head真正因果负责消歧。

方法：
1. 遍历L8的所有head，逐个归零（zero-ablation）
2. 比较消融前后bank/apple的末层编码变化
3. 对每个head计算因果效力分数

使用Qwen3。
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_language_shared import load_qwen3_model, discover_layers
from multimodel_language_shared import encode_to_device, free_model

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'glm5', 'temp')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_target_position(tokenizer, token_ids, target_word):
    tokens = [tokenizer.convert_ids_to_tokens(t) for t in token_ids]
    for pos in range(len(tokens) - 1, -1, -1):
        if target_word in tokens[pos].lower():
            return pos
    return len(tokens) - 1


def get_attention_module(model, layer_idx):
    """获取指定层的attention模块"""
    layers = discover_layers(model)
    layer = layers[layer_idx]
    # Qwen3的attention在不同位置，尝试多种命名
    if hasattr(layer, 'self_attn'):
        return layer.self_attn
    elif hasattr(layer, 'attn'):
        return layer.attn
    elif hasattr(layer, 'attention'):
        return layer.attention
    else:
        print(f"  WARNING: layer {layer_idx} has no attention module, attrs: {[a for a in dir(layer) if not a.startswith('_')]}")
        return None


def create_head_ablation_hook(head_idx, num_heads, ablation_type='zero'):
    """创建一个hook来消融特定的attention head输出"""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output

        # attn_output shape: (batch, seq_len, num_heads * head_dim)
        head_dim = attn_output.shape[-1] // num_heads
        batch_size = attn_output.shape[0]
        seq_len = attn_output.shape[1]

        modified = attn_output.clone()
        start = head_idx * head_dim
        end = start + head_dim
        modified[:, :, start:end] = 0  # zero ablation

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    return hook_fn


def experiment1_head_ablation(model, tokenizer, n_layers):
    """实验1：L8逐head消融——找因果head"""
    print(f"\n{'='*60}")
    print(f"  实验1：L8 Attention Head逐个消融")
    print(f"{'='*60}")

    target_layer = min(8, n_layers - 1)
    attn_module = get_attention_module(model, target_layer)

    if attn_module is None:
        print("  SKIP: 无法获取attention模块")
        return {}

    # 确定head数量
    num_heads = None
    for attr_name in ['num_heads', 'n_head', 'num_attention_heads', 'config']:
        if hasattr(attn_module, attr_name):
            val = getattr(attn_module, attr_name)
            if isinstance(val, int) and val > 0:
                num_heads = val
                break
    if num_heads is None and hasattr(model, 'config'):
        for attr_name in ['num_attention_heads', 'n_head', 'num_heads']:
            if hasattr(model.config, attr_name):
                num_heads = getattr(model.config, attr_name)
                break
    if num_heads is None:
        num_heads = 32  # Qwen3-4B默认
    print(f"  detected num_heads={num_heads}")

    source_ctx = "The river bank was muddy and steep"
    target_ctx = "The bank gave me a loan today"
    target_word = "bank"

    source_enc = encode_to_device(model, tokenizer, source_ctx)
    target_enc = encode_to_device(model, tokenizer, target_ctx)
    source_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), target_word)
    target_pos = find_target_position(tokenizer, target_enc["input_ids"][0].tolist(), target_word)

    # 正常前向传播
    with torch.no_grad():
        source_out = model(**source_enc, output_hidden_states=True)
        target_out = model(**target_enc, output_hidden_states=True)

    source_last = source_out.hidden_states[-1][0, source_pos].cpu().float()
    target_last = target_out.hidden_states[-1][0, target_pos].cpu().float()

    # 基线：bank消歧度（source vs target的cosine距离）
    cos_baseline = F.cosine_similarity(source_last.unsqueeze(0), target_last.unsqueeze(0), dim=1).item()
    print(f"  baseline disambiguation cos(source, target) = {cos_baseline:.4f}")

    # 对source语境中bank编码的L2范数（完整性检查）
    source_norm = torch.norm(source_last).item()
    print(f"  source last hidden norm = {source_norm:.4f}")

    # 逐head消融
    print(f"\n  Head ablation results (zeroing each head at L{target_layer}):")
    print(f"  {'Head':>4s} | {'cos_dis':>7s} | {'delta':>7s} | {'norm_ratio':>10s}")
    print(f"  {'-'*4} | {'-'*7} | {'-'*7} | {'-'*10}")

    head_scores = []
    for h in range(num_heads):
        hook = create_head_ablation_hook(h, num_heads)
        handle = attn_module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                ablated_out = model(**source_enc, output_hidden_states=True)
        finally:
            handle.remove()

        ablated_last = ablated_out.hidden_states[-1][0, source_pos].cpu().float()
        cos_dis = F.cosine_similarity(ablated_last.unsqueeze(0), target_last.unsqueeze(0), dim=1).item()
        delta = cos_dis - cos_baseline
        norm_ratio = torch.norm(ablated_last).item() / source_norm

        head_scores.append({"head": h, "cos_disambig": round(cos_dis, 4), "delta": round(delta, 4), "norm_ratio": round(norm_ratio, 4)})

        if delta > 0.005 or abs(delta) > 0.01:  # 只打印有意义的
            marker = "***" if abs(delta) > 0.05 else "   "
            print(f"  H{h:3d}  | {cos_dis:7.4f} | {delta:+7.4f} | {norm_ratio:10.4f} {marker}")

    # 统计
    deltas = [s["delta"] for s in head_scores]
    significant = [s for s in head_scores if abs(s["delta"]) > 0.005]
    print(f"\n  Summary: {len(significant)}/{num_heads} heads with |delta| > 0.005")
    if significant:
        top_head = max(significant, key=lambda x: abs(x["delta"]))
        print(f"  Most impactful: Head {top_head['head']}, delta={top_head['delta']:+.4f}")

    return {
        "num_heads": num_heads,
        "baseline_cos": round(cos_baseline, 4),
        "significant_heads": significant,
        "all_scores": head_scores,
    }


def experiment2_head_attention_patterns(model, tokenizer, n_layers):
    """实验2：提取L8各head的attention pattern——bank关注哪些token"""
    print(f"\n{'='*60}")
    print(f"  实验2：L8 Attention Patterns——bank关注什么")
    print(f"{'='*60}")

    target_layer = min(8, n_layers - 1)

    sentences = {
        "bank_river": "The river bank was muddy and steep",
        "bank_finance": "The bank gave me a loan today",
        "apple_fruit": "The red apple is sweet and delicious",
        "apple_company": "Apple released the new iPhone today",
    }

    results = {}
    for name, sentence in sentences.items():
        encoded = encode_to_device(model, tokenizer, sentence)
        token_ids = encoded["input_ids"][0].tolist()
        tokens = [repr(tokenizer.convert_ids_to_tokens(t)) for t in token_ids]
        target_words = name.split("_")[0]
        target_pos = find_target_position(tokenizer, token_ids, target_words)

        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True, output_attentions=True)

        # 提取attention
        if hasattr(out, 'attentions') and out.attentions is not None:
            attn = out.attentions[target_layer]  # (batch, num_heads, seq, seq)
            attn_np = attn[0].float().cpu().numpy()
        else:
            print(f"  WARNING: No attention output for '{name}'")
            continue

        print(f"\n  '{sentence}'")
        print(f"  target '{target_words}' at pos {target_pos}")

        # 找top-3 attended tokens（从target_pos的角度看）
        print(f"  Top-3 attended tokens by each influential head:")
        num_heads = attn_np.shape[0]
        for h in range(min(num_heads, 32)):
            attn_weights = attn_np[h, target_pos, :]
            top3 = np.argsort(-attn_weights)[:3]
            if attn_weights[top3[0]] > 0.15:  # 只打印有意义的
                top_str = ", ".join([f"pos{t}({attn_weights[t]:.3f})" for t in top3])
                print(f"    H{h:2d}: {top_str}")

        # 找对target_pos贡献最大的head
        max_attn_per_head = attn_np[:, target_pos, :].max(axis=1)  # 每个head在target_pos上的最大attention
        top_heads = np.argsort(-max_attn_per_head)[:5]
        print(f"  Heads with strongest attention to '{target_words}':")
        for h in top_heads:
            top_token = np.argmax(attn_np[h, target_pos, :])
            print(f"    H{h}: max_attn={max_attn_per_head[h]:.3f} -> {tokens[top_token]}")

        results[name] = {"tokens": tokens, "target_pos": target_pos}

    return results


def experiment3_head_parity_check(model, tokenizer, n_layers):
    """实验3：head-level parity check——消歧head在两个语境中是否不同"""
    print(f"\n{'='*60}")
    print(f"  实验3：Head-Level消歧模式——两个语境的attention差异")
    print(f"{'='*60}")

    target_layer = min(8, n_layers - 1)

    source_ctx = "The river bank was muddy and steep"
    target_ctx = "The bank gave me a loan today"

    source_enc = encode_to_device(model, tokenizer, source_ctx)
    target_enc = encode_to_device(model, tokenizer, target_ctx)

    with torch.no_grad():
        source_out = model(**source_enc, output_hidden_states=True, output_attentions=True)
        target_out = model(**target_enc, output_hidden_states=True, output_attentions=True)

    if source_out.attentions is None or target_out.attentions is None:
        print("  SKIP: No attention output")
        return {}

    source_attn = source_out.attentions[target_layer][0].float().cpu().numpy()
    target_attn = target_out.attentions[target_layer][0].float().cpu().numpy()

    source_bank_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), "bank")
    target_bank_pos = find_target_position(tokenizer, target_enc["input_ids"][0].tolist(), "bank")

    print(f"\n  Head-level cosine distance between attention patterns (bank row):")
    print(f"  {'Head':>4s} | {'cos_sim':>7s} | {'L1_diff':>7s}")
    print(f"  {'-'*4} | {'-'*7} | {'-'*7}")

    head_diffs = []
    for h in range(source_attn.shape[0]):
        s_row = source_attn[h, source_bank_pos, :]
        t_row = target_attn[h, target_bank_pos, :]

        cos = F.cosine_similarity(
            torch.tensor(s_row).unsqueeze(0),
            torch.tensor(t_row).unsqueeze(0), dim=1
        ).item()
        l1 = np.abs(s_row - t_row).mean()

        head_diffs.append({"head": h, "cos": round(cos, 4), "l1": round(l1, 6)})

        if cos < 0.95:
            print(f"  H{h:3d} | {cos:7.4f} | {l1:7.6f}")

    # 汇总
    cos_vals = [d["cos"] for d in head_diffs]
    print(f"\n  Stats: mean_cos={np.mean(cos_vals):.4f}, min_cos={np.min(cos_vals):.4f}")
    max_diff_head = min(head_diffs, key=lambda x: x["cos"])
    print(f"  Most different head: H{max_diff_head['head']}, cos={max_diff_head['cos']:.4f}")

    return {"stats": {"mean_cos": round(np.mean(cos_vals), 4), "min_cos": round(np.min(cos_vals), 4)}, "head_diffs": head_diffs}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  stage562: L8 Attention Head因果分析")
    print("=" * 60)

    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    print(f"  Qwen3 n_layers={n_layers}")

    try:
        r1 = experiment1_head_ablation(model, tokenizer, n_layers)
        r2 = experiment2_head_attention_patterns(model, tokenizer, n_layers)
        r3 = experiment3_head_parity_check(model, tokenizer, n_layers)
    finally:
        free_model(model)

    all_results = {"exp1_head_ablation": r1, "exp2_attention_patterns": r2, "exp3_parity": r3}
    out_path = os.path.join(OUTPUT_DIR, "stage562_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  results saved: {out_path}")
    print(f"\n  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
