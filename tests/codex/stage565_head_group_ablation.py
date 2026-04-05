"""
stage565: Head-Group消融——分组消融寻找因果head组合
动机：stage562发现单个head消融无效(0/32)，但H15差异最大(cos=0.60)。
假设：消歧需要多个head协同消融才有因果效力。

实验：
1. 4-head一组消融（8组）
2. 8-head一组消融（4组）
3. 上半head vs 下半head消融
4. 基于stage562差异排序的渐进消融

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
    layers = discover_layers(model)
    layer = layers[layer_idx]
    if hasattr(layer, 'self_attn'):
        return layer.self_attn
    elif hasattr(layer, 'attn'):
        return layer.attn
    elif hasattr(layer, 'attention'):
        return layer.attention
    return None


def create_group_ablation_hook(head_indices, num_heads):
    """消融一组head"""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output
        head_dim = attn_output.shape[-1] // num_heads
        modified = attn_output.clone()
        for h in head_indices:
            start = h * head_dim
            end = start + head_dim
            modified[:, :, start:end] = 0
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    return hook_fn


def get_num_heads(model):
    if hasattr(model, 'config'):
        for attr in ['num_attention_heads', 'n_head', 'num_heads']:
            if hasattr(model.config, attr):
                return getattr(model.config, attr)
    return 32


def run_with_head_ablation(model, encoded, attn_module, head_indices, num_heads):
    """消融指定head组后运行"""
    hook = create_group_ablation_hook(head_indices, num_heads)
    handle = attn_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True)
    finally:
        handle.remove()
    return out


def experiment1_4head_groups(model, tokenizer, n_layers, num_heads):
    """实验1：4-head一组消融"""
    print(f"\n{'='*60}")
    print(f"  实验1：4-Head Group消融 (L8)")
    print(f"{'='*60}")

    target_layer = min(8, n_layers - 1)
    attn_module = get_attention_module(model, target_layer)
    if attn_module is None:
        print("  SKIP: 无法获取attention模块")
        return {}

    source_ctx = "The river bank was muddy and steep"
    target_ctx = "The bank gave me a loan today"

    source_enc = encode_to_device(model, tokenizer, source_ctx)
    target_enc = encode_to_device(model, tokenizer, target_ctx)
    s_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), "bank")
    t_pos = find_target_position(tokenizer, target_enc["input_ids"][0].tolist(), "bank")

    with torch.no_grad():
        s_out = model(**source_enc, output_hidden_states=True)
        t_out = model(**target_enc, output_hidden_states=True)

    s_last = s_out.hidden_states[-1][0, s_pos].cpu().float()
    t_last = t_out.hidden_states[-1][0, t_pos].cpu().float()
    cos_baseline = F.cosine_similarity(s_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()

    # bank编码自身的完整性
    bank_norm = torch.norm(s_last).item()

    print(f"  baseline cos(source, target) = {cos_baseline:.4f}")
    print(f"  bank encoding norm = {bank_norm:.4f}")
    print(f"\n  {'Group':>10s} | {'heads':>15s} | {'cos_dis':>8s} | {'delta':>8s} | {'norm_ratio':>10s}")
    print(f"  {'-'*10} | {'-'*15} | {'-'*8} | {'-'*8} | {'-'*10}")

    group_size = 4
    results = {}
    for g in range(num_heads // group_size):
        heads = list(range(g * group_size, (g + 1) * group_size))
        ablated = run_with_head_ablation(model, source_enc, attn_module, heads, num_heads)
        a_last = ablated.hidden_states[-1][0, s_pos].cpu().float()
        cos_dis = F.cosine_similarity(a_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()
        delta = cos_dis - cos_baseline
        nr = torch.norm(a_last).item() / bank_norm
        print(f"  G{g}: H{heads[0]:2d}-{heads[-1]:2d} | {str(heads):>15s} | {cos_dis:8.4f} | {delta:+8.4f} | {nr:10.4f}")
        results[f"G{g}"] = {"heads": heads, "delta": round(delta, 4)}

    # 消融所有head（sanity check）
    all_ablated = run_with_head_ablation(model, source_enc, attn_module, list(range(num_heads)), num_heads)
    all_last = all_ablated.hidden_states[-1][0, s_pos].cpu().float()
    all_cos = F.cosine_similarity(all_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()
    all_nr = torch.norm(all_last).item() / bank_norm
    print(f"  {'ALL':>10s} | H0-{num_heads-1:2d} | {str(list(range(num_heads))):>15s} | {all_cos:8.4f} | {all_cos-cos_baseline:+8.4f} | {all_nr:10.4f}")
    results["ALL"] = {"delta": round(all_cos - cos_baseline, 4)}

    return {"baseline": round(cos_baseline, 4), "groups": results}


def experiment2_8head_groups(model, tokenizer, n_layers, num_heads):
    """实验2：8-head一组消融"""
    print(f"\n{'='*60}")
    print(f"  实验2：8-Head Group消融 (L8)")
    print(f"{'='*60}")

    target_layer = min(8, n_layers - 1)
    attn_module = get_attention_module(model, target_layer)

    source_enc = encode_to_device(model, tokenizer, "The river bank was muddy and steep")
    target_enc = encode_to_device(model, tokenizer, "The bank gave me a loan today")
    s_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), "bank")
    t_pos = find_target_position(tokenizer, target_enc["input_ids"][0].tolist(), "bank")

    with torch.no_grad():
        s_out = model(**source_enc, output_hidden_states=True)
        t_out = model(**target_enc, output_hidden_states=True)

    s_last = s_out.hidden_states[-1][0, s_pos].cpu().float()
    t_last = t_out.hidden_states[-1][0, t_pos].cpu().float()
    cos_baseline = F.cosine_similarity(s_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()

    print(f"  baseline cos = {cos_baseline:.4f}")
    print(f"\n  {'Group':>8s} | {'heads':>20s} | {'cos_dis':>8s} | {'delta':>8s}")
    print(f"  {'-'*8} | {'-'*20} | {'-'*8} | {'-'*8}")

    results = {}
    for g in range(num_heads // 8):
        heads = list(range(g * 8, (g + 1) * 8))
        ablated = run_with_head_ablation(model, source_enc, attn_module, heads, num_heads)
        a_last = ablated.hidden_states[-1][0, s_pos].cpu().float()
        cos_dis = F.cosine_similarity(a_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()
        delta = cos_dis - cos_baseline
        print(f"  G{g}: H{heads[0]:2d}-{heads[-1]:2d} | {str(heads):>20s} | {cos_dis:8.4f} | {delta:+8.4f}")
        results[f"G{g}"] = round(delta, 4)

    return {"baseline": round(cos_baseline, 4), "groups": results}


def experiment3_half_groups(model, tokenizer, n_layers, num_heads):
    """实验3：上半head vs 下半head消融 + 保留组消融"""
    print(f"\n{'='*60}")
    print(f"  实验3：Half-Head消融 (L8)")
    print(f"{'='*60}")

    target_layer = min(8, n_layers - 1)
    attn_module = get_attention_module(model, target_layer)

    source_enc = encode_to_device(model, tokenizer, "The river bank was muddy and steep")
    target_enc = encode_to_device(model, tokenizer, "The bank gave me a loan today")
    s_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), "bank")
    t_pos = find_target_position(tokenizer, target_enc["input_ids"][0].tolist(), "bank")

    with torch.no_grad():
        s_out = model(**source_enc, output_hidden_states=True)
        t_out = model(**target_enc, output_hidden_states=True)

    s_last = s_out.hidden_states[-1][0, s_pos].cpu().float()
    t_last = t_out.hidden_states[-1][0, t_pos].cpu().float()
    cos_baseline = F.cosine_similarity(s_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()

    half = num_heads // 2
    configs = [
        (list(range(half)), f"消融H0-{half-1}(下半)"),
        (list(range(half, num_heads)), f"消融H{half}-{num_heads-1}(上半)"),
        (list(range(0, num_heads, 2)), "消融偶数head"),
        (list(range(1, num_heads, 2)), "消融奇数head"),
    ]

    print(f"  baseline cos = {cos_baseline:.4f}")
    print(f"\n  {'Config':>20s} | {'cos_dis':>8s} | {'delta':>8s}")
    print(f"  {'-'*20} | {'-'*8} | {'-'*8}")

    results = {}
    for heads, label in configs:
        ablated = run_with_head_ablation(model, source_enc, attn_module, heads, num_heads)
        a_last = ablated.hidden_states[-1][0, s_pos].cpu().float()
        cos_dis = F.cosine_similarity(a_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()
        delta = cos_dis - cos_baseline
        print(f"  {label:>20s} | {cos_dis:8.4f} | {delta:+8.4f}")
        results[label] = round(delta, 4)

    return {"baseline": round(cos_baseline, 4), "configs": results}


def experiment4_importance_ranked_ablation(model, tokenizer, n_layers, num_heads):
    """实验4：基于head差异排序的渐进消融"""
    print(f"\n{'='*60}")
    print(f"  实验4：按重要性渐进消融 (L8, bank)")
    print(f"{'='*60}")

    target_layer = min(8, n_layers - 1)
    attn_module = get_attention_module(model, target_layer)

    source_enc = encode_to_device(model, tokenizer, "The river bank was muddy and steep")
    target_enc = encode_to_device(model, tokenizer, "The bank gave me a loan today")
    s_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), "bank")
    t_pos = find_target_position(tokenizer, target_enc["input_ids"][0].tolist(), "bank")

    with torch.no_grad():
        s_out = model(**source_enc, output_hidden_states=True, output_attentions=True)
        t_out = model(**target_enc, output_hidden_states=True, output_attentions=True)

    s_last = s_out.hidden_states[-1][0, s_pos].cpu().float()
    t_last = t_out.hidden_states[-1][0, t_pos].cpu().float()
    cos_baseline = F.cosine_similarity(s_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()

    # 计算每个head在两个语境中的attention pattern差异
    if s_out.attentions is not None and t_out.attentions is not None:
        s_attn = s_out.attentions[target_layer][0].float().cpu().numpy()
        t_attn = t_out.attentions[target_layer][0].float().cpu().numpy()

        head_importance = []
        for h in range(num_heads):
            cos_h = F.cosine_similarity(
                torch.tensor(s_attn[h, s_pos, :]).unsqueeze(0),
                torch.tensor(t_attn[h, t_pos, :]).unsqueeze(0), dim=1
            ).item()
            head_importance.append((h, cos_h))
        # 按差异排序（cos越小=差异越大）
        head_importance.sort(key=lambda x: x[1])

        print(f"  baseline cos = {cos_baseline:.4f}")
        print(f"\n  Head importance ranking (差异从大到小):")
        for h, cos_h in head_importance[:8]:
            print(f"    H{h}: cos={cos_h:.4f}")
    else:
        print("  No attention output, using default order")
        head_importance = [(h, 1.0) for h in range(num_heads)]

    # 渐进消融
    print(f"\n  渐进消融（按差异从大到小逐个消融）:")
    print(f"  {'消融数':>6s} | {'cos_dis':>8s} | {'delta':>8s}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*8}")

    results = {}
    cumulative_heads = []
    for step in [1, 2, 4, 8, 16, num_heads]:
        if step > num_heads:
            step = num_heads
        cumulative_heads = [h for h, _ in head_importance[:step]]
        ablated = run_with_head_ablation(model, source_enc, attn_module, cumulative_heads, num_heads)
        a_last = ablated.hidden_states[-1][0, s_pos].cpu().float()
        cos_dis = F.cosine_similarity(a_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()
        delta = cos_dis - cos_baseline
        print(f"  {step:6d} | {cos_dis:8.4f} | {delta:+8.4f}")
        results[f"top{step}"] = round(delta, 4)

    return {"baseline": round(cos_baseline, 4), "progressive": results}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  stage565: Head-Group消融")
    print("=" * 60)

    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    num_heads = get_num_heads(model)
    print(f"  Qwen3 n_layers={n_layers}, num_heads={num_heads}")

    try:
        r1 = experiment1_4head_groups(model, tokenizer, n_layers, num_heads)
        r2 = experiment2_8head_groups(model, tokenizer, n_layers, num_heads)
        r3 = experiment3_half_groups(model, tokenizer, n_layers, num_heads)
        r4 = experiment4_importance_ranked_ablation(model, tokenizer, n_layers, num_heads)
    finally:
        free_model(model)

    all_results = {"exp1_4head": r1, "exp2_8head": r2, "exp3_half": r3, "exp4_progressive": r4}
    out_path = os.path.join(OUTPUT_DIR, "stage565_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  results saved: {out_path}")
    print(f"\n  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
