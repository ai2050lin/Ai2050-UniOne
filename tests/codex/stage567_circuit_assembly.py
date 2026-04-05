"""
stage567: 回路组装——因果head+MLP子回路验证
动机：消歧是head协同的(stage565)，但单个head消融无效。
需要找到head→MLP→head的因果链路，组装成子回路。

实验：
1. L8 attention消融 + L8 MLP消融的组合
2. L8 head消融 → L9 hidden state变化（传播测量）
3. head→MLP→head因果链验证（逐层追踪）
4. bank消歧回路的head参与模式

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


def get_layer_modules(model, layer_idx):
    """获取层内的attn和mlp模块"""
    layers = discover_layers(model)
    layer = layers[layer_idx]
    attn = mlp = None
    if hasattr(layer, 'self_attn'):
        attn = layer.self_attn
    elif hasattr(layer, 'attn'):
        attn = layer.attn
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
    elif hasattr(layer, 'feed_forward'):
        mlp = layer.feed_forward
    return attn, mlp


def get_num_heads(model):
    if hasattr(model, 'config'):
        for attr in ['num_attention_heads', 'n_head', 'num_heads']:
            if hasattr(model.config, attr):
                return getattr(model.config, attr)
    return 32


def zero_ablation_hook(module, input, output):
    """将输出归零的hook"""
    if isinstance(output, tuple):
        return (torch.zeros_like(output[0]),) + output[1:]
    return torch.zeros_like(output)


def run_with_ablation(model, encoded, modules_to_ablate):
    """消融指定模块后运行"""
    handles = []
    for mod in modules_to_ablate:
        handles.append(mod.register_forward_hook(zero_ablation_hook))
    try:
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()
    return out


def experiment1_attn_vs_mlp(model, tokenizer, n_layers, num_heads):
    """实验1：L8 attention消融 vs MLP消融 vs 两者同时消融"""
    print(f"\n{'='*60}")
    print(f"  实验1：Attention vs MLP消融（L8, bank）")
    print(f"{'='*60}")

    target_layer = min(8, n_layers - 1)
    attn_mod, mlp_mod = get_layer_modules(model, target_layer)

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

    # bank编码的自cosine（完整性）
    self_cos = F.cosine_similarity(s_last.unsqueeze(0), s_last.unsqueeze(0), dim=1).item()

    configs = [
        ([], "baseline（无消融）"),
        ([attn_mod], "消融L8-Attention"),
        ([mlp_mod], "消融L8-MLP"),
        ([attn_mod, mlp_mod], "消融L8-Attn+MLP"),
    ]

    print(f"  baseline cos(source, target) = {cos_baseline:.4f}")
    print(f"\n  {'Config':>25s} | {'s_cos_tgt':>9s} | {'s_norm':>7s} | {'delta':>8s}")
    print(f"  {'-'*25} | {'-'*9} | {'-'*7} | {'-'*8}")

    results = {}
    for mods, label in configs:
        if mods:
            ablated = run_with_ablation(model, source_enc, mods)
        else:
            ablated = s_out
        a_last = ablated.hidden_states[-1][0, s_pos].cpu().float()
        a_norm = torch.norm(a_last).item()
        cos_tgt = F.cosine_similarity(a_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()
        delta = cos_tgt - cos_baseline
        results[label] = {"cos_to_target": round(cos_tgt, 4), "delta": round(delta, 4), "norm": round(a_norm, 2)}
        print(f"  {label:>25s} | {cos_tgt:9.4f} | {a_norm:7.2f} | {delta:+8.4f}")

    return results


def experiment2_multi_layer_attn_ablation(model, tokenizer, n_layers, num_heads):
    """实验2：多层attention消融——消歧涉及哪些层"""
    print(f"\n{'='*60}")
    print(f"  实验2：逐层Attention消融（bank→target变化）")
    print(f"{'='*60}")

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
    print(f"\n  {'Layer':>5s} | {'cos_dis':>8s} | {'delta':>8s} | {'norm':>7s}")
    print(f"  {'-'*5} | {'-'*8} | {'-'*8} | {'-'*7}")

    results = {}
    for li in list(range(0, min(n_layers, 36), 3)):
        attn_mod, _ = get_layer_modules(model, li)
        if attn_mod is None:
            continue
        ablated = run_with_ablation(model, source_enc, [attn_mod])
        a_last = ablated.hidden_states[-1][0, s_pos].cpu().float()
        cos_dis = F.cosine_similarity(a_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()
        delta = cos_dis - cos_baseline
        a_norm = torch.norm(a_last).item()
        results[f"L{li}"] = {"delta": round(delta, 4), "norm": round(a_norm, 2)}
        if abs(delta) > 0.01 or li % 6 == 0:
            print(f"  L{li:3d}   | {cos_dis:8.4f} | {delta:+8.4f} | {a_norm:7.2f}")

    return {"baseline": round(cos_baseline, 4), "layers": results}


def experiment3_propagation_measure(model, tokenizer, n_layers, num_heads):
    """实验3：L8 head消融→L9 hidden变化（信息传播测量）"""
    print(f"\n{'='*60}")
    print(f"  实验3：L8消融→下游层传播（逐层hidden state变化）")
    print(f"{'='*60}")

    target_layer = 8
    source_enc = encode_to_device(model, tokenizer, "The river bank was muddy and steep")
    s_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), "bank")

    # 正常前向传播
    with torch.no_grad():
        s_out = model(**source_enc, output_hidden_states=True)

    # 消融L8的attention
    attn_mod, _ = get_layer_modules(model, target_layer)
    ablated_out = run_with_ablation(model, source_enc, [attn_mod])

    print(f"  逐层hidden state变化（消融L8-Attn后）:")
    print(f"  {'Layer':>5s} | {'cos_diff':>8s} | {'L2_ratio':>9s}")
    print(f"  {'-'*5} | {'-'*8} | {'-'*9}")

    results = {}
    for li in range(target_layer, min(target_layer + 15, n_layers)):
        orig_h = s_out.hidden_states[li][0, s_pos].cpu().float()
        ablated_h = ablated_out.hidden_states[li][0, s_pos].cpu().float()

        cos_diff = F.cosine_similarity(orig_h.unsqueeze(0), ablated_h.unsqueeze(0), dim=1).item()
        l2_ratio = torch.norm(ablated_h).item() / (torch.norm(orig_h).item() + 1e-10)

        results[f"L{li}"] = {"cos_diff": round(cos_diff, 4), "l2_ratio": round(l2_ratio, 4)}
        print(f"  L{li:3d}   | {cos_diff:8.4f} | {l2_ratio:9.4f}")

    return results


def experiment4_attn_mlp_cross_layer(model, tokenizer, n_layers, num_heads):
    """实验4：L8 Attn消融 vs L8 MLP消融——在apple上对比"""
    print(f"\n{'='*60}")
    print(f"  实验4：Attn vs MLP消融（apple, 多层对比）")
    print(f"{'='*60}")

    source_enc = encode_to_device(model, tokenizer, "The red apple is sweet and delicious")
    target_enc = encode_to_device(model, tokenizer, "Apple released the new iPhone today")
    s_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), "apple")
    t_pos = find_target_position(tokenizer, target_enc["input_ids"][0].tolist(), "apple")

    with torch.no_grad():
        s_out = model(**source_enc, output_hidden_states=True)
        t_out = model(**target_enc, output_hidden_states=True)

    s_last = s_out.hidden_states[-1][0, s_pos].cpu().float()
    t_last = t_out.hidden_states[-1][0, t_pos].cpu().float()
    cos_baseline = F.cosine_similarity(s_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()

    print(f"  apple baseline cos = {cos_baseline:.4f}")

    key_layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
    key_layers = [l for l in key_layers if l < n_layers]

    print(f"\n  {'Layer':>5s} | {'attn_delta':>10s} | {'mlp_delta':>9s} | {'both_delta':>10s}")
    print(f"  {'-'*5} | {'-'*10} | {'-'*9} | {'-'*10}")

    results = {}
    for li in key_layers:
        attn_mod, mlp_mod = get_layer_modules(model, li)
        if attn_mod is None or mlp_mod is None:
            continue

        # 只消融attn
        a_out = run_with_ablation(model, source_enc, [attn_mod])
        a_last = a_out.hidden_states[-1][0, s_pos].cpu().float()
        attn_delta = F.cosine_similarity(a_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item() - cos_baseline

        # 只消融mlp
        m_out = run_with_ablation(model, source_enc, [mlp_mod])
        m_last = m_out.hidden_states[-1][0, s_pos].cpu().float()
        mlp_delta = F.cosine_similarity(m_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item() - cos_baseline

        # 两者都消融
        b_out = run_with_ablation(model, source_enc, [attn_mod, mlp_mod])
        b_last = b_out.hidden_states[-1][0, s_pos].cpu().float()
        both_delta = F.cosine_similarity(b_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item() - cos_baseline

        results[f"L{li}"] = {"attn": round(attn_delta, 4), "mlp": round(mlp_delta, 4), "both": round(both_delta, 4)}
        print(f"  L{li:3d}   | {attn_delta:+10.4f} | {mlp_delta:+9.4f} | {both_delta:+10.4f}")

    return {"baseline": round(cos_baseline, 4), "layers": results}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  stage567: 回路组装")
    print("=" * 60)

    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    num_heads = get_num_heads(model)
    print(f"  Qwen3 n_layers={n_layers}, num_heads={num_heads}")

    try:
        r1 = experiment1_attn_vs_mlp(model, tokenizer, n_layers, num_heads)
        r2 = experiment2_multi_layer_attn_ablation(model, tokenizer, n_layers, num_heads)
        r3 = experiment3_propagation_measure(model, tokenizer, n_layers, num_heads)
        r4 = experiment4_attn_mlp_cross_layer(model, tokenizer, n_layers, num_heads)
    finally:
        free_model(model)

    out_path = os.path.join(OUTPUT_DIR, "stage567_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"exp1": r1, "exp2": r2, "exp3": r3, "exp4": r4}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  results saved: {out_path}")
    print(f"\n  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
