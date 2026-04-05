"""
stage564: 多层联合Hook——寻找因果效力最强的层组合
动机：单层hook bank的gain仅+0.23，apple的gain+0.59。
假设：bank的消歧需要多层联合干预，因为"bank"的消歧信息分布在多个层。

实验：
1. 单层hook效力曲线（复现stage561）
2. 双层组合hook（C(36,2)太多，用代表性层对）
3. 三层组合hook（L4+L8+L12等关键组合）
4. bank vs apple的差异——为什么apple单层就够？

使用Qwen3。
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations

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


def get_layer_module(model, layer_idx):
    layers = discover_layers(model)
    return layers[layer_idx]


def hook_patch_hidden_state(patch_value, target_pos, device):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        modified = hs.clone()
        modified[0, target_pos, :] = patch_value.to(modified.device, dtype=modified.dtype)
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    return hook_fn


def run_with_multi_hooks(model, encoded, layer_patch_pairs, target_pos):
    """在多个层同时hook"""
    handles = []
    for layer_idx, patch_value in layer_patch_pairs:
        layer_module = get_layer_module(model, layer_idx)
        handle = layer_module.register_forward_hook(
            hook_patch_hidden_state(patch_value, target_pos, 'cuda')
        )
        handles.append(handle)
    try:
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()
    return out


def get_target_hidden(model, tokenizer, sentence, target_word, layer_idx):
    """获取指定层的target编码"""
    enc = encode_to_device(model, tokenizer, sentence)
    pos = find_target_position(tokenizer, enc["input_ids"][0].tolist(), target_word)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.hidden_states[layer_idx][0, pos].clone(), enc, pos


def experiment1_single_layer_curve(model, tokenizer, n_layers):
    """实验1：精确的逐层单hook效力曲线"""
    print(f"\n{'='*60}")
    print(f"  实验1：逐层单Hook效力曲线")
    print(f"{'='*60}")

    experiments = {
        "bank": ("The river bank was muddy and steep", "The bank gave me a loan today", "bank"),
        "apple": ("The red apple is sweet and delicious", "Apple released the new iPhone today", "apple"),
    }

    results = {}
    for name, (src, tgt, tw) in experiments.items():
        source_enc = encode_to_device(model, tokenizer, src)
        target_enc = encode_to_device(model, tokenizer, tgt)
        s_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), tw)
        t_pos = find_target_position(tokenizer, target_enc["input_ids"][0].tolist(), tw)

        with torch.no_grad():
            s_out = model(**source_enc, output_hidden_states=True)
            t_out = model(**target_enc, output_hidden_states=True)

        s_last = s_out.hidden_states[-1][0, s_pos].cpu().float()
        t_last = t_out.hidden_states[-1][0, t_pos].cpu().float()
        cos_orig = F.cosine_similarity(s_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()

        print(f"\n  {name}: original cos={cos_orig:.4f}")
        print(f"  {'Layer':>5s} | {'cos':>8s} | {'gain':>8s} | {'L8_cos_d':>8s}")
        print(f"  {'-'*5} | {'-'*8} | {'-'*8} | {'-'*8}")

        layer_results = {}
        for li in range(n_layers):
            t_h = t_out.hidden_states[li][0, t_pos].clone()
            hooked = run_with_multi_hooks(model, source_enc, [(li, t_h)], s_pos)
            h_last = hooked.hidden_states[-1][0, s_pos].cpu().float()
            cos_h = F.cosine_similarity(h_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()
            gain = cos_h - cos_orig

            # L8处的编码差异
            l8_diff = abs(F.cosine_similarity(
                s_out.hidden_states[min(8, n_layers-1)][0, s_pos].cpu().float().unsqueeze(0),
                t_out.hidden_states[min(8, n_layers-1)][0, t_pos].cpu().float().unsqueeze(0), dim=1
            ).item())

            layer_results[f"L{li}"] = round(gain, 4)
            if li % 2 == 0 or gain > 0.05:
                print(f"  L{li:3d}   | {cos_h:8.4f} | {gain:+8.4f} | {l8_diff:8.4f}")

        results[name] = {"original_cos": round(cos_orig, 4), "layers": layer_results}

    return results


def experiment2_double_layer_combination(model, tokenizer, n_layers):
    """实验2：关键层对的双层联合Hook"""
    print(f"\n{'='*60}")
    print(f"  实验2：双层联合Hook组合")
    print(f"{'='*60}")

    src = "The river bank was muddy and steep"
    tgt = "The bank gave me a loan today"
    tw = "bank"

    source_enc = encode_to_device(model, tokenizer, src)
    target_enc = encode_to_device(model, tokenizer, tgt)
    s_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), tw)
    t_pos = find_target_position(tokenizer, target_enc["input_ids"][0].tolist(), tw)

    with torch.no_grad():
        s_out = model(**source_enc, output_hidden_states=True)
        t_out = model(**target_enc, output_hidden_states=True)

    s_last = s_out.hidden_states[-1][0, s_pos].cpu().float()
    t_last = t_out.hidden_states[-1][0, t_pos].cpu().float()
    cos_orig = F.cosine_similarity(s_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()

    # 选代表性层：0, 4, 8, 12, 16, 20, 24, 28, 32, 35
    key_layers = list(range(0, n_layers, 4))
    if key_layers[-1] != n_layers - 1:
        key_layers.append(n_layers - 1)

    # 预获取所有层的target编码
    target_hs = {}
    for li in key_layers:
        target_hs[li] = t_out.hidden_states[li][0, t_pos].clone()

    # 单层baseline
    print(f"  original cos = {cos_orig:.4f}")
    print(f"\n  单层Hook baseline:")
    single_gains = {}
    for li in key_layers:
        hooked = run_with_multi_hooks(model, source_enc, [(li, target_hs[li])], s_pos)
        h_last = hooked.hidden_states[-1][0, s_pos].cpu().float()
        gain = F.cosine_similarity(h_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item() - cos_orig
        single_gains[li] = gain
        print(f"    L{li:2d}: gain={gain:+.4f}")

    # 双层组合
    print(f"\n  双层组合Hook (top pairs):")
    combos = list(combinations(key_layers, 2))
    combo_results = []
    for l1, l2 in combos:
        patch_pairs = [(l1, target_hs[l1]), (l2, target_hs[l2])]
        hooked = run_with_multi_hooks(model, source_enc, patch_pairs, s_pos)
        h_last = hooked.hidden_states[-1][0, s_pos].cpu().float()
        gain = F.cosine_similarity(h_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item() - cos_orig
        combo_results.append((l1, l2, gain))

    # 找超加性组合（gain > max(single_gains[l1], single_gains[l2])）
    combo_results.sort(key=lambda x: -x[2])
    print(f"  {'L1':>4s} | {'L2':>4s} | {'gain':>8s} | {'single_max':>10s} | {'synergy':>8s}")
    print(f"  {'-'*4} | {'-'*4} | {'-'*8} | {'-'*10} | {'-'*8}")
    for l1, l2, gain in combo_results[:10]:
        sm = max(single_gains[l1], single_gains[l2])
        synergy = gain - sm
        print(f"  L{l1:2d} | L{l2:2d} | {gain:+8.4f} | {sm:+10.4f} | {synergy:+8.4f}")

    return {"original_cos": round(cos_orig, 4), "single_gains": {f"L{k}": round(v, 4) for k, v in single_gains.items()},
            "top_combos": [(f"L{a}", f"L{b}", round(g, 4)) for a, b, g in combo_results[:10]]}


def experiment3_triple_layer(model, tokenizer, n_layers):
    """实验3：三层联合Hook"""
    print(f"\n{'='*60}")
    print(f"  实验3：三层联合Hook")
    print(f"{'='*60}")

    experiments = {
        "bank": ("The river bank was muddy and steep", "The bank gave me a loan today", "bank"),
        "apple": ("The red apple is sweet and delicious", "Apple released the new iPhone today", "apple"),
    }

    results = {}
    for name, (src, tgt, tw) in experiments.items():
        source_enc = encode_to_device(model, tokenizer, src)
        target_enc = encode_to_device(model, tokenizer, tgt)
        s_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), tw)
        t_pos = find_target_position(tokenizer, target_enc["input_ids"][0].tolist(), tw)

        with torch.no_grad():
            s_out = model(**source_enc, output_hidden_states=True)
            t_out = model(**target_enc, output_hidden_states=True)

        s_last = s_out.hidden_states[-1][0, s_pos].cpu().float()
        t_last = t_out.hidden_states[-1][0, t_pos].cpu().float()
        cos_orig = F.cosine_similarity(s_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()

        print(f"\n  {name}: original cos={cos_orig:.4f}")

        # 几个有意义的层组合
        combos = [
            ([4, 8, 12], "early+peak+mid"),
            ([8, 12, 16], "peak+mid+mid"),
            ([12, 16, 20], "mid组合"),
            ([4, 8, 35], "early+peak+last"),
            ([0, 8, 35], "first+peak+last"),
            ([8, 20, 32], "peak+late组合"),
        ]

        print(f"  {'Layers':>18s} | {'label':>14s} | {'gain':>8s} | {'cos':>8s}")
        print(f"  {'-'*18} | {'-'*14} | {'-'*8} | {'-'*8}")

        for layers, label in combos:
            valid_layers = [l for l in layers if l < n_layers]
            patch_pairs = [(l, t_out.hidden_states[l][0, t_pos].clone()) for l in valid_layers]
            hooked = run_with_multi_hooks(model, source_enc, patch_pairs, s_pos)
            h_last = hooked.hidden_states[-1][0, s_pos].cpu().float()
            cos_h = F.cosine_similarity(h_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()
            gain = cos_h - cos_orig
            print(f"  {str(valid_layers):>18s} | {label:>14s} | {gain:+8.4f} | {cos_h:8.4f}")

        results[name] = {"original_cos": round(cos_orig, 4)}

    return results


def experiment4_gradual_replacement(model, tokenizer, n_layers):
    """实验4：逐步替换更多层——从单层到全层"""
    print(f"\n{'='*60}")
    print(f"  实验4：逐步增加Hook层数（bank）")
    print(f"{'='*60}")

    src = "The river bank was muddy and steep"
    tgt = "The bank gave me a loan today"
    tw = "bank"

    source_enc = encode_to_device(model, tokenizer, src)
    target_enc = encode_to_device(model, tokenizer, tgt)
    s_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), tw)
    t_pos = find_target_position(tokenizer, target_enc["input_ids"][0].tolist(), tw)

    with torch.no_grad():
        s_out = model(**source_enc, output_hidden_states=True)
        t_out = model(**target_enc, output_hidden_states=True)

    s_last = s_out.hidden_states[-1][0, s_pos].cpu().float()
    t_last = t_out.hidden_states[-1][0, t_pos].cpu().float()
    cos_orig = F.cosine_similarity(s_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()

    # 从L8开始，逐步向两侧扩展
    center = 8
    print(f"  original cos = {cos_orig:.4f}")
    print(f"\n  {'Hooked Layers':>20s} | {'gain':>8s} | {'cos':>8s}")
    print(f"  {'-'*20} | {'-'*8} | {'-'*8}")

    for radius in range(0, 19):
        layers = list(range(max(0, center - radius), min(n_layers, center + radius + 1)))
        patch_pairs = [(l, t_out.hidden_states[l][0, t_pos].clone()) for l in layers]
        hooked = run_with_multi_hooks(model, source_enc, patch_pairs, s_pos)
        h_last = hooked.hidden_states[-1][0, s_pos].cpu().float()
        cos_h = F.cosine_similarity(h_last.unsqueeze(0), t_last.unsqueeze(0), dim=1).item()
        gain = cos_h - cos_orig
        if radius % 3 == 0 or gain > 0.4:
            print(f"  {str(layers):>20s} | {gain:+8.4f} | {cos_h:8.4f}")

    return {"original_cos": round(cos_orig, 4)}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  stage564: 多层联合Hook")
    print("=" * 60)

    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    print(f"  Qwen3 n_layers={n_layers}")

    try:
        r1 = experiment1_single_layer_curve(model, tokenizer, n_layers)
        r2 = experiment2_double_layer_combination(model, tokenizer, n_layers)
        r3 = experiment3_triple_layer(model, tokenizer, n_layers)
        r4 = experiment4_gradual_replacement(model, tokenizer, n_layers)
    finally:
        free_model(model)

    out_path = os.path.join(OUTPUT_DIR, "stage564_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"exp1": r1, "exp2": r2, "exp3": r3, "exp4": {"summary": "见输出"}}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  results saved: {out_path}")
    print(f"\n  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
