"""
stage561: Forward Hook因果干预——真正的因果验证
核心区别：stage557的"静态修补"只替换了编码向量中的值，但模型的后处理层看到的是
"经过attention和MLP重新计算后的新值"，而不是修补后的值。
本脚本用register_forward_hook在真实前向传播中拦截和替换隐藏状态，
使得后续层确实"看到"了修改后的编码。

三组实验：
1. L8 hook修补——在消歧峰值层真实替换bank/apple的隐藏状态
2. 跨层hook修补——L4/L8/L16/L24/L32各层hook
3. attention输出hook——直接替换attention head的输出

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


def get_layer_module(model, layer_idx):
    """获取指定层的模块"""
    layers = discover_layers(model)
    return layers[layer_idx]


def hook_patch_hidden_state(target_hidden, target_pos, device):
    """创建一个forward hook，替换指定位置的隐藏状态"""
    def hook_fn(module, input, output):
        # output是tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        # 逐层处理：只修改batch 0
        modified = hs.clone()
        modified[0, target_pos, :] = target_hidden.to(modified.device, dtype=modified.dtype)
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    return hook_fn


def run_with_hook(model, encoded, layer_idx, patch_value, target_pos):
    """用hook运行前向传播，在指定层替换target_pos的隐藏状态"""
    layer_module = get_layer_module(model, layer_idx)
    handle = layer_module.register_forward_hook(
        hook_patch_hidden_state(patch_value, target_pos, 'cuda')
    )
    try:
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True)
    finally:
        handle.remove()
    return out


def experiment1_hook_patch(model, tokenizer, n_layers):
    """实验1：L8真实hook修补——验证因果效力"""
    print(f"\n{'='*60}")
    print(f"  实验1：L8 Forward Hook修补（真实因果验证）")
    print(f"{'='*60}")

    patch_layer = min(8, n_layers - 1)
    experiments = {
        "bank": {
            "source_ctx": "The river bank was muddy and steep",
            "target_ctx": "The bank gave me a loan today",
            "target_word": "bank",
            "probe_words": ["river", "water", "money", "loan", "deposit", "financial"],
        },
        "apple": {
            "source_ctx": "The red apple is sweet and delicious",
            "target_ctx": "Apple released the new iPhone today",
            "target_word": "apple",
            "probe_words": ["fruit", "sweet", "delicious", "phone", "iPhone", "technology"],
        },
    }

    results = {}
    for word, config in experiments.items():
        print(f"\n  === {word} ===")

        # 编码两个语境
        source_enc = encode_to_device(model, tokenizer, config["source_ctx"])
        target_enc = encode_to_device(model, tokenizer, config["target_ctx"])
        source_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), config["target_word"])
        target_pos = find_target_position(tokenizer, target_enc["input_ids"][0].tolist(), config["target_word"])

        # 正常前向传播
        with torch.no_grad():
            source_out = model(**source_enc, output_hidden_states=True)
            target_out = model(**target_enc, output_hidden_states=True)

        source_last = source_out.hidden_states[-1][0, source_pos].cpu().float()
        target_last = target_out.hidden_states[-1][0, target_pos].cpu().float()

        # source patch层编码
        source_patch_h = source_out.hidden_states[patch_layer][0, source_pos].clone()
        target_patch_h = target_out.hidden_states[patch_layer][0, target_pos].clone()

        # 原始末层cosine
        cos_orig = F.cosine_similarity(source_last.unsqueeze(0), target_last.unsqueeze(0), dim=1).item()
        print(f"  source last cos to target: {cos_orig:.4f}")

        # Hook修补：用target的L8编码替换source的L8编码，然后继续前向传播到末层
        hooked_out = run_with_hook(model, source_enc, patch_layer, target_patch_h, source_pos)
        hooked_last = hooked_out.hidden_states[-1][0, source_pos].cpu().float()
        cos_hooked = F.cosine_similarity(hooked_last.unsqueeze(0), target_last.unsqueeze(0), dim=1).item()
        print(f"  hooked L{patch_layer} last cos: {cos_hooked:.4f}, gain={cos_hooked-cos_orig:+.4f}")

        # 对照：hook同一层的编码但不修改（检查hook本身的影响）
        control_out = run_with_hook(model, source_enc, patch_layer, source_patch_h, source_pos)
        control_last = control_out.hidden_states[-1][0, source_pos].cpu().float()
        cos_control = F.cosine_similarity(control_last.unsqueeze(0), source_last.unsqueeze(0), dim=1).item()
        print(f"  control (self-hook): cos={cos_control:.4f} (should be ~1.0)")

        # Probe词分析：比较hook前后source对probe词的相似度变化
        print(f"\n  Probe word similarity (source baseline -> hooked):")
        probe_shifts = {}
        for pw in config["probe_words"]:
            pw_enc = encode_to_device(model, tokenizer, pw)
            with torch.no_grad():
                pw_out = model(**pw_enc, output_hidden_states=True)
            pw_last = pw_out.hidden_states[-1][0, -1].cpu().float()

            cos_baseline = F.cosine_similarity(source_last.unsqueeze(0), pw_last.unsqueeze(0), dim=1).item()
            cos_after = F.cosine_similarity(hooked_last.unsqueeze(0), pw_last.unsqueeze(0), dim=1).item()
            shift = cos_after - cos_baseline
            probe_shifts[pw] = {"baseline": round(cos_baseline, 4), "after_hook": round(cos_after, 4), "shift": round(shift, 4)}
            print(f"    {pw:12s}: {cos_baseline:.4f} -> {cos_after:.4f} ({shift:+.4f})")

        # 逐层看hook的影响传播
        print(f"\n  Hook影响传播（逐层cosine to target）:")
        for li in range(patch_layer, min(patch_layer + 10, n_layers)):
            h_hooked = hooked_out.hidden_states[li][0, source_pos].cpu().float()
            h_target = target_out.hidden_states[li][0, target_pos].cpu().float()
            cos_li = F.cosine_similarity(h_hooked.unsqueeze(0), h_target.unsqueeze(0), dim=1).item()
            print(f"    L{li}: cos_to_target={cos_li:.4f}")

        results[word] = {
            "original_cos": round(cos_orig, 4),
            "hooked_cos": round(cos_hooked, 4),
            "gain": round(cos_hooked - cos_orig, 4),
            "control_cos": round(cos_control, 4),
            "probe_shifts": probe_shifts,
        }

    return results


def experiment2_cross_layer_hook(model, tokenizer, n_layers):
    """实验2：不同层的hook修补效果——找因果效力最强的层"""
    print(f"\n{'='*60}")
    print(f"  实验2：跨层Hook修补——因果效力逐层扫描")
    print(f"{'='*60}")

    source_ctx = "The river bank was muddy and steep"
    target_ctx = "The bank gave me a loan today"
    target_word = "bank"

    source_enc = encode_to_device(model, tokenizer, source_ctx)
    target_enc = encode_to_device(model, tokenizer, target_ctx)
    source_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), target_word)
    target_pos = find_target_position(tokenizer, target_enc["input_ids"][0].tolist(), target_word)

    with torch.no_grad():
        source_out = model(**source_enc, output_hidden_states=True)
        target_out = model(**target_enc, output_hidden_states=True)

    source_last = source_out.hidden_states[-1][0, source_pos].cpu().float()
    target_last = target_out.hidden_states[-1][0, target_pos].cpu().float()
    cos_orig = F.cosine_similarity(source_last.unsqueeze(0), target_last.unsqueeze(0), dim=1).item()

    print(f"  original cos(source_last, target_last) = {cos_orig:.4f}")
    print(f"\n  {'Layer':>5s} | {'patch_cos':>8s} | {'gain':>8s} | {'control_cos':>11s}")
    print(f"  {'-'*5} | {'-'*8} | {'-'*8} | {'-'*11}")

    results = {}
    for li in evenly_spaced_layers(n_layers, count=12):
        # 用target的该层编码替换source的
        target_h = target_out.hidden_states[li][0, target_pos].clone()
        source_h = source_out.hidden_states[li][0, source_pos].clone()

        hooked_out = run_with_hook(model, source_enc, li, target_h, source_pos)
        hooked_last = hooked_out.hidden_states[-1][0, source_pos].cpu().float()
        cos_hooked = F.cosine_similarity(hooked_last.unsqueeze(0), target_last.unsqueeze(0), dim=1).item()
        gain = cos_hooked - cos_orig

        # control
        control_out = run_with_hook(model, source_enc, li, source_h, source_pos)
        control_last = control_out.hidden_states[-1][0, source_pos].cpu().float()
        cos_control = F.cosine_similarity(control_last.unsqueeze(0), source_last.unsqueeze(0), dim=1).item()

        print(f"  L{li:3d}   | {cos_hooked:8.4f} | {gain:+8.4f} | {cos_control:11.4f}")
        results[f"L{li}"] = {"patched_cos": round(cos_hooked, 4), "gain": round(gain, 4)}

    return {"original_cos": round(cos_orig, 4), "layer_results": results}


def experiment3_selective_neuron_hook(model, tokenizer, n_layers):
    """实验3：选择性神经元hook——只替换消歧神经元而非整个向量"""
    print(f"\n{'='*60}")
    print(f"  实验3：选择性神经元Hook vs 全向量Hook")
    print(f"{'='*60}")

    patch_layer = min(8, n_layers - 1)
    source_ctx = "The river bank was muddy and steep"
    target_ctx = "The bank gave me a loan today"
    target_word = "bank"

    source_enc = encode_to_device(model, tokenizer, source_ctx)
    target_enc = encode_to_device(model, tokenizer, target_ctx)
    source_pos = find_target_position(tokenizer, source_enc["input_ids"][0].tolist(), target_word)
    target_pos = find_target_position(tokenizer, target_enc["input_ids"][0].tolist(), target_word)

    with torch.no_grad():
        source_out = model(**source_enc, output_hidden_states=True)
        target_out = model(**target_enc, output_hidden_states=True)

    source_last = source_out.hidden_states[-1][0, source_pos].cpu().float()
    target_last = target_out.hidden_states[-1][0, target_pos].cpu().float()
    cos_orig = F.cosine_similarity(source_last.unsqueeze(0), target_last.unsqueeze(0), dim=1).item()

    source_h = source_out.hidden_states[patch_layer][0, source_pos].clone()
    target_h = target_out.hidden_states[patch_layer][0, target_pos].clone()

    # 找消歧神经元（CV最大的）
    stacked = torch.stack([source_h, target_h], dim=0).cpu().float()
    n_std = stacked.std(dim=0)
    n_mean = stacked.mean(dim=0).abs()
    cv = n_std / (n_mean + 1e-10)
    top_neurons = torch.argsort(cv, descending=True)

    print(f"  original cos: {cos_orig:.4f}")
    print(f"\n  {'K':>6s} | {'hook_cos':>8s} | {'gain':>8s} | {'vs_full_vec':>11s}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*11}")

    # 全向量hook baseline
    full_hooked = run_with_hook(model, source_enc, patch_layer, target_h, source_pos)
    full_last = full_hooked.hidden_states[-1][0, source_pos].cpu().float()
    full_cos = F.cosine_similarity(full_last.unsqueeze(0), target_last.unsqueeze(0), dim=1).item()

    results = {}
    for k in [5, 10, 20, 50, 100, 200, 500, 1000]:
        # 只替换top-K神经元
        selective_h = source_h.clone()
        neuron_indices = top_neurons[:k]
        for idx in neuron_indices:
            selective_h[idx] = target_h[idx]

        selective_hooked = run_with_hook(model, source_enc, patch_layer, selective_h, source_pos)
        selective_last = selective_hooked.hidden_states[-1][0, source_pos].cpu().float()
        sel_cos = F.cosine_similarity(selective_last.unsqueeze(0), target_last.unsqueeze(0), dim=1).item()
        gain = sel_cos - cos_orig
        vs_full = sel_cos - full_cos

        print(f"  K={k:4d} | {sel_cos:8.4f} | {gain:+8.4f} | {vs_full:+11.4f}")
        results[f"K={k}"] = {"cos": round(sel_cos, 4), "gain": round(gain, 4)}

    print(f"\n  全向量hook: cos={full_cos:.4f}, gain={full_cos-cos_orig:+.4f}")
    results["full_vector"] = {"cos": round(full_cos, 4), "gain": round(full_cos - cos_orig, 4)}

    return {"original_cos": round(cos_orig, 4), "full_vector_cos": round(full_cos, 4), "selective": results}


def evenly_spaced_layers(n_layers, count=10):
    if n_layers <= count:
        return list(range(n_layers))
    return [int(i * (n_layers - 1) / (count - 1)) for i in range(count)]


def main():
    t0 = time.time()
    print("=" * 60)
    print("  stage561: Forward Hook因果干预")
    print("=" * 60)

    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    print(f"  Qwen3 n_layers={n_layers}")

    try:
        r1 = experiment1_hook_patch(model, tokenizer, n_layers)
        r2 = experiment2_cross_layer_hook(model, tokenizer, n_layers)
        r3 = experiment3_selective_neuron_hook(model, tokenizer, n_layers)
    finally:
        free_model(model)

    all_results = {"exp1_hook_patch": r1, "exp2_cross_layer": r2, "exp3_selective": r3}
    out_path = os.path.join(OUTPUT_DIR, "stage561_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  results saved: {out_path}")
    print(f"\n  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
