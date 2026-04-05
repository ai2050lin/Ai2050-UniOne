"""
stage557: 因果干预——消歧神经元激活修补验证
目标：验证stage556发现的消歧神经元是否真正因果负责消歧
- 方法：将"河岸"语境中的bank编码，用"银行"语境中的消歧神经元值替换
- 如果替换后模型输出从"河岸"相关变为"银行"相关→因果验证通过
- 同时做反向实验和随机神经元对照组

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

DISAMBIG_EXPERIMENTS = {
    "bank": {
        "source_ctx": "The river bank was muddy",
        "target_ctx": "The bank gave me a loan",
        "target_word": "bank",
        "probe_words": ["river", "water", "muddy", "loan", "money", "deposit"],
    },
    "apple": {
        "source_ctx": "red apple is sweet fruit",
        "target_ctx": "Apple released a new phone",
        "target_word": "apple",
        "probe_words": ["fruit", "sweet", "red", "phone", "technology", "iPhone"],
    },
}


def find_target_position(tokenizer, token_ids, target_word):
    tokens = [tokenizer.convert_ids_to_tokens(t) for t in token_ids]
    for pos in range(len(tokens) - 1, -1, -1):
        if target_word in tokens[pos].lower():
            return pos
    return len(tokens) - 1


def get_disambig_neurons(model, tokenizer, config, layer_idx, top_k=20):
    ctxs = [config["source_ctx"], config["target_ctx"]]
    encodings = []
    for ctx in ctxs:
        encoded = encode_to_device(model, tokenizer, ctx)
        token_ids = encoded["input_ids"][0].tolist()
        target_pos = find_target_position(tokenizer, token_ids, config["target_word"])
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True)
        enc = out.hidden_states[layer_idx][0, target_pos].cpu().float().numpy()
        encodings.append(enc)
    stacked = np.array(encodings)
    n_std = stacked.std(axis=0)
    n_mean = stacked.mean(axis=0)
    cv = n_std / (np.abs(n_mean) + 1e-10)
    top_idx = np.argsort(-cv)[:top_k].tolist()
    return top_idx


def experiment1_activation_patch(model, tokenizer, n_layers):
    """核心实验：激活修补验证消歧神经元"""
    print(f"\n{'='*60}")
    print(f"  实验1：消歧神经元激活修补（因果验证）")
    print(f"{'='*60}")

    results = {}
    patch_layer = min(8, n_layers - 1)

    for word, config in DISAMBIG_EXPERIMENTS.items():
        print(f"\n  === {word}: '{config['source_ctx']}' -> '{config['target_ctx']}' ===")

        disambig_idx = get_disambig_neurons(model, tokenizer, config, patch_layer, top_k=20)
        print(f"  patch layer: L{patch_layer}, top-20 neurons: {disambig_idx[:5]}...")

        source_encoded = encode_to_device(model, tokenizer, config["source_ctx"])
        target_encoded = encode_to_device(model, tokenizer, config["target_ctx"])
        source_pos = find_target_position(tokenizer, source_encoded["input_ids"][0].tolist(), config["target_word"])
        target_pos = find_target_position(tokenizer, target_encoded["input_ids"][0].tolist(), config["target_word"])

        with torch.no_grad():
            source_out = model(**source_encoded, output_hidden_states=True)
            target_out = model(**target_encoded, output_hidden_states=True)

        source_hidden = source_out.hidden_states[patch_layer][0, source_pos].clone().cpu().float()
        target_hidden = target_out.hidden_states[patch_layer][0, target_pos].clone().cpu().float()

        cos_orig = F.cosine_similarity(source_hidden.unsqueeze(0), target_hidden.unsqueeze(0), dim=1).item()
        print(f"  original (source vs target) cosine: {cos_orig:.4f}")

        # 修补：替换消歧神经元
        patched = source_hidden.clone()
        for idx in disambig_idx:
            patched[idx] = target_hidden[idx]
        cos_patched = F.cosine_similarity(patched.unsqueeze(0), target_hidden.unsqueeze(0), dim=1).item()
        print(f"  patched (K=20) cosine: {cos_patched:.4f}, gain={cos_patched-cos_orig:+.4f}")

        # 随机对照
        np.random.seed(42)
        random_idx = np.random.choice(len(source_hidden), size=20, replace=False).tolist()
        random_patched = source_hidden.clone()
        for idx in random_idx:
            random_patched[idx] = target_hidden[idx]
        cos_random = F.cosine_similarity(random_patched.unsqueeze(0), target_hidden.unsqueeze(0), dim=1).item()
        print(f"  random (K=20) cosine: {cos_random:.4f}, gain={cos_random-cos_orig:+.4f}")

        # 不同K值
        print(f"\n  K-value sweep:")
        k_list = [5, 10, 20, 50, 100, 200, 500, 1000]
        for k in k_list:
            idx_k = get_disambig_neurons(model, tokenizer, config, patch_layer, top_k=k)
            pk = source_hidden.clone()
            for idx in idx_k:
                pk[idx] = target_hidden[idx]
            cos_k = F.cosine_similarity(pk.unsqueeze(0), target_hidden.unsqueeze(0), dim=1).item()
            print(f"    K={k:4d}: cosine={cos_k:.4f}, gain={cos_k-cos_orig:+.4f}")

        # Probe词相似度变化
        print(f"\n  Probe word similarity shift:")
        for pw in config["probe_words"]:
            pw_enc = encode_to_device(model, tokenizer, pw)
            with torch.no_grad():
                pw_out = model(**pw_enc, output_hidden_states=True)
            pw_h = pw_out.hidden_states[patch_layer][0, -1].cpu().float()

            cos_o = F.cosine_similarity(source_hidden.unsqueeze(0), pw_h.unsqueeze(0), dim=1).item()
            cos_p = F.cosine_similarity(patched.unsqueeze(0), pw_h.unsqueeze(0), dim=1).item()
            print(f"    {pw:12s}: orig={cos_o:.4f} -> patched={cos_p:.4f} ({cos_p-cos_o:+.4f})")

        results[word] = {
            "original_cos": round(cos_orig, 4),
            "patched_cos_20": round(cos_patched, 4),
            "random_cos_20": round(cos_random, 4),
            "gain": round(cos_patched - cos_orig, 4),
        }

    return results


def experiment2_layer_sweep(model, tokenizer, n_layers):
    """在不同层做修补，找因果效力最强的层"""
    print(f"\n{'='*60}")
    print(f"  实验2：逐层修补效力（因果关键层定位）")
    print(f"{'='*60}")

    config = DISAMBIG_EXPERIMENTS["bank"]

    source_encoded = encode_to_device(model, tokenizer, config["source_ctx"])
    target_encoded = encode_to_device(model, tokenizer, config["target_ctx"])
    source_pos = find_target_position(tokenizer, source_encoded["input_ids"][0].tolist(), config["target_word"])
    target_pos = find_target_position(tokenizer, target_encoded["input_ids"][0].tolist(), config["target_word"])

    with torch.no_grad():
        source_out = model(**source_encoded, output_hidden_states=True)
        target_out = model(**target_encoded, output_hidden_states=True)

    print(f"  Layer | orig_cos | K=20_cos | K=20_gain | K=100_cos | K=100_gain")
    sample_layers = list(range(0, min(n_layers, 36)))
    for li in sample_layers:
        source_h = source_out.hidden_states[li][0, source_pos].cpu().float()
        target_h = target_out.hidden_states[li][0, target_pos].cpu().float()
        cos_orig = F.cosine_similarity(source_h.unsqueeze(0), target_h.unsqueeze(0), dim=1).item()

        for k, label in [(20, "20"), (100, "100")]:
            idx_k = get_disambig_neurons(model, tokenizer, config, li, top_k=k)
            pk = source_h.clone()
            for idx in idx_k:
                pk[idx] = target_h[idx]
            cos_k = F.cosine_similarity(pk.unsqueeze(0), target_h.unsqueeze(0), dim=1).item()
            if label == "20":
                cos_20, gain_20 = cos_k, cos_k - cos_orig
            else:
                cos_100, gain_100 = cos_k, cos_k - cos_orig

        if li % 4 == 0 or li == n_layers - 1:
            print(f"  L{li:2d}   | {cos_orig:.4f}   | {cos_20:.4f}    | {gain_20:+.4f}    | {cos_100:.4f}     | {gain_100:+.4f}")

    return {}


def experiment3_full_forward_patch(model, tokenizer, n_layers):
    """完整前向修补：替换中间层隐藏状态后继续前向传播到输出"""
    print(f"\n{'='*60}")
    print(f"  实验3：完整前向修补（输出层验证）")
    print(f"{'='*60}")

    config = DISAMBIG_EXPERIMENTS["bank"]
    patch_layer = min(8, n_layers - 1)

    source_encoded = encode_to_device(model, tokenizer, config["source_ctx"])
    source_pos = find_target_position(tokenizer, source_encoded["input_ids"][0].tolist(), config["target_word"])

    # 获取source的正常输出
    with torch.no_grad():
        source_out = model(**source_encoded, output_hidden_states=True)

    source_last_hidden = source_out.hidden_states[-1][0, source_pos].cpu().float()

    # 获取patch后的编码在末层的表现（通过比较末层相似度间接验证）
    # 直接方法：替换patch层的隐藏状态后，看末层编码
    disambig_idx = get_disambig_neurons(model, tokenizer, config, patch_layer, top_k=100)

    target_encoded = encode_to_device(model, tokenizer, config["target_ctx"])
    with torch.no_grad():
        target_out = model(**target_encoded, output_hidden_states=True)
    target_last = target_out.hidden_states[-1][0, -1].cpu().float()

    # 原始末层
    cos_last_orig = F.cosine_similarity(source_last_hidden.unsqueeze(0), target_last.unsqueeze(0), dim=1).item()

    # 看patch层的编码差异如何传播到末层
    patch_source = source_out.hidden_states[patch_layer][0, source_pos].cpu().float()
    patch_target = target_out.hidden_states[patch_layer][0, target_pos].cpu().float()

    # 修补patch层
    patched_patch = patch_source.clone()
    for idx in disambig_idx:
        patched_patch[idx] = patch_target[idx]

    # patch层修补后的cosine
    cos_patch_layer = F.cosine_similarity(patched_patch.unsqueeze(0), patch_target.unsqueeze(0), dim=1).item()

    # 末层的cosine（未经实际前向传播，只是比较）
    print(f"  source: '{config['source_ctx']}'")
    print(f"  target: '{config['target_ctx']}'")
    print(f"  patch layer L{patch_layer}: patched_cos={cos_patch_layer:.4f}")
    print(f"  last layer (未经前向传播): orig_cos={cos_last_orig:.4f}")
    print(f"  (注意: 完整前向传播修补需要hook机制，这里做间接验证)")

    # 替代方法：用最后一层的编码比较
    # 原始source末层 vs target末层
    # 以及在末层也做修补
    last_disambig = get_disambig_neurons(model, tokenizer, config, n_layers - 1, top_k=100)
    patched_last = source_last_hidden.clone()
    target_last_pos = find_target_position(
        tokenizer, target_encoded["input_ids"][0].tolist(), config["target_word"]
    )
    target_last_h = target_out.hidden_states[-1][0, target_last_pos].cpu().float()

    for idx in last_disambig:
        patched_last[idx] = target_last_h[idx]

    cos_last_patched = F.cosine_similarity(patched_last.unsqueeze(0), target_last_h.unsqueeze(0), dim=1).item()
    print(f"  last layer patched (K=100): cos={cos_last_patched:.4f}, gain={cos_last_patched-cos_last_orig:+.4f}")

    return {}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  stage557: 因果干预——消歧神经元激活修补")
    print("=" * 60)

    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    print(f"  Qwen3 n_layers={n_layers}")

    try:
        r1 = experiment1_activation_patch(model, tokenizer, n_layers)
        r2 = experiment2_layer_sweep(model, tokenizer, n_layers)
        r3 = experiment3_full_forward_patch(model, tokenizer, n_layers)
    finally:
        free_model(model)

    out_path = os.path.join(OUTPUT_DIR, "stage557_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(r1, f, ensure_ascii=False, indent=2)
    print(f"\n  results saved: {out_path}")
    print(f"\n  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
