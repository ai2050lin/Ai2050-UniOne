#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage606-607-608: 逐层MLP vs Attention旋转贡献 + 全息编码验证 + MLP注入消融

Stage606: 逐层追踪消歧方向，用hook分别阻断MLP和Attention，测量各自旋转贡献
Stage607: 全息编码假说——unembed矩阵能否从均匀分散的hidden state中提取消歧信息
Stage608: MLP层注入/消融——在特定MLP层注入/删除消歧方向，验证MLP是旋转执行者

用法: python stage606_607_608_mlp_holographic_ablation.py [qwen3|deepseek7b|glm4|gemma4]
"""

from __future__ import annotations
import sys, json, time, gc, torch, os
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    load_model_bundle, free_model, discover_layers
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


def cos_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def safe_get_device(model):
    for attr in [None, 'model', 'model.model']:
        try:
            obj = model
            if attr:
                for part in attr.split('.'):
                    obj = getattr(obj, part)
            return next(obj.parameters()).device
        except (StopIteration, AttributeError):
            continue
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def move_to_device(batch, model):
    device = safe_get_device(model)
    if hasattr(batch, 'to'):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
    return batch


# ============ Stage606: 逐层MLP vs Attention旋转贡献 ============

def find_mlp_and_attn_components(layer_module):
    """找出transformer层中的MLP和Attention子模块"""
    attn_module = None
    mlp_module = None

    for name, child in layer_module.named_children():
        name_lower = name.lower()
        if 'attn' in name_lower or 'attention' in name_lower or 'self_attn' in name_lower:
            attn_module = child
        if 'mlp' in name_lower or 'feed_forward' in name_lower or 'ffn' in name_lower:
            mlp_module = child

    return attn_module, mlp_module


def run_stage606(model, tokenizer, model_key):
    """
    逐层追踪消歧方向，通过比较相邻层间的方向变化来分离MLP和Attention的贡献。
    
    新方法：不用hook，而是用隐藏状态差来推断
    - 对于层i，输出 = Attention(输入) + MLP(Attention(输入))
    - 如果知道层的输入和输出，可以分离两者贡献
    - 但hidden_states给出的是每层输出（已经包含了residual），所以无法直接分离
    
    改用方法：计算每层的"旋转角度"（方向余弦变化），逐层分析旋转速度
    """
    print(f"\n  --- Stage606: 逐层消歧方向旋转速度分析 ---")
    t0 = time.time()
    device = safe_get_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)

    disamb_pairs = [
        ("The river bank was muddy.", "The bank approved the loan.", "bank"),
        ("She ate a red apple.", "Apple released the iPhone.", "apple"),
        ("The factory plant employs workers.", "She watered the plant.", "plant"),
    ]

    # Get all hidden states
    word_states = {}
    for s1, s2, word in disamb_pairs:
        h1s, h2s = [], []
        for s in [s1, s2]:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
            enc = move_to_device(enc, model)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            for h in out.hidden_states:
                hv = h[0, -1, :].float().cpu()
                if s == s1:
                    h1s.append(hv)
                else:
                    h2s.append(hv)
        word_states[word] = (h1s, h2s)

    results = {}
    for word in word_states:
        h1s, h2s = word_states[word]

        # Find peak layer
        max_d = 0
        peak_l = 0
        for li in range(n_layers):
            d = 1 - cos_sim(h1s[li], h2s[li])
            if d > max_d:
                max_d = d
                peak_l = li

        # Compute per-layer rotation analysis
        layer_data = []
        for li in range(1, n_layers):
            d_prev = h1s[li-1] - h2s[li-1]
            d_curr = h1s[li] - h2s[li]
            d_prev_norm = F.normalize(d_prev, dim=0)
            d_curr_norm = F.normalize(d_curr, dim=0)

            dir_cos = cos_sim(d_prev_norm, d_curr_norm)
            mag_change = torch.norm(d_curr).item() / max(torch.norm(d_prev).item(), 1e-10)
            disamb_curr = 1 - cos_sim(h1s[li], h2s[li])
            disamb_prev = 1 - cos_sim(h1s[li-1], h2s[li-1])

            # "Rotation angle" = arccos(dir_cos), in degrees
            rot_angle = np.degrees(np.arccos(np.clip(dir_cos, -1, 1)))

            # Energy of parallel and orthogonal components
            proj_par = torch.dot(d_curr, d_prev_norm)
            v_par = proj_par * d_prev_norm
            v_orth = d_curr - v_par
            par_energy = torch.norm(v_par).item() ** 2
            orth_energy = torch.norm(v_orth).item() ** 2
            total_energy = torch.norm(d_curr).item() ** 2

            layer_data.append({
                "layer": li,
                "dir_cos": round(dir_cos, 6),
                "rotation_deg": round(rot_angle, 2),
                "mag_change": round(mag_change, 4),
                "disamb": round(disamb_curr, 6),
                "disamb_change": round(disamb_curr - disamb_prev, 6),
                "par_energy_ratio": round(par_energy / max(total_energy, 1e-10), 4),
                "orth_energy_ratio": round(orth_energy / max(total_energy, 1e-10), 4),
            })

        # Summarize: identify layers with maximum rotation
        max_rot = max(layer_data, key=lambda x: x["rotation_deg"])
        max_mag = max(layer_data, key=lambda x: x["mag_change"])

        # Identify rotation phases
        # Pre-peak: layers before peak with increasing rotation
        # Post-peak: layers after peak
        pre_peak = [ld for ld in layer_data if ld["layer"] <= peak_l]
        post_peak = [ld for ld in layer_data if ld["layer"] > peak_l]

        avg_rot_pre = np.mean([ld["rotation_deg"] for ld in pre_peak]) if pre_peak else 0
        avg_rot_post = np.mean([ld["rotation_deg"] for ld in post_peak]) if post_peak else 0
        avg_mag_pre = np.mean([ld["mag_change"] for ld in pre_peak]) if pre_peak else 0
        avg_mag_post = np.mean([ld["mag_change"] for ld in post_peak]) if post_peak else 0

        results[word] = {
            "peak_layer": peak_l,
            "peak_disamb": round(max_d, 4),
            "max_rotation_layer": max_rot["layer"],
            "max_rotation_deg": max_rot["rotation_deg"],
            "max_magnitude_layer": max_mag["layer"],
            "max_magnitude_ratio": max_mag["mag_change"],
            "avg_rotation_pre_peak": round(avg_rot_pre, 2),
            "avg_rotation_post_peak": round(avg_rot_post, 2),
            "avg_magnitude_pre_peak": round(avg_mag_pre, 4),
            "avg_magnitude_post_peak": round(avg_mag_post, 4),
            "layer_data": layer_data,
        }

        print(f"    {word}: peak=L{peak_l}, max_rot=L{max_rot['layer']}({max_rot['rotation_deg']:.1f}deg), "
              f"avg_rot pre={avg_rot_pre:.1f} post={avg_rot_post:.1f} deg")

    elapsed = time.time() - t0
    print(f"  Stage606 done in {elapsed:.1f}s")
    return {"rotation_speed_analysis": results}


# ============ Stage607: 全息编码假说验证 ============

def run_stage607(model, tokenizer, model_key):
    """
    全息编码假说验证：如果消歧信息被均匀分散到所有维度，
    那么unembed矩阵应该能通过线性组合提取出这些信息。
    
    算法：
    1. 获取末层hidden state的差异向量 d = h(ctx1) - h(ctx2)
    2. 将 d 投影到unembed矩阵的行空间
    3. 检查投影后向量的top-k成分是否对应语义相关的token
    4. 测量"信息可提取度"——投影后保留了多少差异信息
    """
    print(f"\n  --- Stage607: 全息编码假说验证 ---")
    t0 = time.time()
    device = safe_get_device(model)

    disamb_pairs = [
        ("The river bank was muddy.", "The bank approved the loan.", "bank"),
        ("She ate a red apple.", "Apple released the iPhone.", "apple"),
        ("The factory plant employs workers.", "She watered the plant.", "plant"),
        ("The hot spring resort.", "Spring is beautiful.", "spring"),
        ("He hit the nail with a hammer.", "She painted her fingernail.", "nail"),
    ]

    # Get unembed matrix
    unembed = None
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        unembed = model.lm_head.weight.data.float()
    elif hasattr(model, 'get_output_embeddings'):
        oe = model.get_output_embeddings()
        if oe is not None:
            unembed = oe.weight.data.float()

    if unembed is None:
        print("  WARNING: No unembed matrix found, skipping Stage607")
        return {"error": "no unembed matrix"}

    unembed_cpu = unembed.cpu()
    vocab_size, hidden_dim = unembed_cpu.shape

    results = {}

    for s1, s2, word in disamb_pairs:
        # Get last layer hidden states
        h1_last = None
        h2_last = None
        for s in [s1, s2]:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
            enc = move_to_device(enc, model)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :].float().cpu()
            if s == s1:
                h1_last = h
            else:
                h2_last = h

        d = h1_last - h2_last  # disambiguation direction vector

        # Compute logits for both contexts
        logits1 = unembed_cpu @ h1_last  # [vocab_size]
        logits2 = unembed_cpu @ h2_last  # [vocab_size]
        logits_diff = logits1 - logits2   # difference in logit space

        # The "recovered" information: how much of d is captured by unembed?
        # Project d onto unembed's row space (which is the same as unembed @ unembed.T @ d_norm)
        d_norm = F.normalize(d, dim=0)
        # Projection: for each token, its contribution to recovering d is unembed[i] * (unembed[i] . d)
        projections = unembed_cpu @ d  # [vocab_size] - dot product of each token's embedding with d

        # Sort by absolute projection value
        top_k = 20
        top_vals, top_ids = torch.topk(projections.abs(), top_k)
        top_tokens = [tokenizer.decode([idx]).strip() for idx in top_ids.tolist()]

        # Also compute the "reconstruction ratio":
        # How much of d's energy is in the row space of unembed?
        # If unembed has rank = min(vocab_size, hidden_dim), and hidden_dim << vocab_size,
        # then unembed.T @ unembed is a hidden_dim x hidden_dim matrix
        U = unembed_cpu  # [vocab, hidden]
        d_f = d.float()
        energy_total = torch.norm(d_f).item() ** 2
        # Since vocab_size >> hidden_dim, unembed's row space spans all of hidden space
        # Instead, measure how much the logit DIFFERENCE captures the disambiguation
        logits_diff = U @ d_f  # [vocab_size]
        logit_diff_energy = torch.norm(logits_diff).item() ** 2

        # Information concentration: what fraction of logit_diff energy is in top-k tokens?
        topk_vals, _ = torch.topk(logits_diff.abs(), 100)
        top100_energy = torch.norm(topk_vals).item() ** 2
        concentration_ratio = top100_energy / max(logit_diff_energy, 1e-10)
        # top100 concentration

        # Logit difference analysis
        top_logit_diff_vals, top_logit_diff_ids = torch.topk(logits_diff.abs(), 10)
        top_logit_tokens = [tokenizer.decode([idx]).strip() for idx in top_logit_diff_ids.tolist()]
        top_logit_signs = [float(logits1[idx].item() - logits2[idx].item()) for idx in top_logit_diff_ids.tolist()]

        # Compare: what does model actually generate differently?
        top_logits1_ids = torch.topk(logits1, 5).indices.tolist()
        top_logits2_ids = torch.topk(logits2, 5).indices.tolist()
        gen1_top = [tokenizer.decode([idx]).strip() for idx in top_logits1_ids]
        gen2_top = [tokenizer.decode([idx]).strip() for idx in top_logits2_ids]

        results[word] = {
            "energy_total": round(energy_total, 2),
            "logit_diff_top100_concentration": round(concentration_ratio, 4),
            "top_projections": list(zip(top_tokens, [round(float(v), 4) for v in top_vals.tolist()])),
            "top_logit_diffs": list(zip(top_logit_tokens,
                                        [round(float(v), 2) for v in top_logit_diff_vals.tolist()],
                                        [round(s, 2) for s in top_logit_signs])),
            "gen1_top5": gen1_top,
            "gen2_top5": gen2_top,
            "hidden_dim": hidden_dim,
            "vocab_size": vocab_size,
        }

        print(f"    {word}: top100_concentration={results[word]['logit_diff_top100_concentration']:.4f}, "
              f"logit_diff_top={top_logit_tokens[:3]}")

    elapsed = time.time() - t0
    print(f"  Stage607 done in {elapsed:.1f}s")
    return {"holographic_analysis": results}


# ============ Stage608: MLP层注入消融 ============

def run_stage608(model, tokenizer, model_key):
    """
    在特定MLP层注入/删除消歧方向，验证MLP是旋转执行者。
    
    算法：
    1. 获取消歧峰值层的消歧方向
    2. 在后续各MLP层的输入端注入该方向
    3. 在MLP层的输出端删除该方向的分量
    4. 比较生成结果，确定MLP对旋转的贡献
    """
    print(f"\n  --- Stage608: MLP层注入消融 ---")
    t0 = time.time()
    device = safe_get_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)

    test_words = [
        ("The river bank was muddy.", "The bank approved the loan.", "bank"),
        ("She ate a red apple.", "Apple released the iPhone.", "apple"),
    ]

    results = []

    for s1, s2, word in test_words:
        # Get peak layer and disambiguation direction
        enc1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=128)
        enc1 = move_to_device(enc1, model)
        enc2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=128)
        enc2 = move_to_device(enc2, model)

        with torch.no_grad():
            out1 = model(**enc1, output_hidden_states=True)
            out2 = model(**enc2, output_hidden_states=True)

        peak_l = 0
        max_d = 0
        for li in range(n_layers):
            d = 1 - cos_sim(out1.hidden_states[li][0, -1, :].float(),
                           out2.hidden_states[li][0, -1, :].float())
            if d > max_d:
                max_d = d
                peak_l = li

        d_peak = (out1.hidden_states[peak_l][0, -1, :].float() -
                  out2.hidden_states[peak_l][0, -1, :].float())
        d_peak_f = F.normalize(d_peak, dim=0).to(device)

        # Baseline generation for ctx1
        with torch.no_grad():
            out_base = model.generate(
                **enc1, max_new_tokens=8, do_sample=False, temperature=0.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen_base = tokenizer.decode(
            out_base[0][enc1["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()[:20]

        # Test: for 3 layers after peak, inject d_peak into MLP input
        test_layer_offsets = [0, 3, 6]  # relative to peak
        layer_results = {}

        for offset in test_layer_offsets:
            target_l = min(peak_l + offset, n_layers - 1)
            if target_l >= n_layers:
                continue

            layer_mod = layers[target_l]
            _, mlp_mod = find_mlp_and_attn_components(layer_mod)

            if mlp_mod is None:
                continue

            # Test 1: Add d_peak at MLP input (pre-MLP injection)
            def make_pre_mlp_hook(direction, scale):
                def hook_fn(module, input):
                    if isinstance(input, tuple) and len(input) > 0:
                        x = input[0]
                        norm = x[0, -1, :].float().norm()
                        modified = x.clone()
                        modified[0, -1, :] = modified[0, -1, :] + (scale * direction * norm).to(modified.dtype)
                        return (modified,) + input[1:]
                    return input
                return hook_fn

            try:
                h = mlp_mod.register_forward_pre_hook(make_pre_mlp_hook(d_peak_f, 1.0))
                with torch.no_grad():
                    out_inj = model.generate(
                        **enc1, max_new_tokens=8, do_sample=False, temperature=0.0,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                h.remove()
                gen_pre_inj = tokenizer.decode(
                    out_inj[0][enc1["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()[:20]
            except Exception as e:
                gen_pre_inj = f"ERR: {str(e)[:30]}"

            # Test 2: Remove d_peak component at MLP output (post-MLP ablation)
            def make_post_mlp_hook(direction):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        x = output[0]
                    else:
                        x = output
                    modified = x.clone()
                    last_tok = modified[0, -1, :].float()
                    proj = torch.dot(last_tok, direction.float()) * direction.float()
                    modified[0, -1, :] = modified[0, -1, :] - proj.to(modified.dtype)
                    if isinstance(output, tuple):
                        return (modified,) + output[1:]
                    return modified
                return hook_fn

            try:
                h = mlp_mod.register_forward_hook(make_post_mlp_hook(d_peak_f))
                with torch.no_grad():
                    out_abl = model.generate(
                        **enc1, max_new_tokens=8, do_sample=False, temperature=0.0,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                h.remove()
                gen_post_abl = tokenizer.decode(
                    out_abl[0][enc1["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()[:20]
            except Exception as e:
                gen_post_abl = f"ERR: {str(e)[:30]}"

            layer_results[f"L{target_l}(peak+{offset})"] = {
                "pre_mlp_inject": gen_pre_inj,
                "post_mlp_ablate": gen_post_abl,
                "pre_changed": gen_pre_inj != gen_base,
                "post_changed": gen_post_abl != gen_base,
            }

        results.append({
            "word": word, "peak_layer": peak_l, "peak_disamb": round(max_d, 4),
            "gen_baseline": gen_base,
            "layer_results": layer_results,
        })

        pre_changed = sum(1 for v in layer_results.values() if v["pre_changed"])
        post_changed = sum(1 for v in layer_results.values() if v["post_changed"])
        print(f"    {word}: peak=L{peak_l}, base='{gen_base}' "
              f"pre_mlp_changed={pre_changed}/{len(layer_results)}, "
              f"post_mlp_changed={post_changed}/{len(layer_results)}")

    elapsed = time.time() - t0
    print(f"  Stage608 done in {elapsed:.1f}s")
    return {"mlp_ablation_analysis": results}


# ============ Main ============

MODEL_KEYS = ["qwen3", "deepseek7b", "glm4", "gemma4"]


def run_single_model(mk):
    print(f"\n{'='*60}")
    print(f"  Loading {mk}...")
    print(f"{'='*60}")
    t0 = time.time()
    model, tokenizer = load_model_bundle(mk)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    try:
        s606 = run_stage606(model, tokenizer, mk)
        s607 = run_stage607(model, tokenizer, mk)
        s608 = run_stage608(model, tokenizer, mk)
        result = {"stage606": s606, "stage607": s607, "stage608": s608}
    except Exception as e:
        import traceback
        print(f"  ERROR in {mk}: {e}")
        traceback.print_exc()
        result = {"error": str(e)}
    finally:
        free_model(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        time.sleep(3)
    return result


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target not in MODEL_KEYS:
            print(f"Unknown model: {target}. Use one of: {MODEL_KEYS}")
            return
        models_to_run = [target]
    else:
        models_to_run = MODEL_KEYS

    combined_path = OUTPUT_DIR / f"stage606_607_608_combined_{TIMESTAMP}.json"
    combined = {"timestamp": TIMESTAMP, "models": {}}

    existing_files = sorted(OUTPUT_DIR.glob("stage606_607_608_combined_*.json"),
                            key=lambda x: x.stat().st_mtime, reverse=True)
    if existing_files and len(sys.argv) == 1:
        try:
            with open(existing_files[0], "r", encoding="utf-8") as f:
                prev = json.load(f)
            combined = prev
            combined_path = existing_files[0]
            print(f"Resuming from {existing_files[0].name}")
        except:
            pass

    for mk in models_to_run:
        if mk in combined["models"] and "error" not in combined["models"][mk]:
            print(f"\n  Skipping {mk} (already completed)")
            continue

        result = run_single_model(mk)
        combined["models"][mk] = result

        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to {combined_path}")

    print(f"\nAll done. Results: {combined_path}")


if __name__ == "__main__":
    main()
