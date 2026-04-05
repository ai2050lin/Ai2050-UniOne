#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage603-604-605: 随机探针分析 + 因果消融(注入实验) + Gemma4 config排查
每次运行一个模型以避免GPU内存问题。

Stage603: 1000随机方向探针覆盖hidden空间，找旋转方向的投影分布
Stage604: 在末层注入峰值消歧方向，验证是否能提升生成准确率（因果消融）
Stage605: Gemma4 config完整排查 + head数量vs消歧度分析

用法: python stage603_604_605_random_probe_causal_config.py [qwen3|deepseek7b|glm4|gemma4]
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


# ============ Stage603: 随机探针分析 ============

def run_stage603(model, tokenizer, model_key):
    """
    用1000个随机方向探针覆盖hidden空间，测量旋转方向与它们的对齐度。
    如果旋转方向与所有随机探针的对齐都接近0，说明信息被分散编码到高维空间。
    如果存在显著高于随机的对齐，说明旋转方向在特定子空间。
    """
    print(f"\n  --- Stage603: 随机探针分析 ---")
    t0 = time.time()
    layers = discover_layers(model)
    n_layers = len(layers)
    device = safe_get_device(model)

    disamb_pairs = [
        ("The river bank was muddy.", "The bank approved the loan.", "bank"),
        ("She ate a red apple.", "Apple released the iPhone.", "apple"),
        ("The factory plant employs workers.", "She watered the plant.", "plant"),
        ("The hot spring resort.", "Spring is beautiful.", "spring"),
        ("He hit the nail with a hammer.", "She painted her fingernail.", "nail"),
    ]

    # Step 1: Get hidden states for all pairs
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

    # Step 2: Find peak layer for each word
    word_peaks = {}
    for word in word_states:
        h1s, h2s = word_states[word]
        max_d = 0
        best_l = 0
        for li in range(n_layers):
            d = 1 - cos_sim(h1s[li], h2s[li])
            if d > max_d:
                max_d = d
                best_l = li
        word_peaks[word] = {"peak_layer": best_l, "peak_disamb": max_d}

    # Step 3: For each word, compute rotation direction at peak+75%
    # and test against 1000 random probes
    np.random.seed(42)
    n_probes = 1000
    analysis = {}

    for word in word_states:
        h1s, h2s = word_states[word]
        pk = word_peaks[word]
        peak_l = pk["peak_layer"]
        target_l = min(peak_l + int((n_layers - peak_l) * 0.75), n_layers - 1)
        if target_l <= peak_l:
            target_l = min(peak_l + 1, n_layers - 1)

        # Peak direction
        d_peak = h1s[peak_l] - h2s[peak_l]
        d_peak_norm = F.normalize(d_peak, dim=0)

        # Target direction
        d_target = h1s[target_l] - h2s[target_l]
        d_target_norm = F.normalize(d_target, dim=0)

        # Orthogonal component (what's new after rotation)
        proj_par = torch.dot(d_target, d_peak_norm)
        v_orth = d_target - proj_par * d_peak_norm
        v_orth_norm = F.normalize(v_orth, dim=0)

        # Also compute parallel component
        v_par = proj_par * d_peak_norm
        v_par_norm = F.normalize(v_par, dim=0) if torch.norm(v_par) > 1e-8 else None

        # Generate 1000 random probes in same dimensionality
        dim = d_peak.shape[0]
        random_probes = torch.randn(n_probes, dim)
        random_probes = F.normalize(random_probes, dim=1)

        # Compute cosines
        orth_cosines = []
        par_cosines = []
        target_cosines = []
        for i in range(n_probes):
            orth_cosines.append(cos_sim(v_orth_norm, random_probes[i]))
            if v_par_norm is not None:
                par_cosines.append(cos_sim(v_par_norm, random_probes[i]))
            target_cosines.append(cos_sim(d_target_norm, random_probes[i]))

        orth_cosines = np.array(orth_cosines)
        par_cosines = np.array(par_cosines) if par_cosines else np.array([0.0])
        target_cosines = np.array(target_cosines)

        # Analysis: compare to theoretical distribution
        # For random unit vectors in d-dimensional space,
        # the distribution of cosine similarity is approximately
        # N(0, 1/sqrt(d)) for large d
        theoretical_std = 1.0 / np.sqrt(dim)

        # Check for "hot spots" - probes with |cos| > 3*theoretical_std
        hotspot_threshold = 3 * theoretical_std
        n_hotspots_orth = np.sum(np.abs(orth_cosines) > hotspot_threshold)
        n_hotspots_target = np.sum(np.abs(target_cosines) > hotspot_threshold)

        # Also test: rank of the orthogonal component
        # Use SVD to check how many principal components capture 90% of variance
        # (We only have one vector, so we project it onto random orthogonal subspaces)

        # Energy analysis
        orth_energy = torch.norm(v_orth).item() ** 2
        par_energy = torch.norm(v_par).item() ** 2
        total_energy = torch.norm(d_target).item() ** 2

        # Direction cosine
        dir_cos = cos_sim(d_peak_norm, d_target_norm)

        analysis[word] = {
            "peak_layer": peak_l,
            "target_layer": target_l,
            "dir_cos": round(dir_cos, 6),
            "orth_energy_ratio": round(orth_energy / max(total_energy, 1e-10), 4),
            "par_energy_ratio": round(par_energy / max(total_energy, 1e-10), 4),
            "hidden_dim": dim,
            "theoretical_std": round(theoretical_std, 6),
            "orth_cosine_stats": {
                "mean": round(float(np.mean(orth_cosines)), 6),
                "std": round(float(np.std(orth_cosines)), 6),
                "max_abs": round(float(np.max(np.abs(orth_cosines))), 6),
                "p95_abs": round(float(np.percentile(np.abs(orth_cosines), 95)), 6),
                "p99_abs": round(float(np.percentile(np.abs(orth_cosines), 99)), 6),
                "n_hotspots": int(n_hotspots_orth),
            },
            "target_cosine_stats": {
                "mean": round(float(np.mean(target_cosines)), 6),
                "std": round(float(np.std(target_cosines)), 6),
                "max_abs": round(float(np.max(np.abs(target_cosines))), 6),
                "n_hotspots": int(n_hotspots_target),
            },
            "par_cosine_stats": {
                "mean": round(float(np.mean(par_cosines)), 6),
                "std": round(float(np.std(par_cosines)), 6),
                "max_abs": round(float(np.max(np.abs(par_cosines))), 6),
            } if v_par_norm is not None else {"mean": 0, "std": 0, "max_abs": 0},
        }

        print(f"    {word}: orth_std={np.std(orth_cosines):.4f} (theo={theoretical_std:.4f}), "
              f"hotspots={n_hotspots_orth}, dir_cos={dir_cos:.4f}")

    elapsed = time.time() - t0
    print(f"  Stage603 done in {elapsed:.1f}s")
    return {"random_probe_analysis": analysis}


# ============ Stage604: 因果消融(注入实验) ============

def run_stage604(model, tokenizer, model_key):
    """
    因果消融实验：在模型生成时，修改末层hidden state，
    注入峰值层的消歧方向，看是否能提升生成准确率。

    算法原理：
    1. 对消歧失败的词（Gemma4的大部分词），获取其末层hidden state
    2. 同时获取消歧成功的上下文（如Qwen3的同一词）在峰值层的消歧方向
    3. 将该消歧方向注入到末层hidden state中
    4. 比较注入前后的生成准确率变化

    由于我们无法跨模型注入（维度不同），改为：
    对同一模型，在末层注入/减去峰值消歧方向的scaled版本，
    观察生成结果的变化方向。
    """
    print(f"\n  --- Stage604: 因果消融(注入实验) ---")
    t0 = time.time()
    device = safe_get_device(model)

    # Use 5 core disambiguation words
    test_cases = [
        ("The river bank was muddy.", "bank", "river"),
        ("The bank approved the loan.", "bank", "finance"),
        ("She ate a red apple.", "apple", "fruit"),
        ("Apple released the iPhone.", "apple", "tech"),
        ("The factory plant employs workers.", "plant", "factory"),
        ("She watered the plant.", "plant", "botany"),
        ("The hot spring resort.", "spring", "water"),
        ("Spring is beautiful.", "spring", "season"),
    ]

    results = []

    for sentence, word, ctx_type in test_cases:
        enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
        enc = move_to_device(enc, model)

        with torch.no_grad():
            # Normal generation (baseline)
            out_base = model.generate(
                **enc, max_new_tokens=8, do_sample=False, temperature=0.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            gen_base = tokenizer.decode(
                out_base[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            # Get hidden states to find peak disambiguation direction
            out_hs = model(**enc, output_hidden_states=True)
            hiddens = out_hs.hidden_states
            n_hs = len(hiddens)

            # Get the last token's hidden states
            last_hiddens = [h[0, -1, :].float() for h in hiddens]

        # For this word, we need a contrastive context to define the disambiguation direction
        contrast_map = {
            "bank": {"river": "The bank approved the loan.", "finance": "The river bank was muddy."},
            "apple": {"fruit": "Apple released the iPhone.", "tech": "She ate a red apple."},
            "plant": {"factory": "She watered the plant.", "botany": "The factory plant employs workers."},
            "spring": {"water": "Spring is beautiful.", "season": "The hot spring resort."},
        }

        if word not in contrast_map or ctx_type not in contrast_map[word]:
            results.append({
                "word": word, "ctx": ctx_type, "sentence": sentence,
                "gen_baseline": gen_base[:20], "gen_inject": gen_base[:20],
                "gen_subtract": gen_base[:20], "changed": False, "direction_change": "N/A",
            })
            continue

        contrast_sentence = contrast_map[word][ctx_type]
        enc2 = tokenizer(contrast_sentence, return_tensors="pt", truncation=True, max_length=128)
        enc2 = move_to_device(enc2, model)

        with torch.no_grad():
            out_hs2 = model(**enc2, output_hidden_states=True)
            hiddens2 = out_hs2.hidden_states
            last_hiddens2 = [h[0, -1, :].float() for h in hiddens2]

        # Find peak disambiguation layer
        max_disamb = 0
        best_l = 0
        for li in range(n_hs):
            d = 1 - cos_sim(last_hiddens[li], last_hiddens2[li])
            if d > max_disamb:
                max_disamb = d
                best_l = li

        # Get disambiguation direction at peak layer
        d_peak = last_hiddens[best_l] - last_hiddens2[best_l]
        d_peak_norm = F.normalize(d_peak.to(device), dim=0)

        # Now inject this direction at the final layer with varying scales
        # We do a hook-based injection
        scales = [0.1, 0.5, 1.0, 2.0]
        gen_injected = {}
        gen_subtracted = {}

        for scale in scales:
            # Inject
            def make_hook(direction, s):
                def hook_fn(module, input, output):
                    # output is tuple: (hidden_states,) for some models
                    if isinstance(output, tuple):
                        hs = output[0]
                    else:
                        hs = output
                    # Modify last token's hidden state
                    new_hs = hs.clone()
                    new_hs[0, -1, :] = hs[0, -1, :] + s * direction * hs[0, -1, :].norm()
                    if isinstance(output, tuple):
                        return (new_hs,) + output[1:]
                    return new_hs
                return hook_fn

            # Find the last layer to hook
            layers = discover_layers(model)
            last_layer = layers[-1] if layers else None
            if last_layer is None:
                continue

            try:
                # Inject
                h = last_layer.register_forward_hook(make_hook(d_peak_norm, scale))
                with torch.no_grad():
                    out_inj = model.generate(
                        **enc, max_new_tokens=8, do_sample=False, temperature=0.0,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                h.remove()
                gen_injected[scale] = tokenizer.decode(
                    out_inj[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()[:20]

                # Subtract
                h2 = last_layer.register_forward_hook(make_hook(d_peak_norm, -scale))
                with torch.no_grad():
                    out_sub = model.generate(
                        **enc, max_new_tokens=8, do_sample=False, temperature=0.0,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                h2.remove()
                gen_subtracted[scale] = tokenizer.decode(
                    out_sub[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()[:20]
            except Exception as e:
                gen_injected[scale] = f"ERROR: {str(e)[:50]}"
                gen_subtracted[scale] = f"ERROR: {str(e)[:50]}"

        # Check if any injection changed the output
        changed = any(
            gen_injected.get(s, "") != gen_base[:20] for s in scales
        )
        direction_change = "mixed"
        if changed:
            # Check direction of change
            inj_changed = [s for s in scales if gen_injected.get(s, "") != gen_base[:20]]
            sub_changed = [s for s in scales if gen_subtracted.get(s, "") != gen_base[:20]]
            if inj_changed and not sub_changed:
                direction_change = "inject_only"
            elif sub_changed and not inj_changed:
                direction_change = "subtract_only"
            elif inj_changed and sub_changed:
                direction_change = "both_directions"

        results.append({
            "word": word, "ctx": ctx_type, "sentence": sentence,
            "peak_layer": best_l, "peak_disamb": round(max_disamb, 4),
            "gen_baseline": gen_base[:20],
            "gen_inject": {str(s): v for s, v in gen_injected.items()},
            "gen_subtract": {str(s): v for s, v in gen_subtracted.items()},
            "changed": changed,
            "direction_change": direction_change,
        })

        status = "CHANGED" if changed else "stable"
        print(f"    {word}/{ctx_type}: peak=L{best_l}, baseline='{gen_base[:15]}' [{status}]")

    # Summary
    n_changed = sum(1 for r in results if r["changed"])
    total = len(results)
    print(f"  Stage604: {n_changed}/{total} cases changed by injection")

    elapsed = time.time() - t0
    print(f"  Stage604 done in {elapsed:.1f}s")
    return {"causal_ablation": results, "n_changed": n_changed, "total": total}


# ============ Stage605: Gemma4 config排查 + 架构分析 ============

def run_stage605(model, tokenizer, model_key):
    """
    排查Gemma4的config信息，并做完整的架构参数vs消歧度分析。
    """
    print(f"\n  --- Stage605: Config排查 + 架构分析 ---")
    t0 = time.time()

    # 1. Full config dump
    config_info = {}
    if hasattr(model, 'config'):
        cfg = model.config
        # Dump all config attributes
        for attr in dir(cfg):
            if not attr.startswith('_'):
                try:
                    val = getattr(cfg, attr)
                    if not callable(val):
                        config_info[attr] = str(val) if not isinstance(val, (int, float, bool, str)) else val
                except:
                    pass

    # 2. Try to get architecture info from model structure
    struct_info = {}
    try:
        # Model type
        struct_info["model_type"] = type(model).__name__
        if hasattr(model, 'model'):
            struct_info["inner_type"] = type(model.model).__name__
            if hasattr(model.model, 'model'):
                struct_info["inner_inner_type"] = type(model.model.model).__name__

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        struct_info["total_params"] = total_params
        struct_info["total_params_B"] = round(total_params / 1e9, 2)

        # Try to find attention layers and count heads
        layers = discover_layers(model)
        struct_info["n_discovered_layers"] = len(layers)

        if layers:
            first_layer = layers[0]
            struct_info["first_layer_type"] = type(first_layer).__name__

            # Get all parameter shapes
            param_shapes = {}
            for name, p in first_layer.named_parameters():
                param_shapes[name] = list(p.shape)
            struct_info["first_layer_params"] = param_shapes

            # Infer architecture
            for name, shape in param_shapes.items():
                if "q_proj" in name or "query" in name:
                    struct_info["q_proj_shape"] = shape
                if "k_proj" in name or "key" in name:
                    struct_info["k_proj_shape"] = shape
                if "v_proj" in name or "value" in name:
                    struct_info["v_proj_shape"] = shape
                if "o_proj" in name or "output" in name or "dense" in name:
                    if "attn" in name or "self" in name:
                        struct_info["o_proj_shape"] = shape

    except Exception as e:
        struct_info["error"] = str(e)

    # 3. Cross-model architecture comparison (hardcoded from known results)
    arch_comparison = {
        "qwen3": {"num_heads": 32, "num_kv_heads": 8, "hidden_size": 2560, "num_layers": 36,
                  "intermediate_size": 9728, "final_disamb": 0.133, "gen_accuracy": 0.700},
        "deepseek7b": {"num_heads": 28, "num_kv_heads": 4, "hidden_size": 3584, "num_layers": 28,
                       "intermediate_size": 18944, "final_disamb": 0.107, "gen_accuracy": 0.525},
        "glm4": {"num_heads": 32, "num_kv_heads": 2, "hidden_size": 4096, "num_layers": 40,
                 "intermediate_size": 13696, "final_disamb": 0.150, "gen_accuracy": 0.700},
        "gemma4": {"num_heads": None, "num_kv_heads": None, "hidden_size": None, "num_layers": None,
                   "intermediate_size": None, "final_disamb": 0.033, "gen_accuracy": 0.150},
    }

    # Compute correlations between architecture params and disambiguation
    corr_analysis = {}
    models_with_data = ["qwen3", "deepseek7b", "glm4"]
    for param in ["num_heads", "num_kv_heads", "hidden_size", "num_layers", "intermediate_size"]:
        xs = [arch_comparison[m][param] for m in models_with_data]
        ys = [arch_comparison[m]["final_disamb"] for m in models_with_data]
        if all(x is not None for x in xs):
            r = np.corrcoef(xs, ys)[0, 1]
            corr_analysis[param] = {
                "values": {m: arch_comparison[m][param] for m in models_with_data},
                "correlation_with_disamb": round(float(r), 4),
                "direction": "positive" if r > 0 else "negative",
            }
        else:
            corr_analysis[param] = {"values": {m: arch_comparison[m][param] for m in models_with_data}, "correlation_with_disamb": None}

    # KV head ratio analysis
    kv_ratios = {}
    for m in models_with_data:
        nh = arch_comparison[m]["num_heads"]
        nkv = arch_comparison[m]["num_kv_heads"]
        if nh and nkv:
            kv_ratios[m] = round(nh / nkv, 2)
    if kv_ratios:
        kv_xs = [kv_ratios[m] for m in models_with_data]
        kv_ys = [arch_comparison[m]["final_disamb"] for m in models_with_data]
        kv_r = np.corrcoef(kv_xs, kv_ys)[0, 1]
        corr_analysis["kv_head_ratio"] = {
            "values": kv_ratios,
            "correlation_with_disamb": round(float(kv_r), 4),
            "description": "num_heads / num_kv_heads (higher = more query heads per KV head)",
        }

    # MLP ratio
    mlp_ratios = {}
    for m in models_with_data:
        h = arch_comparison[m]["hidden_size"]
        inter = arch_comparison[m]["intermediate_size"]
        if h and inter:
            mlp_ratios[m] = round(inter / h, 2)
    if mlp_ratios:
        mlp_xs = [mlp_ratios[m] for m in models_with_data]
        mlp_ys = [arch_comparison[m]["final_disamb"] for m in models_with_data]
        mlp_r = np.corrcoef(mlp_xs, mlp_ys)[0, 1]
        corr_analysis["mlp_ratio"] = {
            "values": mlp_ratios,
            "correlation_with_disamb": round(float(mlp_r), 4),
            "description": "intermediate_size / hidden_size",
        }

    print(f"  Config attributes found: {len(config_info)}")
    print(f"  Model structure: {struct_info.get('model_type', 'N/A')}")
    for param, info in corr_analysis.items():
        r = info.get("correlation_with_disamb", "N/A")
        print(f"  {param} vs disamb: r={r}")

    elapsed = time.time() - t0
    print(f"  Stage605 done in {elapsed:.1f}s")
    return {
        "config_dump": config_info,
        "structure_info": struct_info,
        "architecture_comparison": arch_comparison,
        "correlation_analysis": corr_analysis,
    }


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
        s603 = run_stage603(model, tokenizer, mk)
        s604 = run_stage604(model, tokenizer, mk)
        s605 = run_stage605(model, tokenizer, mk)
        result = {"stage603": s603, "stage604": s604, "stage605": s605}
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
        time.sleep(2)
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

    combined_path = OUTPUT_DIR / f"stage603_604_605_combined_{TIMESTAMP}.json"
    combined = {"timestamp": TIMESTAMP, "models": {}}

    # Resume from existing file
    existing_files = sorted(OUTPUT_DIR.glob("stage603_604_605_combined_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
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
