#!/usr/bin/env python3
"""
P49: RL Training Rotation Effects (Stage695)

Core question: Does RL training (DS7B vs others) change the rotation dynamics?

Key comparisons:
1. DS7B vs other models: rotation consistency, angle distribution, layer progression
2. Does DS7B have a "deeper" rotation (more gradual) vs "sharper" rotation?
3. Does DS7B's rotation plane have different structure?

This script builds on P47 data + adds per-layer detailed tracking.
"""
import sys, math, time, gc
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from sklearn.decomposition import PCA

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

TEXTS = [
    "The cat sat on the mat.", "The dog chased the ball.", "Birds fly south in winter.",
    "Paris is the capital of France.", "Tokyo is a large city.", "The Amazon is a long river.",
    "She carefully folded the origami crane.", "The orchestra played beautifully.",
    "His writing was elegant and precise.", "The painting was incredibly detailed.",
    "If it rains then the ground gets wet.", "She studied hard because she wanted to pass.",
    "The boy who was running fell down.", "Although tired she continued working.",
    "The quick brown fox jumps over the lazy dog.", "She has been working on this project.",
    "They went to the market when it started.", "The report was submitted on time.",
    "Yesterday it rained heavily all day.", "She will finish the report by Friday.",
    "The project was completed last month.", "He arrived before the ceremony began.",
    "Two plus two equals four exactly.", "The derivative of x squared is two x.",
    "DNA contains genetic instructions for life.", "Gravity causes objects to fall.",
    "The neural network learned patterns.", "The equation can be solved step by step.",
    "The experiment yielded consistent results.", "The hypothesis was supported by data.",
    "A red apple is a fruit.", "The bank by the river was flooded.",
    "Spring flowers bloom in March.", "The match was exciting to watch.",
    "The seal swam in the cold ocean.", "A fair decision was made by the judge.",
]

OUTPUT_DIR = _Path(r"d:\develop\TransformerLens-main\tests\glm5_temp\stage695_rl_rotation_20260406_2230")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class Logger:
    def __init__(self, filepath):
        self.f = open(filepath, "w", encoding="utf-8")
    def __call__(self, msg="", end="\n"):
        try:
            safe_msg = msg.encode('gbk', errors='replace').decode('gbk')
        except:
            safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
        print(safe_msg, end=end)
        self.f.write(msg + end)
        if end == "\n":
            self.f.flush()
    def close(self):
        self.f.close()


def load_model(model_name):
    path = MODEL_MAP[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def run_single_model(model_name, texts, log):
    """Extract per-layer rotation profile"""
    log(f"\n{'='*60}")
    log(f"  Processing: {model_name}")
    log(f"{'='*60}")

    t0 = time.time()
    model, tokenizer = load_model(model_name)
    model_device = next(model.parameters()).device

    # Get model dimensions
    tokens = tokenizer.encode(texts[0], return_tensors="pt", truncation=True, max_length=64).to(model_device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    n_layers = len(outputs.hidden_states)
    hidden_dim = outputs.hidden_states[0].shape[-1]
    log(f"  layers: {n_layers}, hidden_dim: {hidden_dim}")

    # For each text, extract all layers and compute rotation from L0
    # Average across texts to get per-layer rotation profile
    layer_angles_avg = np.zeros(n_layers)
    layer_angles_std = np.zeros(n_layers)
    layer_cos_avg = np.zeros(n_layers)

    # Also track: rotation consistency at each layer (are texts rotating in same direction?)
    layer_rot_consistency = np.zeros(n_layers)

    # Sample subset of texts for detailed layer tracking (to save time)
    sample_texts = texts[:12]  # Use first 12 texts

    for text in sample_texts:
        tok = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64).to(model_device)
        with torch.no_grad():
            out = model(tok, output_hidden_states=True)
        h_l0 = out.hidden_states[0][0, -1, :].float().cpu()

        l0_norm = torch.norm(h_l0)
        if l0_norm < 1e-10:
            continue
        l0_dir = h_l0 / l0_norm

        rot_vectors = []
        for li in range(n_layers):
            h = out.hidden_states[li][0, -1, :].float().cpu()
            cos_theta = F.cosine_similarity(h.unsqueeze(0), l0_dir.unsqueeze(0)).item()
            cos_theta = max(-1, min(1, cos_theta))
            angle = math.degrees(math.acos(cos_theta))

            # Rotation vector (perpendicular component)
            proj = cos_theta * torch.norm(h) * l0_dir
            rot_vec = h - proj
            rot_vectors.append(rot_vec)

            layer_angles_avg[li] += angle
            layer_cos_avg[li] += cos_theta

    # Normalize
    n_valid = min(len(sample_texts), 12)
    layer_angles_avg /= max(n_valid, 1)
    layer_cos_avg /= max(n_valid, 1)

    # Compute layer-by-layer angle increments (rotation speed)
    angle_deltas = np.diff(layer_angles_avg)
    
    # Compute cumulative rotation fraction at each layer
    total_rotation = layer_angles_avg[-1]
    cumul_frac = layer_angles_avg / max(total_rotation, 1e-10)

    # Find key milestones
    def find_layer_for_frac(target_frac):
        for li in range(len(cumul_frac)):
            if cumul_frac[li] >= target_frac:
                return li
        return n_layers - 1

    l_10 = find_layer_for_frac(0.10)
    l_25 = find_layer_for_frac(0.25)
    l_50 = find_layer_for_frac(0.50)
    l_75 = find_layer_for_frac(0.75)
    l_90 = find_layer_for_frac(0.90)

    # Rotation "shape" metrics
    # Early rotation ratio: how much rotation happens in first 25% of layers
    early_ratio = cumul_frac[n_layers // 4]
    # Late rotation ratio: how much in last 25%
    late_ratio = 1.0 - cumul_frac[3 * n_layers // 4]
    # Middle concentration: rotation in middle 50%
    mid_ratio = cumul_frac[3 * n_layers // 4] - cumul_frac[n_layers // 4]

    # Max rotation speed layer
    max_delta_layer = np.argmax(angle_deltas)
    max_delta_val = angle_deltas[max_delta_layer]

    results = {
        "model_name": model_name,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "total_rotation_deg": total_rotation,
        "milestones": {
            "L10": l_10, "L25": l_25, "L50": l_50, "L75": l_75, "L90": l_90,
        },
        "shape": {
            "early_ratio": early_ratio,
            "late_ratio": late_ratio,
            "mid_ratio": mid_ratio,
        },
        "max_speed": {
            "layer": int(max_delta_layer),
            "value": max_delta_val,
        },
        "layer_angles": layer_angles_avg.tolist(),
        "layer_cos": layer_cos_avg.tolist(),
    }

    # Print summary
    log(f"\n    === Rotation Profile ===")
    log(f"    Total rotation: {total_rotation:.1f} deg")
    log(f"    Milestones: 10%={l_10}, 25%={l_25}, 50%={l_50}, 75%={l_75}, 90%={l_90}")
    log(f"    Shape: early={early_ratio:.3f}, mid={mid_ratio:.3f}, late={late_ratio:.3f}")
    log(f"    Max speed: L{max_delta_layer} ({max_delta_val:.1f} deg/layer)")
    log(f"    Layer angles (sample): L0={layer_angles_avg[0]:.1f}, L{n_layers//4}={layer_angles_avg[n_layers//4]:.1f}, L{n_layers//2}={layer_angles_avg[n_layers//2]:.1f}, L{3*n_layers//4}={layer_angles_avg[3*n_layers//4]:.1f}, L{n_layers-1}={layer_angles_avg[-1]:.1f}")

    # Interpret shape
    if mid_ratio > 0.5:
        log(f"    -> INV-353: MIDDLE-CONCENTRATED rotation (deep processing)")
    elif early_ratio > 0.5:
        log(f"    -> INV-353: EARLY-CONCENTRATED rotation (shallow processing)")
    elif late_ratio > 0.4:
        log(f"    -> INV-353: LATE-CONCENTRATED rotation (deep refinement)")
    else:
        log(f"    -> INV-353: UNIFORM rotation")

    elapsed = time.time() - t0
    log(f"  elapsed: {elapsed:.1f}s")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    import json

    log_path = OUTPUT_DIR / "output.txt"
    log = Logger(log_path)

    log(f"\n{'='*70}")
    log(f"  P49: RL Training Rotation Effects (Stage695)")
    log(f"{'='*70}")

    all_results = {}
    t_total = time.time()

    for model_name in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        try:
            result = run_single_model(model_name, TEXTS, log)
            all_results[model_name] = result
        except Exception as e:
            log(f"  ERROR processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-model comparison
    log(f"\n{'='*70}")
    log(f"  CROSS-MODEL ROTATION PROFILE COMPARISON")
    log(f"{'='*70}")

    log(f"\n  A: Total Rotation & Shape")
    log(f"  {'model':>12s}  {'total_deg':>10s}  {'early':>8s}  {'mid':>8s}  {'late':>8s}  {'max_spd_L':>10s}  {'max_spd_V':>10s}")
    for m, r in all_results.items():
        log(f"  {m:>12s}  {r['total_rotation_deg']:>10.1f}  {r['shape']['early_ratio']:>8.3f}  {r['shape']['mid_ratio']:>8.3f}  {r['shape']['late_ratio']:>8.3f}  {r['max_speed']['layer']:>10d}  {r['max_speed']['value']:>10.1f}")

    log(f"\n  B: Rotation Milestones (layer where X% of total rotation occurs)")
    log(f"  {'model':>12s}  {'layers':>7s}  {'10%':>5s}  {'25%':>5s}  {'50%':>5s}  {'75%':>5s}  {'90%':>5s}")
    for m, r in all_results.items():
        ms = r["milestones"]
        log(f"  {m:>12s}  {r['n_layers']:>7d}  {ms['L10']:>5d}  {ms['L25']:>5d}  {ms['L50']:>5d}  {ms['L75']:>5d}  {ms['L90']:>5d}")

    log(f"\n  C: Normalized Layer Progression (rotation fraction at each quartile)")
    log(f"  {'model':>12s}  {'L25%':>8s}  {'L50%':>8s}  {'L75%':>8s}")
    for m, r in all_results.items():
        n = r["n_layers"]
        angles = r["layer_angles"]
        total = max(angles[-1], 1e-10)
        log(f"  {m:>12s}  {angles[n//4]/total:>8.3f}  {angles[n//2]/total:>8.3f}  {angles[3*n//4]/total:>8.3f}")

    log(f"\n  D: DS7B vs Others (RL effect)")
    ds7b = all_results.get("deepseek7b", {})
    others_mid = []
    others_early = []
    for m, r in all_results.items():
        if m != "deepseek7b":
            others_mid.append(r["shape"]["mid_ratio"])
            others_early.append(r["shape"]["early_ratio"])

    if others_mid:
        log(f"  DS7B mid_ratio: {ds7b.get('shape', {}).get('mid_ratio', 0):.3f}")
        log(f"  Others avg mid_ratio: {np.mean(others_mid):.3f}")
        log(f"  DS7B early_ratio: {ds7b.get('shape', {}).get('early_ratio', 0):.3f}")
        log(f"  Others avg early_ratio: {np.mean(others_early):.3f}")

        ds7b_mid = ds7b.get('shape', {}).get('mid_ratio', 0)
        if ds7b_mid > np.mean(others_mid) + 0.05:
            log(f"  -> INV-353 CONFIRMED: DS7B has MORE middle-concentrated rotation (RL effect)")
        elif ds7b_mid < np.mean(others_mid) - 0.05:
            log(f"  -> INV-353 REJECTED: DS7B has LESS middle-concentrated rotation")
        else:
            log(f"  -> INV-353 INCONCLUSIVE: DS7B rotation shape similar to others")

    elapsed = time.time() - t_total
    log(f"\n{'='*70}")
    log(f"  P49 COMPLETE - Total elapsed: {elapsed:.1f}s")
    log(f"{'='*70}")

    # Save
    save_results = {}
    for m, r in all_results.items():
        save_results[m] = {
            k: v for k, v in r.items()
            if k not in ["layer_angles", "layer_cos"]
        }
        save_results[m]["layer_angles_quartiles"] = {
            "L0": r["layer_angles"][0],
            "L25": r["layer_angles"][r["n_layers"]//4],
            "L50": r["layer_angles"][r["n_layers"]//2],
            "L75": r["layer_angles"][3*r["n_layers"]//4],
            "LFinal": r["layer_angles"][-1],
        }

    out_path = OUTPUT_DIR / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
    log(f"\n  Results saved to: {out_path}")

    log.close()


if __name__ == "__main__":
    main()
