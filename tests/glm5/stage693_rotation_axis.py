#!/usr/bin/env python3
"""
P47: Rotation Axis / Rotation Plane Analysis (Stage693)

Core question: When hidden states rotate from L0 to final layer,
is there a consistent "rotation plane" shared across texts and models?

Method:
1. For each text, compute h_L0 and h_final (last token)
2. Compute rotation component: r = h_final - cos(θ)*h_L0 (perpendicular to L0)
3. Check if rotation components are consistent across texts (same rotation plane)
4. Extract the principal rotation plane via PCA on rotation vectors
5. Compare rotation planes across models via unembed projection

Key metrics:
- rotation_consistency: how aligned are rotation vectors across texts
- rotation_plane_dim: effective dimensionality of the rotation plane
- cross_model_rotation_alignment: do models rotate in similar "directions"?
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

OUTPUT_DIR = _Path(r"d:\develop\TransformerLens-main\tests\glm5_temp\stage693_rotation_axis_20260406_2215")
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
    print(f"  loading model: {model_name} from {path.name}")
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


def get_hidden_states_pair(model, tokenizer, text):
    """Get L0 and final hidden state for last token"""
    model_device = next(model.parameters()).device
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64)
    tokens = tokens.to(model_device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    n_layers = len(outputs.hidden_states)
    h_l0 = outputs.hidden_states[0][0, -1, :].float().cpu()
    h_final = outputs.hidden_states[n_layers - 1][0, -1, :].float().cpu()
    return h_l0, h_final


def compute_rotation_vector(h_l0, h_final):
    """
    Compute the rotation component: component of h_final perpendicular to h_l0.
    This represents the "direction of rotation" from L0 to final.
    """
    # Normalize L0
    l0_norm = torch.norm(h_l0)
    if l0_norm < 1e-10:
        return None, 0.0, 0.0
    l0_dir = h_l0 / l0_norm

    # Component of h_final along l0_dir
    cos_theta = F.cosine_similarity(h_final.unsqueeze(0), l0_dir.unsqueeze(0)).item()
    proj = cos_theta * torch.norm(h_final) * l0_dir

    # Rotation component (perpendicular to L0)
    rot_vec = h_final - proj

    rot_norm = torch.norm(rot_vec).item()
    angle = math.degrees(math.acos(max(-1, min(1, cos_theta))))

    return rot_vec, angle, cos_theta


def analyze_rotation_plane(model, tokenizer, texts, log):
    """Analyze rotation plane for one model"""
    log("  Extracting rotation vectors...")
    rot_vectors = []
    angles = []
    cos_thetas = []

    for text in texts:
        h_l0, h_final = get_hidden_states_pair(model, tokenizer, text)
        rot_vec, angle, cos_theta = compute_rotation_vector(h_l0, h_final)
        if rot_vec is not None:
            rot_vectors.append(rot_vec)
            angles.append(angle)
            cos_thetas.append(cos_theta)

    n_valid = len(rot_vectors)
    log(f"    Valid texts: {n_valid}")

    if n_valid < 3:
        return None

    rot_matrix = torch.stack(rot_vectors)  # (N, d)

    # A: Rotation consistency - are rotation vectors aligned?
    rot_norms = torch.norm(rot_matrix, dim=1)
    mean_rot_norm = rot_norms.mean().item()
    std_rot_norm = rot_norms.std().item()

    # Normalize rotation vectors
    rot_dirs = rot_matrix / rot_norms.unsqueeze(1)
    mean_rot_dir = rot_dirs.mean(dim=0)
    mean_rot_dir_norm = torch.norm(mean_rot_dir).item()

    # Pairwise cosine similarity of rotation directions
    cos_matrix = torch.mm(rot_dirs, rot_dirs.t())
    n = cos_matrix.shape[0]
    upper_tri_indices = torch.triu_indices(n, n, offset=1)
    pairwise_cos = cos_matrix[upper_tri_indices[0], upper_tri_indices[1]]
    mean_pairwise_cos = pairwise_cos.mean().item()
    std_pairwise_cos = pairwise_cos.std().item()

    # B: PCA on rotation vectors to find rotation plane
    pca = PCA(n_components=min(10, n_valid - 1))
    pca.fit(rot_matrix.numpy())

    # C: Rotation angle statistics
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    mean_cos = np.mean(cos_thetas)

    # D: Project rotation vectors onto unembed for cross-model comparison
    log("  Projecting rotation plane onto unembed...")
    if hasattr(model, 'lm_head'):
        unembed = model.lm_head.weight.detach().float().cpu()
    else:
        unembed = model.get_input_embeddings().weight.detach().float().cpu()

    # Normalize unembed rows
    ue_norms = torch.norm(unembed, dim=1, keepdim=True)
    ue_norms = torch.clamp(ue_norms, min=1e-10)
    ue_dirs = unembed / ue_norms

    # Project mean rotation direction onto unembed space
    rot_ue_cos = torch.mm(ue_dirs, mean_rot_dir.unsqueeze(1)).squeeze(1)
    top_indices = torch.topk(rot_ue_cos.abs(), k=min(20, len(rot_ue_cos)))
    top_tokens = []
    for idx in top_indices.indices:
        try:
            tok = tokenizer.convert_ids_to_tokens(idx.item())
            top_tokens.append(f"{tok}({rot_ue_cos[idx].item():.3f})")
        except:
            top_tokens.append(f"?({rot_ue_cos[idx].item():.3f})")

    # E: Layer-by-layer rotation tracking
    log("  Tracking rotation layer by layer...")
    model_device = next(model.parameters()).device
    sample_tokens = tokenizer.encode(texts[0], return_tensors="pt", truncation=True, max_length=64).to(model_device)
    with torch.no_grad():
        outputs = model(sample_tokens, output_hidden_states=True)
    n_layers = len(outputs.hidden_states)

    layer_angles = []
    layer_cos = []
    for li in range(n_layers):
        h = outputs.hidden_states[li][0, -1, :].float().cpu()
        _, angle, cos_theta = compute_rotation_vector(outputs.hidden_states[0][0, -1, :].float().cpu(), h)
        layer_angles.append(angle)
        layer_cos.append(cos_theta)

    results = {
        "n_texts": n_valid,
        "rotation_consistency": {
            "mean_pairwise_cos": mean_pairwise_cos,
            "std_pairwise_cos": std_pairwise_cos,
            "mean_rot_dir_norm": mean_rot_dir_norm,
        },
        "rotation_magnitude": {
            "mean_rot_norm": mean_rot_norm,
            "std_rot_norm": std_rot_norm,
            "mean_norm_ratio": mean_rot_norm / max(mean_rot_norm + std_rot_norm, 1e-10),
        },
        "rotation_angles": {
            "mean": mean_angle,
            "std": std_angle,
            "min": min(angles),
            "max": max(angles),
            "mean_cos": mean_cos,
        },
        "pca_explained": pca.explained_variance_ratio_[:10].tolist(),
        "pca90": int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90) + 1),
        "pca95": int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1),
        "top_unembed_tokens": top_tokens[:10],
        "layer_angles": layer_angles,
        "layer_cos": layer_cos,
        "n_layers": n_layers,
    }

    # Print summary
    log(f"\n    === Rotation Plane Summary ===")
    log(f"    Rotation angle: {mean_angle:.1f} +/- {std_angle:.1f} deg (cos={mean_cos:.4f})")
    log(f"    Pairwise rotation-dir cos: {mean_pairwise_cos:.4f} +/- {std_pairwise_cos:.4f}")
    log(f"    Mean rot dir magnitude: {mean_rot_dir_norm:.4f}")
    log(f"    Rotation PCA: top-5 = {[f'{v:.4f}' for v in pca.explained_variance_ratio_[:5]]}")
    log(f"    PCA90={results['pca90']}, PCA95={results['pca95']}")
    log(f"    Rotation magnitude: {mean_rot_norm:.2f} +/- {std_rot_norm:.2f}")
    # Skip top_tokens print (may contain non-GBK chars)
    try:
        safe_tokens = [t.encode('ascii', errors='replace').decode('ascii') for t in top_tokens[:5]]
        log(f"    Top unembed alignment: {safe_tokens}")
    except:
        log(f"    Top unembed alignment: (encoding error, see JSON)")

    # Rotation consistency interpretation
    if mean_pairwise_cos > 0.5:
        log(f"    -> INV-350 CONFIRMED: Rotation plane is HIGHLY consistent across texts")
    elif mean_pairwise_cos > 0.2:
        log(f"    -> INV-350 PARTIAL: Rotation plane shows moderate consistency")
    else:
        log(f"    -> INV-350 REJECTED: Rotation directions are diverse (no shared plane)")

    return results


def run_single_model(model_name, texts, log):
    """Run rotation analysis for one model"""
    log(f"\n{'='*60}")
    log(f"  Processing: {model_name}")
    log(f"{'='*60}")

    t0 = time.time()
    model, tokenizer = load_model(model_name)

    model_device = next(model.parameters()).device
    tokens = tokenizer.encode(texts[0], return_tensors="pt", truncation=True, max_length=64).to(model_device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    total_layers = len(outputs.hidden_states)
    hidden_dim = outputs.hidden_states[0].shape[-1]
    log(f"  layers: {total_layers}, hidden_dim: {hidden_dim}")

    results = analyze_rotation_plane(model, tokenizer, texts, log)
    if results:
        results["model_name"] = model_name
        results["hidden_dim"] = hidden_dim
        results["total_layers"] = total_layers

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
    log(f"  P47: Rotation Plane Analysis (Stage693)")
    log(f"{'='*70}")
    log(f"  models: {list(MODEL_MAP.keys())}")
    log(f"  texts: {len(TEXTS)}")

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
    log(f"  CROSS-MODEL ROTATION COMPARISON")
    log(f"{'='*70}")

    log(f"\n  A: Rotation Consistency Comparison")
    log(f"  {'model':>12s}  {'rot_consistency':>15s}  {'rot_std':>10s}  {'mean_angle':>12s}  {'angle_std':>10s}  {'PCA90':>6s}  {'PCA95':>6s}")
    for m, r in all_results.items():
        rc = r["rotation_consistency"]
        ra = r["rotation_angles"]
        log(f"  {m:>12s}  {rc['mean_pairwise_cos']:>15.4f}  {rc['std_pairwise_cos']:>10.4f}  {ra['mean']:>12.1f}  {ra['std']:>10.1f}  {r['pca90']:>6d}  {r['pca95']:>6d}")

    log(f"\n  B: Layer-by-Layer Rotation Progression")
    for m, r in all_results.items():
        angles = r["layer_angles"]
        cos_vals = r["layer_cos"]
        # Find where rotation is fastest (largest angle change per layer)
        deltas = [angles[i+1] - angles[i] for i in range(len(angles)-1)]
        max_delta_layer = deltas.index(max(deltas))
        log(f"  {m:>12s}: L0 angle=0.0, L{max_delta_layer} delta={max(deltas):.1f}(fastest), Final angle={angles[-1]:.1f}")

    log(f"\n  C: Rotation Plane vs Unembed (top-3 tokens per model)")
    for m, r in all_results.items():
        top3 = r["top_unembed_tokens"][:3]
        try:
            safe = [t.encode('ascii', errors='replace').decode('ascii') for t in top3]
            log(f"  {m:>12s}: {safe}")
        except:
            log(f"  {m:>12s}: (see JSON)")

    log(f"\n  D: INV-350 Cross-Model Rotation Consistency")
    consistencies = [r["rotation_consistency"]["mean_pairwise_cos"] for r in all_results.values()]
    avg_consistency = np.mean(consistencies)
    log(f"  Mean cross-text rotation consistency: {avg_consistency:.4f}")
    if avg_consistency > 0.3:
        log(f"  -> INV-350 CONFIRMED: Models share consistent rotation planes")
    else:
        log(f"  -> INV-350 REJECTED: Rotation planes are text-dependent")

    elapsed = time.time() - t_total
    log(f"\n{'='*70}")
    log(f"  P47 COMPLETE - Total elapsed: {elapsed:.1f}s")
    log(f"{'='*70}")

    # Save results
    save_results = {}
    for m, r in all_results.items():
        save_results[m] = {
            k: v for k, v in r.items()
            if k not in ["layer_angles", "layer_cos", "top_unembed_tokens"]
        }
        save_results[m]["layer_angles_summary"] = {
            "L0": r["layer_angles"][0],
            "L25%": r["layer_angles"][len(r["layer_angles"])//4],
            "L50%": r["layer_angles"][len(r["layer_angles"])//2],
            "L75%": r["layer_angles"][3*len(r["layer_angles"])//4],
            "Final": r["layer_angles"][-1],
        }
        save_results[m]["top_unembed_tokens"] = r["top_unembed_tokens"][:5]

    out_path = OUTPUT_DIR / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
    log(f"\n  Results saved to: {out_path}")

    log.close()


if __name__ == "__main__":
    main()
