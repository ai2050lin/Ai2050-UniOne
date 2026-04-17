"""
Phase CLXXIII: W_U SVD Natural Subspace Analysis (Optimized)
=============================================================
Optimized version: 
  - P740: SVD spectrum via eigendecomposition (fast, no U computation)
  - P741: Compute U columns on-demand for top-50 components
  - P742: SVD logit reconstruction without storing full U
"""

import argparse
import json
import numpy as np
import torch
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_utils import load_model, get_model_info, release_model


def to_numpy(tensor_or_array):
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    return tensor_or_array.detach().cpu().float().numpy().astype(np.float32)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ===== P740: W_U SVD Spectrum =====
def p740_svd_spectrum(W_U, model_name, results, W_U_gpu=None):
    """Analyze W_U's SVD spectrum using eigendecomposition of W_U^T @ W_U"""
    print("\n--- P740: W_U SVD Spectrum ---")

    if W_U_gpu is not None:
        n_vocab, d_model = W_U_gpu.shape
    else:
        n_vocab, d_model = W_U.shape
    print(f"  W_U shape: {n_vocab} x {d_model}")

    # Eigendecomposition of W_U^T @ W_U using GPU for speed
    print("  Computing eigendecomposition of W_U^T @ W_U ...", flush=True)
    if W_U_gpu is not None and torch.cuda.is_available():
        print("  Using GPU for W_U^T @ W_U ...", flush=True)
        chunk_size = 50000
        n_chunks = (n_vocab + chunk_size - 1) // chunk_size
        WtW = torch.zeros(d_model, d_model, dtype=torch.float32, device='cuda')
        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min((ci + 1) * chunk_size, n_vocab)
            chunk = W_U_gpu[start:end].float()  # already on GPU
            WtW += chunk.T @ chunk
            print(f"    Chunk {ci+1}/{n_chunks} done", flush=True)
        
        # Do eigendecomposition on GPU too!
        print("  Computing eigendecomposition on GPU ...", flush=True)
        eigenvalues_t, eigenvectors_t = torch.linalg.eigh(WtW)
        # Sort descending
        idx = torch.argsort(eigenvalues_t, descending=True)
        eigenvalues_t = eigenvalues_t[idx]
        eigenvectors_t = eigenvectors_t[:, idx]
        # Move to CPU/numpy
        S = torch.sqrt(torch.maximum(eigenvalues_t, torch.zeros_like(eigenvalues_t))).cpu().numpy()
        Vt = eigenvectors_t.T.cpu().numpy()  # [d_model, d_model]
        del WtW, eigenvalues_t, eigenvectors_t
        torch.cuda.empty_cache()
    else:
        print("  Using CPU for W_U^T @ W_U (slow!) ...", flush=True)
        WtW_np = W_U.T @ W_U  # CPU, very slow for large matrices
        eigenvalues, eigenvectors = np.linalg.eigh(WtW_np)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        S = np.sqrt(np.maximum(eigenvalues, 0))
        Vt = eigenvectors.T
    
    print(f"  Eigendecomposition done. Top 10 SVs: {S[:10].tolist()}")

    # Spectrum analysis
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy

    # ratio(k) = cumulative energy of top-k components
    key_k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    ratio_at_k = {}
    for k in key_k_values:
        if k <= len(S):
            ratio_at_k[k] = float(cumulative_energy[k-1])
    print(f"  ratio(k): {ratio_at_k}")

    # Fit power law: S(k) ~ k^(-alpha)
    valid_range = slice(5, min(500, len(S)//2))
    log_k = np.log(np.arange(1, len(S)+1)[valid_range])
    log_S = np.log(S[valid_range] + 1e-30)
    coeffs = np.polyfit(log_k, log_S, 1)
    alpha_svd = -coeffs[0]
    r_squared = 1 - np.sum((log_S - np.polyval(coeffs, log_k))**2) / np.sum((log_S - np.mean(log_S))**2)
    print(f"  SVD power law: S(k) ~ k^(-{alpha_svd:.3f}), R2={r_squared:.3f}")

    # Effective dimension
    p = S**2 / total_energy
    eff_dim = 1.0 / np.sum(p**2)
    print(f"  Effective dimension: {eff_dim:.1f} (out of {d_model})")

    # Energy concentration
    top1_pct = float(S[0]**2 / total_energy * 100)
    top10_pct = float(np.sum(S[:10]**2) / total_energy * 100)
    top100_pct = float(np.sum(S[:100]**2) / total_energy * 100)
    top500_pct = float(np.sum(S[:500]**2) / total_energy * 100)
    print(f"  Energy: top1={top1_pct:.1f}%, top10={top10_pct:.1f}%, top100={top100_pct:.1f}%, top500={top500_pct:.1f}%")

    # k for 90%/95%/99% energy
    cumulative = np.cumsum(S**2) / total_energy
    k90 = int(np.searchsorted(cumulative, 0.9)) + 1
    k95 = int(np.searchsorted(cumulative, 0.95)) + 1
    k99 = int(np.searchsorted(cumulative, 0.99)) + 1
    print(f"  k for 90%/95%/99% energy: {k90}/{k95}/{k99}")

    results["p740_svd_spectrum"] = {
        "n_vocab": n_vocab,
        "d_model": d_model,
        "eff_dim": float(eff_dim),
        "alpha_svd": float(alpha_svd),
        "r_squared_svd": float(r_squared),
        "ratio_at_k": ratio_at_k,
        "top_singular_values": {f"sv_{i}": float(S[i]) for i in range(min(20, len(S)))},
        "energy_concentration": {
            "top1_pct": top1_pct, "top10_pct": top10_pct,
            "top100_pct": top100_pct, "top500_pct": top500_pct,
        },
        "k_for_90pct": k90,
        "k_for_95pct": k95,
        "k_for_99pct": k99,
    }

    return S, Vt, results


# ===== P741: Natural Subspace Interpretation (lightweight) =====
def p741_natural_subspace_interp(W_U, S, Vt, tokenizer, model_name, results, W_U_gpu=None):
    """Interpret top-50 natural subspaces by computing U columns on-demand"""
    print("\n--- P741: Natural Subspace Interpretation ---")

    n_components = min(50, len(S))
    component_words = {}

    # Pre-compute W_U @ Vt[:50].T on GPU
    if W_U_gpu is not None and torch.cuda.is_available():
        print("  Computing W_U @ Vt on GPU ...", flush=True)
        Vt_torch = torch.tensor(Vt[:n_components], dtype=torch.float32, device='cuda')
        chunk_size = 50000
        n_chunks = (W_U_gpu.shape[0] + chunk_size - 1) // chunk_size
        WV_chunks = []
        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min((ci + 1) * chunk_size, W_U_gpu.shape[0])
            chunk = W_U_gpu[start:end].float()
            wv_chunk = chunk @ Vt_torch.T  # [chunk, n_components]
            WV_chunks.append(wv_chunk.cpu().numpy())
        WV = np.concatenate(WV_chunks, axis=0)  # [n_vocab, n_components]
    else:
        WV = W_U @ Vt[:n_components].T  # CPU fallback

    for k in range(n_components):
        # U[:, k] = WV[:, k] / S[k]
        if S[k] > 1e-10:
            u_k = WV[:, k] / S[k]
        else:
            continue

        # Scale by singular value
        scaled_u = u_k * S[k]

        # Top positive and negative words
        top_pos_idx = np.argsort(scaled_u)[-10:][::-1]
        top_neg_idx = np.argsort(scaled_u)[:10]

        top_pos = []
        for idx in top_pos_idx:
            w = tokenizer.decode([idx]).strip().encode('ascii', 'replace').decode()
            top_pos.append({"word": w, "value": float(scaled_u[idx])})

        top_neg = []
        for idx in top_neg_idx:
            w = tokenizer.decode([idx]).strip().encode('ascii', 'replace').decode()
            top_neg.append({"word": w, "value": float(scaled_u[idx])})

        component_words[f"component_{k}"] = {
            "singular_value": float(S[k]),
            "energy_pct": float(S[k]**2 / np.sum(S**2) * 100),
            "top_positive": top_pos,
            "top_negative": top_neg,
        }

        if k < 10:
            pos_str = ", ".join([w["word"] for w in top_pos[:5]])
            neg_str = ", ".join([w["word"] for w in top_neg[:5]])
            print(f"  Component {k}: sv={S[k]:.2f}, energy={S[k]**2/np.sum(S**2)*100:.2f}%")
            print(f"    Top+: {pos_str}")
            print(f"    Top-: {neg_str}")

    results["p741_natural_subspace"] = {
        "component_words": component_words,
        "n_components_analyzed": n_components,
    }

    return results


# ===== P742: SVD Logit Decomposition (memory-efficient) =====
def p742_svd_logit_decomposition(model, tokenizer, S, Vt, device, model_name, results):
    """Decompose logits using SVD natural subspaces"""
    print("\n--- P742: SVD Logit Decomposition ---")

    sentences = [
        "The cat sat on the",
        "I went to the store to buy",
        "She looked at him with a",
        "The weather was very cold and",
        "He opened the door and saw",
        "In the morning she always drinks",
        "The children played in the park",
        "After dinner they went for a",
        "The book was about a young",
        "They decided to go to the",
    ]

    # Register forward hook
    layer_outputs = {}
    def hook_fn(module, input, output):
        layer_outputs["last_hidden"] = input[0]

    layers = model.model.layers
    last_layer = layers[-1]
    handle = last_layer.register_forward_hook(hook_fn)

    k_values = [1, 5, 10, 20, 50, 100, 200, 500]
    accuracy_by_k = {k: {"top1_match": 0, "top5_overlap": 0, "top10_overlap": 0, "cosine": 0.0}
                     for k in k_values}
    n_valid = 0

    # Keep W_U on GPU for fast matmul
    W_U_gpu = model.lm_head.weight.data.float()  # [n_vocab, d_model] on GPU
    n_vocab = W_U_gpu.shape[0]

    for sent_idx, sentence in enumerate(sentences):
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        h_final = to_numpy(layer_outputs["last_hidden"][0, -1, :])

        # Original logits using GPU
        with torch.no_grad():
            h_torch = torch.tensor(h_final, dtype=torch.float32, device=device)
            original_logits_torch = h_torch @ W_U_gpu.T
            original_logits = original_logits_torch.cpu().numpy()
        top10_original = set(np.argsort(original_logits)[-10:][::-1])
        top5_original = set(np.argsort(original_logits)[-5:][::-1])
        top1_original = np.argmax(original_logits)

        # h projected onto right singular vectors
        h_proj_vt = h_final @ Vt.T  # [d_model] - CPU, small

        for k in k_values:
            # Reconstruct on GPU:
            # h_proj_topk = Vt[:k].T @ h_proj_vt[:k]
            # recon_logits = W_U @ h_proj_topk
            h_proj_topk = Vt[:k].T @ h_proj_vt[:k]  # [d_model], CPU
            with torch.no_grad():
                h_proj_torch = torch.tensor(h_proj_topk, dtype=torch.float32, device=device)
                recon_logits_torch = W_U_gpu @ h_proj_torch
                recon_logits = recon_logits_torch.cpu().numpy()

            top10_recon = set(np.argsort(recon_logits)[-10:][::-1])
            top5_recon = set(np.argsort(recon_logits)[-5:][::-1])
            top1_recon = np.argmax(recon_logits)

            top1_match = 1 if top1_recon == top1_original else 0
            top5_overlap = len(top5_original & top5_recon) / 5.0
            top10_overlap = len(top10_original & top10_recon) / 10.0

            norm_orig = np.linalg.norm(original_logits)
            norm_recon = np.linalg.norm(recon_logits)
            cos = np.dot(original_logits, recon_logits) / (norm_orig * norm_recon + 1e-30) if norm_orig > 0 and norm_recon > 0 else 0

            accuracy_by_k[k]["top1_match"] += top1_match
            accuracy_by_k[k]["top5_overlap"] += top5_overlap
            accuracy_by_k[k]["top10_overlap"] += top10_overlap
            accuracy_by_k[k]["cosine"] += cos

        n_valid += 1
        print(f"  Sentence {sent_idx+1}: done")

    handle.remove()

    # Average
    for k in k_values:
        for metric in ["top1_match", "top5_overlap", "top10_overlap", "cosine"]:
            accuracy_by_k[k][metric] /= n_valid

    print("\n  SVD Logit Reconstruction Accuracy:")
    print(f"  {'k':>6} | {'Top1':>6} | {'Top5':>6} | {'Top10':>6} | {'Cosine':>8}")
    print("  " + "-" * 42)
    for k in k_values:
        r = accuracy_by_k[k]
        print(f"  {k:>6} | {r['top1_match']:>6.2f} | {r['top5_overlap']:>6.2f} | {r['top10_overlap']:>6.2f} | {r['cosine']:>8.4f}")

    k_for_top1 = None
    for k in k_values:
        if accuracy_by_k[k]["top1_match"] > 0:
            k_for_top1 = k
            break

    k_for_cos90 = None
    for k in k_values:
        if accuracy_by_k[k]["cosine"] > 0.9:
            k_for_cos90 = k
            break

    print(f"\n  Min k for Top-1 match: {k_for_top1}")
    print(f"  Min k for cosine > 0.9: {k_for_cos90}")

    # Per-component contribution for first sentence
    inputs = tokenizer(sentences[0], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    h_final = to_numpy(layer_outputs["last_hidden"][0, -1, :])
    h_proj_vt = h_final @ Vt.T

    # Use GPU for W_U
    with torch.no_grad():
        h_torch = torch.tensor(h_final, dtype=torch.float32, device=device)
        original_logits_torch = h_torch @ W_U_gpu.T
        original_logits = original_logits_torch.cpu().numpy()
    top5_idx = np.argsort(original_logits)[-5:][::-1]

    # Compute W_U @ Vt[k, :] on GPU for top-20 components
    Vt20_torch = torch.tensor(Vt[:20], dtype=torch.float32, device=device)
    with torch.no_grad():
        WV20 = W_U_gpu @ Vt20_torch.T  # [n_vocab, 20]
        WV20_np = WV20.cpu().numpy()

    component_contributions = {}
    for rank, word_idx in enumerate(top5_idx):
        word_name = tokenizer.decode([word_idx]).strip().encode('ascii', 'replace').decode()
        contribs = []
        for k in range(min(20, len(S))):
            # Contribution of component k to word i:
            # c_k = h_proj_vt[k] * WV20[word_idx, k]
            c = h_proj_vt[k] * WV20_np[word_idx, k]
            contribs.append({
                "k": k, "contribution": float(c),
                "sv": float(S[k]), "h_proj": float(h_proj_vt[k]),
                "w_vk_i": float(WV20_np[word_idx, k])
            })
        component_contributions[f"rank{rank}_{word_name}"] = contribs

    results["p742_svd_logit"] = {
        "accuracy_by_k": {str(k): v for k, v in accuracy_by_k.items()},
        "k_for_top1": k_for_top1,
        "k_for_cos90": k_for_cos90,
        "component_contributions": component_contributions,
    }

    return results


# ===== Main =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()

    model_name = args.model
    print(f"Phase CLXXIII: W_U SVD Natural Subspace Analysis -- {model_name}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)

    # Get W_U reference (keep on GPU for speed)
    W_U_gpu = model.lm_head.weight.data  # [n_vocab, d_model] on GPU, bfloat16
    n_vocab, d_model = W_U_gpu.shape
    print(f"  W_U shape: {n_vocab} x {d_model}")

    results = {"model": model_name, "timestamp": datetime.now().isoformat()}

    # P740: SVD Spectrum (uses GPU for W_U^T @ W_U)
    S, Vt, results = p740_svd_spectrum(None, model_name, results, W_U_gpu=W_U_gpu)

    # P741: Natural Subspace Interpretation
    results = p741_natural_subspace_interp(None, S, Vt, tokenizer, model_name, results, W_U_gpu=W_U_gpu)

    # P742: SVD Logit Decomposition
    results = p742_svd_logit_decomposition(model, tokenizer, S, Vt, device, model_name, results)

    # Save
    out_dir = Path(f"results/phase_clxxiii")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nResults saved to {out_file}")

    release_model(model)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
