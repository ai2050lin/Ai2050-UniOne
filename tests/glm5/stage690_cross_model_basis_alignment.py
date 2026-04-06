#!/usr/bin/env python3
"""
P44: Cross-Model Semantic Basis Alignment (Stage690)

Core questions:
1. Do different models share the same "semantic basis direction"?
2. What is the cosine similarity between models' basis directions?
3. How does the basis direction relate to the unembed matrix?
4. Do basis directions at L0 vs final layer show different cross-model alignment?

Method:
1. For each model, extract semantic basis direction at L0 and final layer
2. Compute pairwise cosine similarity between models' basis directions
3. Analyze basis direction vs unembed matrix relationship
4. Test INV-343: cross-model basis alignment > random expectation

Usage: python tests/glm5/stage690_cross_model_basis_alignment.py
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


def get_hidden_state(model, tokenizer, text, layer_idx):
    """Get hidden state at specified layer for last token"""
    model_device = next(model.parameters()).device
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64)
    tokens = tokens.to(model_device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    hs = outputs.hidden_states[layer_idx][0, -1, :].float().cpu()
    return hs


def extract_basis_direction(model, tokenizer, texts, layer_idx):
    """Extract semantic basis direction at given layer"""
    directions = []
    for text in texts:
        hs = get_hidden_state(model, tokenizer, text, layer_idx)
        norm = torch.norm(hs).item()
        if norm < 1e-10:
            continue
        direction = hs / norm
        directions.append(direction)
    
    if not directions:
        return None, []
    
    # Mean direction = semantic basis
    direction_matrix = torch.stack(directions)
    mean_dir = direction_matrix.mean(dim=0)
    mean_norm = torch.norm(mean_dir).item()
    if mean_norm < 1e-10:
        return None, []
    mean_dir = mean_dir / mean_norm
    
    # Individual alignments
    alignments = []
    for d in directions:
        cos_val = F.cosine_similarity(d.unsqueeze(0), mean_dir.unsqueeze(0)).item()
        alignments.append(cos_val)
    
    # PCA
    pca = PCA(n_components=min(5, len(directions) - 1))
    pca.fit(direction_matrix.numpy())
    
    return mean_dir, {
        "alignments": alignments,
        "mean_alignment": float(np.mean(alignments)),
        "pca_explained": pca.explained_variance_ratio_.tolist(),
        "direction_matrix": direction_matrix.numpy(),
    }


def basis_vs_unembed(model, tokenizer, basis_dir):
    """Analyze relationship between basis direction and unembed matrix"""
    # Get unembed weight matrix (transpose of lm_head)
    if hasattr(model, 'lm_head'):
        unembed = model.lm_head.weight.detach().float().cpu()
    else:
        # For tied embeddings
        unembed = model.get_input_embeddings().weight.detach().float().cpu()
    
    # Normalize basis direction
    basis_norm = torch.norm(basis_dir)
    if basis_norm < 1e-10:
        return None
    basis_n = basis_dir / basis_norm
    
    # Compute cosine of basis with each unembed row
    # Do it in chunks to save memory
    chunk_size = 10000
    max_cos = -999
    min_cos = 999
    cos_sum = 0
    cos_sq_sum = 0
    count = 0
    top5_cos = []
    
    for i in range(0, unembed.shape[0], chunk_size):
        chunk = unembed[i:i+chunk_size]
        chunk_norms = torch.norm(chunk, dim=1, keepdim=True)
        chunk_norms = torch.clamp(chunk_norms, min=1e-10)
        chunk_n = chunk / chunk_norms
        
        cos_vals = (chunk_n @ basis_n)
        cos_sum += cos_vals.sum().item()
        cos_sq_sum += (cos_vals ** 2).sum().item()
        count += chunk.shape[0]
        
        batch_max = cos_vals.max().item()
        batch_min = cos_vals.min().item()
        if batch_max > max_cos:
            max_cos = batch_max
        if batch_min < min_cos:
            min_cos = batch_min
        
        # Top-5 closest unembed directions
        if len(top5_cos) < 5:
            top5_idx = torch.topk(cos_vals, min(5, len(cos_vals)))
            for idx, val in zip(top5_idx.indices, top5_idx.values):
                top5_cos.append((idx.item() + i, val.item()))
        else:
            top5_cos.sort(key=lambda x: -x[1])
            for idx, val in zip(top5_idx.indices, top5_idx.values):
                if val.item() > top5_cos[-1][1]:
                    top5_cos[-1] = (idx.item() + i, val.item())
                    top5_cos.sort(key=lambda x: -x[1])
    
    mean_cos = cos_sum / max(count, 1)
    std_cos = math.sqrt(max(cos_sq_sum / max(count, 1) - mean_cos ** 2, 0))
    
    # Token IDs to tokens for top-5
    top5_tokens = []
    for idx, val in top5_cos[:5]:
        try:
            token = tokenizer.decode([idx])
            top5_tokens.append(f"{repr(token)}(cos={val:.4f})")
        except:
            top5_tokens.append(f"idx={idx}(cos={val:.4f})")
    
    return {
        "mean_cos": mean_cos,
        "std_cos": std_cos,
        "max_cos": max_cos,
        "min_cos": min_cos,
        "unembed_dim": unembed.shape[0],
        "top5_tokens": top5_tokens,
        # Expected for random directions in high-dim: mean=0, std=1/sqrt(d)
        "expected_std": 1.0 / math.sqrt(unembed.shape[1]) if unembed.shape[1] > 0 else 0,
    }


def run_single_model(model_name, texts, p=print):
    """Extract basis at L0 and final layer for one model"""
    p(f"\n{'='*60}")
    p(f"  Processing: {model_name}")
    p(f"{'='*60}")
    
    t0 = time.time()
    model, tokenizer = load_model(model_name)
    
    # Get total layers
    model_device = next(model.parameters()).device
    tokens = tokenizer.encode(texts[0], return_tensors="pt", truncation=True, max_length=64).to(model_device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    total_layers = len(outputs.hidden_states)
    hidden_dim = outputs.hidden_states[0].shape[-1]
    
    p(f"  layers: {total_layers}, hidden_dim: {hidden_dim}")
    
    results = {
        "model_name": model_name,
        "hidden_dim": hidden_dim,
        "total_layers": total_layers,
    }
    
    # L0 basis
    p(f"\n  Extracting L0 basis...")
    l0_basis, l0_data = extract_basis_direction(model, tokenizer, texts, 0)
    if l0_basis is not None:
        results["l0_basis"] = l0_basis.numpy()
        results["l0_alignment"] = l0_data["mean_alignment"]
        results["l0_pca"] = l0_data["pca_explained"]
        p(f"    L0 alignment: {l0_data['mean_alignment']:.4f}")
        p(f"    L0 PCA top-3: {[f'{v:.4f}' for v in l0_data['pca_explained'][:3]]}")
    
    # Final layer basis
    p(f"  Extracting L{total_layers-1} basis...")
    final_basis, final_data = extract_basis_direction(model, tokenizer, texts, total_layers - 1)
    if final_basis is not None:
        results["final_basis"] = final_basis.numpy()
        results["final_alignment"] = final_data["mean_alignment"]
        results["final_pca"] = final_data["pca_explained"]
        p(f"    Final alignment: {final_data['mean_alignment']:.4f}")
        p(f"    Final PCA top-3: {[f'{v:.4f}' for v in final_data['pca_explained'][:3]]}")
    
    # Basis vs unembed
    if l0_basis is not None:
        p(f"  Analyzing L0 basis vs unembed...")
        l0_ue = basis_vs_unembed(model, tokenizer, l0_basis)
        if l0_ue:
            results["l0_unembed"] = l0_ue
            p(f"    mean_cos={l0_ue['mean_cos']:.6f}, std_cos={l0_ue['std_cos']:.6f}")
            p(f"    max_cos={l0_ue['max_cos']:.6f}, range=[{l0_ue['min_cos']:.6f}, {l0_ue['max_cos']:.6f}]")
            p(f"    expected_std(1/sqrt(d))={l0_ue['expected_std']:.6f}")
            p(f"    top-5: {l0_ue['top5_tokens'][:3]}")
    
    if final_basis is not None:
        p(f"  Analyzing final basis vs unembed...")
        final_ue = basis_vs_unembed(model, tokenizer, final_basis)
        if final_ue:
            results["final_unembed"] = final_ue
            p(f"    mean_cos={final_ue['mean_cos']:.6f}, std_cos={final_ue['std_cos']:.6f}")
            p(f"    max_cos={final_ue['max_cos']:.6f}")
            p(f"    top-5: {final_ue['top5_tokens'][:3]}")
    
    # Cross-layer basis drift
    if l0_basis is not None and final_basis is not None:
        cross_cos = F.cosine_similarity(l0_basis.unsqueeze(0), final_basis.unsqueeze(0)).item()
        results["l0_final_cross_cos"] = cross_cos
        p(f"  L0 vs Final basis cos: {cross_cos:.6f}")
    
    elapsed = time.time() - t0
    p(f"  elapsed: {elapsed:.1f}s")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def cross_model_analysis(all_results, p=print):
    """Analyze cross-model basis alignment"""
    p(f"\n{'='*70}")
    p(f"  CROSS-MODEL BASIS ALIGNMENT ANALYSIS")
    p(f"{'='*70}")
    
    model_names = list(all_results.keys())
    
    # Pairwise L0 basis cosine similarity
    p(f"\n  A: L0 Basis Direction Cosine Similarity")
    p(f"  {'':>12s}", end="")
    for m in model_names:
        p(f"  {m:>10s}", end="")
    p()
    
    l0_cos_matrix = {}
    for m1 in model_names:
        p(f"  {m1:>12s}", end="")
        l0_cos_matrix[m1] = {}
        for m2 in model_names:
            if m1 == m2:
                p(f"  {'1.000':>10s}", end="")
                l0_cos_matrix[m1][m2] = 1.0
            else:
                d1 = all_results[m1]["hidden_dim"]
                d2 = all_results[m2]["hidden_dim"]
                if d1 == d2:
                    b1 = torch.tensor(all_results[m1]["l0_basis"])
                    b2 = torch.tensor(all_results[m2]["l0_basis"])
                    cos = F.cosine_similarity(b1.unsqueeze(0), b2.unsqueeze(0)).item()
                else:
                    cos = float('nan')
                l0_cos_matrix[m1][m2] = cos
                if math.isnan(cos):
                    p(f"  {'N/A':>10s}", end="")
                else:
                    p(f"  {cos:>10.4f}", end="")
        p()
    
    # Pairwise Final layer basis cosine similarity
    p(f"\n  B: Final Layer Basis Direction Cosine Similarity")
    p(f"  {'':>12s}", end="")
    for m in model_names:
        p(f"  {m:>10s}", end="")
    p()
    
    final_cos_matrix = {}
    for m1 in model_names:
        p(f"  {m1:>12s}", end="")
        final_cos_matrix[m1] = {}
        for m2 in model_names:
            if m1 == m2:
                p(f"  {'1.000':>10s}", end="")
                final_cos_matrix[m1][m2] = 1.0
            else:
                d1 = all_results[m1]["hidden_dim"]
                d2 = all_results[m2]["hidden_dim"]
                if d1 == d2:
                    b1 = torch.tensor(all_results[m1]["final_basis"])
                    b2 = torch.tensor(all_results[m2]["final_basis"])
                    cos = F.cosine_similarity(b1.unsqueeze(0), b2.unsqueeze(0)).item()
                else:
                    cos = float('nan')
                final_cos_matrix[m1][m2] = cos
                if math.isnan(cos):
                    p(f"  {'N/A':>10s}", end="")
                else:
                    p(f"  {cos:>10.4f}", end="")
        p()
    
    # Unembed alignment comparison
    p(f"\n  C: Basis vs Unembed Alignment")
    p(f"  {'model':>12s}  {'L0 mean_cos':>12s}  {'L0 std_cos':>12s}  {'L0 max_cos':>12s}  {'Final mean_cos':>14s}  {'Final std_cos':>14s}  {'Final max_cos':>14s}")
    for m in model_names:
        l0_ue = all_results[m].get("l0_unembed", {})
        f_ue = all_results[m].get("final_unembed", {})
        p(f"  {m:>12s}  {l0_ue.get('mean_cos', 0):>12.6f}  {l0_ue.get('std_cos', 0):>12.6f}  {l0_ue.get('max_cos', 0):>12.6f}  {f_ue.get('mean_cos', 0):>14.6f}  {f_ue.get('std_cos', 0):>14.6f}  {f_ue.get('max_cos', 0):>14.6f}")
    
    # Cross-layer drift comparison
    p(f"\n  D: L0 vs Final Layer Basis Drift")
    for m in model_names:
        cross = all_results[m].get("l0_final_cross_cos", 0)
        p(f"  {m:>12s}: L0-Final cos = {cross:.6f}")
    
    # Hidden dimension comparison
    p(f"\n  E: Hidden Dimensions (direct comparison only possible for same-dim models)")
    for m in model_names:
        p(f"  {m:>12s}: d={all_results[m]['hidden_dim']}, layers={all_results[m]['total_layers']}")
    
    # INV-343: Cross-model alignment analysis
    p(f"\n  F: INV-343 Analysis")
    valid_cos = []
    for m1 in model_names:
        for m2 in model_names:
            if m1 < m2:
                c = l0_cos_matrix[m1].get(m2, float('nan'))
                if not math.isnan(c):
                    valid_cos.append(c)
    
    cross_analysis = {}
    
    if valid_cos:
        mean_cross = np.mean(valid_cos)
        cross_analysis["valid_l0_pairs"] = len(valid_cos)
        cross_analysis["mean_l0_cos"] = float(mean_cross)
        p(f"  Valid L0 cross-model pairs: {len(valid_cos)}")
        p(f"  Mean cross-model L0 basis cos: {mean_cross:.6f}")
        p(f"  Expected (random): 0.0000")
        if abs(mean_cross) > 0.05:
            p(f"  -> [INV-343] Cross-model basis alignment is SIGNIFICANTLY above random")
            p(f"  -> Models share a common 'semantic basis direction' at L0")
            cross_analysis["inv_343"] = "CONFIRMED - shared basis"
        else:
            p(f"  -> [INV-343] Cross-model basis alignment is NOT above random")
            p(f"  -> Each model learns its OWN basis direction")
            cross_analysis["inv_343"] = "REJECTED - model-specific"
    else:
        p(f"  No same-dim model pairs for direct comparison")
        cross_analysis["inv_343"] = "N/A - different dimensions"
        cross_analysis["note"] = "All 4 models have different hidden_dims -> need indirect comparison via unembed projection"
    
    # Same-dim pairs for final layer
    valid_final = []
    for m1 in model_names:
        for m2 in model_names:
            if m1 < m2:
                c = final_cos_matrix[m1].get(m2, float('nan'))
                if not math.isnan(c):
                    valid_final.append(c)
    
    if valid_final:
        mean_final = np.mean(valid_final)
        cross_analysis["mean_final_cos"] = float(mean_final)
        p(f"\n  Mean cross-model FINAL basis cos: {mean_final:.6f}")
        if abs(mean_final) > 0.05:
            p(f"  -> Final layer basis also shows cross-model alignment")
        else:
            p(f"  -> Final layer basis is model-specific")
    
    # Indirect comparison: project basis onto shared unembed space
    # Since all models share similar vocab (BPE), compare unembed alignment patterns
    p(f"\n  G: Indirect Cross-Model Comparison (via unembed alignment)")
    for m in model_names:
        l0_ue = all_results[m].get("l0_unembed", {})
        f_ue = all_results[m].get("final_unembed", {})
        l0_top = l0_ue.get("top5_tokens", [])
        f_top = f_ue.get("top5_tokens", [])
        cross_analysis[f"{m}_l0_top_tokens"] = l0_top
        cross_analysis[f"{m}_final_top_tokens"] = f_top
        p(f"  {m:>12s}: L0 top={[t.split('(')[0] for t in l0_top[:3]]}, Final top={[t.split('(')[0] for t in f_top[:3]]}")
    
    # Compare unembed z-scores (how many std above mean the max alignment is)
    p(f"\n  H: Unembed Alignment Z-scores (cross-model comparable)")
    for m in model_names:
        l0_ue = all_results[m].get("l0_unembed", {})
        f_ue = all_results[m].get("final_unembed", {})
        l0_z = l0_ue.get("max_cos", 0) / max(l0_ue.get("std_cos", 1e-10), 1e-10)
        f_z = f_ue.get("max_cos", 0) / max(f_ue.get("std_cos", 1e-10), 1e-10)
        cross_analysis[f"{m}_l0_zscore"] = float(l0_z)
        cross_analysis[f"{m}_final_zscore"] = float(f_z)
        p(f"  {m:>12s}: L0 z={l0_z:.2f}, Final z={f_z:.2f}")
    
    return cross_analysis


def main():
    import json
    
    OUTPUT_DIR = _Path(r"d:\develop\TransformerLens-main\tests\glm5_temp\stage690_cross_model_basis_20260406_2145")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    log_path = OUTPUT_DIR / "output.txt"
    
    class Logger:
        def __init__(self, filepath):
            self.f = open(filepath, "w", encoding="utf-8")
            self._buf = ""
        def __call__(self, msg="", end="\n"):
            print(msg, end=end)
            self.f.write(msg + end)
            if end == "\n":
                self.f.flush()
        def close(self):
            self.f.close()
    
    log = Logger(log_path)
    
    log(f"\n{'='*70}")
    log(f"  P44: Cross-Model Semantic Basis Alignment (Stage690)")
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
    
    # Cross-model analysis
    cross_analysis = cross_model_analysis(all_results, log)
    
    elapsed = time.time() - t_total
    log(f"\n{'='*70}")
    log(f"  P44 COMPLETE - Total elapsed: {elapsed:.1f}s")
    log(f"{'='*70}")
    
    # Save results to JSON
    save_results = {}
    for m, r in all_results.items():
        save_results[m] = {
            "hidden_dim": r["hidden_dim"],
            "total_layers": r["total_layers"],
            "l0_alignment": r.get("l0_alignment"),
            "final_alignment": r.get("final_alignment"),
            "l0_pca": r.get("l0_pca"),
            "final_pca": r.get("final_pca"),
            "l0_unembed": {k: v for k, v in r.get("l0_unembed", {}).items() if k != "top5_tokens"},
            "final_unembed": {k: v for k, v in r.get("final_unembed", {}).items() if k != "top5_tokens"},
            "l0_final_cross_cos": r.get("l0_final_cross_cos"),
            "l0_unembed_top5": r.get("l0_unembed", {}).get("top5_tokens", []),
            "final_unembed_top5": r.get("final_unembed", {}).get("top5_tokens", []),
        }
    save_results["cross_analysis"] = cross_analysis
    
    out_path = OUTPUT_DIR / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
    log(f"\n  Results saved to: {out_path}")
    
    log.close()


if __name__ == "__main__":
    main()
