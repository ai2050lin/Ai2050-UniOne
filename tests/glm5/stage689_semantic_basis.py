#!/usr/bin/env python3
"""
P43: Semantic Basis Direction Extraction（Stage689）

Core question: Is there a shared "semantic basis direction" that all texts align to?
Method:
1. Collect hidden states from many diverse texts
2. Compute the mean direction -> "semantic basis"
3. Project individual directions onto the basis -> measure alignment
4. Analyze what variance remains after removing the basis
5. Cross-model comparison: is the basis similar across models?

Usage: python tests/glm5/stage689_semantic_basis.py <model_name>
"""
import sys, math, statistics, time
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

def load_model(model_name):
    path = MODEL_MAP.get(model_name, _Path(model_name))
    print(f"  loading model: {path.name}")
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

def get_hidden_state(model, tokenizer, text, layer=-1):
    """Get hidden state at specified layer for last token"""
    model_device = next(model.parameters()).device
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64)
    tokens = tokens.to(model_device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    if layer == -1:
        hs = outputs.hidden_states[-1][0, -1, :].float().cpu()
    else:
        hs = outputs.hidden_states[layer][0, -1, :].float().cpu()
    return hs

def extract_semantic_basis(model, tokenizer, texts):
    """Extract the shared semantic basis direction"""
    directions = []
    norms = []
    
    for text in texts:
        hs = get_hidden_state(model, tokenizer, text)
        norm = torch.norm(hs).item()
        direction = hs / max(norm, 1e-10)
        directions.append(direction)
        norms.append(norm)
    
    # Mean direction = semantic basis
    mean_dir = torch.stack(directions).mean(dim=0)
    mean_dir = mean_dir / max(torch.norm(mean_dir).item(), 1e-10)
    
    # Alignment of each text with the basis
    alignments = []
    for d in directions:
        cos_val = F.cosine_similarity(d.unsqueeze(0), mean_dir.unsqueeze(0)).item()
        alignments.append(cos_val)
    
    # Variance explained by basis (PC1)
    direction_matrix = torch.stack(directions).numpy()
    pca = PCA(n_components=5)
    pca.fit(direction_matrix)
    
    return {
        "mean_direction": mean_dir,
        "alignments": alignments,
        "mean_alignment": statistics.mean(alignments),
        "std_alignment": statistics.stdev(alignments),
        "norms": norms,
        "mean_norm": statistics.mean(norms),
        "norm_cv": statistics.stdev(norms) / statistics.mean(norms) if norms else 0,
        "pca_explained": pca.explained_variance_ratio_.tolist(),
        "direction_matrix": direction_matrix,
    }

def analyze_residual_after_basis(basis_data):
    """Analyze what information remains after removing the basis direction"""
    mean_dir = basis_data["mean_direction"]
    matrix = basis_data["direction_matrix"]
    
    # Remove projection onto basis
    residuals = []
    for i in range(matrix.shape[0]):
        d = torch.tensor(matrix[i])
        proj = torch.dot(d, mean_dir) * mean_dir
        residual = d - proj
        residuals.append(residual)
    
    residual_matrix = torch.stack(residuals).numpy()
    
    # PCA on residuals
    pca = PCA(n_components=min(5, residual_matrix.shape[0] - 1))
    pca.fit(residual_matrix)
    
    # Norm of residuals
    residual_norms = [torch.norm(r).item() for r in residuals]
    
    return {
        "residual_norm_mean": statistics.mean(residual_norms),
        "residual_norm_std": statistics.stdev(residual_norms),
        "residual_pca_explained": pca.explained_variance_ratio_.tolist(),
        "relative_residual": statistics.mean(residual_norms),  # relative to unit vectors
    }

def cross_model_basis_similarity(model_name, basis_data):
    """Save basis direction for cross-model comparison"""
    np.save(f'd:\\develop\\TransformerLens-main\\tests\\glm5_temp\\basis_{model_name}.npy',
            basis_data["mean_direction"].numpy())
    return basis_data["mean_direction"].numpy()

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    print(f"\n{'='*60}")
    print(f"  P43: Semantic Basis Direction Extraction")
    print(f"  model: {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer = load_model(model_name)
    
    # 30 diverse texts
    texts = [
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
    ]
    
    t0 = time.time()
    
    # Extract basis at different layers
    print("\n" + "="*60)
    print("  A: Semantic basis at different layers")
    print("="*60)
    
    # Get all layers
    model_device = next(model.parameters()).device
    tokens = tokenizer.encode(texts[0], return_tensors="pt", truncation=True, max_length=64).to(model_device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    total_layers = len(outputs.hidden_states)
    print(f"  total layers: {total_layers}")
    
    layer_results = {}
    sample_layers = [0, total_layers//4, total_layers//2, 3*total_layers//4, total_layers-1]
    
    for layer_idx in sample_layers:
        directions = []
        for text in texts:
            hs = get_hidden_state(model, tokenizer, text, layer=layer_idx)
            norm = torch.norm(hs).item()
            direction = hs / max(norm, 1e-10)
            directions.append(direction)
        
        mean_dir = torch.stack(directions).mean(dim=0)
        mean_dir = mean_dir / max(torch.norm(mean_dir).item(), 1e-10)
        
        alignments = [F.cosine_similarity(d.unsqueeze(0), mean_dir.unsqueeze(0)).item() for d in directions]
        
        layer_results[layer_idx] = {
            "mean_alignment": statistics.mean(alignments),
            "std_alignment": statistics.stdev(alignments),
        }
        print(f"  L{layer_idx:3d}: alignment = {statistics.mean(alignments):.4f} +/- {statistics.stdev(alignments):.4f}")
    
    print("\n" + "="*60)
    print("  B: Final layer basis analysis")
    print("="*60)
    
    basis_data = extract_semantic_basis(model, tokenizer, texts)
    
    print(f"  mean alignment with basis: {basis_data['mean_alignment']:.4f} +/- {basis_data['std_alignment']:.4f}")
    print(f"  norm CV: {basis_data['norm_cv']:.4f}")
    print(f"  PCA explained variance (top-5): {[f'{v:.4f}' for v in basis_data['pca_explained'][:5]]}")
    print(f"  PC1 alone: {basis_data['pca_explained'][0]:.4f} ({basis_data['pca_explained'][0]*100:.1f}%)")
    
    print("\n" + "="*60)
    print("  C: Residual after removing basis")
    print("="*60)
    
    residual_data = analyze_residual_after_basis(basis_data)
    print(f"  residual norm (mean): {residual_data['residual_norm_mean']:.4f}")
    print(f"  residual norm (std): {residual_data['residual_norm_std']:.4f}")
    print(f"  residual PCA explained: {[f'{v:.4f}' for v in residual_data['residual_pca_explained'][:5]]}")
    
    # The residual norm tells us how much directional information is NOT in the basis
    # If residual is small -> almost all information is along the basis direction
    # If residual is large -> significant information in perpendicular directions
    print(f"\n  -> Basis captures: {(1 - residual_data['relative_residual']**2/2)*100:.1f}% of directional variance")
    print(f"  -> Residual (perpendicular) encodes: {residual_data['relative_residual']**2/2*100:.1f}% of directional variance")
    
    # Save basis for cross-model comparison
    basis_vec = cross_model_basis_similarity(model_name, basis_data)
    
    elapsed = time.time() - t0
    
    print(f"\n{'='*60}")
    print(f"  P43 Summary")
    print(f"{'='*60}")
    print(f"  final layer alignment: {basis_data['mean_alignment']:.4f}")
    print(f"  PC1 explained: {basis_data['pca_explained'][0]*100:.1f}%")
    print(f"  norm CV: {basis_data['norm_cv']*100:.1f}%")
    print(f"  basis dimension: 1 (dominant) + residual in {residual_data['residual_pca_explained'][0]*100:.1f}% perpendicular")
    print(f"  elapsed: {elapsed:.1f}s")
    
    # INV-339: Alignment increases with depth
    early_align = layer_results.get(sample_layers[0], {}).get('mean_alignment', 0)
    late_align = layer_results.get(sample_layers[-1], {}).get('mean_alignment', 0)
    if late_align > early_align:
        print(f"\n  -> [INV-339] Alignment increases with depth: L{sample_layers[0]}={early_align:.4f} -> L{sample_layers[-1]}={late_align:.4f} [OK]")
    else:
        print(f"\n  -> [INV-339] Alignment does NOT increase: L{sample_layers[0]}={early_align:.4f} -> L{sample_layers[-1]}={late_align:.4f}")
    
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
