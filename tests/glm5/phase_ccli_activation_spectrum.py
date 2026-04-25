"""
Phase CCLI: Activation Spectrum & Information Bottleneck Analysis
================================================================
Tests whether the "brute force → precision" model has information-theoretic basis.

Key predictions from IB theory:
1. Effective dimensionality (participation ratio) should decrease with depth
   - Shallow: many active dimensions = "brute force"
   - Deep: fewer active dimensions = "compression"
2. Spectral entropy should decrease with depth (more concentrated spectrum)
3. Feature directions should align better with top PCs in deeper layers
4. The "information compression" should correlate with encoding efficiency

Method:
- Collect activations for ~40 sentences at each layer
- SVD of centered activation matrix
- Compute: participation ratio, spectral entropy, top-k variance explained
- Measure alignment between feature directions and top PCs
"""

import argparse
import json
import os
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_CONFIGS = {
    'qwen3': {
        'name': 'Qwen3-4B',
        'path': 'D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c',
        'n_layers': 36,
        'd_model': 2560,
        'dtype': 'bf16',
    },
    'deepseek7b': {
        'name': 'DeepSeek-R1-Distill-Qwen-7B',
        'path': 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'n_layers': 28,
        'd_model': 3584,
        'dtype': '8bit',
    },
    'glm4': {
        'name': 'GLM4-9B-Chat',
        'path': 'D:/develop/model/hub/models--zai-org--glm-4-9b-chat-hf/download',
        'n_layers': 40,
        'd_model': 4096,
        'dtype': '8bit',
    }
}

# Diverse sentences for activation collection
DIVERSE_SENTENCES = [
    "The cat sat on the mat and looked out the window.",
    "She walked to the store to buy some groceries.",
    "The children played in the park after school ended.",
    "He read the book carefully before writing his review.",
    "The rain fell steadily throughout the long night.",
    "They danced together at the annual celebration.",
    "The scientist published her research in a journal.",
    "We climbed the mountain and enjoyed the view.",
    "The artist painted a beautiful landscape scene.",
    "The students studied hard for their final exams.",
    "A bird sang from the top of the old tree.",
    "The chef prepared an elaborate dinner for guests.",
    "She opened the door and stepped inside quietly.",
    "The dog chased the ball across the green field.",
    "He fixed the broken chair with some nails.",
    "The flowers bloomed in the warm spring sunlight.",
    "They traveled to Paris for their summer vacation.",
    "The musician played a sad melody on piano.",
    "She wrote a letter to her old friend.",
    "The wind blew the leaves across the yard.",
]

# Feature pairs
TENSE_PAIRS = [
    ("She walks to school", "She walked to school"),
    ("He runs in the park", "He ran in the park"),
    ("They play football", "They played football"),
    ("The cat sleeps on the mat", "The cat slept on the mat"),
    ("She sings beautifully", "She sang beautifully"),
    ("He writes a letter", "He wrote a letter"),
    ("The bird flies south", "The bird flew south"),
    ("She reads the book", "She read the book"),
    ("They build houses", "They built houses"),
    ("The wind blows hard", "The wind blew hard"),
]

QUESTION_PAIRS = [
    ("She walks to school", "Does she walk to school"),
    ("He runs in the park", "Does he run in the park"),
    ("They play football", "Do they play football"),
    ("The cat sleeps on the mat", "Does the cat sleep on the mat"),
    ("She sings beautifully", "Does she sing beautifully"),
    ("He writes a letter", "Does he write a letter"),
    ("The bird flies south", "Does the bird fly south"),
    ("She reads the book", "Does she read the book"),
    ("They build houses", "Do they build houses"),
    ("The wind blows hard", "Does the wind blow hard"),
]


def load_model(model_key, device='cuda'):
    config = MODEL_CONFIGS[model_key]
    path = config['path']
    dtype = config['dtype']

    print(f"  Loading: {config['name']} ({dtype})")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dtype == '8bit':
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )
    elif dtype == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )

    model.eval()
    return model, tokenizer


def get_target_layer(model, layer_idx):
    model_type = model.config.model_type
    if model_type in ['qwen2', 'qwen3']:
        return model.model.layers[layer_idx]
    elif model_type in ['chatglm', 'glm4']:
        return model.transformer.encoder.layers[layer_idx]
    else:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers[layer_idx]
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'encoder'):
            return model.transformer.encoder.layers[layer_idx]
        else:
            raise ValueError(f"Cannot find layers for model type {model_type}")


def get_hidden_at_layer(model, tokenizer, target_layer, sentence):
    """Get hidden state at a specific layer."""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    seq_len = attention_mask.sum().item()
    last_pos = seq_len - 1

    hidden_state = [None]

    def capture_hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        hidden_state[0] = h[0, last_pos, :].detach().cpu().float().clone()
        return output

    h = target_layer.register_forward_hook(capture_hook)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    h.remove()

    return hidden_state[0].numpy()


def collect_all_activations(model, tokenizer, target_layer, sentences):
    """Collect activations for all sentences at a layer."""
    acts = []
    for sent in sentences:
        h = get_hidden_at_layer(model, tokenizer, target_layer, sent)
        acts.append(h)
    return np.array(acts)


def compute_spectrum(acts_matrix):
    """Compute spectral properties of activation matrix."""
    n, d = acts_matrix.shape
    mean = acts_matrix.mean(axis=0, keepdims=True)
    centered = acts_matrix - mean

    # Use Gram matrix approach for n < d
    if n < d:
        gram = centered @ centered.T / n
        eigenvalues, _ = np.linalg.eigh(gram)
        eigenvalues = np.maximum(eigenvalues, 0)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues * n
    else:
        cov = centered.T @ centered / n
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 0)
        eigenvalues = np.sort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return {
            'participation_ratio': 0, 'spectral_entropy': 0, 'normalized_entropy': 0,
            'top1_var': 0, 'top5_var': 0, 'top10_var': 0, 'top20_var': 0,
            'n_effective_dims': 0, 'total_variance': 0, 'n_nonzero_dims': 0,
        }

    total_var = eigenvalues.sum()
    props = eigenvalues / total_var

    # Participation ratio = (sum λ)^2 / sum(λ^2)
    pr = (eigenvalues.sum())**2 / (eigenvalues**2).sum()

    # Spectral entropy
    log_props = np.log(props + 1e-30)
    spectral_entropy = -np.sum(props * log_props)
    max_entropy = np.log(len(eigenvalues))
    normalized_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0

    cumvar = np.cumsum(props)
    top1 = cumvar[0] if len(cumvar) >= 1 else 0
    top5 = cumvar[min(4, len(cumvar)-1)] if len(cumvar) >= 1 else 0
    top10 = cumvar[min(9, len(cumvar)-1)] if len(cumvar) >= 1 else 0
    top20 = cumvar[min(19, len(cumvar)-1)] if len(cumvar) >= 1 else 0

    return {
        'participation_ratio': float(pr),
        'spectral_entropy': float(spectral_entropy),
        'normalized_entropy': float(normalized_entropy),
        'top1_var': float(top1),
        'top5_var': float(top5),
        'top10_var': float(top10),
        'top20_var': float(top20),
        'n_effective_dims': float(pr),
        'total_variance': float(total_var),
        'n_nonzero_dims': int(len(eigenvalues)),
    }


def compute_feature_pc_alignment(acts_matrix, feature_dir, n_top_pcs=10):
    """Compute alignment between a feature direction and top PCs."""
    n, d = acts_matrix.shape
    mean = acts_matrix.mean(axis=0, keepdims=True)
    centered = acts_matrix - mean

    feat_norm = np.linalg.norm(feature_dir)
    if feat_norm < 1e-10:
        return {'top1': 0, 'top5': 0, 'top10': 0, 'max': 0, 'weighted': 0}
    feat_unit = feature_dir / feat_norm

    # Compute PCs via SVD
    if n < d:
        gram = centered @ centered.T
        eigvals, eigvecs = np.linalg.eigh(gram)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Convert to PC directions
        pcs = (eigvecs.T @ centered)
        pc_norms = np.linalg.norm(pcs, axis=1, keepdims=True)
        pc_norms = np.maximum(pc_norms, 1e-10)
        pcs = pcs / pc_norms
        sv_vals = np.sqrt(np.maximum(eigvals, 0))
    else:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pcs = Vt
        pc_norms = np.linalg.norm(pcs, axis=1, keepdims=True)
        pc_norms = np.maximum(pc_norms, 1e-10)
        pcs = pcs / pc_norms
        sv_vals = S

    cos_sims = np.abs(pcs @ feat_unit)

    sv_top = sv_vals[:len(cos_sims)]
    if sv_top.sum() > 0:
        weights = sv_top / sv_top.sum()
        weighted_alignment = np.sum(weights * cos_sims)
    else:
        weighted_alignment = 0

    return {
        'top1': float(cos_sims[0]) if len(cos_sims) >= 1 else 0,
        'top5': float(np.max(cos_sims[:5])) if len(cos_sims) >= 1 else 0,
        'top10': float(np.max(cos_sims[:10])) if len(cos_sims) >= 1 else 0,
        'max': float(np.max(cos_sims)) if len(cos_sims) >= 1 else 0,
        'weighted': float(weighted_alignment),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_sentences', type=int, default=20, help='Number of diverse sentences')
    args = parser.parse_args()

    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]

    out_dir = f"results/causal_fiber/{model_key}_ccli"
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, 'run.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f"=== Phase CCLI: Activation Spectrum & Information Bottleneck ===")
    log(f"Model: {cfg['name']}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")

    n_layers = cfg['n_layers']

    # Test at key positions
    test_layers = [
        max(1, n_layers // 8),
        max(2, n_layers // 4),
        max(3, n_layers // 2),
        max(4, 3 * n_layers // 4),
        max(5, n_layers - 2),
    ]
    test_layers = sorted(set([l for l in test_layers if 1 <= l < n_layers]))

    # All sentences: diverse + feature pairs
    all_sentences = list(DIVERSE_SENTENCES[:args.n_sentences])
    # Add feature pair sentences
    for p, n in TENSE_PAIRS:
        if p not in all_sentences:
            all_sentences.append(p)
        if n not in all_sentences:
            all_sentences.append(n)
    for p, n in QUESTION_PAIRS:
        if p not in all_sentences:
            all_sentences.append(p)
        if n not in all_sentences:
            all_sentences.append(n)

    n_sent = len(all_sentences)
    log(f"Using {n_sent} sentences total")

    results = {}
    pr_list = []
    entropy_list = []
    top5_var_list = []
    top10_var_list = []

    for layer_idx in test_layers:
        target_layer = get_target_layer(model, layer_idx)
        log(f"\n  === Layer {layer_idx} ===")

        # Collect all activations
        acts = collect_all_activations(model, tokenizer, target_layer, all_sentences)

        # Spectrum analysis
        spec = compute_spectrum(acts)

        pr_list.append(spec['participation_ratio'])
        entropy_list.append(spec['normalized_entropy'])
        top5_var_list.append(spec['top5_var'])
        top10_var_list.append(spec['top10_var'])

        log(f"    Participation Ratio: {spec['participation_ratio']:.2f}")
        log(f"    Normalized Entropy: {spec['normalized_entropy']:.4f}")
        log(f"    Top-5 var: {spec['top5_var']:.4f}, Top-10: {spec['top10_var']:.4f}, Top-20: {spec['top20_var']:.4f}")
        log(f"    N nonzero dims: {spec['n_nonzero_dims']}, Total var: {spec['total_variance']:.2f}")

        # Feature direction norms
        tense_dirs = []
        quest_dirs = []
        act_norms = []

        for i, s in enumerate(all_sentences):
            act_norms.append(np.linalg.norm(acts[i]))

        for p, n in TENSE_PAIRS:
            if p in all_sentences and n in all_sentences:
                pi = all_sentences.index(p)
                ni = all_sentences.index(n)
                tense_dirs.append(acts[pi] - acts[ni])

        for p, n in QUESTION_PAIRS:
            if p in all_sentences and n in all_sentences:
                pi = all_sentences.index(p)
                ni = all_sentences.index(n)
                quest_dirs.append(acts[pi] - acts[ni])

        mean_tense_norm = np.mean([np.linalg.norm(d) for d in tense_dirs]) if tense_dirs else 0
        mean_quest_norm = np.mean([np.linalg.norm(d) for d in quest_dirs]) if quest_dirs else 0
        mean_h_norm = np.mean(act_norms)

        log(f"    ||tense_dir||={mean_tense_norm:.2f}, ||quest_dir||={mean_quest_norm:.2f}, ||h||={mean_h_norm:.2f}")

        # Feature-PC alignment
        alignments = {}
        if tense_dirs:
            tense_dir = np.mean(tense_dirs, axis=0)
            t_align = compute_feature_pc_alignment(acts, tense_dir)
            alignments['tense'] = t_align
            log(f"    Tense-PC alignment: top1={t_align['top1']:.4f}, top5={t_align['top5']:.4f}, "
                f"top10={t_align['top10']:.4f}, weighted={t_align['weighted']:.4f}")

        if quest_dirs:
            quest_dir = np.mean(quest_dirs, axis=0)
            q_align = compute_feature_pc_alignment(acts, quest_dir)
            alignments['question'] = q_align
            log(f"    Quest-PC alignment: top1={q_align['top1']:.4f}, top5={q_align['top5']:.4f}, "
                f"top10={q_align['top10']:.4f}, weighted={q_align['weighted']:.4f}")

        results[layer_idx] = {
            'spectrum': spec,
            'feature_alignment': alignments,
            'tense_dir_norm': float(mean_tense_norm),
            'quest_dir_norm': float(mean_quest_norm),
            'activation_norm': float(mean_h_norm),
        }

    # Correlation analysis
    log(f"\n{'='*70}")
    log(f"ANALYSIS")
    log(f"{'='*70}")

    layer_indices = list(range(len(test_layers)))

    corr_pr, p_pr = spearmanr(layer_indices, pr_list)
    corr_ent, p_ent = spearmanr(layer_indices, entropy_list)
    corr_top5, p_top5 = spearmanr(layer_indices, top5_var_list)
    corr_top10, p_top10 = spearmanr(layer_indices, top10_var_list)

    log(f"\n  Depth Correlations:")
    log(f"    Corr(L, participation_ratio) = {corr_pr:.3f} (p={p_pr:.4f})")
    log(f"    Corr(L, normalized_entropy) = {corr_ent:.3f} (p={p_ent:.4f})")
    log(f"    Corr(L, top5_var_explained) = {corr_top5:.3f} (p={p_top5:.4f})")
    log(f"    Corr(L, top10_var_explained) = {corr_top10:.3f} (p={p_top10:.4f})")

    # Feature alignment correlations
    tense_weighted = [results[l]['feature_alignment'].get('tense', {}).get('weighted', 0) for l in test_layers]
    quest_weighted = [results[l]['feature_alignment'].get('question', {}).get('weighted', 0) for l in test_layers]
    tense_top10 = [results[l]['feature_alignment'].get('tense', {}).get('top10', 0) for l in test_layers]
    quest_top10 = [results[l]['feature_alignment'].get('question', {}).get('top10', 0) for l in test_layers]

    corr_t_w, p_t_w = spearmanr(layer_indices, tense_weighted)
    corr_q_w, p_q_w = spearmanr(layer_indices, quest_weighted)
    corr_t10, p_t10 = spearmanr(layer_indices, tense_top10)
    corr_q10, p_q10 = spearmanr(layer_indices, quest_top10)

    log(f"\n  Feature-PC Alignment Correlations:")
    log(f"    Corr(L, tense_weighted) = {corr_t_w:.3f} (p={p_t_w:.4f})")
    log(f"    Corr(L, quest_weighted) = {corr_q_w:.3f} (p={p_q_w:.4f})")
    log(f"    Corr(L, tense_top10) = {corr_t10:.3f} (p={p_t10:.4f})")
    log(f"    Corr(L, quest_top10) = {corr_q10:.3f} (p={p_q10:.4f})")

    # Feature direction norm correlations
    tense_norms_list = [results[l]['tense_dir_norm'] for l in test_layers]
    quest_norms_list = [results[l]['quest_dir_norm'] for l in test_layers]
    h_norms_list = [results[l]['activation_norm'] for l in test_layers]

    corr_tn, _ = spearmanr(layer_indices, tense_norms_list)
    corr_qn, _ = spearmanr(layer_indices, quest_norms_list)
    corr_hn, _ = spearmanr(layer_indices, h_norms_list)

    log(f"\n  Norm Correlations:")
    log(f"    Corr(L, ||tense_dir||) = {corr_tn:.3f}")
    log(f"    Corr(L, ||quest_dir||) = {corr_qn:.3f}")
    log(f"    Corr(L, ||h||) = {corr_hn:.3f}")

    # Summary table
    log(f"\n  Summary Table:")
    log(f"  {'L':>4} {'PR':>6} {'Entropy':>8} {'Top5':>6} {'Top10':>6} "
        f"{'T_w':>6} {'Q_w':>6} {'T_t10':>6} {'Q_t10':>6}")

    for i, l in enumerate(test_layers):
        log(f"  {l:>4} {pr_list[i]:>6.1f} {entropy_list[i]:>8.4f} {top5_var_list[i]:>6.4f} "
            f"{top10_var_list[i]:>6.4f} {tense_weighted[i]:>6.4f} {quest_weighted[i]:>6.4f} "
            f"{tense_top10[i]:>6.4f} {quest_top10[i]:>6.4f}")

    # Key predictions
    log(f"\n  IB PREDICTIONS:")
    pr_decreases = corr_pr < -0.3
    ent_decreases = corr_ent < -0.3
    top5_increases = corr_top5 > 0.3

    log(f"    {'SUPPORTED' if pr_decreases else 'NOT SUPPORTED'}: PR decreases with depth (Corr={corr_pr:.3f})")
    log(f"    {'SUPPORTED' if ent_decreases else 'NOT SUPPORTED'}: Entropy decreases with depth (Corr={corr_ent:.3f})")
    log(f"    {'SUPPORTED' if top5_increases else 'NOT SUPPORTED'}: Top-5 variance increases with depth (Corr={corr_top5:.3f})")
    log(f"    Feature-PC alignment increases with depth (tense: Corr={corr_t_w:.3f}, quest: Corr={corr_q_w:.3f})")

    # Save results
    save_data = {
        'model': model_key,
        'layers': test_layers,
        'participation_ratios': pr_list,
        'normalized_entropies': entropy_list,
        'top5_variances': top5_var_list,
        'top10_variances': top10_var_list,
        'tense_alignment_weighted': tense_weighted,
        'quest_alignment_weighted': quest_weighted,
        'tense_alignment_top10': tense_top10,
        'quest_alignment_top10': quest_top10,
        'tense_dir_norms': tense_norms_list,
        'quest_dir_norms': quest_norms_list,
        'activation_norms': h_norms_list,
        'correlations': {
            'pr_vs_depth': float(corr_pr),
            'entropy_vs_depth': float(corr_ent),
            'top5_vs_depth': float(corr_top5),
            'top10_vs_depth': float(corr_top10),
            'tense_weighted_vs_depth': float(corr_t_w),
            'quest_weighted_vs_depth': float(corr_q_w),
            'tense_top10_vs_depth': float(corr_t10),
            'quest_top10_vs_depth': float(corr_q10),
            'tense_norm_vs_depth': float(corr_tn),
            'quest_norm_vs_depth': float(corr_qn),
            'h_norm_vs_depth': float(corr_hn),
        },
    }

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    del model
    torch.cuda.empty_cache()

    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Results saved to {out_dir}/results.json")


if __name__ == '__main__':
    main()
