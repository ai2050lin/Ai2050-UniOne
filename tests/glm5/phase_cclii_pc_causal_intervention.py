"""
Phase CCLII: Principal Component Causal Intervention
=====================================================
CCLI found: feature-PC alignment increases with depth.
But we haven't verified that deep PCs actually CONTROL semantic output.

This test does causal intervention along PC directions:
1. At each layer, compute top PCs from a batch of sentences
2. Perturb activations along PC directions (add/subtract)
3. Measure: (a) how much output changes (KL divergence)
           (b) whether the change is semantically targeted
           (c) compare with random direction perturbation

Key predictions from "Feature Principalization" model:
- Deep PCs should produce more semantically targeted changes
- The ratio of "feature-aligned output change" / "total output change"
  should be higher for deep PCs
- Intervening along a "tense-aligned PC" in deep layers should
  change the output's tense more effectively than in shallow layers
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

# Sentences for PC computation
PC_SENTENCES = [
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
    "The river flows through the entire valley.",
    "He drives his car to work every morning.",
    "The stars shine brightly on clear nights.",
    "She teaches mathematics at the local school.",
    "They build houses for people in need.",
    "The sun rises early in the summer months.",
    "He paints pictures of the ocean waves.",
    "The bird flies high above the tall trees.",
    "She runs five miles every single morning.",
    "The clock ticks loudly in the silent room.",
]

# Test sentences for intervention (present tense)
TEST_PRESENT = [
    "She walks to school every day",
    "He runs in the park each morning",
    "The cat sleeps on the mat quietly",
    "They play football on Sundays",
    "The bird flies above the trees",
]

# Corresponding past tense (for measuring tense change)
TEST_PAST = [
    "She walked to school every day",
    "He ran in the park each morning",
    "The cat slept on the mat quietly",
    "They played football on Sundays",
    "The bird flew above the trees",
]

# Feature pairs for computing feature directions
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


def get_hidden_and_logits(model, tokenizer, target_layer, sentence):
    """Get hidden state at target layer and final logits."""
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, last_pos, :].detach().cpu().float()
    h.remove()

    return hidden_state[0].numpy(), logits.numpy()


def intervene_and_get_logits(model, tokenizer, target_layer, sentence, perturbation_vec, last_pos_offset=0):
    """Perturb hidden state at target layer and get resulting logits."""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    seq_len = attention_mask.sum().item()
    last_pos = seq_len - 1

    perturb_tensor = torch.tensor(perturbation_vec, dtype=torch.float32, device=model.device)

    def perturb_hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        perturbed = h.clone()
        perturbed[0, last_pos, :] = perturbed[0, last_pos, :].float() + perturb_tensor.to(perturbed.dtype)
        if isinstance(output, tuple):
            return (perturbed,) + output[1:]
        return perturbed

    h = target_layer.register_forward_hook(perturb_hook)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, last_pos, :].detach().cpu().float()
    h.remove()

    return logits.numpy()


def compute_pcs(activations_matrix, n_pcs=10):
    """Compute top n principal components from activation matrix."""
    n, d = activations_matrix.shape
    mean = activations_matrix.mean(axis=0, keepdims=True)
    centered = activations_matrix - mean

    if n < d:
        # Use Gram matrix
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
    else:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pcs = Vt[:n_pcs]
        pc_norms = np.linalg.norm(pcs, axis=1, keepdims=True)
        pc_norms = np.maximum(pc_norms, 1e-10)
        pcs = pcs / pc_norms

    return pcs[:n_pcs]  # [n_pcs, d]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_pcs', type=int, default=5, help='Number of PCs to test')
    parser.add_argument('--eps_rel', type=float, default=0.1, help='Relative perturbation magnitude')
    parser.add_argument('--n_random', type=int, default=5, help='Number of random directions for baseline')
    args = parser.parse_args()

    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]

    out_dir = f"results/causal_fiber/{model_key}_cclii"
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, 'run.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f"=== Phase CCLII: Principal Component Causal Intervention ===")
    log(f"Model: {cfg['name']}")
    log(f"n_pcs: {args.n_pcs}, eps_rel: {args.eps_rel}, n_random: {args.n_random}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")

    n_layers = cfg['n_layers']
    test_layers = [
        max(1, n_layers // 8),
        max(2, n_layers // 4),
        max(3, n_layers // 2),
        max(4, 3 * n_layers // 4),
        max(5, n_layers - 2),
    ]
    test_layers = sorted(set([l for l in test_layers if 1 <= l < n_layers]))

    # Step 1: Collect activations for PC computation
    log(f"\n--- Step 1: Collecting activations for PC computation ---")

    layer_pcs = {}  # layer -> [n_pcs, d_model]
    layer_feature_dirs = {}  # layer -> tense_dir, quest_dir

    for layer_idx in test_layers:
        target_layer = get_target_layer(model, layer_idx)
        log(f"  Layer {layer_idx}: collecting activations...")

        # Collect for PC computation
        pc_acts = []
        for sent in PC_SENTENCES:
            h, _ = get_hidden_and_logits(model, tokenizer, target_layer, sent)
            pc_acts.append(h)
        pc_acts = np.array(pc_acts)

        # Compute PCs
        pcs = compute_pcs(pc_acts, n_pcs=args.n_pcs)
        layer_pcs[layer_idx] = pcs

        # Compute feature directions
        tense_acts = []
        for p, n in TENSE_PAIRS:
            hp, _ = get_hidden_and_logits(model, tokenizer, target_layer, p)
            hn, _ = get_hidden_and_logits(model, tokenizer, target_layer, n)
            tense_acts.append(hp - hn)
        tense_dir = np.mean(tense_acts, axis=0)
        tense_dir = tense_dir / (np.linalg.norm(tense_dir) + 1e-10)

        # Feature-PC alignment
        alignments = []
        for i in range(len(pcs)):
            cos_sim = np.abs(np.dot(pcs[i], tense_dir))
            alignments.append(cos_sim)

        # Find the PC most aligned with tense feature
        best_pc_idx = np.argmax(alignments)
        best_alignment = alignments[best_pc_idx]

        layer_feature_dirs[layer_idx] = {
            'tense_dir': tense_dir,
            'best_pc_idx': best_pc_idx,
            'best_alignment': best_alignment,
            'all_alignments': alignments,
        }

        log(f"    PCs computed: {pcs.shape}, best tense-PC alignment: {best_alignment:.4f} (PC{best_pc_idx+1})")

    # Step 2: Causal intervention experiments
    log(f"\n--- Step 2: Causal Intervention ---")

    results = {}

    for layer_idx in test_layers:
        target_layer = get_target_layer(model, layer_idx)
        pcs = layer_pcs[layer_idx]
        feat_info = layer_feature_dirs[layer_idx]
        tense_dir = feat_info['tense_dir']
        best_pc_idx = feat_info['best_pc_idx']
        best_pc = pcs[best_pc_idx]

        log(f"\n  === Layer {layer_idx} ===")

        layer_results = {
            'pc_interventions': [],
            'random_interventions': [],
            'feature_interventions': [],
            'best_pc_alignment': feat_info['best_alignment'],
        }

        for test_idx, (test_sent, past_sent) in enumerate(zip(TEST_PRESENT, TEST_PAST)):
            # Clean outputs
            h_clean, logits_clean = get_hidden_and_logits(model, tokenizer, target_layer, test_sent)
            _, logits_past = get_hidden_and_logits(model, tokenizer, target_layer, past_sent)

            h_norm = np.linalg.norm(h_clean)
            eps = args.eps_rel * h_norm

            # Get present-tense token logits and past-tense token logits
            # For simplicity, measure how much the output shifts toward the past-tense output
            logit_diff_clean = logits_past - logits_clean  # direction toward past tense

            # 2a. Intervene along each PC
            pc_kl_divs = []
            pc_tense_shifts = []

            for pc_i in range(len(pcs)):
                pc_dir = pcs[pc_i]

                # +perturbation
                logits_plus = intervene_and_get_logits(
                    model, tokenizer, target_layer, test_sent, eps * pc_dir)

                # -perturbation
                logits_minus = intervene_and_get_logits(
                    model, tokenizer, target_layer, test_sent, -eps * pc_dir)

                # KL divergence (average of + and -)
                log_p_plus = logits_plus - np.max(logits_plus)
                log_p_plus = log_p_plus - np.log(np.sum(np.exp(log_p_plus)) + 1e-30)
                log_p_clean = logits_clean - np.max(logits_clean)
                log_p_clean = log_p_clean - np.log(np.sum(np.exp(log_p_clean)) + 1e-30)

                p_plus = np.exp(log_p_plus)
                p_clean = np.exp(log_p_clean)

                kl_plus = np.sum(p_plus * (log_p_plus - log_p_clean + 1e-30))
                kl_minus_val = 0  # compute similarly

                log_p_minus = logits_minus - np.max(logits_minus)
                log_p_minus = log_p_minus - np.log(np.sum(np.exp(log_p_minus)) + 1e-30)
                p_minus = np.exp(log_p_minus)
                kl_minus_val = np.sum(p_minus * (log_p_minus - log_p_clean + 1e-30))

                avg_kl = (kl_plus + kl_minus_val) / 2

                # Tense shift: how much does +perturbation shift toward past tense output?
                diff_plus = logits_plus - logits_clean
                tense_shift_plus = np.dot(diff_plus, logit_diff_clean) / (np.linalg.norm(diff_plus) * np.linalg.norm(logit_diff_clean) + 1e-10)

                diff_minus = logits_minus - logits_clean
                tense_shift_minus = np.dot(diff_minus, logit_diff_clean) / (np.linalg.norm(diff_minus) * np.linalg.norm(logit_diff_clean) + 1e-10)

                # Take max absolute shift (either direction)
                tense_shift = max(abs(tense_shift_plus), abs(tense_shift_minus))

                pc_kl_divs.append(avg_kl)
                pc_tense_shifts.append(tense_shift)

            # 2b. Intervene along random directions
            rand_kl_divs = []
            rand_tense_shifts = []

            for _ in range(args.n_random):
                rand_dir = np.random.randn(h_clean.shape[0])
                rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-10)

                logits_plus = intervene_and_get_logits(
                    model, tokenizer, target_layer, test_sent, eps * rand_dir)
                logits_minus = intervene_and_get_logits(
                    model, tokenizer, target_layer, test_sent, -eps * rand_dir)

                log_p_plus = logits_plus - np.max(logits_plus)
                log_p_plus = log_p_plus - np.log(np.sum(np.exp(log_p_plus)) + 1e-30)
                log_p_clean2 = logits_clean - np.max(logits_clean)
                log_p_clean2 = log_p_clean2 - np.log(np.sum(np.exp(log_p_clean2)) + 1e-30)
                p_plus = np.exp(log_p_plus)
                p_clean2 = np.exp(log_p_clean2)
                kl_plus = np.sum(p_plus * (log_p_plus - log_p_clean2 + 1e-30))

                log_p_minus = logits_minus - np.max(logits_minus)
                log_p_minus = log_p_minus - np.log(np.sum(np.exp(log_p_minus)) + 1e-30)
                p_minus = np.exp(log_p_minus)
                kl_minus = np.sum(p_minus * (log_p_minus - log_p_clean2 + 1e-30))

                avg_kl = (kl_plus + kl_minus) / 2

                diff_plus = logits_plus - logits_clean
                tense_shift_plus = np.dot(diff_plus, logit_diff_clean) / (np.linalg.norm(diff_plus) * np.linalg.norm(logit_diff_clean) + 1e-10)

                diff_minus = logits_minus - logits_clean
                tense_shift_minus = np.dot(diff_minus, logit_diff_clean) / (np.linalg.norm(diff_minus) * np.linalg.norm(logit_diff_clean) + 1e-10)

                tense_shift = max(abs(tense_shift_plus), abs(tense_shift_minus))

                rand_kl_divs.append(avg_kl)
                rand_tense_shifts.append(tense_shift)

            # 2c. Intervene along feature direction directly
            feat_kl_divs = []
            feat_tense_shifts = []

            logits_plus = intervene_and_get_logits(
                model, tokenizer, target_layer, test_sent, eps * tense_dir)
            logits_minus = intervene_and_get_logits(
                model, tokenizer, target_layer, test_sent, -eps * tense_dir)

            log_p_plus = logits_plus - np.max(logits_plus)
            log_p_plus = log_p_plus - np.log(np.sum(np.exp(log_p_plus)) + 1e-30)
            log_p_clean3 = logits_clean - np.max(logits_clean)
            log_p_clean3 = log_p_clean3 - np.log(np.sum(np.exp(log_p_clean3)) + 1e-30)
            p_plus = np.exp(log_p_plus)
            p_clean3 = np.exp(log_p_clean3)
            kl_plus = np.sum(p_plus * (log_p_plus - log_p_clean3 + 1e-30))

            log_p_minus = logits_minus - np.max(logits_minus)
            log_p_minus = log_p_minus - np.log(np.sum(np.exp(log_p_minus)) + 1e-30)
            p_minus = np.exp(log_p_minus)
            kl_minus = np.sum(p_minus * (log_p_minus - log_p_clean3 + 1e-30))

            feat_avg_kl = (kl_plus + kl_minus) / 2

            diff_plus = logits_plus - logits_clean
            feat_tense_shift_plus = np.dot(diff_plus, logit_diff_clean) / (np.linalg.norm(diff_plus) * np.linalg.norm(logit_diff_clean) + 1e-10)
            diff_minus = logits_minus - logits_clean
            feat_tense_shift_minus = np.dot(diff_minus, logit_diff_clean) / (np.linalg.norm(diff_minus) * np.linalg.norm(logit_diff_clean) + 1e-10)
            feat_tense_shift = max(abs(feat_tense_shift_plus), abs(feat_tense_shift_minus))

            # Store per-sentence results
            test_result = {
                'sentence': test_sent,
                'pc_kl_mean': float(np.mean(pc_kl_divs)),
                'pc_kl_max': float(np.max(pc_kl_divs)),
                'pc_tense_shift_mean': float(np.mean(pc_tense_shifts)),
                'pc_tense_shift_best_pc': float(pc_tense_shifts[best_pc_idx]),
                'rand_kl_mean': float(np.mean(rand_kl_divs)),
                'rand_tense_shift_mean': float(np.mean(rand_tense_shifts)),
                'feat_kl': float(feat_avg_kl),
                'feat_tense_shift': float(feat_tense_shift),
                'kl_ratio_pc_rand': float(np.mean(pc_kl_divs) / (np.mean(rand_kl_divs) + 1e-10)),
                'tense_ratio_pc_rand': float(np.mean(pc_tense_shifts) / (np.mean(rand_tense_shifts) + 1e-10)),
                'tense_ratio_feat_rand': float(feat_tense_shift / (np.mean(rand_tense_shifts) + 1e-10)),
                'tense_ratio_bestpc_rand': float(pc_tense_shifts[best_pc_idx] / (np.mean(rand_tense_shifts) + 1e-10)),
            }

            layer_results['pc_interventions'].append(test_result)

        # Aggregate across test sentences
        agg = {}
        for key in ['pc_kl_mean', 'pc_kl_max', 'pc_tense_shift_mean', 'pc_tense_shift_best_pc',
                     'rand_kl_mean', 'rand_tense_shift_mean', 'feat_kl', 'feat_tense_shift',
                     'kl_ratio_pc_rand', 'tense_ratio_pc_rand', 'tense_ratio_feat_rand',
                     'tense_ratio_bestpc_rand']:
            vals = [r[key] for r in layer_results['pc_interventions']]
            agg[key] = float(np.mean(vals))

        layer_results['aggregated'] = agg
        results[layer_idx] = layer_results

        log(f"    PC KL (mean): {agg['pc_kl_mean']:.4f}, Random KL: {agg['rand_kl_mean']:.4f}, "
            f"KL ratio: {agg['kl_ratio_pc_rand']:.3f}")
        log(f"    PC tense shift: {agg['pc_tense_shift_mean']:.4f}, Random: {agg['rand_tense_shift_mean']:.4f}, "
            f"ratio: {agg['tense_ratio_pc_rand']:.3f}")
        log(f"    Best-PC tense shift: {agg['pc_tense_shift_best_pc']:.4f}, ratio vs rand: {agg['tense_ratio_bestpc_rand']:.3f}")
        log(f"    Feature tense shift: {agg['feat_tense_shift']:.4f}, ratio vs rand: {agg['tense_ratio_feat_rand']:.3f}")
        log(f"    Best-PC alignment: {feat_info['best_alignment']:.4f}")

    # Step 3: Depth correlation analysis
    log(f"\n{'='*70}")
    log(f"ANALYSIS")
    log(f"{'='*70}")

    layers = sorted(results.keys())
    layer_indices = list(range(len(layers)))

    # Key metrics across depth
    pc_kl_ratio = [results[l]['aggregated']['kl_ratio_pc_rand'] for l in layers]
    pc_tense_ratio = [results[l]['aggregated']['tense_ratio_pc_rand'] for l in layers]
    bestpc_tense_ratio = [results[l]['aggregated']['tense_ratio_bestpc_rand'] for l in layers]
    feat_tense_ratio = [results[l]['aggregated']['tense_ratio_feat_rand'] for l in layers]
    best_alignment = [results[l]['best_pc_alignment'] for l in layers]
    rand_kl = [results[l]['aggregated']['rand_kl_mean'] for l in layers]
    pc_kl = [results[l]['aggregated']['pc_kl_mean'] for l in layers]

    log(f"\n  Summary Table:")
    log(f"  {'L':>4} {'PC/ran KL':>9} {'PC/ran ts':>9} {'BPC/ran':>8} {'Feat/ran':>8} {'Align':>6}")
    for i, l in enumerate(layers):
        log(f"  {l:>4} {pc_kl_ratio[i]:>9.3f} {pc_tense_ratio[i]:>9.3f} "
            f"{bestpc_tense_ratio[i]:>8.3f} {feat_tense_ratio[i]:>8.3f} {best_alignment[i]:>6.4f}")

    # Correlations
    corr_kl_ratio, _ = spearmanr(layer_indices, pc_kl_ratio) if len(layers) > 2 else (0, 1)
    corr_tense_ratio, _ = spearmanr(layer_indices, pc_tense_ratio) if len(layers) > 2 else (0, 1)
    corr_bestpc_ratio, _ = spearmanr(layer_indices, bestpc_tense_ratio) if len(layers) > 2 else (0, 1)
    corr_feat_ratio, _ = spearmanr(layer_indices, feat_tense_ratio) if len(layers) > 2 else (0, 1)
    corr_alignment, _ = spearmanr(layer_indices, best_alignment) if len(layers) > 2 else (0, 1)
    corr_rand_kl, _ = spearmanr(layer_indices, rand_kl) if len(layers) > 2 else (0, 1)

    log(f"\n  Depth Correlations:")
    log(f"    Corr(L, PC/random KL ratio) = {corr_kl_ratio:.3f}")
    log(f"    Corr(L, PC/random tense ratio) = {corr_tense_ratio:.3f}")
    log(f"    Corr(L, bestPC/random tense ratio) = {corr_bestpc_ratio:.3f}")
    log(f"    Corr(L, feature/random tense ratio) = {corr_feat_ratio:.3f}")
    log(f"    Corr(L, best-PC alignment) = {corr_alignment:.3f}")
    log(f"    Corr(L, random KL) = {corr_rand_kl:.3f}")

    # Key predictions
    log(f"\n  PREDICTIONS:")
    log(f"    {'SUPPORTED' if corr_bestpc_ratio > 0.3 else 'NOT SUPPORTED'}: "
        f"Best-PC tense shift ratio increases with depth (Corr={corr_bestpc_ratio:.3f})")
    log(f"    {'SUPPORTED' if corr_feat_ratio > 0.3 else 'NOT SUPPORTED'}: "
        f"Feature direction tense shift ratio increases with depth (Corr={corr_feat_ratio:.3f})")
    log(f"    {'SUPPORTED' if corr_alignment > 0.3 else 'NOT SUPPORTED'}: "
        f"Best-PC alignment increases with depth (Corr={corr_alignment:.3f})")

    # Interpretation
    n_supported = sum([
        corr_bestpc_ratio > 0.3,
        corr_feat_ratio > 0.3,
        corr_alignment > 0.3,
    ])

    log(f"\n  INTERPRETATION:")
    if n_supported >= 2:
        log(f"  => Feature Principalization model SUPPORTED!")
        log(f"  => Deep PCs produce more semantically targeted interventions")
        log(f"  => This confirms that deep PCs capture language features, not just statistics")
    elif n_supported >= 1:
        log(f"  => Feature Principalization model PARTIALLY supported")
        log(f"  => Some evidence that deep PCs are semantically meaningful")
    else:
        log(f"  => Feature Principalization model NOT supported by causal intervention")
        log(f"  => PC alignment doesn't translate to causal control")

    # Save results
    save_data = {
        'model': model_key,
        'layers': layers,
        'pc_kl_ratios': pc_kl_ratio,
        'pc_tense_ratios': pc_tense_ratio,
        'bestpc_tense_ratios': bestpc_tense_ratio,
        'feat_tense_ratios': feat_tense_ratio,
        'best_alignments': best_alignment,
        'correlations': {
            'kl_ratio_vs_depth': float(corr_kl_ratio),
            'tense_ratio_vs_depth': float(corr_tense_ratio),
            'bestpc_ratio_vs_depth': float(corr_bestpc_ratio),
            'feat_ratio_vs_depth': float(corr_feat_ratio),
            'alignment_vs_depth': float(corr_alignment),
            'rand_kl_vs_depth': float(corr_rand_kl),
        },
        'layer_results': {str(k): v['aggregated'] for k, v in results.items()},
    }

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    del model
    torch.cuda.empty_cache()

    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Results saved to {out_dir}/results.json")


if __name__ == '__main__':
    main()
