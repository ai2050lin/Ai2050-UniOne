"""
Phase CCLIV: Multi-Feature Causal Geometry Analysis
=====================================================
CCLIII found: Question (syntactic) standardizes, Tense (lexical) doesn't.
But we only tested 2 features! This test validates the dual-track model
with 5 feature types:

1. Tense (lexical/morphological): walk→walked - verb-specific
2. Question (syntactic): statement→question - structural
3. Negation (syntactic): affirmative→negative - structural  
4. Number (morphological): singular→plural - partly lexical
5. Voice (syntactic): active→passive - structural

Predictions:
- Syntactic features (question, negation, voice): CR increases with depth
- Lexical features (tense, number): CR does NOT increase with depth
- All features: efficacy_ratio increases with depth (from CCLII)
- Syntactic features: PR decreases with depth (causal direction collapse)
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

# ============================================================
# Feature pairs for each feature type
# ============================================================

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
    ("I think about it", "I thought about it"),
    ("We drink coffee", "We drank coffee"),
    ("He drives fast", "He drove fast"),
    ("She knows the answer", "She knew the answer"),
    ("The dog barks loudly", "The dog barked loudly"),
    ("They grow vegetables", "They grew vegetables"),
    ("I feel happy today", "I felt happy today"),
    ("She brings lunch", "She brought lunch"),
    ("He catches the ball", "He caught the ball"),
    ("The bell rings twice", "The bell rang twice"),
]

QUESTION_PAIRS = [
    ("The sky is blue", "Is the sky blue?"),
    ("Cats like fish", "Do cats like fish?"),
    ("She is happy", "Is she happy?"),
    ("The door is open", "Is the door open?"),
    ("He works hard", "Does he work hard?"),
    ("Water boils at 100 degrees", "Does water boil at 100 degrees?"),
    ("The car is red", "Is the car red?"),
    ("They live nearby", "Do they live nearby?"),
    ("The book is long", "Is the book long?"),
    ("She speaks French", "Does she speak French?"),
    ("The sun is bright", "Is the sun bright?"),
    ("Birds can fly", "Can birds fly?"),
    ("The room is cold", "Is the room cold?"),
    ("He plays guitar", "Does he play guitar?"),
    ("The food is ready", "Is the food ready?"),
    ("They know the truth", "Do they know the truth?"),
    ("The test is hard", "Is the test hard?"),
    ("She loves music", "Does she love music?"),
    ("The lake is deep", "Is the lake deep?"),
    ("He runs every day", "Does he run every day?"),
]

NEGATION_PAIRS = [
    ("She walks to school", "She does not walk to school"),
    ("He runs in the park", "He does not run in the park"),
    ("They play football", "They do not play football"),
    ("The cat sleeps on the mat", "The cat does not sleep on the mat"),
    ("She sings beautifully", "She does not sing beautifully"),
    ("He writes a letter", "He does not write a letter"),
    ("The bird flies south", "The bird does not fly south"),
    ("She reads the book", "She does not read the book"),
    ("They build houses", "They do not build houses"),
    ("The wind blows hard", "The wind does not blow hard"),
    ("I think about it", "I do not think about it"),
    ("We drink coffee", "We do not drink coffee"),
    ("He drives fast", "He does not drive fast"),
    ("She knows the answer", "She does not know the answer"),
    ("The dog barks loudly", "The dog does not bark loudly"),
    ("They grow vegetables", "They do not grow vegetables"),
    ("I feel happy today", "I do not feel happy today"),
    ("She brings lunch", "She does not bring lunch"),
    ("He catches the ball", "He does not catch the ball"),
    ("The bell rings twice", "The bell does not ring twice"),
]

NUMBER_PAIRS = [
    ("The cat sleeps on the mat", "The cats sleep on the mat"),
    ("The dog barks loudly", "The dogs bark loudly"),
    ("A bird flies south", "Birds fly south"),
    ("The child plays outside", "The children play outside"),
    ("A fish swims in the pond", "Fish swim in the pond"),
    ("The student reads the book", "The students read the book"),
    ("A tree grows in the garden", "Trees grow in the garden"),
    ("The flower blooms in spring", "The flowers bloom in spring"),
    ("A car drives down the street", "Cars drive down the street"),
    ("The star shines bright tonight", "The stars shine bright tonight"),
    ("A house stands on the hill", "Houses stand on the hill"),
    ("The book lies on the table", "The books lie on the table"),
    ("A cloud floats in the sky", "Clouds float in the sky"),
    ("The river flows through town", "Rivers flow through town"),
    ("A mountain rises in the east", "Mountains rise in the east"),
    ("The leaf falls from the tree", "The leaves fall from the tree"),
    ("A sheep grazes in the field", "Sheep graze in the field"),
    ("The bus stops at the corner", "The buses stop at the corner"),
    ("A mouse runs across the floor", "Mice run across the floor"),
    ("The leaf turns red in autumn", "The leaves turn red in autumn"),
]

VOICE_PAIRS = [
    ("The cat chased the mouse", "The mouse was chased by the cat"),
    ("She wrote the letter", "The letter was written by her"),
    ("He fixed the car", "The car was fixed by him"),
    ("They built the house", "The house was built by them"),
    ("The dog bit the man", "The man was bitten by the dog"),
    ("She cooked the dinner", "The dinner was cooked by her"),
    ("He painted the wall", "The wall was painted by him"),
    ("They found the treasure", "The treasure was found by them"),
    ("The wind broke the window", "The window was broken by the wind"),
    ("She opened the door", "The door was opened by her"),
    ("He read the report", "The report was read by him"),
    ("They won the game", "The game was won by them"),
    ("The teacher praised the student", "The student was praised by the teacher"),
    ("She cleaned the room", "The room was cleaned by her"),
    ("He drove the bus", "The bus was driven by him"),
    ("They caught the fish", "The fish was caught by them"),
    ("The rain ruined the crops", "The crops were ruined by the rain"),
    ("She sang the song", "The song was sung by her"),
    ("He wrote the code", "The code was written by him"),
    ("They sold the house", "The house was sold by them"),
]

FEATURE_TYPES = {
    'tense': {'pairs': TENSE_PAIRS, 'type': 'lexical'},
    'question': {'pairs': QUESTION_PAIRS, 'type': 'syntactic'},
    'negation': {'pairs': NEGATION_PAIRS, 'type': 'syntactic'},
    'number': {'pairs': NUMBER_PAIRS, 'type': 'morphological'},
    'voice': {'pairs': VOICE_PAIRS, 'type': 'syntactic'},
}


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


def get_hidden_state(model, tokenizer, target_layer, sentence):
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
        model(input_ids=input_ids, attention_mask=attention_mask)
    h.remove()

    return hidden_state[0].numpy()


def get_hidden_and_logits(model, tokenizer, target_layer, sentence):
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


def intervene_and_get_logits(model, tokenizer, target_layer, sentence, perturbation_vec):
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


def compute_feature_metrics(diffs):
    """Compute CR, PR, mean alignment for a set of difference vectors."""
    n, d = diffs.shape
    mean_dir = diffs.mean(axis=0)
    mean_norm = np.linalg.norm(mean_dir)
    diff_norms = np.linalg.norm(diffs, axis=1)

    # CR
    cr = mean_norm**2 / (np.mean(diff_norms**2) + 1e-10)

    # Mean alignment
    alignments = []
    for i in range(n):
        if diff_norms[i] > 1e-10 and mean_norm > 1e-10:
            cos = np.dot(diffs[i], mean_dir) / (diff_norms[i] * mean_norm)
            alignments.append(cos)
        else:
            alignments.append(0.0)
    mean_alignment = np.mean(alignments)

    # PR of differences
    centered = diffs - diffs.mean(axis=0, keepdims=True)
    if n > 1 and n < d:
        gram = centered @ centered.T
        eigvals, _ = np.linalg.eigh(gram)
        eigvals = np.sort(np.maximum(eigvals, 0))[::-1]
        svs = np.sqrt(eigvals)
        pr = np.sum(svs**2)**2 / (np.sum(svs**4) + 1e-10)
        total = np.sum(svs**2)
        cumvar = np.cumsum(svs**2) / (total + 1e-10)
        dim90 = np.searchsorted(cumvar, 0.9) + 1 if len(cumvar) > 0 else 1
    else:
        pr = 1.0
        dim90 = 1

    return {
        'cr': float(cr),
        'mean_alignment': float(mean_alignment),
        'pr': float(pr),
        'dim90': int(dim90),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=20)
    parser.add_argument('--n_test', type=int, default=10)
    parser.add_argument('--eps_rel', type=float, default=0.1)
    args = parser.parse_args()

    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]

    out_dir = f"results/causal_fiber/{model_key}_ccliv"
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, 'run.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f"=== Phase CCLIV: Multi-Feature Causal Geometry ===")
    log(f"Model: {cfg['name']}")
    log(f"Features: tense(lexical), question(syntactic), negation(syntactic), number(morphological), voice(syntactic)")
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

    results = {}

    for layer_idx in test_layers:
        target_layer = get_target_layer(model, layer_idx)
        log(f"\n=== Layer {layer_idx} ===")

        layer_results = {}
        feature_dirs = {}

        # ============================================================
        # Part 1: Collect all feature directions
        # ============================================================
        for feat_name, feat_info in FEATURE_TYPES.items():
            pairs = feat_info['pairs'][:args.n_pairs]
            diffs = []
            for sent_a, sent_b in pairs:
                h_a = get_hidden_state(model, tokenizer, target_layer, sent_a)
                h_b = get_hidden_state(model, tokenizer, target_layer, sent_b)
                diffs.append(h_b - h_a)
            diffs = np.array(diffs)

            metrics = compute_feature_metrics(diffs)
            metrics['type'] = feat_info['type']

            # Store mean direction (normalized)
            mean_dir = diffs.mean(axis=0)
            mean_norm = np.linalg.norm(mean_dir)
            if mean_norm > 1e-10:
                feature_dirs[feat_name] = mean_dir / mean_norm
            else:
                feature_dirs[feat_name] = np.zeros(cfg['d_model'])

            layer_results[feat_name] = metrics
            log(f"  {feat_name:>10}: CR={metrics['cr']:.3f}, PR={metrics['pr']:.1f}, "
                f"align={metrics['mean_alignment']:.3f}, dim90={metrics['dim90']}")

        # ============================================================
        # Part 2: Cross-feature orthogonality matrix
        # ============================================================
        feat_names = list(FEATURE_TYPES.keys())
        cos_matrix = np.zeros((len(feat_names), len(feat_names)))
        for i, fn_i in enumerate(feat_names):
            for j, fn_j in enumerate(feat_names):
                ni = np.linalg.norm(feature_dirs[fn_i])
                nj = np.linalg.norm(feature_dirs[fn_j])
                if ni > 1e-10 and nj > 1e-10:
                    cos_matrix[i, j] = np.dot(feature_dirs[fn_i], feature_dirs[fn_j]) / (ni * nj)

        log(f"  Cross-feature cos matrix:")
        header = "         " + "".join(f"{fn:>10}" for fn in feat_names)
        log(f"  {header}")
        for i, fn_i in enumerate(feat_names):
            row = f"  {fn_i:>10}" + "".join(f"{cos_matrix[i,j]:>10.3f}" for j in range(len(feat_names)))
            log(row)

        layer_results['cos_matrix'] = cos_matrix.tolist()
        layer_results['feat_names'] = feat_names

        # ============================================================
        # Part 3: Causal efficacy for each feature
        # ============================================================
        # Use a common set of test sentences and measure perturbation effect
        # For each feature, use first n_test pairs
        test_present = [
            "She walks to school every day",
            "He runs in the park each morning",
            "The cat sleeps on the mat quietly",
            "They play football on Sundays",
            "The bird flies above the trees",
            "She sings in the choir weekly",
            "He writes reports for work",
            "The dog barks at the mailman",
            "We drink coffee every morning",
            "The wind blows through the window",
        ][:args.n_test]

        for feat_name, feat_info in FEATURE_TYPES.items():
            pairs = feat_info['pairs'][:args.n_test]
            feat_dir = feature_dirs[feat_name]

            if np.linalg.norm(feat_dir) < 1e-10:
                layer_results[feat_name]['efficacy'] = 0.0
                layer_results[feat_name]['efficacy_ratio'] = 0.0
                continue

            # Measure feature-direction efficacy
            feat_effs = []
            rand_effs = []

            for i, (sent_a, sent_b) in enumerate(pairs):
                # Get logit shift direction
                _, logits_a = get_hidden_and_logits(model, tokenizer, target_layer, sent_a)
                _, logits_b = get_hidden_and_logits(model, tokenizer, target_layer, sent_b)
                logit_diff = logits_b - logits_a

                h_a = get_hidden_state(model, tokenizer, target_layer, sent_a)
                h_norm = np.linalg.norm(h_a)
                eps = args.eps_rel * h_norm

                # Feature perturbation
                logits_plus = intervene_and_get_logits(
                    model, tokenizer, target_layer, sent_a, eps * feat_dir)
                diff = logits_plus - logits_a
                if np.linalg.norm(diff) > 1e-10 and np.linalg.norm(logit_diff) > 1e-10:
                    feat_eff = np.dot(diff, logit_diff) / (np.linalg.norm(diff) * np.linalg.norm(logit_diff))
                else:
                    feat_eff = 0.0
                feat_effs.append(abs(feat_eff))

                # Random baseline (3 random directions)
                for _ in range(3):
                    rand_dir = np.random.randn(cfg['d_model'])
                    rand_dir = rand_dir / np.linalg.norm(rand_dir)
                    logits_rand = intervene_and_get_logits(
                        model, tokenizer, target_layer, sent_a, eps * rand_dir)
                    diff_r = logits_rand - logits_a
                    if np.linalg.norm(diff_r) > 1e-10 and np.linalg.norm(logit_diff) > 1e-10:
                        rand_eff = np.dot(diff_r, logit_diff) / (np.linalg.norm(diff_r) * np.linalg.norm(logit_diff))
                    else:
                        rand_eff = 0.0
                    rand_effs.append(abs(rand_eff))

            mean_feat_eff = np.mean(feat_effs)
            mean_rand_eff = np.mean(rand_effs)
            eff_ratio = mean_feat_eff / (mean_rand_eff + 1e-10)

            layer_results[feat_name]['efficacy'] = float(mean_feat_eff)
            layer_results[feat_name]['efficacy_ratio'] = float(eff_ratio)

        log(f"  Efficacy summary:")
        for fn in feat_names:
            r = layer_results[fn]
            log(f"    {fn:>10}: eff={r['efficacy']:.3f}, E/R={r['efficacy_ratio']:.2f}")

        results[layer_idx] = layer_results

    # ============================================================
    # Analysis
    # ============================================================
    log(f"\n{'='*70}")
    log(f"ANALYSIS")
    log(f"{'='*70}")

    layers = sorted(results.keys())
    layer_indices = list(range(len(layers)))

    feat_names = list(FEATURE_TYPES.keys())
    feat_types = {fn: FEATURE_TYPES[fn]['type'] for fn in feat_names}

    log(f"\n  CR across depth:")
    header = f"  {'L':>4}" + "".join(f"  {fn:>10}" for fn in feat_names)
    log(header)

    cr_by_feature = {}
    pr_by_feature = {}
    eff_ratio_by_feature = {}
    alignment_by_feature = {}

    for fn in feat_names:
        cr_by_feature[fn] = [results[l][fn]['cr'] for l in layers]
        pr_by_feature[fn] = [results[l][fn]['pr'] for l in layers]
        eff_ratio_by_feature[fn] = [results[l][fn]['efficacy_ratio'] for l in layers]
        alignment_by_feature[fn] = [results[l][fn]['mean_alignment'] for l in layers]

    for i, l in enumerate(layers):
        row = f"  {l:>4}" + "".join(f"  {cr_by_feature[fn][i]:>10.3f}" for fn in feat_names)
        log(row)

    # Correlations
    log(f"\n  Depth correlations:")
    log(f"  {'Feature':>12} {'Type':>14} {'Corr_CR':>8} {'Corr_PR':>8} {'Corr_ER':>8} {'Corr_Al':>8}")

    corr_results = {}
    for fn in feat_names:
        corr_cr, _ = spearmanr(layer_indices, cr_by_feature[fn]) if len(layers) > 2 else (0, 1)
        corr_pr, _ = spearmanr(layer_indices, pr_by_feature[fn]) if len(layers) > 2 else (0, 1)
        corr_er, _ = spearmanr(layer_indices, eff_ratio_by_feature[fn]) if len(layers) > 2 else (0, 1)
        corr_al, _ = spearmanr(layer_indices, alignment_by_feature[fn]) if len(layers) > 2 else (0, 1)

        corr_results[fn] = {
            'cr': float(corr_cr),
            'pr': float(corr_pr),
            'efficacy_ratio': float(corr_er),
            'alignment': float(corr_al),
        }

        log(f"  {fn:>12} {feat_types[fn]:>14} {corr_cr:>+8.3f} {corr_pr:>+8.3f} {corr_er:>+8.3f} {corr_al:>+8.3f}")

    # Group by type
    log(f"\n  Group analysis (syntactic vs lexical/morphological):")
    
    syntactic_features = [fn for fn in feat_names if feat_types[fn] == 'syntactic']
    lexical_features = [fn for fn in feat_names if feat_types[fn] in ('lexical', 'morphological')]

    syn_cr_corrs = [corr_results[fn]['cr'] for fn in syntactic_features]
    lex_cr_corrs = [corr_results[fn]['cr'] for fn in lexical_features]
    syn_er_corrs = [corr_results[fn]['efficacy_ratio'] for fn in syntactic_features]
    lex_er_corrs = [corr_results[fn]['efficacy_ratio'] for fn in lexical_features]
    syn_pr_corrs = [corr_results[fn]['pr'] for fn in syntactic_features]
    lex_pr_corrs = [corr_results[fn]['pr'] for fn in lexical_features]

    log(f"  Syntactic features: {syntactic_features}")
    log(f"    Mean Corr(CR): {np.mean(syn_cr_corrs):+.3f}")
    log(f"    Mean Corr(PR): {np.mean(syn_pr_corrs):+.3f}")
    log(f"    Mean Corr(E/R): {np.mean(syn_er_corrs):+.3f}")
    log(f"  Lexical/Morphological features: {lexical_features}")
    log(f"    Mean Corr(CR): {np.mean(lex_cr_corrs):+.3f}")
    log(f"    Mean Corr(PR): {np.mean(lex_pr_corrs):+.3f}")
    log(f"    Mean Corr(E/R): {np.mean(lex_er_corrs):+.3f}")

    # Key predictions
    log(f"\n  PREDICTIONS:")
    
    # 1. Syntactic CR increases
    syn_cr_supported = np.mean(syn_cr_corrs) > 0.3
    log(f"    {'SUPPORTED' if syn_cr_supported else 'NOT SUPPORTED'}: "
        f"Syntactic CR increases with depth (mean Corr={np.mean(syn_cr_corrs):+.3f})")
    
    # 2. Lexical CR does NOT increase
    lex_cr_not_supported = np.mean(lex_cr_corrs) < 0.3
    log(f"    {'SUPPORTED' if lex_cr_not_supported else 'NOT SUPPORTED'}: "
        f"Lexical/Morph CR does NOT increase with depth (mean Corr={np.mean(lex_cr_corrs):+.3f})")
    
    # 3. All E/R increases
    all_er_supported = np.mean(syn_er_corrs + lex_er_corrs) > 0.3
    log(f"    {'SUPPORTED' if all_er_supported else 'NOT SUPPORTED'}: "
        f"All features E/R increases with depth (mean Corr={np.mean(syn_er_corrs + lex_er_corrs):+.3f})")
    
    # 4. Syntactic PR decreases
    syn_pr_supported = np.mean(syn_pr_corrs) < -0.3
    log(f"    {'SUPPORTED' if syn_pr_supported else 'NOT SUPPORTED'}: "
        f"Syntactic PR decreases with depth (mean Corr={np.mean(syn_pr_corrs):+.3f})")

    n_supported = sum([syn_cr_supported, lex_cr_not_supported, all_er_supported, syn_pr_supported])

    log(f"\n  INTERPRETATION:")
    if n_supported >= 3:
        log(f"  => Syntax-Lexical Dual-Track model STRONGLY SUPPORTED!")
    elif n_supported >= 2:
        log(f"  => Syntax-Lexical Dual-Track model PARTIALLY supported")
    elif n_supported >= 1:
        log(f"  => Weak evidence for dual-track model")
    else:
        log(f"  => Dual-Track model NOT supported")
        log(f"  => Feature type may not be the right classification axis")

    # E/R table across depth
    log(f"\n  Efficacy Ratio across depth:")
    header = f"  {'L':>4}" + "".join(f"  {fn:>10}" for fn in feat_names)
    log(header)
    for i, l in enumerate(layers):
        row = f"  {l:>4}" + "".join(f"  {eff_ratio_by_feature[fn][i]:>10.2f}" for fn in feat_names)
        log(row)

    # Save results
    save_data = {
        'model': model_key,
        'layers': layers,
        'feature_types': feat_types,
        'correlations': corr_results,
        'group_analysis': {
            'syntactic_features': syntactic_features,
            'lexical_features': lexical_features,
            'syn_mean_cr_corr': float(np.mean(syn_cr_corrs)),
            'lex_mean_cr_corr': float(np.mean(lex_cr_corrs)),
            'syn_mean_er_corr': float(np.mean(syn_er_corrs)),
            'lex_mean_er_corr': float(np.mean(lex_er_corrs)),
            'syn_mean_pr_corr': float(np.mean(syn_pr_corrs)),
            'lex_mean_pr_corr': float(np.mean(lex_pr_corrs)),
        },
        'layer_results': {},
    }

    for l in layers:
        lr = {}
        for fn in feat_names:
            lr[fn] = {
                'cr': results[l][fn]['cr'],
                'pr': results[l][fn]['pr'],
                'alignment': results[l][fn]['mean_alignment'],
                'dim90': results[l][fn]['dim90'],
                'efficacy': results[l][fn]['efficacy'],
                'efficacy_ratio': results[l][fn]['efficacy_ratio'],
                'type': results[l][fn]['type'],
            }
        save_data['layer_results'][str(l)] = lr

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    del model
    torch.cuda.empty_cache()

    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Results saved to {out_dir}/results.json")


if __name__ == '__main__':
    main()
