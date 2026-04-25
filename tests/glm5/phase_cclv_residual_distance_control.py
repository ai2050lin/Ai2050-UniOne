"""
Phase CCLV: Residual Distance Control Analysis
================================================
CCLIV found: E/R increases with depth for ALL 5 features.
But is this just because deeper layers are closer to the output?

If E/R increase is trivial (just distance to output):
  - E/R should be proportional to 1/remaining_layers
  - Or E/R should scale linearly with remaining depth

If E/R increase reflects genuine feature encoding:
  - E/R should increase FASTER than distance predicts
  - There should be a "super-linear" component

This test:
1. Measures E/R at each layer (same as CCLIV)
2. Computes the "distance model" prediction: E/R_pred = a / (N_remaining + b)
3. Compares actual E/R with distance prediction
4. If actual >> distance prediction in deep layers → genuine encoding
5. Also tests: feature independence (tense + question combined perturbation)

Key predictions:
- If genuine: E/R grows super-linearly, not just linearly with 1/distance
- If trivial: E/R is well-explained by distance to output
"""

import argparse
import json
import os
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr
from scipy.optimize import curve_fit

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
    ("I think about it", "I thought about it"),
    ("We drink coffee", "We drank coffee"),
    ("He drives fast", "He drove fast"),
    ("She knows the answer", "She knew the answer"),
    ("The dog barks loudly", "The dog barked loudly"),
]

QUESTION_PAIRS = [
    ("The sky is blue", "Is the sky blue?"),
    ("Cats like fish", "Do cats like fish?"),
    ("She is happy", "Is she happy?"),
    ("The door is open", "Is the door open?"),
    ("He works hard", "Does he work hard?"),
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
]

# Test sentences for independence analysis
INDEPENDENCE_TEST = [
    ("She walks to school every day", "She walked to school every day", "Does she walk to school every day?"),
    ("He runs in the park each morning", "He ran in the park each morning", "Does he run in the park each morning?"),
    ("The cat sleeps on the mat quietly", "The cat slept on the mat quietly", "Does the cat sleep on the mat quietly?"),
    ("They play football on Sundays", "They played football on Sundays", "Do they play football on Sundays?"),
    ("The bird flies above the trees", "The bird flew above the trees", "Does the bird fly above the trees?"),
    ("She sings in the choir weekly", "She sang in the choir weekly", "Does she sing in the choir weekly?"),
    ("He writes reports for work", "He wrote reports for work", "Does he write reports for work?"),
    ("The dog barks at the mailman", "The dog barked at the mailman", "Does the dog bark at the mailman?"),
    ("We drink coffee every morning", "We drank coffee every morning", "Do we drink coffee every morning?"),
    ("The wind blows through the window", "The wind blew through the window", "Does the wind blow through the window?"),
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
            trust_remote_code=True, local_files_only=True)
    elif dtype == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto", trust_remote_code=True, local_files_only=True)
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
        raise ValueError(f"Cannot find layers for model type {model_type}")


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


def compute_feature_direction(model, tokenizer, target_layer, pairs):
    """Compute mean feature direction from pairs."""
    diffs = []
    for sent_a, sent_b in pairs:
        h_a = get_hidden_state(model, tokenizer, target_layer, sent_a)
        h_b = get_hidden_state(model, tokenizer, target_layer, sent_b)
        diffs.append(h_b - h_a)
    diffs = np.array(diffs)
    mean_dir = diffs.mean(axis=0)
    norm = np.linalg.norm(mean_dir)
    if norm > 1e-10:
        return mean_dir / norm
    return np.zeros(diffs.shape[1])


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--eps_rel', type=float, default=0.1)
    args = parser.parse_args()

    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]
    out_dir = f"results/causal_fiber/{model_key}_cclv"
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, 'run.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f"=== Phase CCLV: Residual Distance Control Analysis ===")
    log(f"Model: {cfg['name']}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")

    n_layers = cfg['n_layers']
    d_model = cfg['d_model']

    # Test ALL layers (not just 5 sample layers) for better curve fitting
    # But to save time, test every 3rd layer
    test_layers = list(range(1, n_layers - 1, max(1, n_layers // 12)))
    # Ensure last layer is included
    if (n_layers - 2) not in test_layers:
        test_layers.append(n_layers - 2)
    test_layers = sorted(set(test_layers))

    log(f"Testing layers: {test_layers}")

    results = {}

    for layer_idx in test_layers:
        target_layer = get_target_layer(model, layer_idx)
        remaining = n_layers - layer_idx
        log(f"\n=== Layer {layer_idx} (remaining: {remaining}) ===")

        # Compute feature directions
        tense_dir = compute_feature_direction(model, tokenizer, target_layer, TENSE_PAIRS)
        quest_dir = compute_feature_direction(model, tokenizer, target_layer, QUESTION_PAIRS)

        layer_data = {
            'remaining': remaining,
            'tense': {},
            'question': {},
            'independence': {},
        }

        # ============================================================
        # Part 1: E/R for tense and question (using independence test sentences)
        # ============================================================
        for feat_name, feat_dir, pairs_for_test in [
            ('tense', tense_dir, [(s, p, q) for s, p, q in INDEPENDENCE_TEST]),
            ('question', quest_dir, [(s, p, q) for s, p, q in INDEPENDENCE_TEST]),
        ]:
            feat_effs = []
            rand_effs = []

            for sent_pres, sent_past, sent_quest in pairs_for_test:
                # Target logit direction depends on feature
                if feat_name == 'tense':
                    _, logits_a = get_hidden_and_logits(model, tokenizer, target_layer, sent_pres)
                    _, logits_b = get_hidden_and_logits(model, tokenizer, target_layer, sent_past)
                else:
                    _, logits_a = get_hidden_and_logits(model, tokenizer, target_layer, sent_pres)
                    _, logits_b = get_hidden_and_logits(model, tokenizer, target_layer, sent_quest)

                logit_diff = logits_b - logits_a

                h_a = get_hidden_state(model, tokenizer, target_layer, sent_pres)
                h_norm = np.linalg.norm(h_a)
                eps = args.eps_rel * h_norm

                # Feature perturbation
                logits_plus = intervene_and_get_logits(
                    model, tokenizer, target_layer, sent_pres, eps * feat_dir)
                diff = logits_plus - logits_a
                if np.linalg.norm(diff) > 1e-10 and np.linalg.norm(logit_diff) > 1e-10:
                    feat_eff = abs(np.dot(diff, logit_diff) / (np.linalg.norm(diff) * np.linalg.norm(logit_diff)))
                else:
                    feat_eff = 0.0
                feat_effs.append(feat_eff)

                # Random baseline (3 directions)
                for _ in range(3):
                    rand_dir = np.random.randn(d_model)
                    rand_dir = rand_dir / np.linalg.norm(rand_dir)
                    logits_rand = intervene_and_get_logits(
                        model, tokenizer, target_layer, sent_pres, eps * rand_dir)
                    diff_r = logits_rand - logits_a
                    if np.linalg.norm(diff_r) > 1e-10 and np.linalg.norm(logit_diff) > 1e-10:
                        rand_eff = abs(np.dot(diff_r, logit_diff) / (np.linalg.norm(diff_r) * np.linalg.norm(logit_diff)))
                    else:
                        rand_eff = 0.0
                    rand_effs.append(rand_eff)

            mean_feat = np.mean(feat_effs)
            mean_rand = np.mean(rand_effs)
            er = mean_feat / (mean_rand + 1e-10)

            layer_data[feat_name] = {
                'mean_efficacy': float(mean_feat),
                'mean_random': float(mean_rand),
                'efficacy_ratio': float(er),
            }

            log(f"  {feat_name}: eff={mean_feat:.3f}, rand={mean_rand:.3f}, E/R={er:.2f}")

        # ============================================================
        # Part 2: Feature independence test
        # ============================================================
        # For each test sentence, perturb along:
        # (a) tense only, (b) question only, (c) both combined
        # If independent: combined = tense + question (in logit space)

        tense_only_shifts = []
        quest_only_shifts = []
        combined_shifts = []
        predicted_combined = []

        for sent_pres, sent_past, sent_quest in INDEPENDENCE_TEST:
            _, logits_clean = get_hidden_and_logits(model, tokenizer, target_layer, sent_pres)
            _, logits_past = get_hidden_and_logits(model, tokenizer, target_layer, sent_past)
            _, logits_quest = get_hidden_and_logits(model, tokenizer, target_layer, sent_quest)

            h_clean = get_hidden_state(model, tokenizer, target_layer, sent_pres)
            h_norm = np.linalg.norm(h_clean)
            eps = args.eps_rel * h_norm

            # Tense perturbation
            logits_t = intervene_and_get_logits(
                model, tokenizer, target_layer, sent_pres, eps * tense_dir)
            tense_shift = logits_t - logits_clean

            # Question perturbation
            logits_q = intervene_and_get_logits(
                model, tokenizer, target_layer, sent_pres, eps * quest_dir)
            quest_shift = logits_q - logits_clean

            # Combined perturbation (both directions)
            logits_combined = intervene_and_get_logits(
                model, tokenizer, target_layer, sent_pres, eps * tense_dir + eps * quest_dir)
            combined_shift = logits_combined - logits_clean

            # Predicted combined (if independent = additive)
            predicted = tense_shift + quest_shift

            # Measure how well predicted matches actual
            if np.linalg.norm(combined_shift) > 1e-10:
                cos_pred_actual = np.dot(predicted, combined_shift) / (
                    np.linalg.norm(predicted) * np.linalg.norm(combined_shift) + 1e-10)
            else:
                cos_pred_actual = 0.0

            # Ratio of norms
            norm_ratio = np.linalg.norm(combined_shift) / (np.linalg.norm(predicted) + 1e-10)

            tense_only_shifts.append(tense_shift)
            quest_only_shifts.append(quest_shift)
            combined_shifts.append(combined_shift)
            predicted_combined.append(predicted)

        # Compute independence metrics
        cos_independence = []
        norm_ratios = []
        for i in range(len(combined_shifts)):
            cs = combined_shifts[i]
            ps = predicted_combined[i]
            if np.linalg.norm(cs) > 1e-10:
                cos_ip = np.dot(ps, cs) / (np.linalg.norm(ps) * np.linalg.norm(cs) + 1e-10)
                cos_independence.append(cos_ip)
                norm_ratios.append(np.linalg.norm(cs) / (np.linalg.norm(ps) + 1e-10))

        mean_cos_indep = np.mean(cos_independence) if cos_independence else 0.0
        mean_norm_ratio = np.mean(norm_ratios) if norm_ratios else 0.0

        layer_data['independence'] = {
            'cos_predicted_actual': float(mean_cos_indep),
            'norm_ratio_actual_predicted': float(mean_norm_ratio),
        }

        log(f"  Independence: cos(pred,actual)={mean_cos_indep:.3f}, norm_ratio={mean_norm_ratio:.3f}")

        results[layer_idx] = layer_data

    # ============================================================
    # Analysis: Distance model vs actual E/R
    # ============================================================
    log(f"\n{'='*70}")
    log(f"ANALYSIS")
    log(f"{'='*70}")

    layers = sorted(results.keys())
    remaining = [results[l]['remaining'] for l in layers]
    tense_er = [results[l]['tense']['efficacy_ratio'] for l in layers]
    quest_er = [results[l]['question']['efficacy_ratio'] for l in layers]
    cos_indep = [results[l]['independence']['cos_predicted_actual'] for l in layers]
    norm_rat = [results[l]['independence']['norm_ratio_actual_predicted'] for l in layers]

    log(f"\n  Data table:")
    log(f"  {'L':>4} {'rem':>4} {'ER_T':>6} {'ER_Q':>6} {'cos_ind':>8} {'nr':>6}")
    for l in layers:
        r = results[l]
        log(f"  {l:>4} {r['remaining']:>4} {r['tense']['efficacy_ratio']:>6.2f} "
            f"{r['question']['efficacy_ratio']:>6.2f} {r['independence']['cos_predicted_actual']:>8.3f} "
            f"{r['independence']['norm_ratio_actual_predicted']:>6.3f}")

    # Distance model: E/R = a / (remaining + b) + c
    # Simplified: E/R = a * (1/remaining)
    # Also try: E/R = a * exp(b * layer_idx)

    layer_indices = list(range(len(layers)))

    # Model 1: Linear with 1/remaining
    inv_remaining = [1.0 / (r + 0.5) for r in remaining]  # +0.5 to avoid div by 0

    # Fit linear model: E/R = alpha * (1/remaining) + beta
    from numpy.polynomial import polynomial as P
    try:
        # Tense
        coeffs_t = np.polyfit(inv_remaining, tense_er, 1)
        pred_t = np.polyval(coeffs_t, inv_remaining)
        ss_res_t = np.sum((np.array(tense_er) - pred_t)**2)
        ss_tot_t = np.sum((np.array(tense_er) - np.mean(tense_er))**2)
        r2_distance_tense = 1 - ss_res_t / (ss_tot_t + 1e-10)

        # Question
        coeffs_q = np.polyfit(inv_remaining, quest_er, 1)
        pred_q = np.polyval(coeffs_q, inv_remaining)
        ss_res_q = np.sum((np.array(quest_er) - pred_q)**2)
        ss_tot_q = np.sum((np.array(quest_er) - np.mean(quest_er))**2)
        r2_distance_quest = 1 - ss_res_q / (ss_tot_q + 1e-10)
    except:
        r2_distance_tense = 0.0
        r2_distance_quest = 0.0

    # Model 2: Linear with layer index
    try:
        coeffs_t2 = np.polyfit(layer_indices, tense_er, 1)
        pred_t2 = np.polyval(coeffs_t2, layer_indices)
        ss_res_t2 = np.sum((np.array(tense_er) - pred_t2)**2)
        r2_linear_tense = 1 - ss_res_t2 / (ss_tot_t + 1e-10)

        coeffs_q2 = np.polyfit(layer_indices, quest_er, 1)
        pred_q2 = np.polyval(coeffs_q2, layer_indices)
        ss_res_q2 = np.sum((np.array(quest_er) - pred_q2)**2)
        r2_linear_quest = 1 - ss_res_q2 / (ss_tot_q + 1e-10)
    except:
        r2_linear_tense = 0.0
        r2_linear_quest = 0.0

    # Model 3: Exponential with layer index
    try:
        log_t = np.log(np.array(tense_er) + 0.01)
        log_q = np.log(np.array(quest_er) + 0.01)
        coeffs_t3 = np.polyfit(layer_indices, log_t, 1)
        pred_t3 = np.exp(np.polyval(coeffs_t3, layer_indices))
        ss_res_t3 = np.sum((np.array(tense_er) - pred_t3)**2)
        r2_exp_tense = 1 - ss_res_t3 / (ss_tot_t + 1e-10)

        coeffs_q3 = np.polyfit(layer_indices, log_q, 1)
        pred_q3 = np.exp(np.polyval(coeffs_q3, layer_indices))
        ss_res_q3 = np.sum((np.array(quest_er) - pred_q3)**2)
        r2_exp_quest = 1 - ss_res_q3 / (ss_tot_q + 1e-10)
    except:
        r2_exp_tense = 0.0
        r2_exp_quest = 0.0

    log(f"\n  Model comparison (R-squared):")
    log(f"  {'Model':>30} {'Tense':>8} {'Question':>8}")
    log(f"  {'1/remaining (distance)':>30} {r2_distance_tense:>8.3f} {r2_distance_quest:>8.3f}")
    log(f"  {'Linear with depth':>30} {r2_linear_tense:>8.3f} {r2_linear_quest:>8.3f}")
    log(f"  {'Exponential with depth':>30} {r2_exp_tense:>8.3f} {r2_exp_quest:>8.3f}")

    # Super-linear test: does E/R grow faster than 1/remaining?
    # Compare residuals of distance model vs linear model
    # If distance model underestimates deep layers → super-linear (genuine encoding)
    # If distance model fits well → trivial (just distance)

    log(f"\n  Distance model residuals (actual - predicted):")
    log(f"  {'L':>4} {'rem':>4} {'resid_T':>8} {'resid_Q':>8}")
    for i, l in enumerate(layers):
        resid_t = tense_er[i] - pred_t[i] if 'pred_t' in dir() else 0
        resid_q = quest_er[i] - pred_q[i] if 'pred_q' in dir() else 0
        log(f"  {l:>4} {remaining[i]:>4} {resid_t:>+8.3f} {resid_q:>+8.3f}")

    # Check if residuals increase with depth (super-linear)
    if 'pred_t' in dir() and len(layers) > 3:
        resid_t_arr = np.array(tense_er) - pred_t
        resid_q_arr = np.array(quest_er) - pred_q
        corr_resid_t, _ = spearmanr(layer_indices, resid_t_arr)
        corr_resid_q, _ = spearmanr(layer_indices, resid_q_arr)
    else:
        corr_resid_t = 0
        corr_resid_q = 0

    log(f"\n  Residual-depth correlation:")
    log(f"    Corr(L, residual_tense)    = {corr_resid_t:+.3f}")
    log(f"    Corr(L, residual_question) = {corr_resid_q:+.3f}")

    # Key predictions
    log(f"\n  KEY PREDICTIONS:")

    distance_fits = max(r2_distance_tense, r2_distance_quest) > 0.8
    super_linear = corr_resid_t > 0.3 or corr_resid_q > 0.3
    exp_better = r2_exp_tense > r2_distance_tense + 0.1 or r2_exp_quest > r2_distance_quest + 0.1

    if distance_fits and not super_linear:
        log(f"  => E/R increase is TRIVIAL (well-explained by distance to output)")
        log(f"  => No evidence for genuine feature encoding advantage in deep layers")
    elif super_linear:
        log(f"  => E/R increase is SUPER-LINEAR (faster than distance predicts)")
        log(f"  => Deep layers have GENUINE feature encoding advantage!")
    elif exp_better:
        log(f"  => E/R increase is EXPONENTIAL (much faster than distance)")
        log(f"  => Strong evidence for genuine encoding advantage!")
    else:
        log(f"  => E/R increase is PARTIALLY explained by distance")
        log(f"  => Some genuine encoding advantage may exist")

    # Independence analysis
    log(f"\n  FEATURE INDEPENDENCE:")
    corr_indep, _ = spearmanr(layer_indices, cos_indep) if len(layers) > 2 else (0, 1)
    corr_norm, _ = spearmanr(layer_indices, norm_rat) if len(layers) > 2 else (0, 1)

    log(f"    Mean cos(predicted, actual): {np.mean(cos_indep):.3f}")
    log(f"    Mean norm_ratio: {np.mean(norm_rat):.3f}")
    log(f"    Corr(L, cos_independence) = {corr_indep:+.3f}")
    log(f"    Corr(L, norm_ratio) = {corr_norm:+.3f}")

    if np.mean(cos_indep) > 0.9:
        log(f"  => Features are INDEPENDENT (combined ≈ sum of individuals)")
        log(f"  => Tense and question directions do not interfere")
    elif np.mean(cos_indep) > 0.7:
        log(f"  => Features are PARTIALLY independent")
        log(f"  => Some interference between tense and question directions")
    else:
        log(f"  => Features are NOT independent")
        log(f"  => Significant interference between features")

    # Save results
    save_data = {
        'model': model_key,
        'n_layers': n_layers,
        'layers': layers,
        'model_comparison': {
            'r2_distance_tense': float(r2_distance_tense),
            'r2_distance_quest': float(r2_distance_quest),
            'r2_linear_tense': float(r2_linear_tense),
            'r2_linear_quest': float(r2_linear_quest),
            'r2_exp_tense': float(r2_exp_tense),
            'r2_exp_quest': float(r2_exp_quest),
        },
        'residual_correlations': {
            'tense': float(corr_resid_t),
            'question': float(corr_resid_q),
        },
        'independence': {
            'mean_cos': float(np.mean(cos_indep)),
            'mean_norm_ratio': float(np.mean(norm_rat)),
            'corr_cos_depth': float(corr_indep),
            'corr_norm_depth': float(corr_norm),
        },
        'layer_results': {str(k): v for k, v in results.items()},
    }

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    del model
    torch.cuda.empty_cache()

    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Results saved to {out_dir}/results.json")


if __name__ == '__main__':
    main()
