"""
Phase CCXLIX: Feature-Aligned Curvature Analysis
=================================================
CCXLVIII showed: shallow curvature is more anisotropic (varies more across directions).
But WHAT are the high-curvature directions? Do they correspond to linguistic features?

Key hypothesis:
- If shallow anisotropy is "semantically structured", then:
  Directions that encode linguistic features (tense, question, number) should have
  HIGHER curvature than random directions, ESPECIALLY in shallow layers.

Test design:
1. Create contrastive sentence pairs that differ in ONE feature:
   - Tense: "She walks" vs "She walked"
   - Question: "She walks" vs "Does she walk?"
   - Number: "She walks" vs "They walk"

2. At each layer, compute the "feature direction":
   feature_dir = (h_pos - h_neg) / ||h_pos - h_neg||
   where h_pos/neg are hidden states for the two sentences.

3. Measure curvature along feature directions vs random directions.

4. Test: Is curvature along feature directions higher than random?
   And does this difference decrease with depth (as anisotropy decreases)?
"""

import argparse
import json
import os
import numpy as np
from datetime import datetime

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

# Contrastive pairs for each feature
TENSE_PAIRS = [
    ("She walks to school", "She walked to school"),
    ("He runs in the park", "He ran in the park"),
    ("They play football", "They played football"),
    ("The cat sleeps on the mat", "The cat slept on the mat"),
    ("She sings beautifully", "She sang beautifully"),
    ("He writes a letter", "He wrote a letter"),
    ("The bird flies south", "The bird flew south"),
    ("She reads the book", "She read the book"),
]

QUESTION_PAIRS = [
    ("She walks to school", "Does she walk to school"),
    ("He runs in the park", "Does he run in the park"),
    ("They play football", "Do they play football"),
    ("The cat sleeps on the mat", "Does the cat sleep on the mat"),
    ("She sings beautifully", "Does she sing beautifully"),
    ("He writes a letter", "Does he write a letter"),
]

NUMBER_PAIRS = [
    ("She walks to school", "They walk to school"),
    ("He runs in the park", "They run in the park"),
    ("The cat sleeps on the mat", "The cats sleep on the mat"),
    ("She sings beautifully", "They sing beautifully"),
    ("He writes a letter", "They write a letter"),
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
    """Get hidden state at a specific layer for a sentence."""
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


def measure_curvature_along_direction(model, tokenizer, target_layer, sentence,
                                       direction, eps_rel=0.05):
    """Measure relative curvature along a specific direction."""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    seq_len = attention_mask.sum().item()
    last_pos = seq_len - 1

    dir_tensor = torch.tensor(direction, dtype=torch.float32, device=model.device)

    # Get clean output
    clean_output = [None]
    h_norm_val = [None]

    def make_clean_hooks():
        def pre_hook(module, args):
            hidden = args[0]
            h_norm_val[0] = torch.norm(hidden[0, last_pos, :].float()).item()
            return args
        def post_hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            clean_output[0] = h[0, last_pos, :].detach().cpu().float().clone()
            return output
        return pre_hook, post_hook

    h1 = target_layer.register_forward_pre_hook(make_clean_hooks()[0])
    h2 = target_layer.register_forward_hook(make_clean_hooks()[1])
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    h1.remove()
    h2.remove()

    f_clean = clean_output[0].numpy()
    h_norm = h_norm_val[0]
    eps = eps_rel * h_norm

    # +perturbation
    output_plus = [None]
    output_minus = [None]

    def make_perturb_hooks(ve, ep, lp):
        def pre_plus(module, args):
            hidden = args[0]
            perturbed = hidden.clone()
            perturbed[0, lp, :] = perturbed[0, lp, :].float() + ep * ve.to(perturbed.dtype)
            return (perturbed,) + args[1:]
        def post_plus(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            output_plus[0] = h[0, lp, :].detach().cpu().float().clone()
            return output
        def pre_minus(module, args):
            hidden = args[0]
            perturbed = hidden.clone()
            perturbed[0, lp, :] = perturbed[0, lp, :].float() - ep * ve.to(perturbed.dtype)
            return (perturbed,) + args[1:]
        def post_minus(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            output_minus[0] = h[0, lp, :].detach().cpu().float().clone()
            return output
        return pre_plus, post_plus, pre_minus, post_minus

    ph_p, poh_p, ph_m, poh_m = make_perturb_hooks(dir_tensor, eps, last_pos)

    h3 = target_layer.register_forward_pre_hook(ph_p)
    h4 = target_layer.register_forward_hook(poh_p)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    h3.remove()
    h4.remove()

    h5 = target_layer.register_forward_pre_hook(ph_m)
    h6 = target_layer.register_forward_hook(poh_m)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    h5.remove()
    h6.remove()

    f_plus = output_plus[0].numpy()
    f_minus = output_minus[0].numpy()

    # Jacobian and Hessian along this direction
    jac_v = (f_plus - f_minus) / (2 * eps)
    hess_v = (f_plus - 2 * f_clean + f_minus) / (eps ** 2)

    jac_norm = np.linalg.norm(jac_v)
    hess_norm = np.linalg.norm(hess_v)

    rel_curv = hess_norm / (jac_norm + 1e-10)

    return rel_curv, jac_norm, hess_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_random', type=int, default=10)
    parser.add_argument('--eps_rel', type=float, default=0.05)
    args = parser.parse_args()

    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]

    out_dir = f"results/causal_fiber/{model_key}_ccxlix"
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, 'run.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f"=== Phase CCXLIX: Feature-Aligned Curvature ===")
    log(f"Model: {cfg['name']}, n_random: {args.n_random}, eps_rel: {args.eps_rel}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")

    n_layers = cfg['n_layers']
    d_model = cfg['d_model']

    # Test at key positions
    test_layers = [
        max(1, n_layers // 8),
        max(2, n_layers // 4),
        max(3, n_layers // 2),
        max(4, 3 * n_layers // 4),
        max(5, n_layers - 2),
    ]
    test_layers = sorted(set([l for l in test_layers if 1 <= l < n_layers]))

    feature_types = {
        'tense': TENSE_PAIRS,
        'question': QUESTION_PAIRS,
        'number': NUMBER_PAIRS,
    }

    results = {}

    for layer_idx in test_layers:
        target_layer = get_target_layer(model, layer_idx)
        log(f"\n  === Layer {layer_idx} ===")

        layer_result = {
            'layer': layer_idx,
            'features': {},
            'random_curvatures': [],
        }

        # 1. Measure curvature along feature directions
        for feat_name, pairs in feature_types.items():
            feat_curvatures = []
            feat_dirs_norms = []

            for sent_pos, sent_neg in pairs[:4]:  # Use first 4 pairs
                # Get hidden states for both sentences
                h_pos = get_hidden_at_layer(model, tokenizer, target_layer, sent_pos)
                h_neg = get_hidden_at_layer(model, tokenizer, target_layer, sent_neg)

                # Feature direction
                feat_dir = h_pos - h_neg
                feat_dir_norm = np.linalg.norm(feat_dir)
                feat_dirs_norms.append(feat_dir_norm)

                if feat_dir_norm < 1e-6:
                    continue

                feat_dir_unit = feat_dir / feat_dir_norm

                # Measure curvature along feature direction (use positive sentence as base)
                curv, jac, hess = measure_curvature_along_direction(
                    model, tokenizer, target_layer, sent_pos, feat_dir_unit, args.eps_rel)
                feat_curvatures.append(curv)

            mean_curv = np.mean(feat_curvatures) if feat_curvatures else 0
            mean_dir_norm = np.mean(feat_dirs_norms) if feat_dirs_norms else 0

            layer_result['features'][feat_name] = {
                'mean_curvature': float(mean_curv),
                'mean_dir_norm': float(mean_dir_norm),
                'curvatures': [float(c) for c in feat_curvatures],
            }
            log(f"    {feat_name:>10}: curv={mean_curv:.4f}, dir_norm={mean_dir_norm:.2f}")

        # 2. Measure curvature along random directions (baseline)
        # Use first sentence of tense pairs as base
        base_sentence = TENSE_PAIRS[0][0]
        random_curvatures = []

        for _ in range(args.n_random):
            rand_dir = np.random.randn(d_model)
            rand_dir = rand_dir / np.linalg.norm(rand_dir)

            curv, jac, hess = measure_curvature_along_direction(
                model, tokenizer, target_layer, base_sentence, rand_dir, args.eps_rel)
            random_curvatures.append(curv)

        mean_random_curv = np.mean(random_curvatures)
        layer_result['random_curvatures'] = [float(c) for c in random_curvatures]
        layer_result['mean_random_curvature'] = float(mean_random_curv)
        log(f"    {'random':>10}: curv={mean_random_curv:.4f}")

        # 3. Compute "feature alignment ratio"
        for feat_name in feature_types:
            feat_curv = layer_result['features'][feat_name]['mean_curvature']
            ratio = feat_curv / (mean_random_curv + 1e-10)
            layer_result['features'][feat_name]['alignment_ratio'] = float(ratio)
            log(f"    {feat_name:>10} ratio: {ratio:.3f}x random")

        results[layer_idx] = layer_result

    # Analysis
    log(f"\n{'='*70}")
    log(f"ANALYSIS")
    log(f"{'='*70}")

    layers = []
    feat_ratios = {feat: [] for feat in feature_types}
    random_curvs = []
    feat_curvs = {feat: [] for feat in feature_types}
    dir_norms = {feat: [] for feat in feature_types}

    for l in sorted(results.keys()):
        d = results[l]
        layers.append(d['layer'])
        random_curvs.append(d['mean_random_curvature'])
        for feat in feature_types:
            feat_curvs[feat].append(d['features'][feat]['mean_curvature'])
            feat_ratios[feat].append(d['features'][feat]['alignment_ratio'])
            dir_norms[feat].append(d['features'][feat]['mean_dir_norm'])

    layers_arr = np.array(layers, dtype=float)

    log(f"\n  Feature Alignment Ratios (feature_curv / random_curv):")
    log(f"  {'L':>4} {'tense':>8} {'question':>10} {'number':>8} {'random':>8}")

    for i, l in enumerate(layers):
        log(f"  {l:>4} {feat_ratios['tense'][i]:>8.2f}x {feat_ratios['question'][i]:>10.2f}x "
            f"{feat_ratios['number'][i]:>8.2f}x {random_curvs[i]:>8.4f}")

    # Correlation analysis
    log(f"\n  Correlation with Depth:")
    for feat in feature_types:
        corr_ratio = np.corrcoef(layers_arr, np.array(feat_ratios[feat]))[0, 1] if len(layers_arr) > 2 else 0
        corr_curv = np.corrcoef(layers_arr, np.array(feat_curvs[feat]))[0, 1] if len(layers_arr) > 2 else 0
        corr_norm = np.corrcoef(layers_arr, np.array(dir_norms[feat]))[0, 1] if len(layers_arr) > 2 else 0
        log(f"    {feat:>10}: Corr(L, ratio)={corr_ratio:>7.3f}, Corr(L, curv)={corr_curv:>7.3f}, Corr(L, dir_norm)={corr_norm:>7.3f}")

    corr_random = np.corrcoef(layers_arr, np.array(random_curvs))[0, 1] if len(layers_arr) > 2 else 0
    log(f"    {'random':>10}: Corr(L, curv)={corr_random:>7.3f}")

    # Key comparison: does feature curvature decrease FASTER than random?
    log(f"\n  KEY COMPARISON:")
    log(f"  If feature alignment ratio DECREASES with depth =>")
    log(f"  Shallow high-curvature directions are MORE feature-aligned")
    log(f"  => Shallow anisotropy IS semantically structured!")

    for feat in feature_types:
        ratios = np.array(feat_ratios[feat])
        if ratios[0] > ratios[-1]:
            log(f"    {feat:>10}: ratio decreases with depth ({ratios[0]:.2f}x -> {ratios[-1]:.2f}x) => FEATURE ALIGNED")
        else:
            log(f"    {feat:>10}: ratio increases/constant with depth ({ratios[0]:.2f}x -> {ratios[-1]:.2f}x) => NOT feature aligned")

    # Absolute curvature comparison
    log(f"\n  Absolute Curvature by Feature Type:")
    log(f"  {'L':>4} {'tense':>8} {'question':>10} {'number':>8} {'random':>8}")
    for i, l in enumerate(layers):
        log(f"  {l:>4} {feat_curvs['tense'][i]:>8.4f} {feat_curvs['question'][i]:>10.4f} "
            f"{feat_curvs['number'][i]:>8.4f} {random_curvs[i]:>8.4f}")

    # Feature direction norms across layers
    log(f"\n  Feature Direction Norms (||h_pos - h_neg||):")
    log(f"  {'L':>4} {'tense':>8} {'question':>10} {'number':>8}")
    for i, l in enumerate(layers):
        log(f"  {l:>4} {dir_norms['tense'][i]:>8.2f} {dir_norms['question'][i]:>10.2f} "
            f"{dir_norms['number'][i]:>8.2f}")

    for feat in feature_types:
        corr_norm = np.corrcoef(layers_arr, np.array(dir_norms[feat]))[0, 1] if len(layers_arr) > 2 else 0
        log(f"    Corr(L, ||{feat}_dir||) = {corr_norm:.3f}")

    # Interpretation
    log(f"\n  INTERPRETATION:")

    all_ratios_decrease = True
    for feat in feature_types:
        ratios = np.array(feat_ratios[feat])
        if ratios[0] <= ratios[-1]:
            all_ratios_decrease = False

    if all_ratios_decrease:
        log(f"  ALL feature alignment ratios decrease with depth!")
        log(f"  => Shallow anisotropy IS semantically structured.")
        log(f"  => High-curvature directions in shallow layers encode linguistic features.")
    else:
        some_decrease = False
        for feat in feature_types:
            ratios = np.array(feat_ratios[feat])
            corr_r = np.corrcoef(layers_arr, ratios)[0, 1] if len(layers_arr) > 2 else 0
            if corr_r < -0.3:
                some_decrease = True
                log(f"  {feat} alignment ratio decreases with depth => this feature is structured in shallow layers")
        if not some_decrease:
            log(f"  No consistent pattern of decreasing alignment ratio.")
            log(f"  Shallow anisotropy may NOT be primarily feature-driven.")

    # Save results
    save_data = {}
    for l, d in results.items():
        save_data[str(l)] = {k: v for k, v in d.items()}

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'model': cfg['name'], 'results': save_data}, f, indent=2, default=str)

    del model
    torch.cuda.empty_cache()

    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
