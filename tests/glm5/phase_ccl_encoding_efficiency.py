"""
Phase CCL: Encoding Efficiency & Linear Separability
=====================================================
CCXLIX discovered: feature directions have LOWER curvature than random!
This suggests: shallow = "chaotic encoding", deep = "linear representation".

This test QUANTIFIES the encoding-decoupling model:
1. Feature separation ratio: ||feature_dir|| / ||h|| (fraction of space used for features)
2. Encoding efficiency: ||feature_dir|| / (||h|| * rel_curv) (feature sep per unit nonlinearity)
3. Linear separability: can a linear probe classify features at each layer?
4. Curvature-to-separation ratio: rel_curv * ||h|| / ||feature_dir|| (nonlinearity per unit separation)

Predictions:
- Feature separation ratio should INCREASE with depth (features get more separated)
- Encoding efficiency should INCREASE with depth (more feature separation per unit curvature)
- Linear separability should INCREASE with depth (features become linearly decoupled)
- Curvature-to-separation ratio should DECREASE with depth (less nonlinearity per unit separation)
"""

import argparse
import json
import os
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

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

# More pairs for better linear probe training
TENSE_PAIRS_EXTENDED = [
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
    ("He paints pictures", "He painted pictures"),
    ("They dance together", "They danced together"),
    ("The sun rises early", "The sun rose early"),
    ("She plays piano", "She played piano"),
    ("He drives carefully", "He drove carefully"),
    ("The river flows north", "The river flowed north"),
    ("She teaches mathematics", "She taught mathematics"),
    ("They travel abroad", "They traveled abroad"),
    ("The dog barks loudly", "The dog barked loudly"),
    ("She cooks dinner", "She cooked dinner"),
]

QUESTION_PAIRS_EXTENDED = [
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
    ("He paints pictures", "Does he paint pictures"),
    ("They dance together", "Do they dance together"),
    ("The sun rises early", "Does the sun rise early"),
    ("She plays piano", "Does she play piano"),
    ("He drives carefully", "Does he drive carefully"),
    ("The river flows north", "Does the river flow north"),
    ("She teaches mathematics", "Does she teach mathematics"),
    ("They travel abroad", "Do they travel abroad"),
    ("The dog barks loudly", "Does the dog bark loudly"),
    ("She cooks dinner", "Does she cook dinner"),
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


def measure_rel_curvature(model, tokenizer, target_layer, sentence, n_dirs=10, eps_rel=0.05):
    """Measure mean relative curvature at a layer (for a given sentence)."""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    seq_len = attention_mask.sum().item()
    last_pos = seq_len - 1

    d_model = MODEL_CONFIGS[list(MODEL_CONFIGS.keys())[0]]['d_model']
    # Detect d_model from first layer
    for mk, mc in MODEL_CONFIGS.items():
        if hasattr(target_layer, 'parameters'):
            for p in target_layer.parameters():
                d_model = p.shape[-1] if len(p.shape) > 1 else d_model
                break
            break

    curvatures = []

    for _ in range(n_dirs):
        v = np.random.randn(d_model)
        v = v / (np.linalg.norm(v) + 1e-10)
        v_tensor = torch.tensor(v, dtype=torch.float32, device=model.device)

        # Clean output and h_norm
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

        ph_p, poh_p, ph_m, poh_m = make_perturb_hooks(v_tensor, eps, last_pos)
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

        jac_v = (f_plus - f_minus) / (2 * eps)
        hess_v = (f_plus - 2 * f_clean + f_minus) / (eps ** 2)

        jac_norm = np.linalg.norm(jac_v)
        hess_norm = np.linalg.norm(hess_v)

        rel_curv = hess_norm / (jac_norm + 1e-10)
        curvatures.append(rel_curv)

    return np.mean(curvatures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_curv_dirs', type=int, default=8)
    parser.add_argument('--eps_rel', type=float, default=0.05)
    args = parser.parse_args()

    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]

    out_dir = f"results/causal_fiber/{model_key}_ccl"
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, 'run.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f"=== Phase CCL: Encoding Efficiency & Linear Separability ===")
    log(f"Model: {cfg['name']}, n_curv_dirs: {args.n_curv_dirs}, eps_rel: {args.eps_rel}")
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

    results = {}

    for layer_idx in test_layers:
        target_layer = get_target_layer(model, layer_idx)
        log(f"\n  === Layer {layer_idx} ===")

        # 1. Collect hidden states for linear probe
        tense_pos_hidden = []
        tense_neg_hidden = []
        quest_pos_hidden = []
        quest_neg_hidden = []

        for sent_pos, sent_neg in TENSE_PAIRS_EXTENDED[:12]:
            h_pos = get_hidden_at_layer(model, tokenizer, target_layer, sent_pos)
            h_neg = get_hidden_at_layer(model, tokenizer, target_layer, sent_neg)
            tense_pos_hidden.append(h_pos)
            tense_neg_hidden.append(h_neg)

        for sent_pos, sent_neg in QUESTION_PAIRS_EXTENDED[:12]:
            h_pos = get_hidden_at_layer(model, tokenizer, target_layer, sent_pos)
            h_neg = get_hidden_at_layer(model, tokenizer, target_layer, sent_neg)
            quest_pos_hidden.append(h_pos)
            quest_neg_hidden.append(h_neg)

        # 2. Feature direction norms
        tense_dirs = [p - n for p, n in zip(tense_pos_hidden, tense_neg_hidden)]
        quest_dirs = [p - n for p, n in zip(quest_pos_hidden, quest_neg_hidden)]

        tense_dir_norms = [np.linalg.norm(d) for d in tense_dirs]
        quest_dir_norms = [np.linalg.norm(d) for d in quest_dirs]

        mean_tense_dir = np.mean(tense_dir_norms)
        mean_quest_dir = np.mean(quest_dir_norms)

        # 3. Activation norms
        all_hidden = tense_pos_hidden + tense_neg_hidden + quest_pos_hidden + quest_neg_hidden
        act_norms = [np.linalg.norm(h) for h in all_hidden]
        mean_act_norm = np.mean(act_norms)

        # 4. Feature separation ratio
        tense_sep_ratio = mean_tense_dir / (mean_act_norm + 1e-10)
        quest_sep_ratio = mean_quest_dir / (mean_act_norm + 1e-10)

        # 5. Linear separability (logistic regression)
        # Tense classification
        X_tense = np.array(tense_pos_hidden + tense_neg_hidden)
        y_tense = np.array([1]*len(tense_pos_hidden) + [0]*len(tense_neg_hidden))
        try:
            clf_tense = LogisticRegression(max_iter=1000, C=1.0)
            scores_tense = cross_val_score(clf_tense, X_tense, y_tense, cv=3, scoring='accuracy')
            acc_tense = float(np.mean(scores_tense))
        except Exception:
            acc_tense = 0.5

        # Question classification
        X_quest = np.array(quest_pos_hidden + quest_neg_hidden)
        y_quest = np.array([1]*len(quest_pos_hidden) + [0]*len(quest_neg_hidden))
        try:
            clf_quest = LogisticRegression(max_iter=1000, C=1.0)
            scores_quest = cross_val_score(clf_quest, X_quest, y_quest, cv=3, scoring='accuracy')
            acc_quest = float(np.mean(scores_quest))
        except Exception:
            acc_quest = 0.5

        # 6. Relative curvature (mean across a few sentences)
        curv_estimates = []
        for sent in [TENSE_PAIRS_EXTENDED[0][0], TENSE_PAIRS_EXTENDED[1][0],
                     TENSE_PAIRS_EXTENDED[2][0]]:
            c = measure_rel_curvature(model, tokenizer, target_layer, sent,
                                      n_dirs=args.n_curv_dirs, eps_rel=args.eps_rel)
            curv_estimates.append(c)
        mean_curv = np.mean(curv_estimates)

        # 7. Encoding efficiency = feature_separation / curvature
        # Higher = more feature separation per unit of nonlinearity
        tense_efficiency = tense_sep_ratio / (mean_curv + 1e-10)
        quest_efficiency = quest_sep_ratio / (mean_curv + 1e-10)

        # 8. Curvature-to-separation ratio = curvature / separation
        # Lower = less nonlinearity per unit of feature separation (more efficient)
        tense_curv_to_sep = mean_curv / (tense_sep_ratio + 1e-10)
        quest_curv_to_sep = mean_curv / (quest_sep_ratio + 1e-10)

        results[layer_idx] = {
            'layer': layer_idx,
            'activation_norm': float(mean_act_norm),
            'tense_dir_norm': float(mean_tense_dir),
            'quest_dir_norm': float(mean_quest_dir),
            'tense_sep_ratio': float(tense_sep_ratio),
            'quest_sep_ratio': float(quest_sep_ratio),
            'tense_accuracy': float(acc_tense),
            'quest_accuracy': float(acc_quest),
            'mean_curvature': float(mean_curv),
            'tense_efficiency': float(tense_efficiency),
            'quest_efficiency': float(quest_efficiency),
            'tense_curv_to_sep': float(tense_curv_to_sep),
            'quest_curv_to_sep': float(quest_curv_to_sep),
        }

        log(f"    ||h||={mean_act_norm:.1f}, tense_dir={mean_tense_dir:.2f}, "
            f"quest_dir={mean_quest_dir:.2f}")
        log(f"    sep_ratio: tense={tense_sep_ratio:.4f}, quest={quest_sep_ratio:.4f}")
        log(f"    accuracy:  tense={acc_tense:.3f}, quest={acc_quest:.3f}")
        log(f"    curvature={mean_curv:.4f}")
        log(f"    efficiency: tense={tense_efficiency:.4f}, quest={quest_efficiency:.4f}")
        log(f"    curv/sep:   tense={tense_curv_to_sep:.2f}, quest={quest_curv_to_sep:.2f}")

    # Analysis
    log(f"\n{'='*70}")
    log(f"ANALYSIS")
    log(f"{'='*70}")

    layers = []
    sep_ratios_t = []
    sep_ratios_q = []
    accs_t = []
    accs_q = []
    curvs = []
    effs_t = []
    effs_q = []
    cs_t = []
    cs_q = []

    for l in sorted(results.keys()):
        d = results[l]
        layers.append(d['layer'])
        sep_ratios_t.append(d['tense_sep_ratio'])
        sep_ratios_q.append(d['quest_sep_ratio'])
        accs_t.append(d['tense_accuracy'])
        accs_q.append(d['quest_accuracy'])
        curvs.append(d['mean_curvature'])
        effs_t.append(d['tense_efficiency'])
        effs_q.append(d['quest_efficiency'])
        cs_t.append(d['tense_curv_to_sep'])
        cs_q.append(d['quest_curv_to_sep'])

    layers_arr = np.array(layers, dtype=float)

    log(f"\n  Summary Table:")
    log(f"  {'L':>4} {'sep_t':>7} {'sep_q':>7} {'acc_t':>6} {'acc_q':>6} "
        f"{'curv':>8} {'eff_t':>7} {'eff_q':>7} {'c/s_t':>7} {'c/s_q':>7}")

    for i, l in enumerate(layers):
        log(f"  {l:>4} {sep_ratios_t[i]:>7.4f} {sep_ratios_q[i]:>7.4f} "
            f"{accs_t[i]:>6.3f} {accs_q[i]:>6.3f} {curvs[i]:>8.4f} "
            f"{effs_t[i]:>7.2f} {effs_q[i]:>7.2f} {cs_t[i]:>7.2f} {cs_q[i]:>7.2f}")

    log(f"\n  Correlations with Depth:")
    metrics = {
        'sep_ratio_tense': (np.array(sep_ratios_t), 'Feature separation ratio (tense)'),
        'sep_ratio_quest': (np.array(sep_ratios_q), 'Feature separation ratio (question)'),
        'acc_tense': (np.array(accs_t), 'Linear separability (tense)'),
        'acc_quest': (np.array(accs_q), 'Linear separability (question)'),
        'curvature': (np.array(curvs), 'Relative curvature'),
        'efficiency_tense': (np.array(effs_t), 'Encoding efficiency (tense)'),
        'efficiency_quest': (np.array(effs_q), 'Encoding efficiency (question)'),
        'curv_to_sep_tense': (np.array(cs_t), 'Curvature/Separation (tense)'),
        'curv_to_sep_quest': (np.array(cs_q), 'Curvature/Separation (question)'),
    }

    for name, (vals, desc) in metrics.items():
        corr = np.corrcoef(layers_arr, vals)[0, 1] if len(layers_arr) > 2 else 0
        arrow = "UP" if corr > 0.3 else ("DOWN" if corr < -0.3 else "~")
        log(f"    Corr(L, {name:>20}) = {corr:>7.3f}  [{arrow}]  {desc}")

    # Key predictions
    log(f"\n  PREDICTION VERIFICATION:")
    log(f"  (Based on encoding-decoupling model)")

    corr_sep_t = np.corrcoef(layers_arr, np.array(sep_ratios_t))[0, 1] if len(layers_arr) > 2 else 0
    corr_sep_q = np.corrcoef(layers_arr, np.array(sep_ratios_q))[0, 1] if len(layers_arr) > 2 else 0
    corr_acc_t = np.corrcoef(layers_arr, np.array(accs_t))[0, 1] if len(layers_arr) > 2 else 0
    corr_acc_q = np.corrcoef(layers_arr, np.array(accs_q))[0, 1] if len(layers_arr) > 2 else 0
    corr_eff_t = np.corrcoef(layers_arr, np.array(effs_t))[0, 1] if len(layers_arr) > 2 else 0
    corr_eff_q = np.corrcoef(layers_arr, np.array(effs_q))[0, 1] if len(layers_arr) > 2 else 0
    corr_cs_t = np.corrcoef(layers_arr, np.array(cs_t))[0, 1] if len(layers_arr) > 2 else 0
    corr_cs_q = np.corrcoef(layers_arr, np.array(cs_q))[0, 1] if len(layers_arr) > 2 else 0

    predictions = [
        ("Feature separation ratio INCREASES with depth", corr_sep_t > 0.3 or corr_sep_q > 0.3,
         f"Corr = {corr_sep_t:.3f}, {corr_sep_q:.3f}"),
        ("Linear separability INCREASES with depth", corr_acc_t > 0.3 or corr_acc_q > 0.3,
         f"Corr = {corr_acc_t:.3f}, {corr_acc_q:.3f}"),
        ("Encoding efficiency INCREASES with depth", corr_eff_t > 0.3 or corr_eff_q > 0.3,
         f"Corr = {corr_eff_t:.3f}, {corr_eff_q:.3f}"),
        ("Curvature/Separation DECREASES with depth", corr_cs_t < -0.3 or corr_cs_q < -0.3,
         f"Corr = {corr_cs_t:.3f}, {corr_cs_q:.3f}"),
    ]

    for pred, supported, evidence in predictions:
        status = "SUPPORTED" if supported else "NOT SUPPORTED"
        log(f"    {status}: {pred}")
        log(f"      Evidence: {evidence}")

    # Interpretation
    n_supported = sum(1 for _, s, _ in predictions if s)
    log(f"\n  INTERPRETATION:")
    log(f"  {n_supported}/4 predictions supported by {cfg['name']}")

    if n_supported >= 3:
        log(f"  => Encoding-decoupling model STRONGLY supported!")
        log(f"  => Deep layers produce features more efficiently (more separation per unit curvature)")
        log(f"  => Shallow high curvature is 'wasted' on chaotic encoding, not feature production")
    elif n_supported >= 2:
        log(f"  => Encoding-decoupling model PARTIALLY supported")
        log(f"  => Some aspects confirmed, others need revision")
    else:
        log(f"  => Encoding-decoupling model NOT supported by this model")
        log(f"  => Need to rethink the relationship between curvature and feature encoding")

    # Save
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'model': cfg['name'], 'results': {str(k): v for k, v in results.items()}},
                  f, indent=2, default=str)

    del model
    torch.cuda.empty_cache()

    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
