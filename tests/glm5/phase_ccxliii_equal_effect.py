"""
Phase CCXLIII: 等效扰动实验 — 验证"浅层播种"是真实还是假象
=============================================================
核心问题:
  CCXLII发现Rel.Curv在浅层L1更高, 但绝对曲率||[A,B]||与L1无关!
  这意味着"浅层播种"可能只是归一化假象——浅层A_eff小导致分母小。

关键区分:
  H1_real: 浅层perturbation确实"纠缠效率更高" — 即使控制A_eff, 
           浅层的Rel.Curv仍然更高
  H2_artifact: "浅层播种"纯粹是归一化假象 — 控制A_eff后,
               Rel.Curv与L1位置无关

实验设计:
  在每个L1位置, 调整α使A_eff ≈ 目标值(如40-50)
  - 浅层L1: 用大α(1.5-3.0)补偿低传播效率
  - 中层L1: 用标准α(1.0)  
  - 深层L1: 用小α(0.3-0.7)补偿高传播效率
  
  然后测量曲率, 看Rel.Curv是否仍然在浅层更高。
  
  如果H1成立: 浅层Rel.Curv > 深层Rel.Curv (即使A_eff相同)
  如果H2成立: Rel.Curv在A_eff相同时与L1无关
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

FOUR_WAY_PAIRS = {
    'tense_x_question': {
        'feature_A': 'tense',
        'feature_B': 'question',
        'sentences': [
            ("She walks to school", "She walked to school", "Does she walk to school", "Did she walk to school"),
            ("He runs in the park", "He ran in the park", "Does he run in the park", "Did he run in the park"),
            ("They play football", "They played football", "Do they play football", "Did they play football"),
            ("The cat sleeps on the mat", "The cat slept on the mat", "Does the cat sleep on the mat", "Did the cat sleep on the mat"),
            ("She sings beautifully", "She sang beautifully", "Does she sing beautifully", "Did she sing beautifully"),
            ("He writes a letter", "He wrote a letter", "Does he write a letter", "Did he write a letter"),
            ("They travel abroad", "They traveled abroad", "Do they travel abroad", "Did they travel abroad"),
            ("The dog barks loudly", "The dog barked loudly", "Does the dog bark loudly", "Did the dog bark loudly"),
            ("She cooks dinner", "She cooked dinner", "Does she cook dinner", "Did she cook dinner"),
            ("He drives carefully", "He drove carefully", "Does he drive carefully", "Did he drive carefully"),
            ("The bird flies south", "The bird flew south", "Does the bird fly south", "Did the bird fly south"),
            ("She reads the book", "She read the book", "Does she read the book", "Did she read the book"),
            ("They build houses", "They built houses", "Do they build houses", "Did they build houses"),
            ("The river flows north", "The river flowed north", "Does the river flow north", "Did the river flow north"),
            ("She teaches mathematics", "She taught mathematics", "Does she teach mathematics", "Did she teach mathematics"),
            ("He fixes computers", "He fixed computers", "Does he fix computers", "Did he fix computers"),
            ("The train arrives early", "The train arrived early", "Does the train arrive early", "Did the train arrive early"),
            ("She paints landscapes", "She painted landscapes", "Does she paint landscapes", "Did she paint landscapes"),
            ("They sell fresh bread", "They sold fresh bread", "Do they sell fresh bread", "Did they sell fresh bread"),
            ("He speaks three languages", "He spoke three languages", "Does he speak three languages", "Did he speak three languages"),
        ]
    },
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


def get_all_hidden_states(model, tokenizer, sentence, n_layers):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    seq_len = attention_mask.sum().item()
    last_hidden = torch.stack([h[0, seq_len-1, :].cpu().float() for h in hidden_states])
    return last_hidden.numpy()


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


def intervene_two_layers(model, tokenizer, sentence, layer1_idx, pert1_vec, layer2_idx, pert2_vec, n_layers):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    seq_len = attention_mask.sum().item()
    last_pos = seq_len - 1

    target_layer1 = get_target_layer(model, layer1_idx)
    target_layer2 = get_target_layer(model, layer2_idx)

    pert1_tensor = torch.tensor(pert1_vec, dtype=torch.float32, device=model.device)
    pert2_tensor = torch.tensor(pert2_vec, dtype=torch.float32, device=model.device)

    def make_hook(pert_tensor, last_pos):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
                hidden[:, last_pos, :] = hidden[:, last_pos, :].float() + pert_tensor.unsqueeze(0).to(hidden.dtype)
                return (hidden,) + rest
            else:
                output[:, last_pos, :] = output[:, last_pos, :].float() + pert_tensor.unsqueeze(0).to(output.dtype)
                return output
        return hook_fn

    hook1 = target_layer1.register_forward_hook(make_hook(pert1_tensor, last_pos))
    hook2 = target_layer2.register_forward_hook(make_hook(pert2_tensor, last_pos))

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        result = hidden_states[n_layers][0, last_pos, :].cpu().float().numpy()
    finally:
        hook1.remove()
        hook2.remove()

    return result


def intervene_single_layer(model, tokenizer, sentence, layer_idx, pert_vec, n_layers):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    seq_len = attention_mask.sum().item()
    last_pos = seq_len - 1

    target_layer = get_target_layer(model, layer_idx)
    pert_tensor = torch.tensor(pert_vec, dtype=torch.float32, device=model.device)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
            hidden[:, last_pos, :] = hidden[:, last_pos, :].float() + pert_tensor.unsqueeze(0).to(hidden.dtype)
            return (hidden,) + rest
        else:
            output[:, last_pos, :] = output[:, last_pos, :].float() + pert_tensor.unsqueeze(0).to(output.dtype)
            return output

    handle = target_layer.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        result = hidden_states[n_layers][0, last_pos, :].cpu().float().numpy()
    finally:
        handle.remove()

    return result


def calibrate_alpha_for_target_effect(model, tokenizer, sentences, all_h00, pc1_A, layer_idx, 
                                       mean_dhA_norm, n_layers, target_effect, n_calib=10):
    """Find alpha such that mean A_eff ≈ target_effect."""
    # Start with α=1.0 and measure baseline effect
    test_alpha = 1.0
    test_effects = []
    
    for i in range(min(n_calib, len(sentences))):
        s00 = sentences[i][0]
        h_clean = all_h00[i, n_layers, :]
        pert_vec = test_alpha * mean_dhA_norm * pc1_A
        h_pert = intervene_single_layer(model, tokenizer, s00, layer_idx, pert_vec, n_layers)
        test_effects.append(np.linalg.norm(h_pert - h_clean))
    
    mean_effect_at_alpha1 = np.mean(test_effects)
    
    if mean_effect_at_alpha1 < 1e-6:
        return 1.0, mean_effect_at_alpha1
    
    # Linear scaling assumption: effect ∝ alpha
    # target_effect = alpha * mean_effect_at_alpha1
    needed_alpha = target_effect / mean_effect_at_alpha1
    
    # Clamp to reasonable range
    needed_alpha = max(0.1, min(5.0, needed_alpha))
    
    return needed_alpha, mean_effect_at_alpha1


def run_equal_effect_experiment(model, tokenizer, model_key, n_pairs=20):
    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg['n_layers']
    d_model = cfg['d_model']
    
    # L1 positions to test
    l1_positions = sorted(set([
        max(1, n_layers // 7),
        max(2, n_layers // 3),
        max(3, n_layers // 2),
        max(4, 2 * n_layers // 3),
    ]))
    
    # L2 fixed at 3L/4
    L2_fixed = 3 * n_layers // 4
    
    results = {
        'model': cfg['name'],
        'n_layers': n_layers,
        'd_model': d_model,
        'L2_fixed': L2_fixed,
        'l1_positions': l1_positions,
        'pairs': {},
    }
    
    for pair_name, pair_data in FOUR_WAY_PAIRS.items():
        feat_A = pair_data['feature_A']
        feat_B = pair_data['feature_B']
        sentences = pair_data['sentences'][:n_pairs]
        n_s = len(sentences)
        
        print(f"\n--- {pair_name} ({feat_A} x {feat_B}), n={n_s} ---")
        print(f"  L1 positions: {l1_positions}, L2={L2_fixed}")
        
        # Step 1: Collect baseline hidden states
        print(f"  Collecting baseline hidden states...")
        all_h00 = []
        all_h10 = []
        all_h01 = []
        
        for i, (s00, s10, s01, s11) in enumerate(sentences):
            h00 = get_all_hidden_states(model, tokenizer, s00, n_layers)
            h10 = get_all_hidden_states(model, tokenizer, s10, n_layers)
            h01 = get_all_hidden_states(model, tokenizer, s01, n_layers)
            all_h00.append(h00)
            all_h10.append(h10)
            all_h01.append(h01)
        
        all_h00 = np.array(all_h00)
        all_h10 = np.array(all_h10)
        all_h01 = np.array(all_h01)
        
        # Step 2: Compute PC1 at each relevant layer
        from sklearn.decomposition import PCA
        
        needed_layers = set(l1_positions + [L2_fixed])
        
        pc1_A_per_layer = {}
        pc1_B_per_layer = {}
        mean_dhA_norm_per_layer = {}
        mean_dhB_norm_per_layer = {}
        
        for l in sorted(needed_layers):
            if l < 1 or l >= n_layers:
                continue
            dh_A = all_h10[:, l, :] - all_h00[:, l, :]
            dh_B = all_h01[:, l, :] - all_h00[:, l, :]
            
            pca_A = PCA(n_components=5)
            pca_A.fit(dh_A)
            pc1_A = pca_A.components_[0] / (np.linalg.norm(pca_A.components_[0]) + 1e-10)
            
            pca_B = PCA(n_components=5)
            pca_B.fit(dh_B)
            pc1_B = pca_B.components_[0] / (np.linalg.norm(pca_B.components_[0]) + 1e-10)
            
            pc1_A_per_layer[l] = pc1_A
            pc1_B_per_layer[l] = pc1_B
            mean_dhA_norm_per_layer[l] = float(np.mean(np.linalg.norm(dh_A, axis=1)))
            mean_dhB_norm_per_layer[l] = float(np.mean(np.linalg.norm(dh_B, axis=1)))
        
        # Step 3: Calibrate alpha for each L1 position
        print(f"\n  Calibrating alpha for each L1 position...")
        
        # First, measure A_eff at α=1.0 for each L1
        baseline_effects = {}
        for l1 in l1_positions:
            if l1 not in pc1_A_per_layer:
                continue
            effects = []
            for i in range(min(5, n_s)):
                s00 = sentences[i][0]
                h_clean = all_h00[i, n_layers, :]
                pert_vec = 1.0 * mean_dhA_norm_per_layer[l1] * pc1_A_per_layer[l1]
                h_pert = intervene_single_layer(model, tokenizer, s00, l1, pert_vec, n_layers)
                effects.append(np.linalg.norm(h_pert - h_clean))
            baseline_effects[l1] = np.mean(effects)
            print(f"    L1={l1}: A_eff at α=1.0 = {baseline_effects[l1]:.2f}")
        
        # Target effect: use the median of baseline effects
        target_effect = np.median(list(baseline_effects.values()))
        print(f"  Target A_eff = {target_effect:.2f} (median)")
        
        # Compute needed alpha for each L1
        needed_alphas = {}
        for l1 in l1_positions:
            if baseline_effects[l1] < 1e-6:
                needed_alphas[l1] = 1.0
            else:
                needed_alphas[l1] = target_effect / baseline_effects[l1]
                needed_alphas[l1] = max(0.1, min(5.0, needed_alphas[l1]))
            print(f"    L1={l1}: needed α = {needed_alphas[l1]:.3f}")
        
        # Step 4: Measure curvature at each L1 with calibrated alpha
        print(f"\n  Measuring curvature with calibrated alpha...")
        
        equal_effect_data = {}
        
        for l1_idx, l1 in enumerate(l1_positions):
            if l1 not in pc1_A_per_layer:
                continue
            
            alpha_A = needed_alphas[l1]
            alpha_B = 1.0  # B always at L2 with α=1.0
            
            scale_A_L1 = alpha_A * mean_dhA_norm_per_layer[l1]
            scale_B_L1 = alpha_B * mean_dhB_norm_per_layer[l1]
            scale_A_L2 = 1.0 * mean_dhA_norm_per_layer[L2_fixed]
            scale_B_L2 = 1.0 * mean_dhB_norm_per_layer[L2_fixed]
            
            curv_norms = []
            single_effects = []
            A_effects = []
            B_effects = []
            
            for i in range(n_s):
                s00 = sentences[i][0]
                h_clean = all_h00[i, n_layers, :]
                
                try:
                    h_AB = intervene_two_layers(model, tokenizer, s00,
                        l1, scale_A_L1 * pc1_A_per_layer[l1],
                        L2_fixed, scale_B_L2 * pc1_B_per_layer[L2_fixed],
                        n_layers)
                    
                    h_BA = intervene_two_layers(model, tokenizer, s00,
                        l1, scale_B_L1 * pc1_B_per_layer[l1],
                        L2_fixed, scale_A_L2 * pc1_A_per_layer[L2_fixed],
                        n_layers)
                    
                    h_A_only = intervene_single_layer(model, tokenizer, s00, l1, scale_A_L1 * pc1_A_per_layer[l1], n_layers)
                    h_B_only = intervene_single_layer(model, tokenizer, s00, L2_fixed, scale_B_L2 * pc1_B_per_layer[L2_fixed], n_layers)
                    
                    curvature = h_AB - h_BA
                    curv_norm = np.linalg.norm(curvature)
                    
                    curv_norms.append(curv_norm)
                    single_effects.append(
                        (np.linalg.norm(h_A_only - h_clean) + np.linalg.norm(h_B_only - h_clean)) / 2
                    )
                    A_effects.append(np.linalg.norm(h_A_only - h_clean))
                    B_effects.append(np.linalg.norm(h_B_only - h_clean))
                    
                except Exception as e:
                    print(f"  Failed at L1={l1}, sample {i}: {e}")
                    continue
            
            if len(curv_norms) > 0:
                mean_curv = np.mean(curv_norms)
                std_curv = np.std(curv_norms) if len(curv_norms) > 1 else 0
                mean_single = np.mean(single_effects)
                mean_A_eff = np.mean(A_effects)
                mean_B_eff = np.mean(B_effects)
                rel_curv = mean_curv / (mean_single + 1e-10)
                
                equal_effect_data[f"L{l1}"] = {
                    'l1': l1,
                    'alpha_A': float(alpha_A),
                    'distance': L2_fixed - l1,
                    'curvature_norm_mean': float(mean_curv),
                    'curvature_norm_std': float(std_curv),
                    'mean_single_effect': float(mean_single),
                    'mean_A_effect': float(mean_A_eff),
                    'mean_B_effect': float(mean_B_eff),
                    'relative_curvature': float(rel_curv),
                    'n_valid': len(curv_norms),
                }
        
        pair_result = {
            'pair': pair_name,
            'feature_A': feat_A,
            'feature_B': feat_B,
            'n_samples': n_s,
            'L2_fixed': L2_fixed,
            'target_A_effect': float(target_effect),
            'needed_alphas': needed_alphas,
            'equal_effect_data': equal_effect_data,
        }
        
        results['pairs'][pair_name] = pair_result
        
        # Print results
        print(f"\n  === Equal-Effect Curvature for {pair_name} ===")
        print(f"  Target A_eff = {target_effect:.2f}")
        print(f"  {'L1':>4} {'α_A':>6} {'Dist':>4} {'A_eff':>8} {'B_eff':>8} {'||[A,B]||':>10} {'Rel.Curv':>10}")
        
        for key in sorted(equal_effect_data.keys(), key=lambda x: equal_effect_data[x]['l1']):
            d = equal_effect_data[key]
            print(f"  L{d['l1']:>2} {d['alpha_A']:>6.3f} {d['distance']:>4} {d['mean_A_effect']:>8.2f} {d['mean_B_effect']:>8.2f} {d['curvature_norm_mean']:>10.2f} {d['relative_curvature']:>10.4f}")
        
        # Analysis
        l1_vals = [d['l1'] for d in equal_effect_data.values()]
        rel_curvs = [d['relative_curvature'] for d in equal_effect_data.values()]
        abs_curvs = [d['curvature_norm_mean'] for d in equal_effect_data.values()]
        a_effs = [d['mean_A_effect'] for d in equal_effect_data.values()]
        
        if len(l1_vals) > 2:
            corr_l1_rel = np.corrcoef(l1_vals, rel_curvs)[0, 1]
            corr_l1_abs = np.corrcoef(l1_vals, abs_curvs)[0, 1]
            
            print(f"\n  Corr(L1, Rel.Curv) = {corr_l1_rel:.3f}")
            print(f"  Corr(L1, ||[A,B]||) = {corr_l1_abs:.3f}")
            
            if corr_l1_rel < -0.3:
                print(f"  -> After controlling A_eff, shallow L1 STILL has higher Rel.Curv -> Shallow seeding is REAL")
            elif corr_l1_rel > 0.3:
                print(f"  -> After controlling A_eff, deep L1 has higher Rel.Curv -> OPPOSITE of shallow seeding!")
            else:
                print(f"  -> After controlling A_eff, Rel.Curv independent of L1 -> Shallow seeding is ARTIFACT")
            
            # Check if A_eff was successfully controlled
            a_eff_std = np.std(a_effs) / np.mean(a_effs)
            print(f"  A_eff variation: std/mean = {a_eff_std:.3f} ({'well-controlled' if a_eff_std < 0.2 else 'poorly controlled'})")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=20)
    args = parser.parse_args()
    
    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]
    
    out_dir = f"results/causal_fiber/{model_key}_ccxliii"
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = os.path.join(out_dir, 'run.log')
    
    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    
    log(f"=== Phase CCXLIII: Equal-Effect Experiment ===")
    log(f"Model: {cfg['name']}, n_pairs: {args.n_pairs}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")
    
    results = run_equal_effect_experiment(model, tokenizer, model_key, n_pairs=args.n_pairs)
    
    results['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    log(f"\n{'='*70}")
    log(f"SUMMARY: {cfg['name']}")
    log(f"{'='*70}")
    
    for pair_name, pair_res in results['pairs'].items():
        log(f"\n  Pair: {pair_name}")
        eed = pair_res.get('equal_effect_data', {})
        target = pair_res.get('target_A_effect', 0)
        alphas = pair_res.get('needed_alphas', {})
        
        log(f"  Target A_eff = {target:.2f}")
        
        for key in sorted(eed.keys(), key=lambda x: eed[x]['l1']):
            d = eed[key]
            log(f"    L1={d['l1']}: α={d['alpha_A']:.3f}, A_eff={d['mean_A_effect']:.2f}, "
                f"||[A,B]||={d['curvature_norm_mean']:.2f}, Rel.Curv={d['relative_curvature']:.3f}")
        
        l1_vals = [d['l1'] for d in eed.values()]
        rel_curvs = [d['relative_curvature'] for d in eed.values()]
        abs_curvs = [d['curvature_norm_mean'] for d in eed.values()]
        
        if len(l1_vals) > 2:
            corr_l1_rel = np.corrcoef(l1_vals, rel_curvs)[0, 1]
            corr_l1_abs = np.corrcoef(l1_vals, abs_curvs)[0, 1]
            log(f"  Corr(L1, Rel.Curv) = {corr_l1_rel:.3f}")
            log(f"  Corr(L1, ||[A,B]||) = {corr_l1_abs:.3f}")
            
            if corr_l1_rel < -0.3:
                log(f"  VERDICT: 'Shallow seeding' is REAL — shallow layers produce more entanglement per unit effect")
            elif abs(corr_l1_rel) < 0.3:
                log(f"  VERDICT: 'Shallow seeding' is ARTIFACT — entanglement is proportional to effect size, not layer position")
            else:
                log(f"  VERDICT: Unexpected — deep layers produce more entanglement per unit effect")
    
    del model
    torch.cuda.empty_cache()
    
    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
