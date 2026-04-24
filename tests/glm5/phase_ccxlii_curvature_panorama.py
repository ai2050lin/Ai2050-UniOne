"""
Phase CCXLII: 曲率全景图谱 — 二维曲率热力图
=============================================
核心问题:
  CCXL和CCXLI分别固定L1或L2, 得到一维截面。
  但曲率真正依赖(L1位置, L2位置, 距离)三个变量, 需要二维全景图。

实验设计:
  在(L1, L2)的网格上系统测量曲率:
  - L1 ∈ {L/6, L/3, L/2, 2L/3}  (4个位置)
  - L2 ∈ {L1+2, L1+L/6, L1+L/3, min(L1+L/2, L-2)} (4个距离)
  - 约16个(L1, L2)组合 (排除L2<=L1)
  - 1个特征对 × 15样本 × α=1.0

  生成二维热力图, 直观显示曲率的"地形":
  - 如果"共振模型"正确 → 中间L1 + 中等距离 = 曲率峰值(山脊)
  - 如果"积分模型"正确 → 任意L1 + 最大距离 = 曲率峰值(右上角)
  - 如果"浅层生成器"正确 → 最小L1 + 任意距离 = 曲率峰值(第一行)
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
        ]
    },
}

ALPHA = 1.0


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


def generate_grid(n_layers):
    """Generate (L1, L2) grid points."""
    L6 = max(1, n_layers // 6)
    L3 = max(2, n_layers // 3)
    L2 = max(3, n_layers // 2)
    L23 = max(4, 2 * n_layers // 3)
    
    l1_candidates = sorted(set([L6, L3, L2, L23]))
    # Ensure all L1 >= 1 and < n_layers - 2
    l1_candidates = [l for l in l1_candidates if 1 <= l < n_layers - 3]
    
    grid = []
    for l1 in l1_candidates:
        remaining = n_layers - l1 - 2  # max distance from l1
        # Distance offsets: small, medium, large
        offsets = sorted(set([
            2,
            max(3, remaining // 3),
            max(4, 2 * remaining // 3),
            remaining,
        ]))
        offsets = [o for o in offsets if 2 <= o <= remaining]
        # Deduplicate
        offsets = list(dict.fromkeys(offsets))
        
        for off in offsets:
            l2 = l1 + off
            if l2 < n_layers - 1:
                grid.append((l1, l2))
    
    return grid


def run_panorama(model, tokenizer, model_key, n_pairs=15):
    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg['n_layers']
    d_model = cfg['d_model']
    
    grid = generate_grid(n_layers)
    
    results = {
        'model': cfg['name'],
        'n_layers': n_layers,
        'd_model': d_model,
        'grid': grid,
        'pairs': {},
    }
    
    for pair_name, pair_data in FOUR_WAY_PAIRS.items():
        feat_A = pair_data['feature_A']
        feat_B = pair_data['feature_B']
        sentences = pair_data['sentences'][:n_pairs]
        n_s = len(sentences)
        
        print(f"\n--- {pair_name} ({feat_A} x {feat_B}), n={n_s} ---")
        print(f"  Grid points: {len(grid)}")
        
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
        
        needed_layers = set()
        for l1, l2 in grid:
            needed_layers.add(l1)
            needed_layers.add(l2)
        
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
        
        # Step 3: Measure curvature at each grid point
        print(f"  Measuring curvature at {len(grid)} grid points...")
        
        panorama_data = {}
        max_l1 = max(l1 for l1, l2 in grid)
        
        for g_idx, (l1, l2) in enumerate(grid):
            pc1_A_L1 = pc1_A_per_layer.get(l1)
            pc1_B_L1 = pc1_B_per_layer.get(l1)
            pc1_A_L2 = pc1_A_per_layer.get(l2)
            pc1_B_L2 = pc1_B_per_layer.get(l2)
            
            if any(v is None for v in [pc1_A_L1, pc1_B_L1, pc1_A_L2, pc1_B_L2]):
                print(f"  Skipping L1={l1}, L2={l2}: missing PC1")
                continue
            
            scale_A_L1 = ALPHA * mean_dhA_norm_per_layer[l1]
            scale_B_L1 = ALPHA * mean_dhB_norm_per_layer[l1]
            scale_A_L2 = ALPHA * mean_dhA_norm_per_layer[l2]
            scale_B_L2 = ALPHA * mean_dhB_norm_per_layer[l2]
            
            curv_norms = []
            single_effects = []
            A_effects = []
            B_effects = []
            
            for i in range(n_s):
                s00 = sentences[i][0]
                h_clean = all_h00[i, n_layers, :]
                
                try:
                    h_AB = intervene_two_layers(model, tokenizer, s00,
                        l1, scale_A_L1 * pc1_A_L1,
                        l2, scale_B_L2 * pc1_B_L2,
                        n_layers)
                    
                    h_BA = intervene_two_layers(model, tokenizer, s00,
                        l1, scale_B_L1 * pc1_B_L1,
                        l2, scale_A_L2 * pc1_A_L2,
                        n_layers)
                    
                    h_A_only = intervene_single_layer(model, tokenizer, s00, l1, scale_A_L1 * pc1_A_L1, n_layers)
                    h_B_only = intervene_single_layer(model, tokenizer, s00, l2, scale_B_L2 * pc1_B_L2, n_layers)
                    
                    curvature = h_AB - h_BA
                    curv_norm = np.linalg.norm(curvature)
                    
                    curv_norms.append(curv_norm)
                    single_effects.append(
                        (np.linalg.norm(h_A_only - h_clean) + np.linalg.norm(h_B_only - h_clean)) / 2
                    )
                    A_effects.append(np.linalg.norm(h_A_only - h_clean))
                    B_effects.append(np.linalg.norm(h_B_only - h_clean))
                    
                except Exception as e:
                    print(f"  Failed at L1={l1}, L2={l2}, sample {i}: {e}")
                    continue
            
            if len(curv_norms) > 0:
                mean_curv = np.mean(curv_norms)
                std_curv = np.std(curv_norms) if len(curv_norms) > 1 else 0
                mean_single = np.mean(single_effects)
                mean_A_eff = np.mean(A_effects)
                mean_B_eff = np.mean(B_effects)
                rel_curv = mean_curv / (mean_single + 1e-10)
                
                key = f"L{l1}_L{l2}"
                panorama_data[key] = {
                    'l1': l1,
                    'l2': l2,
                    'distance': l2 - l1,
                    'curvature_norm_mean': float(mean_curv),
                    'curvature_norm_std': float(std_curv),
                    'mean_single_effect': float(mean_single),
                    'mean_A_effect': float(mean_A_eff),
                    'mean_B_effect': float(mean_B_eff),
                    'relative_curvature': float(rel_curv),
                    'n_valid': len(curv_norms),
                }
            
            if (g_idx + 1) % 4 == 0:
                print(f"  Done {g_idx+1}/{len(grid)} grid points")
        
        pair_result = {
            'pair': pair_name,
            'feature_A': feat_A,
            'feature_B': feat_B,
            'n_samples': n_s,
            'panorama_data': panorama_data,
        }
        
        results['pairs'][pair_name] = pair_result
        
        # Print 2D table
        l1_values = sorted(set(d['l1'] for d in panorama_data.values()))
        
        print(f"\n  === Curvature Panorama: {pair_name} ===")
        print(f"  {'L1\\L2':>6}", end="")
        for l1_val in l1_values:
            print(f"  L{l1_val:>2}", end="")
        print()
        
        for l1_val in l1_values:
            print(f"  L{l1_val:>4}", end="")
            for l2_base in l1_values:
                # Find data with this L1 and L2 near l2_base
                matching = [d for d in panorama_data.values() if d['l1'] == l1_val and abs(d['l2'] - l2_base) < n_layers // 6]
                if matching:
                    best = matching[0]
                    print(f" {best['relative_curvature']:>5.2f}", end="")
                else:
                    print(f"   --", end="")
            print()
        
        # Print detailed table
        print(f"\n  Detailed curvature values:")
        print(f"  {'L1':>4} {'L2':>4} {'Dist':>4} {'||[A,B]||':>10} {'Rel.Curv':>10} {'A_eff':>8} {'B_eff':>8}")
        for key in sorted(panorama_data.keys(), key=lambda k: (panorama_data[k]['l1'], panorama_data[k]['l2'])):
            d = panorama_data[key]
            print(f"  L{d['l1']:>2} L{d['l2']:>2} {d['distance']:>4} {d['curvature_norm_mean']:>10.2f} {d['relative_curvature']:>10.4f} {d['mean_A_effect']:>8.2f} {d['mean_B_effect']:>8.2f}")
        
        # Analyze patterns
        print(f"\n  === Pattern Analysis ===")
        
        data_list = sorted(panorama_data.values(), key=lambda d: (d['l1'], d['l2']))
        
        # 1. Which L1 produces max curvature?
        l1_curvatures = {}
        for d in data_list:
            if d['l1'] not in l1_curvatures:
                l1_curvatures[d['l1']] = []
            l1_curvatures[d['l1']].append(d['relative_curvature'])
        
        l1_mean_curv = {l1: np.mean(curvs) for l1, curvs in l1_curvatures.items()}
        best_l1 = max(l1_mean_curv, key=l1_mean_curv.get)
        print(f"  Best L1 (max mean Rel.Curv): L{best_l1} = {l1_mean_curv[best_l1]:.3f}")
        for l1 in sorted(l1_mean_curv.keys()):
            print(f"    L{l1}: mean Rel.Curv = {l1_mean_curv[l1]:.3f}")
        
        # 2. Which distance produces max curvature?
        dist_curvatures = {}
        for d in data_list:
            dist = d['distance']
            if dist not in dist_curvatures:
                dist_curvatures[dist] = []
            dist_curvatures[dist].append(d['relative_curvature'])
        
        # 3. Find peak curvature point
        peak = max(data_list, key=lambda d: d['relative_curvature'])
        print(f"\n  Peak curvature: L1={peak['l1']}, L2={peak['l2']}, dist={peak['distance']}")
        print(f"    ||[A,B]|| = {peak['curvature_norm_mean']:.2f}, Rel.Curv = {peak['relative_curvature']:.3f}")
        
        # 4. Correlation analysis
        distances = [d['distance'] for d in data_list]
        abs_curvs = [d['curvature_norm_mean'] for d in data_list]
        rel_curvs = [d['relative_curvature'] for d in data_list]
        l1_vals = [d['l1'] for d in data_list]
        l2_vals = [d['l2'] for d in data_list]
        
        if len(data_list) > 3:
            corr_dist_abs = np.corrcoef(distances, abs_curvs)[0, 1]
            corr_dist_rel = np.corrcoef(distances, rel_curvs)[0, 1]
            corr_l1_abs = np.corrcoef(l1_vals, abs_curvs)[0, 1]
            corr_l1_rel = np.corrcoef(l1_vals, rel_curvs)[0, 1]
            
            print(f"\n  Correlations:")
            print(f"    Corr(dist, ||[A,B]||) = {corr_dist_abs:.3f}")
            print(f"    Corr(dist, Rel.Curv) = {corr_dist_rel:.3f}")
            print(f"    Corr(L1, ||[A,B]||) = {corr_l1_abs:.3f}")
            print(f"    Corr(L1, Rel.Curv) = {corr_l1_rel:.3f}")
            
            # Model selection
            if corr_l1_rel > 0.3:
                print(f"  → Deeper L1 → higher curvature: mid/deep layers generate more entanglement")
            elif corr_l1_rel < -0.3:
                print(f"  → Shallower L1 → higher curvature: early layers generate more entanglement ✓")
            else:
                print(f"  → Curvature independent of L1 position")
            
            if abs(corr_dist_rel) < 0.3 and abs(corr_l1_rel) < 0.3:
                print(f"  → 'Resonance model': curvature depends on specific (L1, L2) combination, not simple monotonic trends")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=15)
    args = parser.parse_args()
    
    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]
    
    out_dir = f"results/causal_fiber/{model_key}_ccxlii"
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = os.path.join(out_dir, 'run.log')
    
    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    
    log(f"=== Phase CCXLII: Curvature Panorama ===")
    log(f"Model: {cfg['name']}, n_pairs: {args.n_pairs}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")
    
    grid = generate_grid(cfg['n_layers'])
    log(f"Grid: {len(grid)} points")
    for l1, l2 in grid:
        log(f"  L1={l1}, L2={l2}, dist={l2-l1}")
    
    results = run_panorama(model, tokenizer, model_key, n_pairs=args.n_pairs)
    
    results['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Final summary
    log(f"\n{'='*70}")
    log(f"SUMMARY: {cfg['name']}")
    log(f"{'='*70}")
    
    for pair_name, pair_res in results['pairs'].items():
        log(f"\n  Pair: {pair_name}")
        pd = pair_res.get('panorama_data', {})
        
        if not pd:
            continue
        
        data_list = list(pd.values())
        peak = max(data_list, key=lambda d: d['relative_curvature'])
        
        log(f"  Peak: L1={peak['l1']}, L2={peak['l2']}, dist={peak['distance']}")
        log(f"    ||[A,B]|| = {peak['curvature_norm_mean']:.2f}, Rel.Curv = {peak['relative_curvature']:.3f}")
        
        # L1 position effect
        l1_groups = {}
        for d in data_list:
            l1_groups.setdefault(d['l1'], []).append(d['relative_curvature'])
        for l1 in sorted(l1_groups.keys()):
            mean_rc = np.mean(l1_groups[l1])
            log(f"    L1={l1}: mean Rel.Curv = {mean_rc:.3f}")
    
    del model
    torch.cuda.empty_cache()
    
    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
