"""
Phase CCXLI: 反向传播实验 — 曲率的非对称性
============================================
核心问题:
  CCXL发现固定L1(浅), 变化L2(深)时, 绝对曲率随距离递增。
  现在反过来: 固定L2(深), 变化L1(浅), 曲率是否也随距离递增?

关键区分:
  H1_symmetric: 曲率纯粹是距离的函数, 两个方向都递增
  H2_asymmetric: 浅层perturbation对累积曲率更重要, 只有L1固定时递增
  H3_reverse: 固定L2后, 曲率随L1变浅而增大(浅层是纠缠生成器)

实验设计:
  固定l2 = 3*L/4 (深层), 变化l1从1到l2-2
  对比CCXL: 固定l1=L/4, 变化l2

如果H3成立: l1越浅, 曲率越大 → 浅层确实是"纠缠生成器"
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
            ("The wind blows gently", "The wind blew gently", "Does the wind blow gently", "Did the wind blow gently"),
            ("She types documents", "She typed documents", "Does she type documents", "Did she type documents"),
            ("They harvest crops", "They harvested crops", "Do they harvest crops", "Did they harvest crops"),
            ("He delivers packages", "He delivered packages", "Does he deliver packages", "Did he deliver packages"),
            ("The bell rings loudly", "The bell rang loudly", "Does the bell ring loudly", "Did the bell ring loudly"),
        ]
    },
    'voice_x_question': {
        'feature_A': 'voice',
        'feature_B': 'question',
        'sentences': [
            ("The cat catches the mouse", "The mouse is caught by the cat", "Does the cat catch the mouse", "Is the mouse caught by the cat"),
            ("She writes the report", "The report is written by her", "Does she write the report", "Is the report written by her"),
            ("He fixes the car", "The car is fixed by him", "Does he fix the car", "Is the car fixed by him"),
            ("They build the house", "The house is built by them", "Do they build the house", "Is the house built by them"),
            ("The chef cooks the meal", "The meal is cooked by the chef", "Does the chef cook the meal", "Is the meal cooked by the chef"),
            ("She delivers the speech", "The speech is delivered by her", "Does she deliver the speech", "Is the speech delivered by her"),
            ("He paints the fence", "The fence is painted by him", "Does he paint the fence", "Is the fence painted by him"),
            ("They discover the treasure", "The treasure is discovered by them", "Do they discover the treasure", "Is the treasure discovered by them"),
            ("The teacher explains the lesson", "The lesson is explained by the teacher", "Does the teacher explain the lesson", "Is the lesson explained by the teacher"),
            ("She directs the film", "The film is directed by her", "Does she direct the film", "Is the film directed by her"),
            ("He composes the music", "The music is composed by him", "Does he compose the music", "Is the music composed by him"),
            ("They publish the article", "The article is published by them", "Do they publish the article", "Is the article published by them"),
            ("The company launches the product", "The product is launched by the company", "Does the company launch the product", "Is the product launched by the company"),
            ("She records the song", "The song is recorded by her", "Does she record the song", "Is the song recorded by her"),
            ("He designs the building", "The building is designed by him", "Does he design the building", "Is the building designed by him"),
            ("They organize the event", "The event is organized by them", "Do they organize the event", "Is the event organized by them"),
            ("The police arrest the thief", "The thief is arrested by the police", "Do the police arrest the thief", "Is the thief arrested by the police"),
            ("She washes the dishes", "The dishes are washed by her", "Does she wash the dishes", "Are the dishes washed by her"),
            ("He drives the bus", "The bus is driven by him", "Does he drive the bus", "Is the bus driven by him"),
            ("They clean the room", "The room is cleaned by them", "Do they clean the room", "Is the room cleaned by them"),
            ("The wind blows the leaves", "The leaves are blown by the wind", "Does the wind blow the leaves", "Are the leaves blown by the wind"),
            ("She types the letter", "The letter is typed by her", "Does she type the letter", "Is the letter typed by her"),
            ("He catches the fish", "The fish is caught by him", "Does he catch the fish", "Is the fish caught by him"),
            ("They sell the house", "The house is sold by them", "Do they sell the house", "Is the house sold by them"),
            ("The doctor examines the patient", "The patient is examined by the doctor", "Does the doctor examine the patient", "Is the patient examined by the doctor"),
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


def run_reverse_experiment(model, tokenizer, model_key, n_pairs=25):
    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg['n_layers']
    d_model = cfg['d_model']
    
    # Fix L2 at 3*L/4 (deep layer), vary L1 from 1 to L2-2
    L2_fixed = 3 * n_layers // 4
    
    # Sample L1 values: every few layers from 1 to L2-2
    l1_values = list(range(1, L2_fixed - 1, max(1, n_layers // 10)))
    if l1_values[-1] < L2_fixed - 2:
        l1_values.append(L2_fixed - 2)
    
    results = {
        'model': cfg['name'],
        'n_layers': n_layers,
        'd_model': d_model,
        'L2_fixed': L2_fixed,
        'l1_values': l1_values,
        'pairs': {},
    }
    
    for pair_name, pair_data in FOUR_WAY_PAIRS.items():
        feat_A = pair_data['feature_A']
        feat_B = pair_data['feature_B']
        sentences = pair_data['sentences'][:n_pairs]
        n_s = len(sentences)
        
        print(f"\n--- {pair_name} ({feat_A} x {feat_B}), n={n_s} ---")
        print(f"  L2_fixed={L2_fixed}, l1_values={l1_values}")
        
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
            
            if (i + 1) % 10 == 0:
                print(f"  Collected {i+1}/{n_s} baseline states")
        
        all_h00 = np.array(all_h00)
        all_h10 = np.array(all_h10)
        all_h01 = np.array(all_h01)
        
        # Step 2: Compute PC1 at each relevant layer
        from sklearn.decomposition import PCA
        
        pc1_A_per_layer = {}
        pc1_B_per_layer = {}
        mean_dhA_norm_per_layer = {}
        mean_dhB_norm_per_layer = {}
        
        needed_layers = set([L2_fixed] + l1_values)
        for l in needed_layers:
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
        
        # Step 3: For each l1, measure curvature with fixed L2
        print(f"  Measuring curvature at {len(l1_values)} l1 values...")
        
        reverse_data = {}
        
        for l1_idx, l1 in enumerate(l1_values):
            pc1_A_L1 = pc1_A_per_layer[l1]
            pc1_B_L1 = pc1_B_per_layer[l1]
            pc1_A_L2 = pc1_A_per_layer[L2_fixed]
            pc1_B_L2 = pc1_B_per_layer[L2_fixed]
            
            scale_A_L1 = ALPHA * mean_dhA_norm_per_layer[l1]
            scale_B_L1 = ALPHA * mean_dhB_norm_per_layer[l1]
            scale_A_L2 = ALPHA * mean_dhA_norm_per_layer[L2_fixed]
            scale_B_L2 = ALPHA * mean_dhB_norm_per_layer[L2_fixed]
            
            curv_norms = []
            single_effects = []
            A_effects = []
            B_effects = []
            n_intervene = min(n_s, n_pairs)
            
            for i in range(n_intervene):
                s00 = sentences[i][0]
                h_clean = all_h00[i, n_layers, :]
                
                try:
                    h_AB = intervene_two_layers(model, tokenizer, s00,
                        l1, scale_A_L1 * pc1_A_L1,
                        L2_fixed, scale_B_L2 * pc1_B_L2,
                        n_layers)
                    
                    h_BA = intervene_two_layers(model, tokenizer, s00,
                        l1, scale_B_L1 * pc1_B_L1,
                        L2_fixed, scale_A_L2 * pc1_A_L2,
                        n_layers)
                    
                    h_A_only = intervene_single_layer(model, tokenizer, s00, l1, scale_A_L1 * pc1_A_L1, n_layers)
                    h_B_only = intervene_single_layer(model, tokenizer, s00, L2_fixed, scale_B_L2 * pc1_B_L2, n_layers)
                    
                    curvature = h_AB - h_BA
                    curv_norm = np.linalg.norm(curvature)
                    
                    curv_norms.append(curv_norm)
                    single_effects.append(
                        (np.linalg.norm(h_A_only - h_clean) + np.linalg.norm(h_B_only - h_clean)) / 2
                    )
                    A_effects.append(np.linalg.norm(h_A_only - h_clean))
                    B_effects.append(np.linalg.norm(h_B_only - h_clean))
                    
                except Exception as e:
                    print(f"  Failed at L1={l1}, L2={L2_fixed}, sample {i}: {e}")
                    continue
            
            if len(curv_norms) > 0:
                mean_curv = np.mean(curv_norms)
                std_curv = np.std(curv_norms) if len(curv_norms) > 1 else 0
                mean_single = np.mean(single_effects)
                mean_A_eff = np.mean(A_effects)
                mean_B_eff = np.mean(B_effects)
                rel_curv = mean_curv / (mean_single + 1e-10)
                
                reverse_data[f"L{l1}"] = {
                    'l1': l1,
                    'distance': L2_fixed - l1,
                    'curvature_norm_mean': float(mean_curv),
                    'curvature_norm_std': float(std_curv),
                    'mean_single_effect': float(mean_single),
                    'mean_A_effect': float(mean_A_eff),
                    'mean_B_effect': float(mean_B_eff),
                    'relative_curvature': float(rel_curv),
                    'n_valid': len(curv_norms),
                }
            
            if (l1_idx + 1) % 3 == 0:
                print(f"  Done {l1_idx+1}/{len(l1_values)} l1 values")
        
        pair_result = {
            'pair': pair_name,
            'feature_A': feat_A,
            'feature_B': feat_B,
            'n_samples': n_s,
            'L2_fixed': L2_fixed,
            'reverse_data': reverse_data,
        }
        
        results['pairs'][pair_name] = pair_result
        
        # Print summary
        print(f"\n  === Reverse Propagation Profile for {pair_name} (L2={L2_fixed}) ===")
        print(f"  {'L1':>4} | {'Dist':>4} | {'||[A,B]||':>10} | {'Rel.Curv':>10} | {'A_eff':>8} | {'B_eff':>8}")
        
        for key in sorted(reverse_data.keys(), key=lambda x: reverse_data[x]['l1']):
            d = reverse_data[key]
            print(f"  L{d['l1']:>2} | {d['distance']:>4} | {d['curvature_norm_mean']:>10.4f} | {d['relative_curvature']:>10.4f} | {d['mean_A_effect']:>8.4f} | {d['mean_B_effect']:>8.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=25)
    args = parser.parse_args()
    
    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]
    
    out_dir = f"results/causal_fiber/{model_key}_ccxli"
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = os.path.join(out_dir, 'run.log')
    
    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    
    log(f"=== Phase CCXLI: Reverse Propagation Experiment ===")
    log(f"Model: {cfg['name']}, n_pairs: {args.n_pairs}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")
    
    results = run_reverse_experiment(model, tokenizer, model_key, n_pairs=args.n_pairs)
    
    results['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    log(f"\n{'='*70}")
    log(f"SUMMARY: {cfg['name']}")
    log(f"{'='*70}")
    
    for pair_name, pair_res in results['pairs'].items():
        log(f"\n  Pair: {pair_name}")
        rd = pair_res.get('reverse_data', {})
        
        if not rd:
            continue
        
        distances = [d['distance'] for d in rd.values()]
        abs_curvs = [d['curvature_norm_mean'] for d in rd.values()]
        rel_curvs = [d['relative_curvature'] for d in rd.values()]
        l1_values = [d['l1'] for d in rd.values()]
        
        if len(distances) > 2:
            corr_abs = np.corrcoef(distances, abs_curvs)[0, 1]
            corr_rel = np.corrcoef(distances, rel_curvs)[0, 1]
            corr_l1_abs = np.corrcoef(l1_values, abs_curvs)[0, 1]
        else:
            corr_abs = 0
            corr_rel = 0
            corr_l1_abs = 0
        
        log(f"    Corr(dist, ||[A,B]||) = {corr_abs:.4f}")
        log(f"    Corr(dist, Rel.Curv) = {corr_rel:.4f}")
        log(f"    Corr(L1, ||[A,B]||) = {corr_l1_abs:.4f}")
        
        if corr_abs > 0.3:
            log(f"    Absolute curvature INCREASES with distance (larger gap = more curvature)")
        elif corr_abs < -0.3:
            log(f"    Absolute curvature DECREASES with distance (smaller gap = more curvature)")
        else:
            log(f"    Absolute curvature UNCORRELATED with distance")
        
        if corr_l1_abs > 0.3:
            log(f"    Deeper L1 → more curvature → deep layers are entanglement generators")
        elif corr_l1_abs < -0.3:
            log(f"    Shallower L1 → more curvature → shallow layers are entanglement generators ✓")
        else:
            log(f"    Curvature independent of L1 position")
    
    del model
    torch.cuda.empty_cache()
    
    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
