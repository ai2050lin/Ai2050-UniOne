"""
Phase CCXLIV: Local Perturbation Normalization — The Decisive Test
==================================================================
User's critical insight:
  C_norm = ||[A,B]|| / (A_eff * B_eff) still has "input-side Jacobian" bias.
  
  A_eff ≈ J_{L1->out} * ||delta_L1||
  But the commutator also involves J_{in->L1} (input-side gain).
  
  Shallow L1: J_{in->L1} small, J_{L1->out} small → A_eff small
  Deep L1:    J_{in->L1} large, J_{L1->out} large → A_eff large
  
  C_norm = ||H|| / J_{L1->out}  (where H is the interaction tensor)
  So C_norm varies with L1 even if ||H|| is constant!

Decisive test: Fix ||delta_L1|| = constant (same LOCAL perturbation at each L1)
  α_local(L1) = target_norm / mean_dhA_norm(L1)
  
  Then measure:
  1. ||[A,B]||_fix — raw commutator with fixed local perturbation
  2. C_norm_fix = ||[A,B]||_fix / (A_eff_fix * B_eff)
  3. H_intrinsic = ||[A,B]|| / (||delta_L1|| * B_eff) — fully normalized
  
  If Corr(L1, H_intrinsic) < 0 → shallow layers genuinely more non-commutative
  If Corr(L1, H_intrinsic) ≈ 0 → "shallow seeding" was propagation artifact
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


def run_local_norm_experiment(model, tokenizer, model_key, n_pairs=20):
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

        # Step 2: Compute PCA directions and local perturbation norms
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

        # Step 3: Report local perturbation norms (this reveals J_{in->L1}!)
        print(f"\n  === Local Perturbation Norms (||delta_L1|| at alpha=1) ===")
        print(f"  This reveals J_{{in->L1}} profile!")
        print(f"  {'L1':>4} {'||delta_A||':>12} {'||delta_B||':>12} {'Position':>8}")

        for l1 in l1_positions:
            if l1 in mean_dhA_norm_per_layer:
                pos = f"L/{n_layers/l1:.1f}" if l1 > 0 else "N/A"
                print(f"  L{l1:>2} {mean_dhA_norm_per_layer[l1]:>12.4f} {mean_dhB_norm_per_layer[l1]:>12.4f} {pos:>8}")

        if L2_fixed in mean_dhA_norm_per_layer:
            print(f"  L{L2_fixed:>2} {mean_dhA_norm_per_layer[L2_fixed]:>12.4f} {mean_dhB_norm_per_layer[L2_fixed]:>12.4f} {'L2(fixed)':>8}")

        # Step 4: Compute alpha for fixed local perturbation
        # target_norm = mean of ||delta_L1|| across L1 positions
        l1_norms = [mean_dhA_norm_per_layer[l] for l in l1_positions if l in mean_dhA_norm_per_layer]
        target_local_norm = np.mean(l1_norms)
        
        print(f"\n  Target local perturbation norm: {target_local_norm:.4f}")
        
        # alpha_local(L1) = target / mean_dhA_norm(L1)
        alphas_local = {}
        for l1 in l1_positions:
            if l1 in mean_dhA_norm_per_layer and mean_dhA_norm_per_layer[l1] > 1e-6:
                alphas_local[l1] = target_local_norm / mean_dhA_norm_per_layer[l1]
                alphas_local[l1] = max(0.1, min(5.0, alphas_local[l1]))
            else:
                alphas_local[l1] = 1.0
            print(f"    L1={l1}: ||delta||={mean_dhA_norm_per_layer.get(l1,0):.4f}, alpha_local={alphas_local[l1]:.3f}")

        # Step 5: Measure all metrics at each L1
        # Both NATURAL (alpha=1) and FIXED LOCAL (alpha=alpha_local)
        print(f"\n  Measuring curvature (natural + fixed local)...")

        layer_data = {}

        for l1 in l1_positions:
            if l1 not in pc1_A_per_layer:
                continue

            alpha_nat = 1.0
            alpha_fix = alphas_local[l1]

            # Natural perturbation scales
            scale_A_L1_nat = alpha_nat * mean_dhA_norm_per_layer[l1]
            scale_A_L1_fix = alpha_fix * mean_dhA_norm_per_layer[l1]  # = target_local_norm by construction

            # B perturbation always at L2 with alpha=1
            scale_B_L2 = 1.0 * mean_dhB_norm_per_layer[L2_fixed]

            # Local perturbation norms (what we actually apply)
            local_norm_nat = scale_A_L1_nat  # ||delta_L1|| for natural
            local_norm_fix = scale_A_L1_fix  # ||delta_L1|| for fixed (should = target)

            results_nat = {'curv': [], 'A_eff': [], 'B_eff': []}
            results_fix = {'curv': [], 'A_eff': [], 'B_eff': []}

            for i in range(n_s):
                s00 = sentences[i][0]
                h_clean = all_h00[i, n_layers, :]

                try:
                    # === NATURAL (alpha=1) ===
                    h_AB_nat = intervene_two_layers(model, tokenizer, s00,
                        l1, scale_A_L1_nat * pc1_A_per_layer[l1],
                        L2_fixed, scale_B_L2 * pc1_B_per_layer[L2_fixed],
                        n_layers)

                    h_BA_nat = intervene_two_layers(model, tokenizer, s00,
                        l1, scale_B_L2 * pc1_B_per_layer[l1],  # B at L1
                        L2_fixed, scale_A_L1_nat * pc1_A_per_layer[L2_fixed],  # A at L2 -- wait, this is wrong
                        n_layers)

                    # Actually the AB/BA swap should be:
                    # AB = A(tense) at L1, B(question) at L2
                    # BA = B(question) at L1, A(tense) at L2
                    # But A and B are different features with different PCA directions

                    # Let me re-do this properly:
                    # A = tense perturbation, B = question perturbation
                    # AB: apply tense at L1, question at L2
                    # BA: apply question at L1, tense at L2

                    h_AB_nat = intervene_two_layers(model, tokenizer, s00,
                        l1, scale_A_L1_nat * pc1_A_per_layer[l1],  # A (tense) at L1
                        L2_fixed, scale_B_L2 * pc1_B_per_layer[L2_fixed],  # B (question) at L2
                        n_layers)

                    h_BA_nat = intervene_two_layers(model, tokenizer, s00,
                        l1, scale_B_L2 * pc1_B_per_layer[l1],  # B (question) at L1
                        L2_fixed, scale_A_L1_nat * pc1_A_per_layer[L2_fixed],  # A (tense) at L2
                        n_layers)

                    h_A_only_nat = intervene_single_layer(model, tokenizer, s00, l1,
                        scale_A_L1_nat * pc1_A_per_layer[l1], n_layers)
                    h_B_only_nat = intervene_single_layer(model, tokenizer, s00, L2_fixed,
                        scale_B_L2 * pc1_B_per_layer[L2_fixed], n_layers)

                    curv_nat = np.linalg.norm(h_AB_nat - h_BA_nat)
                    a_eff_nat = np.linalg.norm(h_A_only_nat - h_clean)
                    b_eff_nat = np.linalg.norm(h_B_only_nat - h_clean)

                    results_nat['curv'].append(curv_nat)
                    results_nat['A_eff'].append(a_eff_nat)
                    results_nat['B_eff'].append(b_eff_nat)

                    # === FIXED LOCAL (alpha=alpha_fix) ===
                    h_AB_fix = intervene_two_layers(model, tokenizer, s00,
                        l1, scale_A_L1_fix * pc1_A_per_layer[l1],  # A (tense) at L1 with fixed norm
                        L2_fixed, scale_B_L2 * pc1_B_per_layer[L2_fixed],  # B (question) at L2
                        n_layers)

                    h_BA_fix = intervene_two_layers(model, tokenizer, s00,
                        l1, scale_B_L2 * pc1_B_per_layer[l1],  # B (question) at L1 -- same B regardless
                        L2_fixed, scale_A_L1_fix * pc1_A_per_layer[L2_fixed],  # A (tense) at L2
                        n_layers)

                    h_A_only_fix = intervene_single_layer(model, tokenizer, s00, l1,
                        scale_A_L1_fix * pc1_A_per_layer[l1], n_layers)

                    curv_fix = np.linalg.norm(h_AB_fix - h_BA_fix)
                    a_eff_fix = np.linalg.norm(h_A_only_fix - h_clean)

                    results_fix['curv'].append(curv_fix)
                    results_fix['A_eff'].append(a_eff_fix)
                    results_fix['B_eff'].append(b_eff_nat)  # B_eff doesn't change

                except Exception as e:
                    print(f"  Failed at L1={l1}, sample {i}: {e}")
                    continue

            if len(results_nat['curv']) > 0:
                mean_curv_nat = np.mean(results_nat['curv'])
                mean_A_eff_nat = np.mean(results_nat['A_eff'])
                mean_B_eff_nat = np.mean(results_nat['B_eff'])

                mean_curv_fix = np.mean(results_fix['curv'])
                mean_A_eff_fix = np.mean(results_fix['A_eff'])
                mean_B_eff_fix = np.mean(results_fix['B_eff'])

                # Key metrics
                # C_norm = ||[A,B]|| / (A_eff * B_eff) -- normalizes by output effects
                c_norm_nat = mean_curv_nat / (mean_A_eff_nat * mean_B_eff_nat + 1e-10)
                c_norm_fix = mean_curv_fix / (mean_A_eff_fix * mean_B_eff_fix + 1e-10)

                # H_intrinsic = ||[A,B]|| / (||delta_L1|| * B_eff) -- normalizes by local perturbation
                # This removes BOTH input-side and output-side Jacobian from numerator
                # If this still varies with L1, it's genuine intrinsic non-commutativity
                h_intrinsic_nat = mean_curv_nat / (local_norm_nat * mean_B_eff_nat + 1e-10)
                h_intrinsic_fix = mean_curv_fix / (local_norm_fix * mean_B_eff_fix + 1e-10)

                # Forward gain estimate: J_{L1->out} ≈ A_eff / ||delta_L1||
                j_forward_nat = mean_A_eff_nat / (local_norm_nat + 1e-10)
                j_forward_fix = mean_A_eff_fix / (local_norm_fix + 1e-10)

                layer_data[f"L{l1}"] = {
                    'l1': l1,
                    'distance': L2_fixed - l1,
                    # Natural perturbation (alpha=1)
                    'alpha_nat': 1.0,
                    'local_norm_nat': float(local_norm_nat),
                    'A_eff_nat': float(mean_A_eff_nat),
                    'B_eff_nat': float(mean_B_eff_nat),
                    'curv_norm_nat': float(mean_curv_nat),
                    'c_norm_nat': float(c_norm_nat),
                    'h_intrinsic_nat': float(h_intrinsic_nat),
                    'j_forward_nat': float(j_forward_nat),
                    # Fixed local perturbation
                    'alpha_fix': float(alpha_fix),
                    'local_norm_fix': float(local_norm_fix),
                    'A_eff_fix': float(mean_A_eff_fix),
                    'B_eff_fix': float(mean_B_eff_fix),
                    'curv_norm_fix': float(mean_curv_fix),
                    'c_norm_fix': float(c_norm_fix),
                    'h_intrinsic_fix': float(h_intrinsic_fix),
                    'j_forward_fix': float(j_forward_fix),
                    'n_valid': len(results_nat['curv']),
                }

        # Step 6: Print comprehensive results
        print(f"\n  {'='*90}")
        print(f"  COMPREHENSIVE RESULTS: {pair_name}")
        print(f"  {'='*90}")

        print(f"\n  --- Natural Perturbation (alpha=1) ---")
        print(f"  {'L1':>4} {'||dL1||':>8} {'A_eff':>8} {'B_eff':>8} {'||[A,B]||':>10} {'C_norm':>10} {'H_intr':>10} {'J_fwd':>8}")
        for key in sorted(layer_data.keys(), key=lambda x: layer_data[x]['l1']):
            d = layer_data[key]
            print(f"  L{d['l1']:>2} {d['local_norm_nat']:>8.3f} {d['A_eff_nat']:>8.2f} {d['B_eff_nat']:>8.2f} "
                  f"{d['curv_norm_nat']:>10.2f} {d['c_norm_nat']:>10.6f} {d['h_intrinsic_nat']:>10.6f} {d['j_forward_nat']:>8.3f}")

        print(f"\n  --- Fixed Local Perturbation (||delta_L1|| = {target_local_norm:.3f}) ---")
        print(f"  {'L1':>4} {'alpha':>6} {'||dL1||':>8} {'A_eff':>8} {'B_eff':>8} {'||[A,B]||':>10} {'C_norm':>10} {'H_intr':>10} {'J_fwd':>8}")
        for key in sorted(layer_data.keys(), key=lambda x: layer_data[x]['l1']):
            d = layer_data[key]
            print(f"  L{d['l1']:>2} {d['alpha_fix']:>6.3f} {d['local_norm_fix']:>8.3f} {d['A_eff_fix']:>8.2f} {d['B_eff_fix']:>8.2f} "
                  f"{d['curv_norm_fix']:>10.2f} {d['c_norm_fix']:>10.6f} {d['h_intrinsic_fix']:>10.6f} {d['j_forward_fix']:>8.3f}")

        # Step 7: Correlation analysis
        l1_vals = [d['l1'] for d in layer_data.values()]

        print(f"\n  === CORRELATION ANALYSIS ===")
        print(f"  (Negative = shallow layers have higher values)")

        metrics = {
            '||delta_L1||_nat': [d['local_norm_nat'] for d in layer_data.values()],
            'A_eff_nat': [d['A_eff_nat'] for d in layer_data.values()],
            'J_forward_nat': [d['j_forward_nat'] for d in layer_data.values()],
            '||[A,B]||_nat': [d['curv_norm_nat'] for d in layer_data.values()],
            'C_norm_nat': [d['c_norm_nat'] for d in layer_data.values()],
            'H_intrinsic_nat': [d['h_intrinsic_nat'] for d in layer_data.values()],
            '---': None,
            'A_eff_fix': [d['A_eff_fix'] for d in layer_data.values()],
            'J_forward_fix': [d['j_forward_fix'] for d in layer_data.values()],
            '||[A,B]||_fix': [d['curv_norm_fix'] for d in layer_data.values()],
            'C_norm_fix': [d['c_norm_fix'] for d in layer_data.values()],
            'H_intrinsic_fix': [d['h_intrinsic_fix'] for d in layer_data.values()],
        }

        for metric_name, metric_vals in metrics.items():
            if metric_vals is None:
                print(f"  ---")
                continue
            if len(l1_vals) > 2:
                corr = np.corrcoef(l1_vals, metric_vals)[0, 1]
                print(f"  Corr(L1, {metric_name:>20}) = {corr:>7.3f}", end="")
                if 'H_intrinsic' in metric_name:
                    if corr < -0.3:
                        print(f"  <-- SHALLOW INTRINSIC NON-COMMUTATIVITY STRONGER")
                    elif abs(corr) < 0.3:
                        print(f"  <-- NO L1 DEPENDENCE (propagation artifact)")
                    else:
                        print(f"  <-- DEEP INTRINSIC NON-COMMUTATIVITY STRONGER")
                else:
                    print()

        # Step 8: The decisive comparison
        print(f"\n  === THE DECISIVE COMPARISON ===")
        h_nat = [d['h_intrinsic_nat'] for d in layer_data.values()]
        h_fix = [d['h_intrinsic_fix'] for d in layer_data.values()]

        if len(l1_vals) > 2:
            corr_h_nat = np.corrcoef(l1_vals, h_nat)[0, 1]
            corr_h_fix = np.corrcoef(l1_vals, h_fix)[0, 1]

            print(f"  H_intrinsic_nat: Corr(L1, H) = {corr_h_nat:.3f}")
            print(f"  H_intrinsic_fix: Corr(L1, H) = {corr_h_fix:.3f}")
            print()

            if abs(corr_h_fix) < 0.3:
                print(f"  VERDICT: 'Shallow seeding' is PROPAGATION ARTIFACT")
                print(f"    After fixing local perturbation, intrinsic non-commutativity")
                print(f"    does NOT vary with L1. The apparent 'shallow advantage' was")
                print(f"    entirely due to J_{{in->L1}} and J_{{L1->out}} variation.")
            elif corr_h_fix < -0.3:
                print(f"  VERDICT: 'Shallow seeding' is GENUINE")
                print(f"    After fixing local perturbation, intrinsic non-commutativity")
                print(f"    is STILL stronger at shallow layers. The 'shallow advantage'")
                print(f"    is not just a propagation artifact.")
            else:
                print(f"  VERDICT: UNEXPECTED - deep layers have stronger intrinsic non-commutativity")

            # Also check: does J_forward vary with L1?
            j_nat = [d['j_forward_nat'] for d in layer_data.values()]
            corr_j = np.corrcoef(l1_vals, j_nat)[0, 1] if len(l1_vals) > 2 else 0
            print(f"\n  J_forward (A_eff / ||delta_L1||): Corr(L1, J) = {corr_j:.3f}")
            if corr_j > 0.3:
                print(f"    Forward gain INCREASES with depth → deep perturbations propagate more efficiently")
            elif corr_j < -0.3:
                print(f"    Forward gain DECREASES with depth → shallow perturbations propagate more efficiently")
            else:
                print(f"    Forward gain roughly constant across layers")

        pair_result = {
            'pair': pair_name,
            'feature_A': feat_A,
            'feature_B': feat_B,
            'n_samples': n_s,
            'L2_fixed': L2_fixed,
            'target_local_norm': float(target_local_norm),
            'alphas_local': {str(k): float(v) for k, v in alphas_local.items()},
            'mean_dhA_norm_per_layer': {str(k): float(v) for k, v in mean_dhA_norm_per_layer.items()},
            'mean_dhB_norm_per_layer': {str(k): float(v) for k, v in mean_dhB_norm_per_layer.items()},
            'layer_data': layer_data,
        }

        results['pairs'][pair_name] = pair_result

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=20)
    args = parser.parse_args()

    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]

    out_dir = f"results/causal_fiber/{model_key}_ccxliv"
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, 'run.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f"=== Phase CCXLIV: Local Perturbation Normalization ===")
    log(f"Model: {cfg['name']}, n_pairs: {args.n_pairs}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")

    results = run_local_norm_experiment(model, tokenizer, model_key, n_pairs=args.n_pairs)

    results['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    log(f"\n{'='*70}")
    log(f"SUMMARY: {cfg['name']}")
    log(f"{'='*70}")

    for pair_name, pair_res in results['pairs'].items():
        log(f"\n  Pair: {pair_name}")
        ld = pair_res.get('layer_data', {})
        target = pair_res.get('target_local_norm', 0)
        alphas = pair_res.get('alphas_local', {})

        log(f"  Target ||delta_L1|| = {target:.4f}")

        log(f"\n  Natural perturbation:")
        for key in sorted(ld.keys(), key=lambda x: ld[x]['l1']):
            d = ld[key]
            log(f"    L1={d['l1']}: ||dL1||={d['local_norm_nat']:.4f}, A_eff={d['A_eff_nat']:.2f}, "
                f"||[A,B]||={d['curv_norm_nat']:.2f}, C_norm={d['c_norm_nat']:.6f}, H_intr={d['h_intrinsic_nat']:.6f}")

        log(f"\n  Fixed local perturbation:")
        for key in sorted(ld.keys(), key=lambda x: ld[x]['l1']):
            d = ld[key]
            log(f"    L1={d['l1']}: alpha={d['alpha_fix']:.3f}, ||dL1||={d['local_norm_fix']:.4f}, A_eff={d['A_eff_fix']:.2f}, "
                f"||[A,B]||={d['curv_norm_fix']:.2f}, C_norm={d['c_norm_fix']:.6f}, H_intr={d['h_intrinsic_fix']:.6f}")

        l1_vals = [d['l1'] for d in ld.values()]
        if len(l1_vals) > 2:
            h_nat = [d['h_intrinsic_nat'] for d in ld.values()]
            h_fix = [d['h_intrinsic_fix'] for d in ld.values()]
            c_nat = [d['c_norm_nat'] for d in ld.values()]
            c_fix = [d['c_norm_fix'] for d in ld.values()]
            j_fwd = [d['j_forward_nat'] for d in ld.values()]

            corr_h_nat = np.corrcoef(l1_vals, h_nat)[0, 1]
            corr_h_fix = np.corrcoef(l1_vals, h_fix)[0, 1]
            corr_c_nat = np.corrcoef(l1_vals, c_nat)[0, 1]
            corr_c_fix = np.corrcoef(l1_vals, c_fix)[0, 1]
            corr_j = np.corrcoef(l1_vals, j_fwd)[0, 1]

            log(f"\n  KEY CORRELATIONS:")
            log(f"    Corr(L1, C_norm_nat)      = {corr_c_nat:.3f}")
            log(f"    Corr(L1, C_norm_fix)      = {corr_c_fix:.3f}")
            log(f"    Corr(L1, H_intrinsic_nat) = {corr_h_nat:.3f}")
            log(f"    Corr(L1, H_intrinsic_fix) = {corr_h_fix:.3f}")
            log(f"    Corr(L1, J_forward)       = {corr_j:.3f}")

            if abs(corr_h_fix) < 0.3:
                log(f"  VERDICT: 'Shallow seeding' is PROPAGATION ARTIFACT")
            elif corr_h_fix < -0.3:
                log(f"  VERDICT: 'Shallow seeding' is GENUINE")
            else:
                log(f"  VERDICT: Unexpected pattern")

    del model
    torch.cuda.empty_cache()

    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
