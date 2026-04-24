"""
Phase CCXXXIX: 局部曲率的层级动力学
=====================================
核心目标:
  CCXXXVIII确认了全局曲率(Rel.Curv=1.37-1.83), 但只在L/4→L/2测量。
  现在测量每对相邻层之间的"局部曲率", 分析曲率的层级分布。

数学原理:
  全局曲率 = 从L1到L2的累积曲率
  局部曲率 = 相邻层l和l+1之间的曲率
  
  如果层间变换是F_l: h_l → h_{l+1}, 那么:
  全局: F_{L2-1} ∘ ... ∘ F_{L1} ≠ F_{L1} ∘ ... ∘ F_{L2-1} (一般不可交换)
  局部: [F_l(·+ε_A), F_l(·+ε_B)] 在l层的交换子
  
  实验方法:
  对每个层对(l, l+1):
    Path AB: perturb A at layer l, perturb B at layer l+1
    Path BA: perturb B at layer l, perturb A at layer l+1
    Local Curvature[l] = ||h_last(AB) - h_last(BA)||

关键问题:
  1. 曲率在哪层最大? → 纠缠在哪层最强?
  2. 曲率的层级模式: 递增/递减/U形/均匀?
  3. 曲率与特征类型的关系?

样本设计:
  3个特征对 × 25个样本 × 1个α值(1.0) × N_layers个层对
  限制α=1.0和25样本以控制运行时间
"""

import argparse
import json
import os
import numpy as np
from datetime import datetime
from collections import defaultdict

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
    'tense_x_polarity': {
        'feature_A': 'tense',
        'feature_B': 'polarity',
        'sentences': [
            ("She likes the design", "She liked the design", "She does not like the design", "She did not like the design"),
            ("He enjoys the concert", "He enjoyed the concert", "He does not enjoy the concert", "He did not enjoy the concert"),
            ("They support the idea", "They supported the idea", "They do not support the idea", "They did not support the idea"),
            ("The method works well", "The method worked well", "The method does not work well", "The method did not work well"),
            ("She understands the concept", "She understood the concept", "She does not understand the concept", "She did not understand the concept"),
            ("He appreciates the help", "He appreciated the help", "He does not appreciate the help", "He did not appreciate the help"),
            ("They value the feedback", "They valued the feedback", "They do not value the feedback", "They did not value the feedback"),
            ("The plan makes sense", "The plan made sense", "The plan does not make sense", "The plan did not make sense"),
            ("She trusts his judgment", "She trusted his judgment", "She does not trust his judgment", "She did not trust his judgment"),
            ("He believes the story", "He believed the story", "He does not believe the story", "He did not believe the story"),
            ("They accept the offer", "They accepted the offer", "They do not accept the offer", "They did not accept the offer"),
            ("The food tastes good", "The food tasted good", "The food does not taste good", "The food did not taste good"),
            ("She wants to go home", "She wanted to go home", "She does not want to go home", "She did not want to go home"),
            ("He remembers the event", "He remembered the event", "He does not remember the event", "The event was not remembered by him"),
            ("They need more time", "They needed more time", "They do not need more time", "They did not need more time"),
            ("The system runs smoothly", "The system ran smoothly", "The system does not run smoothly", "The system did not run smoothly"),
            ("She follows the rules", "She followed the rules", "She does not follow the rules", "She did not follow the rules"),
            ("He deserves the award", "He deserved the award", "He does not deserve the award", "He did not deserve the award"),
            ("They own a house", "They owned a house", "They do not own a house", "They did not own a house"),
            ("The evidence convinces him", "The evidence convinced him", "The evidence does not convince him", "The evidence did not convince him"),
            ("She speaks German", "She spoke German", "She does not speak German", "She did not speak German"),
            ("He manages the team", "He managed the team", "He does not manage the team", "He did not manage the team"),
            ("They share the profits", "They shared the profits", "They do not share the profits", "They did not share the profits"),
            ("The bridge is safe", "The bridge was safe", "The bridge is not safe", "The bridge was not safe"),
            ("She knows the answer", "She knew the answer", "She does not know the answer", "She did not know the answer"),
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


def run_layer_curvature_experiment(model, tokenizer, model_key, n_pairs=25):
    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg['n_layers']
    d_model = cfg['d_model']
    
    # Sample layer pairs: skip L0 (embedding), sample every 2 layers to save time
    # Then fill in around peaks
    all_layer_pairs = [(l, l+1) for l in range(1, n_layers - 1)]
    
    # For efficiency, sample every 2nd layer pair initially
    sampled_pairs = all_layer_pairs[::2]
    
    results = {
        'model': cfg['name'],
        'n_layers': n_layers,
        'd_model': d_model,
        'sampled_layer_pairs': sampled_pairs,
        'pairs': {},
    }
    
    for pair_name, pair_data in FOUR_WAY_PAIRS.items():
        feat_A = pair_data['feature_A']
        feat_B = pair_data['feature_B']
        sentences = pair_data['sentences'][:n_pairs]
        n_s = len(sentences)
        
        print(f"\n--- {pair_name} ({feat_A} x {feat_B}), n={n_s} ---")
        
        # Step 1: Collect baseline hidden states at ALL layers
        print(f"  Collecting baseline hidden states...")
        all_h00 = []  # h00[layer] for each sample
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
        
        all_h00 = np.array(all_h00)  # (n_s, n_layers+1, d_model)
        all_h10 = np.array(all_h10)
        all_h01 = np.array(all_h01)
        
        # Step 2: Compute PC1 directions at each layer
        from sklearn.decomposition import PCA
        
        pc1_A_per_layer = {}
        pc1_B_per_layer = {}
        mean_dhA_norm_per_layer = {}
        mean_dhB_norm_per_layer = {}
        
        for l in range(1, n_layers):
            dh_A = all_h10[:, l, :] - all_h00[:, l, :]  # (n_s, d_model)
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
        
        # Also get last layer PC1 for projection
        dh_A_last = all_h10[:, n_layers, :] - all_h00[:, n_layers, :]
        dh_B_last = all_h01[:, n_layers, :] - all_h00[:, n_layers, :]
        pca_A_last = PCA(n_components=5)
        pca_A_last.fit(dh_A_last)
        pc1_A_last = pca_A_last.components_[0] / (np.linalg.norm(pca_A_last.components_[0]) + 1e-10)
        pca_B_last = PCA(n_components=5)
        pca_B_last.fit(dh_B_last)
        pc1_B_last = pca_B_last.components_[0] / (np.linalg.norm(pca_B_last.components_[0]) + 1e-10)
        
        # Step 3: Measure curvature at each sampled layer pair
        print(f"  Measuring curvature at {len(sampled_pairs)} layer pairs...")
        
        layer_curvature_data = {}
        
        for pair_idx, (l1, l2) in enumerate(sampled_pairs):
            pc1_A_l1 = pc1_A_per_layer[l1]
            pc1_B_l1 = pc1_B_per_layer[l1]
            pc1_A_l2 = pc1_A_per_layer[l2]
            pc1_B_l2 = pc1_B_per_layer[l2]
            
            scale_A_l1 = ALPHA * mean_dhA_norm_per_layer[l1]
            scale_B_l1 = ALPHA * mean_dhB_norm_per_layer[l1]
            scale_A_l2 = ALPHA * mean_dhA_norm_per_layer[l2]
            scale_B_l2 = ALPHA * mean_dhB_norm_per_layer[l2]
            
            curv_norms = []
            curv_A_projs = []
            curv_B_projs = []
            single_effects = []
            
            n_intervene = min(n_s, n_pairs)
            
            for i in range(n_intervene):
                s00 = sentences[i][0]
                h_clean = all_h00[i, n_layers, :]
                
                try:
                    # Path AB: A@l1, then B@l2
                    h_AB = intervene_two_layers(model, tokenizer, s00,
                        l1, scale_A_l1 * pc1_A_l1,
                        l2, scale_B_l2 * pc1_B_l2,
                        n_layers)
                    
                    # Path BA: B@l1, then A@l2
                    h_BA = intervene_two_layers(model, tokenizer, s00,
                        l1, scale_B_l1 * pc1_B_l1,
                        l2, scale_A_l2 * pc1_A_l2,
                        n_layers)
                    
                    # Single perturbation reference
                    h_A_only = intervene_single_layer(model, tokenizer, s00, l1, scale_A_l1 * pc1_A_l1, n_layers)
                    h_B_only = intervene_single_layer(model, tokenizer, s00, l1, scale_B_l1 * pc1_B_l1, n_layers)
                    
                    # Curvature
                    curvature = h_AB - h_BA
                    curv_norm = np.linalg.norm(curvature)
                    
                    curv_norms.append(curv_norm)
                    curv_A_projs.append(np.dot(curvature, pc1_A_last))
                    curv_B_projs.append(np.dot(curvature, pc1_B_last))
                    single_effects.append(
                        (np.linalg.norm(h_A_only - h_clean) + np.linalg.norm(h_B_only - h_clean)) / 2
                    )
                    
                except Exception as e:
                    print(f"  Intervention failed at L{l1}-L{l2}, sample {i}: {e}")
                    continue
            
            if len(curv_norms) > 0:
                mean_curv = np.mean(curv_norms)
                std_curv = np.std(curv_norms) if len(curv_norms) > 1 else 0
                mean_single = np.mean(single_effects)
                rel_curv = mean_curv / (mean_single + 1e-10)
                mean_curv_A = np.mean(curv_A_projs)
                mean_curv_B = np.mean(curv_B_projs)
                
                layer_curvature_data[f"L{l1}_L{l2}"] = {
                    'l1': l1, 'l2': l2,
                    'curvature_norm_mean': float(mean_curv),
                    'curvature_norm_std': float(std_curv),
                    'curvature_A_proj_mean': float(mean_curv_A),
                    'curvature_B_proj_mean': float(mean_curv_B),
                    'mean_single_effect': float(mean_single),
                    'relative_curvature': float(rel_curv),
                    'n_valid': len(curv_norms),
                }
            
            if (pair_idx + 1) % 5 == 0:
                print(f"  Done {pair_idx+1}/{len(sampled_pairs)} layer pairs")
        
        # Also compute PC1 cos(A,B) per layer for reference
        cos_AB_per_layer = {}
        for l in range(1, n_layers):
            cos_AB_per_layer[str(l)] = float(np.dot(pc1_A_per_layer[l], pc1_B_per_layer[l]))
        
        pair_result = {
            'pair': pair_name,
            'feature_A': feat_A,
            'feature_B': feat_B,
            'n_samples': n_s,
            'layer_curvature': layer_curvature_data,
            'cos_AB_per_layer': cos_AB_per_layer,
        }
        
        results['pairs'][pair_name] = pair_result
        
        # Print summary for this pair
        print(f"\n  === Layer Curvature Profile for {pair_name} ===")
        print(f"  {'Layer':>8} | {'||[A,B]||':>10} | {'Rel.Curv':>10} | {'cos(A,B)':>10}")
        
        for key in sorted(layer_curvature_data.keys(), key=lambda x: layer_curvature_data[x]['l1']):
            d = layer_curvature_data[key]
            l1 = d['l1']
            cos_ab = cos_AB_per_layer.get(str(l1), 0)
            print(f"  L{l1:>2}-L{l1+1:<2} | {d['curvature_norm_mean']:>10.4f} | {d['relative_curvature']:>10.4f} | {cos_ab:>10.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=25)
    args = parser.parse_args()
    
    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]
    
    out_dir = f"results/causal_fiber/{model_key}_ccxxxix"
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = os.path.join(out_dir, 'run.log')
    
    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    
    log(f"=== Phase CCXXXIX: Layer-wise Local Curvature Dynamics ===")
    log(f"Model: {cfg['name']}, n_pairs: {args.n_pairs}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")
    
    results = run_layer_curvature_experiment(model, tokenizer, model_key, n_pairs=args.n_pairs)
    
    results['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    log(f"\n{'='*70}")
    log(f"SUMMARY: {cfg['name']}")
    log(f"{'='*70}")
    
    for pair_name, pair_res in results['pairs'].items():
        log(f"\n  Pair: {pair_name}")
        lc = pair_res.get('layer_curvature', {})
        
        if not lc:
            continue
        
        # Find peak curvature layer
        max_rel_curv = 0
        peak_layer = -1
        all_rel_curvs = []
        
        for key, d in lc.items():
            rc = d['relative_curvature']
            all_rel_curvs.append(rc)
            if rc > max_rel_curv:
                max_rel_curv = rc
                peak_layer = d['l1']
        
        avg_rel_curv = np.mean(all_rel_curvs) if all_rel_curvs else 0
        
        # Categorize layer curvature profile
        early_rel = np.mean([d['relative_curvature'] for d in lc.values() if d['l1'] < cfg['n_layers'] // 3])
        mid_rel = np.mean([d['relative_curvature'] for d in lc.values() if cfg['n_layers'] // 3 <= d['l1'] < 2 * cfg['n_layers'] // 3])
        late_rel = np.mean([d['relative_curvature'] for d in lc.values() if d['l1'] >= 2 * cfg['n_layers'] // 3])
        
        log(f"    Avg Rel.Curv: {avg_rel_curv:.4f}")
        log(f"    Peak at Layer: {peak_layer} (Rel.Curv={max_rel_curv:.4f})")
        log(f"    Early/Mid/Late: {early_rel:.4f} / {mid_rel:.4f} / {late_rel:.4f}")
        
        if mid_rel > early_rel and mid_rel > late_rel:
            log(f"    Profile: INVERTED-U (peak at middle layers)")
        elif late_rel > mid_rel and late_rel > early_rel:
            log(f"    Profile: INCREASING (curvature grows deeper)")
        elif early_rel > mid_rel and early_rel > late_rel:
            log(f"    Profile: DECREASING (curvature strongest early)")
        else:
            log(f"    Profile: MIXED")
    
    del model
    torch.cuda.empty_cache()
    
    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
