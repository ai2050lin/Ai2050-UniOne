"""
Phase CCXXXVIII: 交换律实验 — 测量曲率!
=========================================
核心目标:
  CCXXXVII确认了辛纠缠(Purity≈0.5-0.99), 但辛守恒不成立。
  现在测量"曲率" — 纤维丛联络的关键特征。

数学原理:
  在微分几何中, 联络∇的曲率 R = [∇_A, ∇_B] = ∇_A∇_B - ∇_B∇_A
  如果 R ≠ 0 → 联络有曲率 → 非平凡的纤维丛结构
  
  在神经网络中:
  - ∇_A = "对隐藏状态做A方向的perturbation后传播"
  - ∇_B = "对隐藏状态做B方向的perturbation后传播"
  - ∇_A∇_B ≠ ∇_B∇_A → "先B后A" ≠ "先A后B"

实验设计:
  在两个不同层做顺序perturbation:
    Path AB: perturb A at L1, then perturb B at L2 (L1 < L2)
    Path BA: perturb B at L1, then perturb A at L2 (L1 < L2)
  
  为什么顺序重要?
  - L1的perturbation改变了L1+1到L2的传播路径
  - 所以L2的隐藏状态依赖于L1做了什么perturbation
  - 在L2添加相同的perturbation, 但基线不同 → 最终输出不同
  
  Curvature = h_output(AB) - h_output(BA)
  如果 |curvature| > 0 → 交换律不成立 → 曲率存在 → 纤维丛结构!

关键假设:
  H1_flat: [∇_A, ∇_B] = 0 → 顺序无关 → 平坦联络 → 独立编码
  H2_curved: [∇_A, ∇_B] ≠ 0 → 顺序重要 → 有曲率 → 纠缠的联络结构
  H3_quantify: 曲率大小与特征纠缠程度正相关
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
            ("She wears a dress", "She wore a dress", "Does she wear a dress", "Did she wear a dress"),
            ("They dance at parties", "They danced at parties", "Do they dance at parties", "Did they dance at parties"),
            ("He catches the ball", "He caught the ball", "Does he catch the ball", "Did he catch the ball"),
            ("The baby cries often", "The baby cried often", "Does the baby cry often", "Did the baby cry often"),
            ("She opens the window", "She opened the window", "Does she open the window", "Did she open the window"),
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
            ("He remembers the event", "He remembered the event", "He does not remember the event", "He did not remember the event"),
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
            ("He likes the movie", "He liked the movie", "He does not like the movie", "He did not like the movie"),
            ("They agree with the plan", "They agreed with the plan", "They do not agree with the plan", "They did not agree with the plan"),
            ("The car works fine", "The car worked fine", "The car does not work fine", "The car did not work fine"),
            ("She enjoys the book", "She enjoyed the book", "She does not enjoy the book", "She did not enjoy the book"),
            ("He understands the problem", "He understood the problem", "He does not understand the problem", "He did not understand the problem"),
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
            ("She bakes the cake", "The cake is baked by her", "Does she bake the cake", "Is the cake baked by her"),
            ("He breaks the window", "The window is broken by him", "Does he break the window", "Is the window broken by him"),
            ("They cut the grass", "The grass is cut by them", "Do they cut the grass", "Is the grass cut by them"),
            ("The judge dismisses the case", "The case is dismissed by the judge", "Does the judge dismiss the case", "Is the case dismissed by the judge"),
            ("She translates the document", "The document is translated by her", "Does she translate the document", "Is the document translated by her"),
        ]
    },
}

ALPHA_VALUES = [1.0, 2.0]


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
    """Get the transformer layer module for hook registration."""
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
    """
    Run model with TWO sequential interventions:
    At layer1: add pert1_vec to hidden state at last valid token
    At layer2: add pert2_vec to hidden state at last valid token
    (layer1 < layer2, so pert1 happens first, then pert2)
    
    Returns the last layer hidden state.
    """
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
    """Run model with single layer intervention."""
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


def run_commutator_experiment(model, tokenizer, model_key, n_pairs=30):
    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg['n_layers']
    d_model = cfg['d_model']
    
    # Two intervention layers: L/4 and L/2
    L1 = n_layers // 4      # early-mid layer
    L2 = n_layers // 2      # mid layer
    
    results = {
        'model': cfg['name'],
        'n_layers': n_layers,
        'd_model': d_model,
        'L1': L1,
        'L2': L2,
        'pairs': {},
    }
    
    for pair_name, pair_data in FOUR_WAY_PAIRS.items():
        feat_A = pair_data['feature_A']
        feat_B = pair_data['feature_B']
        sentences = pair_data['sentences'][:n_pairs]
        n_s = len(sentences)
        
        print(f"\n--- {pair_name} ({feat_A} x {feat_B}), n={n_s} ---")
        print(f"  L1={L1}, L2={L2}")
        
        # Step 1: Collect baseline hidden states at L1, L2, last layer
        h_A0B0_last = []
        h_A0B0_L1 = []
        h_A0B0_L2 = []
        
        # Also collect dh_A and dh_B at L1 and L2 for PC1 computation
        h_A1B0_L1 = []; h_A1B0_L2 = []; h_A1B0_last = []
        h_A0B1_L1 = []; h_A0B1_L2 = []; h_A0B1_last = []
        
        for i, (s00, s10, s01, s11) in enumerate(sentences):
            h00 = get_all_hidden_states(model, tokenizer, s00, n_layers)
            h10 = get_all_hidden_states(model, tokenizer, s10, n_layers)
            h01 = get_all_hidden_states(model, tokenizer, s01, n_layers)
            
            h_A0B0_last.append(h00[n_layers])
            h_A0B0_L1.append(h00[L1])
            h_A0B0_L2.append(h00[L2])
            h_A1B0_L1.append(h10[L1])
            h_A1B0_L2.append(h10[L2])
            h_A1B0_last.append(h10[n_layers])
            h_A0B1_L1.append(h01[L1])
            h_A0B1_L2.append(h01[L2])
            h_A0B1_last.append(h01[n_layers])
            
            if (i + 1) % 10 == 0:
                print(f"  Collected {i+1}/{n_s} baseline hidden states")
        
        h_A0B0_last = np.array(h_A0B0_last)
        h_A0B0_L1 = np.array(h_A0B0_L1)
        h_A0B0_L2 = np.array(h_A0B0_L2)
        h_A1B0_L1 = np.array(h_A1B0_L1)
        h_A1B0_L2 = np.array(h_A1B0_L2)
        h_A1B0_last = np.array(h_A1B0_last)
        h_A0B1_L1 = np.array(h_A0B1_L1)
        h_A0B1_L2 = np.array(h_A0B1_L2)
        h_A0B1_last = np.array(h_A0B1_last)
        
        # Step 2: Compute PC1 directions at L1, L2, and last layer
        from sklearn.decomposition import PCA
        
        pc1_dirs = {}
        for layer_name, dh_A_data, dh_B_data in [
            ('L1', h_A1B0_L1 - h_A0B0_L1, h_A0B1_L1 - h_A0B0_L1),
            ('L2', h_A1B0_L2 - h_A0B0_L2, h_A0B1_L2 - h_A0B0_L2),
            ('last', h_A1B0_last - h_A0B0_last, h_A0B1_last - h_A0B0_last),
        ]:
            pca_A = PCA(n_components=5)
            pca_A.fit(dh_A_data)
            pc1_A = pca_A.components_[0] / (np.linalg.norm(pca_A.components_[0]) + 1e-10)
            
            pca_B = PCA(n_components=5)
            pca_B.fit(dh_B_data)
            pc1_B = pca_B.components_[0] / (np.linalg.norm(pca_B.components_[0]) + 1e-10)
            
            pc1_dirs[layer_name] = {
                'pc1_A': pc1_A, 'pc1_B': pc1_B,
                'mean_dhA_norm': float(np.mean(np.linalg.norm(dh_A_data, axis=1))),
                'mean_dhB_norm': float(np.mean(np.linalg.norm(dh_B_data, axis=1))),
                'cos_pc1A_pc1B': float(np.dot(pc1_A, pc1_B)),
                'pc1_A_var': float(pca_A.explained_variance_ratio_[0]),
                'pc1_B_var': float(pca_B.explained_variance_ratio_[0]),
            }
        
        print(f"  PC1 at L1: cos(A,B)={pc1_dirs['L1']['cos_pc1A_pc1B']:.4f}")
        print(f"  PC1 at L2: cos(A,B)={pc1_dirs['L2']['cos_pc1A_pc1B']:.4f}")
        print(f"  PC1 at last: cos(A,B)={pc1_dirs['last']['cos_pc1A_pc1B']:.4f}")
        
        # Step 3: Commutator experiment
        pc1_A_L1 = pc1_dirs['L1']['pc1_A']
        pc1_B_L1 = pc1_dirs['L1']['pc1_B']
        pc1_A_L2 = pc1_dirs['L2']['pc1_A']
        pc1_B_L2 = pc1_dirs['L2']['pc1_B']
        pc1_A_last = pc1_dirs['last']['pc1_A']
        pc1_B_last = pc1_dirs['last']['pc1_B']
        
        mean_dhA_L1 = pc1_dirs['L1']['mean_dhA_norm']
        mean_dhB_L1 = pc1_dirs['L1']['mean_dhB_norm']
        mean_dhA_L2 = pc1_dirs['L2']['mean_dhA_norm']
        mean_dhB_L2 = pc1_dirs['L2']['mean_dhB_norm']
        
        commutator_results = {alpha: {
            # Two-layer paths
            'AB_last': [], 'BA_last': [],           # full output vectors
            'AB_A_proj': [], 'AB_B_proj': [],       # projections onto PC1_A, PC1_B at last
            'BA_A_proj': [], 'BA_B_proj': [],
            # Single perturbation baselines
            'A_only_L1': [], 'B_only_L1': [],       # single pert at L1
            'A_only_L2': [], 'B_only_L2': [],       # single pert at L2
            # Clean baseline
            'clean_last': [],
            # Curvature metrics
            'curvature_norm': [],                     # ||AB - BA||
            'curvature_A_proj': [],                   # (AB-BA) · PC1_A
            'curvature_B_proj': [],                   # (AB-BA) · PC1_B
            'curvature_cos_A': [],                    # cos(AB-BA, PC1_A)
            'curvature_cos_B': [],                    # cos(AB-BA, PC1_B)
        } for alpha in ALPHA_VALUES}
        
        n_intervene = min(n_s, n_pairs)
        
        for i in range(n_intervene):
            s00 = sentences[i][0]
            h_clean = h_A0B0_last[i]
            
            for alpha in ALPHA_VALUES:
                scale_A_L1 = alpha * mean_dhA_L1
                scale_B_L1 = alpha * mean_dhB_L1
                scale_A_L2 = alpha * mean_dhA_L2
                scale_B_L2 = alpha * mean_dhB_L2
                
                try:
                    # Path AB: A@L1, then B@L2
                    h_AB = intervene_two_layers(model, tokenizer, s00,
                        L1, scale_A_L1 * pc1_A_L1,
                        L2, scale_B_L2 * pc1_B_L2,
                        n_layers)
                    
                    # Path BA: B@L1, then A@L2
                    h_BA = intervene_two_layers(model, tokenizer, s00,
                        L1, scale_B_L1 * pc1_B_L1,
                        L2, scale_A_L2 * pc1_A_L2,
                        n_layers)
                    
                    # Single perturbation baselines
                    h_A_only_L1 = intervene_single_layer(model, tokenizer, s00, L1, scale_A_L1 * pc1_A_L1, n_layers)
                    h_B_only_L1 = intervene_single_layer(model, tokenizer, s00, L1, scale_B_L1 * pc1_B_L1, n_layers)
                    h_A_only_L2 = intervene_single_layer(model, tokenizer, s00, L2, scale_A_L2 * pc1_A_L2, n_layers)
                    h_B_only_L2 = intervene_single_layer(model, tokenizer, s00, L2, scale_B_L2 * pc1_B_L2, n_layers)
                    
                    # Compute curvature = AB - BA
                    curvature = h_AB - h_BA
                    curv_norm = np.linalg.norm(curvature)
                    
                    commutator_results[alpha]['AB_last'].append(h_AB)
                    commutator_results[alpha]['BA_last'].append(h_BA)
                    commutator_results[alpha]['AB_A_proj'].append(np.dot(h_AB - h_clean, pc1_A_last))
                    commutator_results[alpha]['AB_B_proj'].append(np.dot(h_AB - h_clean, pc1_B_last))
                    commutator_results[alpha]['BA_A_proj'].append(np.dot(h_BA - h_clean, pc1_A_last))
                    commutator_results[alpha]['BA_B_proj'].append(np.dot(h_BA - h_clean, pc1_B_last))
                    commutator_results[alpha]['A_only_L1'].append(h_A_only_L1 - h_clean)
                    commutator_results[alpha]['B_only_L1'].append(h_B_only_L1 - h_clean)
                    commutator_results[alpha]['A_only_L2'].append(h_A_only_L2 - h_clean)
                    commutator_results[alpha]['B_only_L2'].append(h_B_only_L2 - h_clean)
                    commutator_results[alpha]['clean_last'].append(h_clean)
                    
                    commutator_results[alpha]['curvature_norm'].append(curv_norm)
                    commutator_results[alpha]['curvature_A_proj'].append(np.dot(curvature, pc1_A_last))
                    commutator_results[alpha]['curvature_B_proj'].append(np.dot(curvature, pc1_B_last))
                    if curv_norm > 1e-10:
                        commutator_results[alpha]['curvature_cos_A'].append(np.dot(curvature, pc1_A_last) / curv_norm)
                        commutator_results[alpha]['curvature_cos_B'].append(np.dot(curvature, pc1_B_last) / curv_norm)
                    
                except Exception as e:
                    print(f"  Intervention failed for sample {i}, alpha={alpha}: {e}")
                    continue
            
            if (i + 1) % 10 == 0:
                print(f"  Interventions done for {i+1}/{n_intervene} samples")
        
        # Step 4: Summarize
        pair_result = {
            'pair': pair_name,
            'feature_A': feat_A,
            'feature_B': feat_B,
            'n_samples': n_s,
            'n_interventions': n_intervene,
            'L1': L1,
            'L2': L2,
            'pc1_dirs': {},
            'alpha_results': {},
        }
        
        for ln in ['L1', 'L2', 'last']:
            pair_result['pc1_dirs'][ln] = {
                'cos_pc1A_pc1B': pc1_dirs[ln]['cos_pc1A_pc1B'],
                'pc1_A_var': pc1_dirs[ln]['pc1_A_var'],
                'pc1_B_var': pc1_dirs[ln]['pc1_B_var'],
                'mean_dhA_norm': pc1_dirs[ln]['mean_dhA_norm'],
                'mean_dhB_norm': pc1_dirs[ln]['mean_dhB_norm'],
            }
        
        print(f"\n  === Commutator Results for {pair_name} ===")
        print(f"  {'Alpha':>6} | {'||[A,B]||':>10} | {'curv·A':>8} | {'curv·B':>8} | {'cos(A)':>8} | {'cos(B)':>8} | {'Rel.Curv':>10}")
        
        for alpha in ALPHA_VALUES:
            cr = commutator_results[alpha]
            alpha_res = {}
            
            curv_norms = cr['curvature_norm']
            curv_A = cr['curvature_A_proj']
            curv_B = cr['curvature_B_proj']
            curv_cos_A = cr['curvature_cos_A']
            curv_cos_B = cr['curvature_cos_B']
            
            if len(curv_norms) > 0:
                mean_curv_norm = np.mean(curv_norms)
                std_curv_norm = np.std(curv_norms) if len(curv_norms) > 1 else 0
                mean_curv_A = np.mean(curv_A)
                mean_curv_B = np.mean(curv_B)
                mean_cos_A = np.mean(curv_cos_A) if len(curv_cos_A) > 0 else 0
                mean_cos_B = np.mean(curv_cos_B) if len(curv_cos_B) > 0 else 0
                
                # Relative curvature: ||[A,B]|| / (||delta_A|| + ||delta_B||)
                # Use mean single-perturbation effect as reference
                mean_A_L1 = np.mean([np.linalg.norm(d) for d in cr['A_only_L1']])
                mean_B_L1 = np.mean([np.linalg.norm(d) for d in cr['B_only_L1']])
                mean_A_L2 = np.mean([np.linalg.norm(d) for d in cr['A_only_L2']])
                mean_B_L2 = np.mean([np.linalg.norm(d) for d in cr['B_only_L2']])
                mean_single_effect = (mean_A_L1 + mean_B_L1 + mean_A_L2 + mean_B_L2) / 4
                relative_curvature = mean_curv_norm / (mean_single_effect + 1e-10)
                
                alpha_res = {
                    'curvature_norm_mean': float(mean_curv_norm),
                    'curvature_norm_std': float(std_curv_norm),
                    'curvature_A_proj_mean': float(mean_curv_A),
                    'curvature_B_proj_mean': float(mean_curv_B),
                    'curvature_cos_A_mean': float(mean_cos_A),
                    'curvature_cos_B_mean': float(mean_cos_B),
                    'mean_single_effect': float(mean_single_effect),
                    'relative_curvature': float(relative_curvature),
                    'n_valid': len(curv_norms),
                    
                    # Also store AB and BA path projections
                    'AB_A_proj_mean': float(np.mean(cr['AB_A_proj'])),
                    'AB_B_proj_mean': float(np.mean(cr['AB_B_proj'])),
                    'BA_A_proj_mean': float(np.mean(cr['BA_A_proj'])),
                    'BA_B_proj_mean': float(np.mean(cr['BA_B_proj'])),
                }
                
                print(f"  {alpha:>6.1f} | {mean_curv_norm:>10.4f} | {mean_curv_A:>8.4f} | {mean_curv_B:>8.4f} | {mean_cos_A:>8.4f} | {mean_cos_B:>8.4f} | {relative_curvature:>10.4f}")
            
            pair_result['alpha_results'][str(alpha)] = alpha_res
        
        results['pairs'][pair_name] = pair_result
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=30)
    args = parser.parse_args()
    
    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]
    
    out_dir = f"results/causal_fiber/{model_key}_ccxxxviii"
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = os.path.join(out_dir, 'run.log')
    
    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    
    log(f"=== Phase CCXXXVIII: Commutator/Curvature Experiment ===")
    log(f"Model: {cfg['name']}, n_pairs: {args.n_pairs}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")
    
    results = run_commutator_experiment(model, tokenizer, model_key, n_pairs=args.n_pairs)
    
    results['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    log(f"\n{'='*70}")
    log(f"SUMMARY: {cfg['name']}")
    log(f"{'='*70}")
    
    all_rel_curv = []
    
    for pair_name, pair_res in results['pairs'].items():
        log(f"\n  Pair: {pair_name}")
        for alpha_str, alpha_res in pair_res.get('alpha_results', {}).items():
            if alpha_res:
                rel_curv = alpha_res.get('relative_curvature', 0)
                curv_norm = alpha_res.get('curvature_norm_mean', 0)
                cos_A = alpha_res.get('curvature_cos_A_mean', 0)
                cos_B = alpha_res.get('curvature_cos_B_mean', 0)
                log(f"    α={alpha_str}: ||[A,B]||={curv_norm:.4f}, rel_curv={rel_curv:.4f}, cos(curv,A)={cos_A:.4f}, cos(curv,B)={cos_B:.4f}")
                all_rel_curv.append(rel_curv)
    
    # Verdict
    log(f"\n  VERDICT:")
    if len(all_rel_curv) > 0:
        avg_rel_curv = np.mean(all_rel_curv)
        log(f"    Average Relative Curvature: {avg_rel_curv:.4f}")
        if avg_rel_curv < 0.01:
            log(f"    → H1_FLAT: Commutator ≈ 0, flat connection, independent encoding")
        elif avg_rel_curv < 0.1:
            log(f"    → H2_WEAK_CURVE: Small but non-zero curvature, weak coupling")
        else:
            log(f"    → H3_CURVED: Significant curvature! Non-trivial connection/fiber bundle structure!")
            log(f"    → Order of feature perturbation matters → features are geometrically coupled")
    
    del model
    torch.cuda.empty_cache()
    
    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
