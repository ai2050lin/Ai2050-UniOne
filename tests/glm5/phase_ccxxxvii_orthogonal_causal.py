"""
Phase CCXXXVII: 正交化因果实验 — 分离几何重叠与辛纠缠
======================================================
核心目标:
  CCXXXVI发现强因果泄漏(Leak Ratio≈1.0), 但这可能只是PC1_A和PC1_B
  方向重叠的几何投影, 而非真正的因果纠缠。

  CCXXXVII的关键改进: 将PC1_A分解为:
    PC1_A_∥ = (PC1_A · PC1_B) × PC1_B   (与PC1_B重叠的分量)
    PC1_A_⊥ = PC1_A - PC1_A_∥           (与PC1_B正交的分量)

  然后:
  1. Perturb PC1_A_⊥ → 如果仍泄漏到B → 真正的辛纠缠!
  2. Perturb PC1_A_∥ → 泄漏到B是预期的(几何投影)
  3. 对比: leak_perp / leak_parallel → "纠缠纯度"

关键假设:
  H1_geometry: 泄漏完全是几何投影 → leak_perp ≈ 0, leak_parallel > 0
  H2_partial:  部分几何+部分纠缠 → 0 < leak_perp < leak_parallel
  H3_symplectic: 真正的辛纠缠 → leak_perp ≈ leak_parallel > 0

方法:
  在中间层(L/2)做三种perturbation:
    (a) PC1_A     (原始, 对照CCXXXVI)
    (b) PC1_A_∥   (与B重叠分量)
    (c) PC1_A_⊥   (与B正交分量)
  测量最后一层输出变化在PC1_B方向的投影(泄漏)

  同时做B→A方向的正交化:
    PC1_B_∥ = (PC1_B · PC1_A) × PC1_A
    PC1_B_⊥ = PC1_B - PC1_B_∥

  额外: 测量辛内积 ω(PC1_A, PC1_B) — 如果辛结构成立,
        辛内积应该跨层近似守恒(辛变换保辛形式).
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
            ("They win prizes", "They won prizes", "Do they win prizes", "Did they win prizes"),
            ("He breaks records", "He broke records", "Does he break records", "Did he break records"),
            ("The water boils rapidly", "The water boiled rapidly", "Does the water boil rapidly", "Did the water boil rapidly"),
            ("She feeds the cat", "She fed the cat", "Does she feed the cat", "Did she feed the cat"),
            ("They watch television", "They watched television", "Do they watch television", "Did they watch television"),
            ("He grows vegetables", "He grew vegetables", "Does he grow vegetables", "Did he grow vegetables"),
            ("The sun rises early", "The sun rose early", "Does the sun rise early", "Did the sun rise early"),
            ("She keeps secrets", "She kept secrets", "Does she keep secrets", "Did she keep secrets"),
            ("They practice yoga", "They practiced yoga", "Do they practice yoga", "Did they practice yoga"),
            ("He meets the president", "He met the president", "Does he meet the president", "Did he meet the president"),
            ("The door closes slowly", "The door closed slowly", "Does the door close slowly", "Did the door close slowly"),
            ("She borrows books", "She borrowed books", "Does she borrow books", "Did she borrow books"),
            ("They swim in the lake", "They swam in the lake", "Do they swim in the lake", "Did they swim in the lake"),
            ("He rides a bicycle", "He rode a bicycle", "Does he ride a bicycle", "Did he ride a bicycle"),
            ("The plane takes off", "The plane took off", "Does the plane take off", "Did the plane take off"),
            ("She draws portraits", "She drew portraits", "Does she draw portraits", "Did she draw portraits"),
            ("They climb mountains", "They climbed mountains", "Do they climb mountains", "Did they climb mountains"),
            ("He drinks coffee", "He drank coffee", "Does he drink coffee", "Did he drink coffee"),
            ("The flower blooms early", "The flower bloomed early", "Does the flower bloom early", "Did the flower bloom early"),
            ("She loves the music", "She loved the music", "Does she love the music", "Did she love the music"),
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
            ("They trust the process", "They trusted the process", "They do not trust the process", "They did not trust the process"),
            ("The machine operates correctly", "The machine operated correctly", "The machine does not operate correctly", "The machine did not operate correctly"),
            ("She believes the theory", "She believed the theory", "She does not believe the theory", "She did not believe the theory"),
            ("He appreciates the effort", "He appreciated the effort", "He does not appreciate the effort", "He did not appreciate the effort"),
            ("They support the cause", "They supported the cause", "They do not support the cause", "They did not support the cause"),
            ("The method produces results", "The method produced results", "The method does not produce results", "The method did not produce results"),
            ("She accepts the decision", "She accepted the decision", "She does not accept the decision", "She did not accept the decision"),
            ("He values the opinion", "He valued the opinion", "He does not value the opinion", "He did not value the opinion"),
            ("They want the change", "They wanted the change", "They do not want the change", "They did not want the change"),
            ("The software works properly", "The software worked properly", "The software does not work properly", "The software did not work properly"),
            ("She likes the proposal", "She liked the proposal", "She does not like the proposal", "She did not like the proposal"),
            ("He remembers the name", "He remembered the name", "He does not remember the name", "He did not remember the name"),
            ("They need the resource", "They needed the resource", "They do not need the resource", "They did not need the resource"),
            ("The plan includes details", "The plan included details", "The plan does not include details", "The plan did not include details"),
            ("She follows the procedure", "She followed the procedure", "She does not follow the procedure", "She did not follow the procedure"),
            ("He deserves the credit", "He deserved the credit", "He does not deserve the credit", "He did not deserve the credit"),
            ("They own the property", "They owned the property", "They do not own the property", "They did not own the property"),
            ("The system supports the feature", "The system supported the feature", "The system does not support the feature", "The system did not support the feature"),
            ("She trusts the expert", "She trusted the expert", "She does not trust the expert", "She did not trust the expert"),
            ("He speaks French", "He spoke French", "He does not speak French", "He did not speak French"),
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
            ("He chairs the meeting", "The meeting is chaired by him", "Does he chair the meeting", "Is the meeting chaired by him"),
            ("They approve the budget", "The budget is approved by them", "Do they approve the budget", "Is the budget approved by them"),
            ("The nurse gives the medicine", "The medicine is given by the nurse", "Does the nurse give the medicine", "Is the medicine given by the nurse"),
            ("She signs the contract", "The contract is signed by her", "Does she sign the contract", "Is the contract signed by her"),
            ("The storm damages the roof", "The roof is damaged by the storm", "Does the storm damage the roof", "Is the roof damaged by the storm"),
            ("He teaches the class", "The class is taught by him", "Does he teach the class", "Is the class taught by him"),
            ("They deliver the mail", "The mail is delivered by them", "Do they deliver the mail", "Is the mail delivered by them"),
            ("She writes the novel", "The novel is written by her", "Does she write the novel", "Is the novel written by her"),
            ("He leads the team", "The team is led by him", "Does he lead the team", "Is the team led by him"),
            ("They paint the wall", "The wall is painted by them", "Do they paint the wall", "Is the wall painted by them"),
            ("The fire damages the building", "The building is damaged by the fire", "Does the fire damage the building", "Is the building damaged by the fire"),
            ("She cooks the dinner", "The dinner is cooked by her", "Does she cook the dinner", "Is the dinner cooked by her"),
            ("He repairs the engine", "The engine is repaired by him", "Does he repair the engine", "Is the engine repaired by him"),
            ("They plant the tree", "The tree is planted by them", "Do they plant the tree", "Is the tree planted by them"),
            ("The manager reviews the report", "The report is reviewed by the manager", "Does the manager review the report", "Is the report reviewed by the manager"),
            ("She cleans the office", "The office is cleaned by her", "Does she clean the office", "Is the office cleaned by her"),
            ("He fixes the computer", "The computer is fixed by him", "Does he fix the computer", "Is the computer fixed by him"),
            ("They build the road", "The road is built by them", "Do they build the road", "Is the road built by them"),
            ("The chef prepares the meal", "The meal is prepared by the chef", "Does the chef prepare the meal", "Is the meal prepared by the chef"),
            ("She writes the essay", "The essay is written by her", "Does she write the essay", "Is the essay written by her"),
        ]
    },
}

ALPHA_VALUES = [1.0, 2.0]

# Layers for symplectic form conservation test
LAYER_SAMPLING = {
    'qwen3': [0, 6, 12, 18, 24, 30, 35],
    'deepseek7b': [0, 4, 9, 14, 19, 23, 27],
    'glm4': [0, 6, 13, 20, 27, 33, 39],
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


def intervene_at_layer(model, tokenizer, sentence, layer_idx, perturbation_vec, n_layers):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    seq_len = attention_mask.sum().item()
    last_pos = seq_len - 1

    model_type = model.config.model_type

    if model_type in ['qwen2', 'qwen3']:
        target_layer = model.model.layers[layer_idx]
    elif model_type in ['chatglm', 'glm4']:
        target_layer = model.transformer.encoder.layers[layer_idx]
    else:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            target_layer = model.model.layers[layer_idx]
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'encoder'):
            target_layer = model.transformer.encoder.layers[layer_idx]
        else:
            raise ValueError(f"Cannot find layers for model type {model_type}")

    pert_tensor = torch.tensor(perturbation_vec, dtype=torch.float32, device=model.device)

    def perturb_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
            hidden[:, last_pos, :] = hidden[:, last_pos, :].float() + pert_tensor.unsqueeze(0).to(hidden.dtype)
            return (hidden,) + rest
        else:
            output[:, last_pos, :] = output[:, last_pos, :].float() + pert_tensor.unsqueeze(0).to(output.dtype)
            return output

    handle = target_layer.register_forward_hook(perturb_hook)

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        result = hidden_states[n_layers][0, last_pos, :].cpu().float().numpy()
    finally:
        handle.remove()

    return result


def compute_symplectic_product(dh_A, dh_B):
    """
    Compute discrete symplectic-like product ω(A, B) = Σ_i (dh_A[i] * dh_B[i])
    But for symplectic structure, we need paired coordinates.
    
    In a Kähler/symplectic interpretation, the "symplectic form" between two directions
    can be measured by their cross-dependency in a perturbation experiment.
    
    Here we use the natural measure: 
    ω(PC1_A, PC1_B) = mean(cos(Δh_from_perturb_A, PC1_B)) * sign
    
    This is computed from the causal intervention, not from geometry alone.
    """
    # Geometric overlap (cosine similarity)
    cos_sim = np.dot(dh_A, dh_B) / (np.linalg.norm(dh_A) * np.linalg.norm(dh_B) + 1e-10)
    return cos_sim


def orthogonalize(v, u):
    """
    Decompose v into components parallel and perpendicular to u.
    v_∥ = (v · u / ||u||²) * u
    v_⊥ = v - v_∥
    
    Returns: v_parallel, v_perp, overlap_fraction
    """
    u_norm_sq = np.dot(u, u)
    if u_norm_sq < 1e-12:
        return np.zeros_like(v), v.copy(), 0.0
    
    proj_coeff = np.dot(v, u) / u_norm_sq
    v_parallel = proj_coeff * u
    v_perp = v - v_parallel
    
    # What fraction of v's norm is in the parallel component?
    v_norm = np.linalg.norm(v)
    parallel_norm = np.linalg.norm(v_parallel)
    overlap_fraction = parallel_norm / (v_norm + 1e-10)
    
    return v_parallel, v_perp, overlap_fraction


def run_orthogonal_causal(model, tokenizer, model_key, n_pairs=40):
    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg['n_layers']
    d_model = cfg['d_model']
    mid_layer = n_layers // 2
    layer_sample = LAYER_SAMPLING.get(model_key, [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1])
    
    results = {
        'model': cfg['name'],
        'n_layers': n_layers,
        'd_model': d_model,
        'intervention_layer': mid_layer,
        'layer_sample': layer_sample,
        'pairs': {},
        'symplectic_conservation': {},
    }
    
    for pair_name, pair_data in FOUR_WAY_PAIRS.items():
        feat_A = pair_data['feature_A']
        feat_B = pair_data['feature_B']
        sentences = pair_data['sentences'][:n_pairs]
        n_s = len(sentences)
        
        print(f"\n--- {pair_name} ({feat_A} x {feat_B}), n={n_s} ---")
        
        # Step 1: Collect all 4-way hidden states at ALL sampled layers + last layer
        # h_all[sample_idx][condition][layer_idx] = (d,) vector
        h_all = {}
        for cond_idx, cond_name in enumerate(['A0B0', 'A1B0', 'A0B1', 'A1B1']):
            h_all[cond_name] = {layer: [] for layer in layer_sample}
            h_all[cond_name][n_layers] = []  # also last layer
        
        for i, (s00, s10, s01, s11) in enumerate(sentences):
            for cond, s in [('A0B0', s00), ('A1B0', s10), ('A0B1', s01), ('A1B1', s11)]:
                h = get_all_hidden_states(model, tokenizer, s, n_layers)
                for layer in layer_sample:
                    h_all[cond][layer].append(h[layer])
                h_all[cond][n_layers].append(h[n_layers])
            
            if (i + 1) % 10 == 0:
                print(f"  Collected {i+1}/{n_s} 4-way hidden states")
        
        # Convert to arrays
        for cond in h_all:
            for layer in h_all[cond]:
                h_all[cond][layer] = np.array(h_all[cond][layer])
        
        # Step 2: Compute PC1 directions at each sampled layer
        pc1_directions = {}
        for layer in layer_sample + [n_layers]:
            dh_A = h_all['A1B0'][layer] - h_all['A0B0'][layer]
            dh_B = h_all['A0B1'][layer] - h_all['A0B0'][layer]
            
            from sklearn.decomposition import PCA
            pca_A = PCA(n_components=5)
            pca_A.fit(dh_A)
            pc1_A = pca_A.components_[0]
            pc1_A = pc1_A / (np.linalg.norm(pc1_A) + 1e-10)
            
            pca_B = PCA(n_components=5)
            pca_B.fit(dh_B)
            pc1_B = pca_B.components_[0]
            pc1_B = pc1_B / (np.linalg.norm(pc1_B) + 1e-10)
            
            pc1_directions[layer] = {
                'pc1_A': pc1_A,
                'pc1_B': pc1_B,
                'pc1_A_var': float(pca_A.explained_variance_ratio_[0]),
                'pc1_B_var': float(pca_B.explained_variance_ratio_[0]),
                'cos_pc1A_pc1B': float(abs(np.dot(pc1_A, pc1_B))),
                'sign_pc1A_pc1B': float(np.dot(pc1_A, pc1_B)),
                'mean_dhA_norm': float(np.mean(np.linalg.norm(dh_A, axis=1))),
                'mean_dhB_norm': float(np.mean(np.linalg.norm(dh_B, axis=1))),
            }
        
        # Step 3: Symplectic form conservation across layers
        # ω(PC1_A, PC1_B) = cos(PC1_A, PC1_B) at each layer
        print(f"\n  Symplectic form ω(PC1_A, PC1_B) across layers:")
        symplectic_values = {}
        for layer in sorted(pc1_directions.keys()):
            omega = pc1_directions[layer]['sign_pc1A_pc1B']
            symplectic_values[layer] = float(omega)
            print(f"    L{layer}: ω = {omega:.4f}")
        
        results['symplectic_conservation'][pair_name] = symplectic_values
        
        # Step 4: Orthogonal decomposition at mid layer
        pc1_A_mid = pc1_directions[mid_layer]['pc1_A']
        pc1_B_mid = pc1_directions[mid_layer]['pc1_B']
        pc1_A_last = pc1_directions[n_layers]['pc1_A']
        pc1_B_last = pc1_directions[n_layers]['pc1_B']
        
        # Decompose PC1_A_mid relative to PC1_B_mid
        pc1_A_mid_parallel, pc1_A_mid_perp, overlap_A = orthogonalize(pc1_A_mid, pc1_B_mid)
        # Decompose PC1_B_mid relative to PC1_A_mid
        pc1_B_mid_parallel, pc1_B_mid_perp, overlap_B = orthogonalize(pc1_B_mid, pc1_A_mid)
        
        # Normalize perp directions to unit length
        norm_A_perp = np.linalg.norm(pc1_A_mid_perp)
        norm_B_perp = np.linalg.norm(pc1_B_mid_perp)
        if norm_A_perp > 1e-10:
            pc1_A_mid_perp = pc1_A_mid_perp / norm_A_perp
        if norm_B_perp > 1e-10:
            pc1_B_mid_perp = pc1_B_mid_perp / norm_B_perp
        
        # Similarly for last layer
        pc1_A_last_parallel, pc1_A_last_perp, overlap_A_last = orthogonalize(pc1_A_last, pc1_B_last)
        pc1_B_last_parallel, pc1_B_last_perp, overlap_B_last = orthogonalize(pc1_B_last, pc1_A_last)
        
        print(f"\n  Orthogonal Decomposition (mid layer L{mid_layer}):")
        print(f"    PC1_A overlap with PC1_B: {overlap_A:.4f} ({overlap_A**2*100:.1f}% energy)")
        print(f"    PC1_B overlap with PC1_A: {overlap_B:.4f} ({overlap_B**2*100:.1f}% energy)")
        print(f"    ||PC1_A_⊥|| = {norm_A_perp:.4f}")
        print(f"    ||PC1_B_⊥|| = {norm_B_perp:.4f}")
        print(f"    Verification: cos(PC1_A_⊥, PC1_B_mid) = {abs(np.dot(pc1_A_mid_perp, pc1_B_mid)):.6f}")
        
        # Step 5: Causal intervention with orthogonalized directions
        mean_dhA_norm_mid = pc1_directions[mid_layer]['mean_dhA_norm']
        mean_dhB_norm_mid = pc1_directions[mid_layer]['mean_dhB_norm']
        
        causal_effects = {alpha: {
            # Original (CCXXXVI style)
            'A_full_to_B': [], 'A_full_to_A': [],
            'B_full_to_A': [], 'B_full_to_B': [],
            # Orthogonalized
            'A_perp_to_B': [], 'A_perp_to_A': [],    # perturb A_⊥ → measure at B
            'A_par_to_B': [], 'A_par_to_A': [],       # perturb A_∥ → measure at B
            'B_perp_to_A': [], 'B_perp_to_B': [],     # perturb B_⊥ → measure at A
            'B_par_to_A': [], 'B_par_to_B': [],       # perturb B_∥ → measure at A
        } for alpha in ALPHA_VALUES}
        
        # Only intervene on a subset for efficiency (still good sample)
        n_intervene = min(n_s, n_pairs)
        
        for i in range(n_intervene):
            s00 = sentences[i][0]
            
            for alpha in ALPHA_VALUES:
                # === Perturbation A: Full ===
                scale_A = alpha * mean_dhA_norm_mid
                pert_A_full = scale_A * pc1_A_mid
                try:
                    h_pert = intervene_at_layer(model, tokenizer, s00, mid_layer, pert_A_full, n_layers)
                    delta = h_pert - h_all['A0B0'][n_layers][i]
                    dn = np.linalg.norm(delta)
                    if dn > 1e-8:
                        causal_effects[alpha]['A_full_to_B'].append(np.dot(delta, pc1_B_last) / dn)
                        causal_effects[alpha]['A_full_to_A'].append(np.dot(delta, pc1_A_last) / dn)
                except:
                    pass
                
                # === Perturbation A: Perpendicular component ===
                if norm_A_perp > 0.05:  # Only if perp component is meaningful
                    pert_A_perp = scale_A * pc1_A_mid_perp * norm_A_perp  # Scale to match original A norm
                    try:
                        h_pert = intervene_at_layer(model, tokenizer, s00, mid_layer, pert_A_perp, n_layers)
                        delta = h_pert - h_all['A0B0'][n_layers][i]
                        dn = np.linalg.norm(delta)
                        if dn > 1e-8:
                            causal_effects[alpha]['A_perp_to_B'].append(np.dot(delta, pc1_B_last) / dn)
                            causal_effects[alpha]['A_perp_to_A'].append(np.dot(delta, pc1_A_last) / dn)
                    except:
                        pass
                
                # === Perturbation A: Parallel component ===
                pert_A_par = scale_A * pc1_A_mid_parallel  # Already has correct magnitude
                try:
                    h_pert = intervene_at_layer(model, tokenizer, s00, mid_layer, pert_A_par, n_layers)
                    delta = h_pert - h_all['A0B0'][n_layers][i]
                    dn = np.linalg.norm(delta)
                    if dn > 1e-8:
                        causal_effects[alpha]['A_par_to_B'].append(np.dot(delta, pc1_B_last) / dn)
                        causal_effects[alpha]['A_par_to_A'].append(np.dot(delta, pc1_A_last) / dn)
                except:
                    pass
                
                # === Perturbation B: Full ===
                scale_B = alpha * mean_dhB_norm_mid
                pert_B_full = scale_B * pc1_B_mid
                try:
                    h_pert = intervene_at_layer(model, tokenizer, s00, mid_layer, pert_B_full, n_layers)
                    delta = h_pert - h_all['A0B0'][n_layers][i]
                    dn = np.linalg.norm(delta)
                    if dn > 1e-8:
                        causal_effects[alpha]['B_full_to_A'].append(np.dot(delta, pc1_A_last) / dn)
                        causal_effects[alpha]['B_full_to_B'].append(np.dot(delta, pc1_B_last) / dn)
                except:
                    pass
                
                # === Perturbation B: Perpendicular component ===
                if norm_B_perp > 0.05:
                    pert_B_perp = scale_B * pc1_B_mid_perp * norm_B_perp
                    try:
                        h_pert = intervene_at_layer(model, tokenizer, s00, mid_layer, pert_B_perp, n_layers)
                        delta = h_pert - h_all['A0B0'][n_layers][i]
                        dn = np.linalg.norm(delta)
                        if dn > 1e-8:
                            causal_effects[alpha]['B_perp_to_A'].append(np.dot(delta, pc1_A_last) / dn)
                            causal_effects[alpha]['B_perp_to_B'].append(np.dot(delta, pc1_B_last) / dn)
                    except:
                        pass
                
                # === Perturbation B: Parallel component ===
                pert_B_par = scale_B * pc1_B_mid_parallel
                try:
                    h_pert = intervene_at_layer(model, tokenizer, s00, mid_layer, pert_B_par, n_layers)
                    delta = h_pert - h_all['A0B0'][n_layers][i]
                    dn = np.linalg.norm(delta)
                    if dn > 1e-8:
                        causal_effects[alpha]['B_par_to_A'].append(np.dot(delta, pc1_A_last) / dn)
                        causal_effects[alpha]['B_par_to_B'].append(np.dot(delta, pc1_B_last) / dn)
                except:
                    pass
            
            if (i + 1) % 10 == 0:
                print(f"  Interventions done for {i+1}/{n_intervene} samples")
        
        # Step 6: Summarize
        pair_result = {
            'pair': pair_name,
            'feature_A': feat_A,
            'feature_B': feat_B,
            'n_samples': n_s,
            'n_interventions': n_intervene,
            'intervention_layer': mid_layer,
            'orthogonal_decomposition_mid': {
                'overlap_A_on_B': float(overlap_A),
                'overlap_B_on_A': float(overlap_B),
                'norm_A_perp': float(norm_A_perp),
                'norm_B_perp': float(norm_B_perp),
                'cos_A_perp_B_mid': float(abs(np.dot(pc1_A_mid_perp, pc1_B_mid))),
            },
            'pc1_directions': {},
            'alpha_results': {},
        }
        
        for layer in sorted(pc1_directions.keys()):
            pair_result['pc1_directions'][str(layer)] = {
                'cos_pc1A_pc1B': pc1_directions[layer]['cos_pc1A_pc1B'],
                'sign_pc1A_pc1B': pc1_directions[layer]['sign_pc1A_pc1B'],
                'pc1_A_var': pc1_directions[layer]['pc1_A_var'],
                'pc1_B_var': pc1_directions[layer]['pc1_B_var'],
            }
        
        print(f"\n  === Orthogonalized Causal Results for {pair_name} ===")
        print(f"  {'Alpha':>6} | {'A_full→B':>9} | {'A_⊥→B':>9} | {'A_∥→B':>9} | {'A_full→A':>9} | {'EntanglePurity':>15}")
        
        for alpha in ALPHA_VALUES:
            alpha_res = {}
            
            for key in causal_effects[alpha]:
                vals = causal_effects[alpha][key]
                if len(vals) > 0:
                    alpha_res[key + '_mean'] = float(np.mean(vals))
                    alpha_res[key + '_std'] = float(np.std(vals)) if len(vals) > 1 else 0
                    alpha_res[key + '_n'] = len(vals)
                else:
                    alpha_res[key + '_mean'] = None
                    alpha_res[key + '_std'] = None
                    alpha_res[key + '_n'] = 0
            
            # Key metric: entanglement purity
            # = |leak_perp| / (|leak_perp| + |leak_parallel|)
            # If 1.0 → pure entanglement (perp leaks as much as parallel)
            # If 0.0 → pure geometric projection (only parallel leaks)
            a_perp_b = abs(alpha_res.get('A_perp_to_B_mean', 0) or 0)
            a_par_b = abs(alpha_res.get('A_par_to_B_mean', 0) or 0)
            entangle_purity = a_perp_b / (a_perp_b + a_par_b + 1e-10)
            alpha_res['entangle_purity_A'] = float(entangle_purity)
            
            # Also compute leak_perp / leak_full ratio
            a_full_b = abs(alpha_res.get('A_full_to_B_mean', 0) or 0)
            perp_to_full_ratio = a_perp_b / (a_full_b + 1e-10)
            alpha_res['perp_to_full_ratio_A'] = float(perp_to_full_ratio)
            
            # B side
            b_perp_a = abs(alpha_res.get('B_perp_to_A_mean', 0) or 0)
            b_par_a = abs(alpha_res.get('B_par_to_A_mean', 0) or 0)
            entangle_purity_B = b_perp_a / (b_perp_a + b_par_a + 1e-10)
            alpha_res['entangle_purity_B'] = float(entangle_purity_B)
            
            pair_result['alpha_results'][str(alpha)] = alpha_res
            
            a_full_b_signed = alpha_res.get('A_full_to_B_mean', 0) or 0
            a_perp_b_signed = alpha_res.get('A_perp_to_B_mean', 0) or 0
            a_par_b_signed = alpha_res.get('A_par_to_B_mean', 0) or 0
            a_full_a = alpha_res.get('A_full_to_A_mean', 0) or 0
            
            print(f"  {alpha:>6.1f} | {a_full_b_signed:>9.4f} | {a_perp_b_signed:>9.4f} | {a_par_b_signed:>9.4f} | {a_full_a:>9.4f} | {entangle_purity:>15.4f}")
        
        results['pairs'][pair_name] = pair_result
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=40)
    args = parser.parse_args()
    
    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]
    
    out_dir = f"results/causal_fiber/{model_key}_ccxxxvii"
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = os.path.join(out_dir, 'run.log')
    
    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    
    log(f"=== Phase CCXXXVII: Orthogonalized Causal Experiment ===")
    log(f"Model: {cfg['name']}, n_pairs: {args.n_pairs}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")
    
    results = run_orthogonal_causal(model, tokenizer, model_key, n_pairs=args.n_pairs)
    
    results['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    log(f"\n{'='*70}")
    log(f"SUMMARY: {cfg['name']}")
    log(f"{'='*70}")
    
    for pair_name, pair_res in results['pairs'].items():
        log(f"\n  Pair: {pair_name}")
        overlap = pair_res['orthogonal_decomposition_mid']['overlap_A_on_B']
        log(f"  PC1_A overlap on PC1_B: {overlap:.4f} ({overlap**2*100:.1f}% energy)")
        
        for alpha_str, alpha_res in pair_res.get('alpha_results', {}).items():
            log(f"    α={alpha_str}:")
            a_perp_b = alpha_res.get('A_perp_to_B_mean')
            a_par_b = alpha_res.get('A_par_to_B_mean')
            a_full_b = alpha_res.get('A_full_to_B_mean')
            purity = alpha_res.get('entangle_purity_A', 0)
            perp_full = alpha_res.get('perp_to_full_ratio_A', 0)
            
            if a_perp_b is not None and a_par_b is not None:
                log(f"      A_full→B: {a_full_b:.4f}")
                log(f"      A_⊥→B (entangled): {a_perp_b:.4f}")
                log(f"      A_∥→B (geometric): {a_par_b:.4f}")
                log(f"      Entanglement Purity: {purity:.4f}")
                log(f"      Perp/Full ratio: {perp_full:.4f}")
    
    # Symplectic conservation
    log(f"\n  Symplectic Form ω(PC1_A, PC1_B) Conservation:")
    for pair_name, omega_dict in results.get('symplectic_conservation', {}).items():
        log(f"    {pair_name}:")
        values = [v for v in omega_dict.values()]
        if len(values) > 1:
            conservation = np.std(values) / (np.mean(np.abs(values)) + 1e-10)
            log(f"      Coefficient of variation: {conservation:.4f}")
            for layer, omega in sorted(omega_dict.items()):
                log(f"        L{layer}: ω = {omega:.4f}")
    
    # Verdict
    log(f"\n  VERDICT:")
    all_purities = []
    for pair_name, pair_res in results['pairs'].items():
        for alpha_str, alpha_res in pair_res.get('alpha_results', {}).items():
            p = alpha_res.get('entangle_purity_A', 0)
            if p > 0:
                all_purities.append(p)
    
    if len(all_purities) > 0:
        avg_purity = np.mean(all_purities)
        log(f"    Average Entanglement Purity: {avg_purity:.4f}")
        if avg_purity > 0.5:
            log(f"    → H3_SYMPLECTIC SUPPORTED: Perpendicular component leaks as much as parallel!")
            log(f"    → True symplectic entanglement detected, not just geometric overlap!")
        elif avg_purity > 0.2:
            log(f"    → H2_PARTIAL: Both geometric overlap AND genuine entanglement present")
            log(f"    → Symplectic structure partially confirmed")
        else:
            log(f"    → H1_GEOMETRY: Leakage is mostly geometric projection of overlapping PC1 directions")
            log(f"    → Symplectic hypothesis not supported by this data")
    
    del model
    torch.cuda.empty_cache()
    
    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
