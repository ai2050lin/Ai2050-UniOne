"""
Phase CCXXXVI: 因果干预实验 — 从相关到因果!
================================================
核心目标:
  前几个阶段都是相关性分析。现在进行真正的因果干预:
  在特征A的表示方向上做perturbation, 观察特征B的表示是否因果性地变化。

关键假设:
  H1_null: 特征A和B独立编码 → perturb A不影响B (cos(Δh_B_effected, PC1_B) ≈ 0)
  H2_redundancy: 冗余消除模型 → perturb A会微弱影响B (|cos| > 0 但很小)
  H3_compression: 亚加性压缩 → perturb A会减少B的表示 (cos < 0)

方法:
  1. 获取特征A和B的PC1方向(从A0B0 vs A1B0 / A0B0 vs A0B1)
  2. 在中间层(L/2)对A0B0条件的隐藏状态做perturbation: h' = h + α·PC1_A
  3. 使用register_forward_hook在中层注入perturbed hidden state
  4. 计算"因果效应": cos(h_output_perturbed - h_output_clean, PC1_B)
  5. 对多个α值(0.5, 1.0, 2.0, 5.0)和多个特征对进行测试
  6. 同时做反向实验: perturb B, 观察A的变化
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

# Use 3 representative pairs to save GPU time
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

ALPHA_VALUES = [0.5, 1.0, 2.0, 5.0]


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
    """Get hidden states at all layers for a sentence. Returns (n_layers+1, d) array."""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (1, seq, d), length n_layers+1
    seq_len = attention_mask.sum().item()
    # Get last valid token position for each layer
    last_hidden = torch.stack([h[0, seq_len-1, :].cpu().float() for h in hidden_states])
    return last_hidden.numpy()  # (n_layers+1, d)


def get_layer_hidden(model, tokenizer, sentence, layer_idx, n_layers):
    """Get hidden state at one specific layer."""
    h = get_all_hidden_states(model, tokenizer, sentence, n_layers)
    return h[layer_idx]  # (d,)


def intervene_at_layer(model, tokenizer, sentence, layer_idx, perturbation_vec, n_layers):
    """
    Run model with intervention: at layer_idx, add perturbation_vec to the hidden state
    at the last valid token position. Return the final layer's hidden state.
    
    Uses register_forward_hook to inject the perturbation.
    """
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    seq_len = attention_mask.sum().item()
    last_pos = seq_len - 1
    
    # Get the model's transformer layers
    # Different models have different layer names
    # Qwen/Qwen2/Qwen3: model.model.layers[i]
    # GLM: model.transformer.encoder.layers[i]
    model_type = model.config.model_type
    
    if model_type in ['qwen2', 'qwen3']:
        target_layer = model.model.layers[layer_idx]
    elif model_type in ['chatglm', 'glm4']:
        target_layer = model.transformer.encoder.layers[layer_idx]
    else:
        # Try common patterns
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            target_layer = model.model.layers[layer_idx]
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'encoder'):
            target_layer = model.transformer.encoder.layers[layer_idx]
        else:
            raise ValueError(f"Cannot find layers for model type {model_type}")
    
    # Convert perturbation to tensor
    pert_tensor = torch.tensor(perturbation_vec, dtype=torch.float32, device=model.device)
    
    # Hook to add perturbation at last token position
    intervened_output = {}
    
    def perturb_hook(module, input, output):
        # output is typically a tuple (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
            # Add perturbation at last valid position
            hidden[:, last_pos, :] = hidden[:, last_pos, :].float() + pert_tensor.unsqueeze(0).to(hidden.dtype)
            return (hidden,) + rest
        else:
            output[:, last_pos, :] = output[:, last_pos, :].float() + pert_tensor.unsqueeze(0).to(output.dtype)
            return output
    
    # Register hook
    handle = target_layer.register_forward_hook(perturb_hook)
    
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        # Return last layer hidden at last valid position
        result = hidden_states[n_layers][0, last_pos, :].cpu().float().numpy()
    finally:
        handle.remove()
    
    return result


def run_causal_intervention(model, tokenizer, model_key, n_pairs=50):
    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg['n_layers']
    d_model = cfg['d_model']
    mid_layer = n_layers // 2  # intervention at middle layer
    
    results = {
        'model': cfg['name'],
        'n_layers': n_layers,
        'd_model': d_model,
        'intervention_layer': mid_layer,
        'pairs': {}
    }
    
    for pair_name, pair_data in FOUR_WAY_PAIRS.items():
        feat_A = pair_data['feature_A']
        feat_B = pair_data['feature_B']
        sentences = pair_data['sentences'][:n_pairs]
        n_s = len(sentences)
        
        print(f"\n--- {pair_name} ({feat_A} x {feat_B}), n={n_s} ---")
        
        # Step 1: Collect all 4-way hidden states at last layer and mid layer
        h_A0B0_last = []
        h_A1B0_last = []
        h_A0B1_last = []
        h_A1B1_last = []
        h_A0B0_mid = []
        h_A1B0_mid = []
        h_A0B1_mid = []
        
        for i, (s00, s10, s01, s11) in enumerate(sentences):
            h_all_00 = get_all_hidden_states(model, tokenizer, s00, n_layers)
            h_all_10 = get_all_hidden_states(model, tokenizer, s10, n_layers)
            h_all_01 = get_all_hidden_states(model, tokenizer, s01, n_layers)
            h_all_11 = get_all_hidden_states(model, tokenizer, s11, n_layers)
            
            h_A0B0_last.append(h_all_00[n_layers])
            h_A1B0_last.append(h_all_10[n_layers])
            h_A0B1_last.append(h_all_01[n_layers])
            h_A1B1_last.append(h_all_11[n_layers])
            h_A0B0_mid.append(h_all_00[mid_layer])
            h_A1B0_mid.append(h_all_10[mid_layer])
            h_A0B1_mid.append(h_all_01[mid_layer])
            
            if (i + 1) % 10 == 0:
                print(f"  Collected {i+1}/{n_s} 4-way hidden states")
        
        h_A0B0_last = np.array(h_A0B0_last)
        h_A1B0_last = np.array(h_A1B0_last)
        h_A0B1_last = np.array(h_A0B1_last)
        h_A1B1_last = np.array(h_A1B1_last)
        h_A0B0_mid = np.array(h_A0B0_mid)
        h_A1B0_mid = np.array(h_A1B0_mid)
        h_A0B1_mid = np.array(h_A0B1_mid)
        
        # Step 2: Compute PC1 directions for features A and B at last layer
        dh_A_last = h_A1B0_last - h_A0B0_last
        dh_B_last = h_A0B1_last - h_A0B0_last
        
        from sklearn.decomposition import PCA
        pca_A = PCA(n_components=5)
        pca_A.fit(dh_A_last)
        pc1_A = pca_A.components_[0]
        pc1_A = pc1_A / (np.linalg.norm(pc1_A) + 1e-10)
        
        pca_B = PCA(n_components=5)
        pca_B.fit(dh_B_last)
        pc1_B = pca_B.components_[0]
        pc1_B = pc1_B / (np.linalg.norm(pc1_B) + 1e-10)
        
        # Mid-layer PC1 directions
        dh_A_mid = h_A1B0_mid - h_A0B0_mid
        dh_B_mid = h_A0B1_mid - h_A0B0_mid
        
        pca_A_mid = PCA(n_components=5)
        pca_A_mid.fit(dh_A_mid)
        pc1_A_mid = pca_A_mid.components_[0]
        pc1_A_mid = pc1_A_mid / (np.linalg.norm(pc1_A_mid) + 1e-10)
        
        pca_B_mid = PCA(n_components=5)
        pca_B_mid.fit(dh_B_mid)
        pc1_B_mid = pca_B_mid.components_[0]
        pc1_B_mid = pc1_B_mid / (np.linalg.norm(pc1_B_mid) + 1e-10)
        
        print(f"  PC1_A var: last={pca_A.explained_variance_ratio_[0]:.3f}, mid={pca_A_mid.explained_variance_ratio_[0]:.3f}")
        print(f"  PC1_B var: last={pca_B.explained_variance_ratio_[0]:.3f}, mid={pca_B_mid.explained_variance_ratio_[0]:.3f}")
        print(f"  cos(PC1_A, PC1_B) at last: {abs(np.dot(pc1_A, pc1_B)):.4f}")
        
        # Step 3: Causal intervention at mid-layer
        mean_dhA_norm_mid = np.mean(np.linalg.norm(dh_A_mid, axis=1))
        mean_dhB_norm_mid = np.mean(np.linalg.norm(dh_B_mid, axis=1))
        
        causal_effects = {alpha: {'A_to_B': [], 'B_to_A': [], 'A_to_A': [], 'B_to_B': []}
                         for alpha in ALPHA_VALUES}
        
        for i in range(n_s):
            s00 = sentences[i][0]
            
            for alpha in ALPHA_VALUES:
                # --- Perturb A at mid-layer, measure effect at last layer ---
                scale_A = alpha * mean_dhA_norm_mid
                perturbation_A = scale_A * pc1_A_mid  # (d,)
                
                try:
                    h_last_perturbed_A = intervene_at_layer(
                        model, tokenizer, s00, mid_layer, perturbation_A, n_layers)
                except Exception as e:
                    print(f"  Intervention A failed for sample {i}, alpha={alpha}: {e}")
                    continue
                
                # Causal effect
                delta_h_A = h_last_perturbed_A - h_A0B0_last[i]
                delta_norm = np.linalg.norm(delta_h_A)
                if delta_norm < 1e-8:
                    continue
                
                cos_A_to_B = np.dot(delta_h_A, pc1_B) / (delta_norm + 1e-10)
                cos_A_to_A = np.dot(delta_h_A, pc1_A) / (delta_norm + 1e-10)
                
                causal_effects[alpha]['A_to_B'].append(cos_A_to_B)
                causal_effects[alpha]['A_to_A'].append(cos_A_to_A)
                
                # --- Perturb B at mid-layer, measure effect at last layer ---
                scale_B = alpha * mean_dhB_norm_mid
                perturbation_B = scale_B * pc1_B_mid
                
                try:
                    h_last_perturbed_B = intervene_at_layer(
                        model, tokenizer, s00, mid_layer, perturbation_B, n_layers)
                except Exception as e:
                    continue
                
                delta_h_B = h_last_perturbed_B - h_A0B0_last[i]
                delta_norm_B = np.linalg.norm(delta_h_B)
                if delta_norm_B < 1e-8:
                    continue
                
                cos_B_to_A = np.dot(delta_h_B, pc1_A) / (delta_norm_B + 1e-10)
                cos_B_to_B = np.dot(delta_h_B, pc1_B) / (delta_norm_B + 1e-10)
                
                causal_effects[alpha]['B_to_A'].append(cos_B_to_A)
                causal_effects[alpha]['B_to_B'].append(cos_B_to_B)
            
            if (i + 1) % 10 == 0:
                print(f"  Interventions done for {i+1}/{n_s} samples")
        
        # Summarize
        pair_result = {
            'pair': pair_name,
            'feature_A': feat_A,
            'feature_B': feat_B,
            'n_samples': n_s,
            'pc1_A_var_last': float(pca_A.explained_variance_ratio_[0]),
            'pc1_B_var_last': float(pca_B.explained_variance_ratio_[0]),
            'pc1_A_var_mid': float(pca_A_mid.explained_variance_ratio_[0]),
            'pc1_B_var_mid': float(pca_B_mid.explained_variance_ratio_[0]),
            'cos_pc1A_pc1B_last': float(abs(np.dot(pc1_A, pc1_B))),
            'mean_dhA_norm_mid': float(mean_dhA_norm_mid),
            'mean_dhB_norm_mid': float(mean_dhB_norm_mid),
            'intervention_layer': mid_layer,
            'alpha_results': {}
        }
        
        print(f"\n  Causal Intervention Results for {pair_name}:")
        print(f"  {'Alpha':>6} | {'A→B cos':>8} | {'A→A cos':>8} | {'B→A cos':>8} | {'B→B cos':>8} | {'Cross/Own':>10}")
        
        for alpha in ALPHA_VALUES:
            a2b = causal_effects[alpha]['A_to_B']
            a2a = causal_effects[alpha]['A_to_A']
            b2a = causal_effects[alpha]['B_to_A']
            b2b = causal_effects[alpha]['B_to_B']
            
            if len(a2b) > 0 and len(a2a) > 0:
                mean_a2b = np.mean(a2b)
                mean_a2a = np.mean(a2a)
                mean_b2a = np.mean(b2a) if len(b2a) > 0 else 0
                mean_b2b = np.mean(b2b) if len(b2b) > 0 else 0
                
                cross_own = abs(mean_a2b) / (abs(mean_a2a) + 1e-10)
                
                pair_result['alpha_results'][str(alpha)] = {
                    'A_to_B_mean': float(mean_a2b),
                    'A_to_B_std': float(np.std(a2b)) if len(a2b) > 1 else 0,
                    'A_to_B_signif': float(abs(mean_a2b) / (np.std(a2b) + 1e-10)) if len(a2b) > 1 else 0,
                    'A_to_A_mean': float(mean_a2a),
                    'A_to_A_std': float(np.std(a2a)) if len(a2a) > 1 else 0,
                    'B_to_A_mean': float(mean_b2a),
                    'B_to_A_std': float(np.std(b2a)) if len(b2a) > 1 else 0,
                    'B_to_B_mean': float(mean_b2b),
                    'B_to_B_std': float(np.std(b2b)) if len(b2b) > 1 else 0,
                    'cross_own_ratio': float(cross_own),
                    'n_valid': len(a2b),
                }
                
                print(f"  {alpha:>6.1f} | {mean_a2b:>8.4f} | {mean_a2a:>8.4f} | {mean_b2a:>8.4f} | {mean_b2b:>8.4f} | {cross_own:>10.4f}")
            else:
                print(f"  {alpha:>6.1f} | Insufficient data")
        
        results['pairs'][pair_name] = pair_result
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=50)
    args = parser.parse_args()
    
    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]
    
    out_dir = f"results/causal_fiber/{model_key}_ccxxxvi"
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = os.path.join(out_dir, 'run.log')
    
    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    
    log(f"=== Phase CCXXXVI: Causal Intervention Experiment ===")
    log(f"Model: {cfg['name']}, n_pairs: {args.n_pairs}")
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model
    model, tokenizer = load_model(model_key)
    log(f"Model loaded successfully")
    
    # Run experiment
    results = run_causal_intervention(model, tokenizer, model_key, n_pairs=args.n_pairs)
    
    # Save results
    results['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    log(f"\n{'='*60}")
    log(f"SUMMARY: {cfg['name']}")
    log(f"{'='*60}")
    
    all_cross_A2B = defaultdict(list)
    all_own_A2A = defaultdict(list)
    all_cross_B2A = defaultdict(list)
    all_own_B2B = defaultdict(list)
    
    for pair_name, pair_res in results['pairs'].items():
        for alpha_str, alpha_res in pair_res.get('alpha_results', {}).items():
            alpha = float(alpha_str)
            all_cross_A2B[alpha].append(alpha_res['A_to_B_mean'])
            all_own_A2A[alpha].append(alpha_res['A_to_A_mean'])
            all_cross_B2A[alpha].append(alpha_res['B_to_A_mean'])
            all_own_B2B[alpha].append(alpha_res['B_to_B_mean'])
    
    log(f"\n{'Alpha':>6} | {'A→B (cross)':>12} | {'A→A (own)':>12} | {'B→A (cross)':>12} | {'B→B (own)':>12} | {'Leak Ratio':>12}")
    for alpha in sorted(all_cross_A2B.keys()):
        cross_a2b = np.mean(all_cross_A2B[alpha])
        own_a2a = np.mean(all_own_A2A[alpha])
        cross_b2a = np.mean(all_cross_B2A[alpha])
        own_b2b = np.mean(all_own_B2B[alpha])
        leak = (abs(cross_a2b) + abs(cross_b2a)) / (abs(own_a2a) + abs(own_b2b) + 1e-10)
        log(f"{alpha:>6.1f} | {cross_a2b:>12.4f} | {own_a2a:>12.4f} | {cross_b2a:>12.4f} | {own_b2b:>12.4f} | {leak:>12.4f}")
    
    # Verdict
    log(f"\nVERDICT:")
    avg_leak = np.mean([
        abs(np.mean(v)) if len(v) > 0 else 0 
        for v in all_cross_A2B.values()
    ])
    avg_own = np.mean([
        abs(np.mean(v)) if len(v) > 0 else 1
        for v in all_own_A2A.values()
    ])
    
    if avg_leak < 0.02:
        log(f"  H1_NULL SUPPORTED: Features are independently encoded (cross_leak={avg_leak:.4f})")
    elif avg_leak < 0.1:
        log(f"  H2_REDUNDANCY SUPPORTED: Weak cross-feature leakage (cross_leak={avg_leak:.4f})")
    else:
        log(f"  STRONG CROSS-FEATURE EFFECT: Significant leakage (cross_leak={avg_leak:.4f})")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    log(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
