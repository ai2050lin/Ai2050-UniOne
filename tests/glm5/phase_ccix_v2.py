"""
Phase CCIX: 因果效应的代数结构 + Median统计 + Position-specific patching
高效版本: 减少层数, 增加统计精度, 一次完成所有测试

核心实验:
  S1. Residual全层扫描 (8层) + 代数拟合 (线性 vs 指数 vs 幂律)
  S2. Attn/MLP贡献分析 (5层) + Median统计
  S3. Position-specific patching (5层, 3位置)
  S4. 语义特征 (sentiment/semantic) patching

输出: results/causal_fiber/{model}_ccix/
"""

import torch
import json
import os
import argparse
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_CONFIGS = {
    "deepseek7b": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "load_kwargs": {"device_map": "auto", "load_in_8bit": True},
        "n_layers": 28,
        "resid_layers": [0, 4, 8, 12, 16, 20, 24, 27],
        "component_layers": [0, 8, 16, 24, 27],
    },
    "qwen3": {
        "name": "Qwen/Qwen3-4B",
        "load_kwargs": {"device_map": "auto", "torch_dtype": torch.bfloat16},
        "n_layers": 36,
        "resid_layers": [0, 6, 12, 18, 24, 30, 35],
        "component_layers": [0, 9, 18, 27, 35],
    },
    "glm4": {
        "name": "THUDM/glm-4-9b-chat",
        "load_kwargs": {"device_map": "auto", "load_in_8bit": True},
        "n_layers": 40,
        "resid_layers": [0, 8, 16, 24, 32, 39],
        "component_layers": [0, 10, 20, 30, 39],
    },
}

# 语法特征对
SYNTAX_PAIRS = {
    "tense": [
        ("The cat sleeps on the mat.", "The cat slept on the mat."),
        ("She walks to school every day.", "She walked to school every day."),
        ("He plays guitar in a band.", "He played guitar in a band."),
        ("The sun rises in the east.", "The sun rose in the east."),
        ("They run three miles daily.", "They ran three miles daily."),
        ("I write letters to friends.", "I wrote letters to friends."),
        ("We cook dinner every night.", "We cooked dinner every night."),
        ("She sings in the choir.", "She sang in the choir."),
        ("The dog barks at strangers.", "The dog barked at strangers."),
        ("He drives to work early.", "He drove to work early."),
        ("The baby cries for milk.", "The baby cried for milk."),
        ("I read books every weekend.", "I read books last weekend."),
        ("They swim in the lake.", "They swam in the lake."),
        ("We plant flowers in spring.", "We planted flowers in spring."),
        ("She paints beautiful landscapes.", "She painted beautiful landscapes."),
        ("The wind blows from the north.", "The wind blew from the north."),
        ("He builds model airplanes.", "He built model airplanes."),
        ("I drink coffee every morning.", "I drank coffee this morning."),
        ("They dance at weddings.", "They danced at the wedding."),
        ("The train arrives at noon.", "The train arrived at noon."),
        ("She teaches math at school.", "She taught math at school."),
        ("We visit grandma on Sundays.", "We visited grandma last Sunday."),
        ("The bird flies south in winter.", "The bird flew south in winter."),
        ("He fixes cars for a living.", "He fixed cars for a living."),
        ("I buy groceries on Fridays.", "I bought groceries last Friday."),
        ("The river flows to the sea.", "The river flowed to the sea."),
        ("They play chess after dinner.", "They played chess after dinner."),
        ("She knits sweaters for winter.", "She knitted sweaters for winter."),
        ("We watch movies on weekends.", "We watched movies last weekend."),
        ("The clock strikes midnight.", "The clock struck midnight."),
    ],
    "polarity": [
        ("The movie was good and exciting.", "The movie was bad and boring."),
        ("She is a kind and generous person.", "She is a cruel and selfish person."),
        ("The weather is beautiful today.", "The weather is terrible today."),
        ("He gave a brilliant performance.", "He gave a terrible performance."),
        ("The food was delicious and fresh.", "The food was disgusting and stale."),
        ("They had a wonderful vacation.", "They had a miserable vacation."),
        ("The book is fascinating and insightful.", "The book is dull and shallow."),
        ("She made an excellent decision.", "She made a terrible decision."),
        ("The garden looks lovely in spring.", "The garden looks ugly in spring."),
        ("He is a smart and capable leader.", "He is a foolish and incompetent leader."),
        ("The concert was amazing and energetic.", "The concert was awful and boring."),
        ("She received a warm welcome.", "She received a cold welcome."),
        ("The project was a great success.", "The project was a complete failure."),
        ("He has a bright and cheerful personality.", "He has a dark and gloomy personality."),
        ("The hotel was clean and comfortable.", "The hotel was dirty and uncomfortable."),
        ("They enjoyed a peaceful evening.", "They endured a chaotic evening."),
        ("The new policy is fair and just.", "The new policy is unfair and unjust."),
        ("She felt happy and content.", "She felt sad and miserable."),
        ("The journey was smooth and pleasant.", "The journey was rough and unpleasant."),
        ("He found a valuable treasure.", "He found a worthless trinket."),
        ("The painting is beautiful and vivid.", "The painting is ugly and faded."),
        ("She gave a generous donation.", "She gave a stingy donation."),
        ("The room was spacious and bright.", "The room was cramped and dark."),
        ("They had a profitable business.", "They had a losing business."),
        ("The solution was elegant and simple.", "The solution was clumsy and complex."),
    ],
    "number": [
        ("The cat sits on the mat.", "The cats sit on the mat."),
        ("A dog runs in the park.", "Dogs run in the park."),
        ("The bird sings in the tree.", "The birds sing in the tree."),
        ("A child plays with the toy.", "Children play with the toy."),
        ("The student reads the book.", "The students read the book."),
        ("A flower blooms in spring.", "Flowers bloom in spring."),
        ("The car drives down the street.", "The cars drive down the street."),
        ("A tree grows in the garden.", "Trees grow in the garden."),
        ("The house stands on the hill.", "The houses stand on the hill."),
        ("A star shines in the sky.", "Stars shine in the sky."),
        ("The fish swims in the river.", "The fish swim in the river."),
        ("A cloud floats in the sky.", "Clouds float in the sky."),
        ("The door opens slowly.", "The doors open slowly."),
        ("A lamp lights the room.", "Lamps light the room."),
        ("The key unlocks the door.", "The keys unlock the door."),
        ("A leaf falls from the tree.", "Leaves fall from the tree."),
        ("The bell rings at noon.", "The bells ring at noon."),
        ("A boat sails on the lake.", "Boats sail on the lake."),
        ("The clock ticks quietly.", "The clocks tick quietly."),
        ("A window looks over the garden.", "Windows look over the garden."),
    ],
}

# 语义/情感特征对
SEMANTIC_PAIRS = {
    "sentiment": [
        ("I love this amazing product!", "I hate this terrible product!"),
        ("The experience was wonderful.", "The experience was awful."),
        ("She felt joyful and grateful.", "She felt angry and resentful."),
        ("This is the best day ever.", "This is the worst day ever."),
        ("The gift was thoughtful and kind.", "The insult was cruel and mean."),
        ("We had a fantastic celebration.", "We had a dreadful argument."),
        ("The performance was outstanding.", "The performance was pathetic."),
        ("He spoke with warmth and compassion.", "He spoke with coldness and contempt."),
        ("The results exceeded our expectations.", "The results fell below our expectations."),
        ("She smiled with genuine happiness.", "She frowned with deep sorrow."),
        ("The music was uplifting and inspiring.", "The noise was annoying and disturbing."),
        ("His advice was helpful and wise.", "His advice was harmful and foolish."),
        ("The meal was a delightful treat.", "The meal was a terrible disappointment."),
        ("They showed great courage and bravery.", "They showed great fear and cowardice."),
        ("The story had a happy ending.", "The story had a tragic ending."),
        ("I admire her dedication and effort.", "I despise her laziness and neglect."),
        ("The garden was a paradise of beauty.", "The dump was a wasteland of ugliness."),
        ("She embraced him with love.", "She rejected him with hatred."),
        ("The news brought hope and relief.", "The news brought despair and panic."),
        ("His actions showed generosity.", "His actions showed greed."),
        ("The atmosphere was friendly and welcoming.", "The atmosphere was hostile and threatening."),
        ("She praised his excellent work.", "She criticized his poor work."),
        ("The day was filled with laughter.", "The day was filled with tears."),
        ("We celebrated our great victory.", "We mourned our bitter defeat."),
        ("The future looks bright and promising.", "The future looks bleak and hopeless."),
    ],
    "semantic_topic": [
        ("The doctor examined the patient carefully.", "The chef prepared the meal carefully."),
        ("Scientists discovered a new particle.", "Artists created a new painting."),
        ("The engine roared to life.", "The orchestra played to life."),
        ("She solved the complex equation.", "She wrote the complex poem."),
        ("The rocket launched into orbit.", "The ship sailed into harbor."),
        ("He programmed the computer algorithm.", "He composed the musical symphony."),
        ("The experiment yielded interesting data.", "The novel yielded interesting insights."),
        ("The telescope observed distant galaxies.", "The microscope observed tiny organisms."),
        ("She calculated the trajectory precisely.", "She choreographed the dance precisely."),
        ("The reactor generated enormous power.", "The storm generated enormous waves."),
        ("The mathematician proved the theorem.", "The philosopher proved the argument."),
        ("The satellite orbited the planet.", "The moon orbited the earth."),
        ("He analyzed the chemical compound.", "He analyzed the literary work."),
        ("The bridge spanned the wide river.", "The rainbow spanned the wide sky."),
        ("The laser cut through the metal.", "The scissors cut through the fabric."),
        ("She researched the historical period.", "She explored the geographical region."),
        ("The microscope revealed cell structures.", "The telescope revealed star patterns."),
        ("The algorithm sorted the database.", "The librarian sorted the collection."),
        ("The vaccine prevented the disease.", "The umbrella prevented the soaking."),
        ("The code compiled without errors.", "The speech delivered without pauses."),
    ],
}

POSITION_TEMPLATES = [
    # (template, pos1_word, pos2_word, key_position_idx)
    # "not" at position 1 vs position 3
    ("The cat did {} like the food.", "not", "really", 1),
    ("She was {} going to the party.", "not", "really", 1),
    ("He is {} the best player.", "not", "quite", 1),
    # "always" vs "never" at different positions
    ("They {} finish their homework.", "always", "never", 1),
    ("She {} remembers to call.", "always", "never", 1),
    # adjective at position 2
    ("The {} dog barked loudly.", "big", "small", 2),
    ("A {} bird sang sweetly.", "beautiful", "ugly", 2),
]


def get_hidden_at_layer(model, input_ids, layer_idx, component=None):
    """Get hidden state at a specific layer for the last token."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    captured = {}

    def hook_fn(module, input, output):
        if component == "attn":
            # self_attn output: (hidden, attn_weights, past_kv)
            captured["h"] = output[0].detach()
        elif component == "mlp":
            captured["h"] = output.detach()
        else:
            # DecoderLayer output: (hidden, ...) 
            captured["h"] = output[0].detach()

    layer = model.model.layers[layer_idx]
    if component == "attn":
        handle = layer.self_attn.register_forward_hook(hook_fn)
    elif component == "mlp":
        handle = layer.mlp.register_forward_hook(hook_fn)
    else:
        handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(input_ids)

    handle.remove()
    return captured["h"][0, -1, :].cpu().float()


def patch_and_measure(model, clean_ids, source_ids, layer_idx, component=None):
    """Patch source hidden state into clean run, measure l2 difference from clean."""
    device = next(model.parameters()).device
    clean_ids = clean_ids.to(device)
    source_ids = source_ids.to(device)

    # Step 1: Get clean hidden
    captured_clean = {}
    def clean_hook(module, input, output):
        if component == "attn":
            captured_clean["h"] = output[0][0, -1, :].detach()
        elif component == "mlp":
            captured_clean["h"] = output[0, -1, :].detach()
        else:
            captured_clean["h"] = output[0][0, -1, :].detach()

    layer = model.model.layers[layer_idx]
    if component == "attn":
        h_clean = layer.self_attn.register_forward_hook(clean_hook)
    elif component == "mlp":
        h_clean = layer.mlp.register_forward_hook(clean_hook)
    else:
        h_clean = layer.register_forward_hook(clean_hook)

    with torch.no_grad():
        _ = model(clean_ids)
    h_clean.remove()
    clean_h = captured_clean["h"].cpu().float()

    # Step 2: Get source hidden
    captured_source = {}
    def source_hook(module, input, output):
        if component == "attn":
            captured_source["h"] = output[0][0, -1, :].detach()
        elif component == "mlp":
            captured_source["h"] = output[0, -1, :].detach()
        else:
            captured_source["h"] = output[0][0, -1, :].detach()

    if component == "attn":
        h_source = layer.self_attn.register_forward_hook(source_hook)
    elif component == "mlp":
        h_source = layer.mlp.register_forward_hook(source_hook)
    else:
        h_source = layer.register_forward_hook(source_hook)

    with torch.no_grad():
        _ = model(source_ids)
    h_source.remove()
    source_h = captured_source["h"].cpu().float()

    # Step 3: Patched run — replace clean hidden with source at this layer
    captured_patched = {}
    def patch_hook(module, input, output):
        if component == "attn":
            h = output[0].clone()
            h[0, -1, :] = source_h.to(h.device)
            captured_patched["h"] = h[0, -1, :].detach()
            return (h,) + output[1:]
        elif component == "mlp":
            h = output.clone()
            h[0, -1, :] = source_h.to(h.device)
            captured_patched["h"] = h[0, -1, :].detach()
            return h
        else:
            h = output[0].clone()
            h[0, -1, :] = source_h.to(h.device)
            captured_patched["h"] = h[0, -1, :].detach()
            return (h,) + output[1:]

    if component == "attn":
        h_patch = layer.self_attn.register_forward_hook(patch_hook)
    elif component == "mlp":
        h_patch = layer.mlp.register_forward_hook(patch_hook)
    else:
        h_patch = layer.register_forward_hook(patch_hook)

    with torch.no_grad():
        _ = model(clean_ids)
    h_patch.remove()
    patched_h = captured_patched["h"].cpu().float()

    # l2 distance: ||patched - clean|| 
    l2 = torch.norm(patched_h - clean_h, p=2).item()
    # cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        patched_h.unsqueeze(0), clean_h.unsqueeze(0)
    ).item()

    return l2, cos_sim


def patch_position_specific(model, tokenizer, template, word1, word2, key_pos_idx, layer_idx, n_positions=3):
    """Patch at specific token positions to measure position-specific causal effect."""
    device = next(model.parameters()).device
    
    sent1 = template.format(word1)
    sent2 = template.format(word2)
    
    ids1 = tokenizer(sent1, return_tensors="pt")["input_ids"]
    ids2 = tokenizer(sent2, return_tensors="pt")["input_ids"]
    
    # Ensure same length
    min_len = min(ids1.shape[1], ids2.shape[1])
    if ids1.shape[1] != ids2.shape[1]:
        # Pad shorter one
        if ids1.shape[1] < min_len:
            ids1 = torch.cat([ids1, torch.zeros(1, min_len - ids1.shape[1], dtype=ids1.dtype)], dim=1)
        else:
            ids2 = torch.cat([ids2, torch.zeros(1, min_len - ids2.shape[1], dtype=ids2.dtype)], dim=1)
    
    n_tokens = ids1.shape[1]
    
    # Get source hidden states at each position
    layer = model.model.layers[layer_idx]
    
    source_hiddens = {}
    def source_hook(module, input, output):
        source_hiddens["h"] = output[0].detach()
    
    h_source = layer.register_forward_hook(source_hook)
    with torch.no_grad():
        _ = model(ids2.to(device))
    h_source.remove()
    source_h = source_hiddens["h"][0].cpu().float()  # [seq_len, d_model]
    
    # Get clean final hidden
    clean_final = {}
    def clean_hook2(module, input, output):
        clean_final["h"] = output[0][0, -1, :].detach()
    h_clean = layer.register_forward_hook(clean_hook2)
    with torch.no_grad():
        _ = model(ids1.to(device))
    h_clean.remove()
    clean_h = clean_final["h"].cpu().float()
    
    # Patch at each position
    results = {}
    for pos_idx in range(min(n_tokens, n_positions)):
        patch_h = source_h[pos_idx].to(device)
        
        patched_final = {}
        def make_patch_hook(pidx, ph):
            def hook_fn(module, input, output):
                h = output[0].clone()
                h[0, pidx, :] = ph.to(h.device)
                patched_final["h"] = h[0, -1, :].detach()
                return (h,) + output[1:]
            return hook_fn
        
        h_patch = layer.register_forward_hook(make_patch_hook(pos_idx, patch_h))
        with torch.no_grad():
            _ = model(ids1.to(device))
        h_patch.remove()
        
        patched_h = patched_final["h"].cpu().float()
        l2 = torch.norm(patched_h - clean_h, p=2).item()
        results[f"pos_{pos_idx}"] = l2
    
    return results


def fit_algebraic_model(layers, l2_values):
    """Fit linear, exponential, and power-law models to l2(layer)."""
    x = np.array(layers, dtype=float)
    y = np.array(l2_values, dtype=float)
    
    results = {}
    
    # Linear: y = a*x + b
    try:
        coeffs_lin = np.polyfit(x, y, 1)
        y_pred_lin = np.polyval(coeffs_lin, x)
        ss_res = np.sum((y - y_pred_lin)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results["linear"] = {"a": coeffs_lin[0], "b": coeffs_lin[1], "r2": r2_lin}
    except:
        results["linear"] = {"a": 0, "b": 0, "r2": 0}
    
    # Exponential: y = a * exp(b * x)
    try:
        y_pos = np.maximum(y, 1e-6)
        log_y = np.log(y_pos)
        coeffs_exp = np.polyfit(x, log_y, 1)
        y_pred_exp = np.exp(coeffs_exp[1]) * np.exp(coeffs_exp[0] * x)
        ss_res = np.sum((y - y_pred_exp)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_exp = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results["exponential"] = {"a": np.exp(coeffs_exp[1]), "b": coeffs_exp[0], "r2": r2_exp}
    except:
        results["exponential"] = {"a": 0, "b": 0, "r2": 0}
    
    # Power law: y = a * x^b
    try:
        x_pos = np.maximum(x, 1e-6)
        y_pos = np.maximum(y, 1e-6)
        log_x = np.log(x_pos)
        log_y = np.log(y_pos)
        valid = np.isfinite(log_x) & np.isfinite(log_y)
        if valid.sum() >= 2:
            coeffs_pow = np.polyfit(log_x[valid], log_y[valid], 1)
            y_pred_pow = np.exp(coeffs_pow[1]) * x_pos ** coeffs_pow[0]
            ss_res = np.sum((y[valid] - y_pred_pow[valid])**2)
            ss_tot = np.sum((y[valid] - np.mean(y[valid]))**2)
            r2_pow = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            results["power_law"] = {"a": np.exp(coeffs_pow[1]), "b": coeffs_pow[0], "r2": r2_pow}
        else:
            results["power_law"] = {"a": 0, "b": 0, "r2": 0}
    except:
        results["power_law"] = {"a": 0, "b": 0, "r2": 0}
    
    return results


def run_experiment(model_key, n_pairs=200):
    """Run full Phase CCIX experiment for one model."""
    config = MODEL_CONFIGS[model_key]
    print(f"\n{'='*60}")
    print(f"Phase CCIX: {model_key}")
    print(f"Model: {config['name']}")
    print(f"{'='*60}")
    
    # Load model
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["name"], trust_remote_code=True, **config["load_kwargs"]
    )
    model.eval()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded!")
    
    # Output directory
    out_dir = f"results/causal_fiber/{model_key}_ccix"
    os.makedirs(out_dir, exist_ok=True)
    
    all_results = {}
    
    # ===== S1: Residual全层扫描 + 代数拟合 =====
    print(f"\n--- S1: Residual全层扫描 ({len(config['resid_layers'])}层, {n_pairs}对) ---")
    resid_results = {}
    
    for layer_idx in config["resid_layers"]:
        layer_key = f"L{layer_idx}"
        resid_results[layer_key] = {}
        
        for feat_name, pairs in SYNTAX_PAIRS.items():
            l2_list = []
            cos_list = []
            n_used = min(n_pairs, len(pairs))
            
            for i in range(n_used):
                sent_clean, sent_source = pairs[i]
                ids_clean = tokenizer(sent_clean, return_tensors="pt")["input_ids"]
                ids_source = tokenizer(sent_source, return_tensors="pt")["input_ids"]
                
                try:
                    l2, cos = patch_and_measure(model, ids_clean, ids_source, layer_idx)
                    l2_list.append(l2)
                    cos_list.append(cos)
                except Exception as e:
                    print(f"  Error at L{layer_idx}/{feat_name}/{i}: {e}")
                    continue
            
            if l2_list:
                resid_results[layer_key][feat_name] = {
                    "mean_l2": float(np.mean(l2_list)),
                    "median_l2": float(np.median(l2_list)),
                    "std_l2": float(np.std(l2_list)),
                    "mean_cos": float(np.mean(cos_list)),
                    "n": len(l2_list),
                    "all_l2": [float(x) for x in l2_list],
                }
                print(f"  L{layer_idx} {feat_name}: mean={np.mean(l2_list):.1f}, median={np.median(l2_list):.1f}, std={np.std(l2_list):.1f}, n={len(l2_list)}")
    
    # 代数拟合
    print(f"\n--- 代数拟合 ---")
    algebraic_fits = {}
    for feat_name in SYNTAX_PAIRS.keys():
        layers = []
        l2_means = []
        l2_medians = []
        for layer_idx in config["resid_layers"]:
            key = f"L{layer_idx}"
            if key in resid_results and feat_name in resid_results[key]:
                layers.append(layer_idx)
                l2_means.append(resid_results[key][feat_name]["mean_l2"])
                l2_medians.append(resid_results[key][feat_name]["median_l2"])
        
        if len(layers) >= 3:
            fits_mean = fit_algebraic_model(layers, l2_means)
            fits_median = fit_algebraic_model(layers, l2_medians)
            algebraic_fits[feat_name] = {
                "mean": fits_mean,
                "median": fits_median,
            }
            print(f"  {feat_name} (mean):  linear R²={fits_mean['linear']['r2']:.4f}, exp R²={fits_mean['exponential']['r2']:.4f}, power R²={fits_mean['power_law']['r2']:.4f}")
            print(f"  {feat_name} (median): linear R²={fits_median['linear']['r2']:.4f}, exp R²={fits_median['exponential']['r2']:.4f}, power R²={fits_median['power_law']['r2']:.4f}")
    
    all_results["resid"] = resid_results
    all_results["algebraic_fits"] = algebraic_fits
    
    # ===== S2: Attn/MLP贡献分析 (5层, 150对) =====
    print(f"\n--- S2: Attn/MLP贡献分析 ({len(config['component_layers'])}层, 150对) ---")
    n_comp = min(150, n_pairs)
    comp_results = {"attn": {}, "mlp": {}}
    
    for layer_idx in config["component_layers"]:
        layer_key = f"L{layer_idx}"
        comp_results["attn"][layer_key] = {}
        comp_results["mlp"][layer_key] = {}
        
        for feat_name, pairs in SYNTAX_PAIRS.items():
            for comp in ["attn", "mlp"]:
                l2_list = []
                n_used = min(n_comp, len(pairs))
                
                for i in range(n_used):
                    sent_clean, sent_source = pairs[i]
                    ids_clean = tokenizer(sent_clean, return_tensors="pt")["input_ids"]
                    ids_source = tokenizer(sent_source, return_tensors="pt")["input_ids"]
                    
                    try:
                        l2, cos = patch_and_measure(model, ids_clean, ids_source, layer_idx, component=comp)
                        l2_list.append(l2)
                    except Exception as e:
                        continue
                
                if l2_list:
                    comp_results[comp][layer_key][feat_name] = {
                        "mean_l2": float(np.mean(l2_list)),
                        "median_l2": float(np.median(l2_list)),
                        "std_l2": float(np.std(l2_list)),
                        "n": len(l2_list),
                    }
            
            # 计算贡献比
            if layer_key in comp_results["attn"] and layer_key in comp_results["mlp"]:
                for feat_name in SYNTAX_PAIRS.keys():
                    if feat_name in comp_results["attn"][layer_key] and feat_name in comp_results["mlp"][layer_key]:
                        a = comp_results["attn"][layer_key][feat_name]["median_l2"]
                        m = comp_results["mlp"][layer_key][feat_name]["median_l2"]
                        total = a + m if (a + m) > 0 else 1
                        comp_results["attn"][layer_key][feat_name]["ratio"] = a / total
                        comp_results["mlp"][layer_key][feat_name]["ratio"] = m / total
        
        # Print
        for feat_name in SYNTAX_PAIRS.keys():
            a_key = layer_key in comp_results["attn"] and feat_name in comp_results["attn"][layer_key]
            m_key = layer_key in comp_results["mlp"] and feat_name in comp_results["mlp"][layer_key]
            if a_key and m_key:
                a_med = comp_results["attn"][layer_key][feat_name]["median_l2"]
                m_med = comp_results["mlp"][layer_key][feat_name]["median_l2"]
                a_r = comp_results["attn"][layer_key][feat_name].get("ratio", 0)
                m_r = comp_results["mlp"][layer_key][feat_name].get("ratio", 0)
                print(f"  L{layer_idx} {feat_name}: attn={a_med:.1f}({a_r:.0%}), mlp={m_med:.1f}({m_r:.0%})")
    
    all_results["components"] = comp_results
    
    # ===== S3: Position-specific patching =====
    print(f"\n--- S3: Position-specific patching (5层) ---")
    pos_results = {}
    
    for layer_idx in config["component_layers"][:5]:  # top 5 layers
        layer_key = f"L{layer_idx}"
        pos_results[layer_key] = {}
        
        for template, word1, word2, key_pos in POSITION_TEMPLATES:
            try:
                pos_l2 = patch_position_specific(
                    model, tokenizer, template, word1, word2, key_pos, layer_idx
                )
                template_name = template[:30].replace(" ", "_")
                pos_results[layer_key][template_name] = pos_l2
                print(f"  L{layer_idx} '{template[:40]}...': {pos_l2}")
            except Exception as e:
                print(f"  L{layer_idx} pos error: {e}")
                continue
    
    all_results["position_specific"] = pos_results
    
    # ===== S4: 语义特征 patching =====
    print(f"\n--- S4: 语义特征 patching ({len(config['component_layers'])}层) ---")
    semantic_results = {}
    n_sem = min(n_pairs, 25)
    
    for layer_idx in config["component_layers"]:
        layer_key = f"L{layer_idx}"
        semantic_results[layer_key] = {}
        
        for feat_name, pairs in SEMANTIC_PAIRS.items():
            l2_list = []
            n_used = min(n_sem, len(pairs))
            
            for i in range(n_used):
                sent_clean, sent_source = pairs[i]
                ids_clean = tokenizer(sent_clean, return_tensors="pt")["input_ids"]
                ids_source = tokenizer(sent_source, return_tensors="pt")["input_ids"]
                
                try:
                    l2, cos = patch_and_measure(model, ids_clean, ids_source, layer_idx)
                    l2_list.append(l2)
                except:
                    continue
            
            if l2_list:
                semantic_results[layer_key][feat_name] = {
                    "mean_l2": float(np.mean(l2_list)),
                    "median_l2": float(np.median(l2_list)),
                    "std_l2": float(np.std(l2_list)),
                    "n": len(l2_list),
                }
                print(f"  L{layer_idx} {feat_name}: mean={np.mean(l2_list):.1f}, median={np.median(l2_list):.1f}")
    
    all_results["semantic"] = semantic_results
    
    # Save
    with open(f"{out_dir}/full_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {model_key} complete! Saved to {out_dir}/full_results.json")
    
    # Free memory
    del model
    torch.cuda.empty_cache()
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["deepseek7b", "qwen3", "glm4"])
    parser.add_argument("--n_pairs", type=int, default=200)
    args = parser.parse_args()
    
    run_experiment(args.model, args.n_pairs)
