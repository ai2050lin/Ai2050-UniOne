"""
Phase CCIX Stepwise: 分步运行，每步保存结果
用法: python phase_ccix_stepwise.py --model deepseek7b --step S1
步骤: S1=resid全层, S2=attn, S3=mlp, S4=position, S5=semantic, S6=fit
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
        "comp_layers": [0, 8, 16, 24, 27],
    },
    "qwen3": {
        "name": "Qwen/Qwen3-4B",
        "load_kwargs": {"device_map": "auto", "torch_dtype": torch.bfloat16},
        "n_layers": 36,
        "resid_layers": [0, 6, 12, 18, 24, 30, 35],
        "comp_layers": [0, 9, 18, 27, 35],
    },
    "glm4": {
        "name": "THUDM/glm-4-9b-chat",
        "load_kwargs": {"device_map": "auto", "load_in_8bit": True},
        "n_layers": 40,
        "resid_layers": [0, 8, 16, 24, 32, 39],
        "comp_layers": [0, 10, 20, 30, 39],
    },
}

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


def patch_and_measure(model, clean_ids, source_ids, layer_idx, component=None):
    device = next(model.parameters()).device
    clean_ids = clean_ids.to(device)
    source_ids = source_ids.to(device)

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
        h = layer.self_attn.register_forward_hook(clean_hook)
    elif component == "mlp":
        h = layer.mlp.register_forward_hook(clean_hook)
    else:
        h = layer.register_forward_hook(clean_hook)

    with torch.no_grad():
        _ = model(clean_ids)
    h.remove()
    clean_h = captured_clean["h"].cpu().float()

    captured_source = {}
    def source_hook(module, input, output):
        if component == "attn":
            captured_source["h"] = output[0][0, -1, :].detach()
        elif component == "mlp":
            captured_source["h"] = output[0, -1, :].detach()
        else:
            captured_source["h"] = output[0][0, -1, :].detach()

    if component == "attn":
        h = layer.self_attn.register_forward_hook(source_hook)
    elif component == "mlp":
        h = layer.mlp.register_forward_hook(source_hook)
    else:
        h = layer.register_forward_hook(source_hook)

    with torch.no_grad():
        _ = model(source_ids)
    h.remove()
    source_h = captured_source["h"].cpu().float()

    captured_patched = {}
    def patch_hook_fn(module, input, output):
        if component == "attn":
            ho = output[0].clone()
            ho[0, -1, :] = source_h.to(ho.device)
            captured_patched["h"] = ho[0, -1, :].detach()
            return (ho,) + output[1:]
        elif component == "mlp":
            ho = output.clone()
            ho[0, -1, :] = source_h.to(ho.device)
            captured_patched["h"] = ho[0, -1, :].detach()
            return ho
        else:
            ho = output[0].clone()
            ho[0, -1, :] = source_h.to(ho.device)
            captured_patched["h"] = ho[0, -1, :].detach()
            return (ho,) + output[1:]

    if component == "attn":
        h = layer.self_attn.register_forward_hook(patch_hook_fn)
    elif component == "mlp":
        h = layer.mlp.register_forward_hook(patch_hook_fn)
    else:
        h = layer.register_forward_hook(patch_hook_fn)

    with torch.no_grad():
        _ = model(clean_ids)
    h.remove()
    patched_h = captured_patched["h"].cpu().float()

    l2 = torch.norm(patched_h - clean_h, p=2).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        patched_h.unsqueeze(0), clean_h.unsqueeze(0)
    ).item()
    return l2, cos_sim


def fit_algebraic_model(layers, l2_values):
    x = np.array(layers, dtype=float)
    y = np.array(l2_values, dtype=float)
    results = {}
    
    # Linear
    try:
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results["linear"] = {"a": float(coeffs[0]), "b": float(coeffs[1]), "r2": float(r2)}
    except:
        results["linear"] = {"a": 0, "b": 0, "r2": 0}
    
    # Exponential
    try:
        y_pos = np.maximum(y, 1e-6)
        log_y = np.log(y_pos)
        coeffs = np.polyfit(x, log_y, 1)
        y_pred = np.exp(coeffs[1]) * np.exp(coeffs[0] * x)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results["exponential"] = {"a": float(np.exp(coeffs[1])), "b": float(coeffs[0]), "r2": float(r2)}
    except:
        results["exponential"] = {"a": 0, "b": 0, "r2": 0}
    
    # Power law
    try:
        x_pos = np.maximum(x, 1.0)
        y_pos = np.maximum(y, 1e-6)
        log_x = np.log(x_pos)
        log_y = np.log(y_pos)
        valid = np.isfinite(log_x) & np.isfinite(log_y)
        if valid.sum() >= 2:
            coeffs = np.polyfit(log_x[valid], log_y[valid], 1)
            y_pred = np.exp(coeffs[1]) * x_pos ** coeffs[0]
            ss_res = np.sum((y[valid] - y_pred[valid])**2)
            ss_tot = np.sum((y[valid] - np.mean(y[valid]))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            results["power_law"] = {"a": float(np.exp(coeffs[1])), "b": float(coeffs[0]), "r2": float(r2)}
        else:
            results["power_law"] = {"a": 0, "b": 0, "r2": 0}
    except:
        results["power_law"] = {"a": 0, "b": 0, "r2": 0}
    
    # Logarithmic: y = a * ln(x) + b
    try:
        x_pos = np.maximum(x, 1.0)
        log_x = np.log(x_pos)
        valid = np.isfinite(log_x) & np.isfinite(y)
        if valid.sum() >= 2:
            coeffs = np.polyfit(log_x[valid], y[valid], 1)
            y_pred = coeffs[0] * log_x + coeffs[1]
            ss_res = np.sum((y[valid] - y_pred[valid])**2)
            ss_tot = np.sum((y[valid] - np.mean(y[valid]))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            results["logarithmic"] = {"a": float(coeffs[0]), "b": float(coeffs[1]), "r2": float(r2)}
        else:
            results["logarithmic"] = {"a": 0, "b": 0, "r2": 0}
    except:
        results["logarithmic"] = {"a": 0, "b": 0, "r2": 0}
    
    return results


def run_step(model_key, step, n_pairs=200):
    config = MODEL_CONFIGS[model_key]
    out_dir = f"results/causal_fiber/{model_key}_ccix"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load existing results
    result_file = f"{out_dir}/full_results.json"
    if os.path.exists(result_file):
        with open(result_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step} for {model_key}")
    
    # Only load model if needed
    if step in ["S1", "S2", "S3", "S4", "S5"]:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(config["name"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config["name"], trust_remote_code=True, **config["load_kwargs"]
        )
        model.eval()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded!")
    
    # S1: Residual全层扫描
    if step == "S1":
        print(f"--- S1: Residual全层扫描 ({len(config['resid_layers'])}层, {n_pairs}对) ---")
        resid_results = {}
        for layer_idx in config["resid_layers"]:
            layer_key = f"L{layer_idx}"
            resid_results[layer_key] = {}
            for feat_name, pairs in SYNTAX_PAIRS.items():
                l2_list = []
                cos_list = []
                n_used = min(n_pairs, len(pairs))
                for i in range(n_used):
                    try:
                        ids_c = tokenizer(pairs[i][0], return_tensors="pt")["input_ids"]
                        ids_s = tokenizer(pairs[i][1], return_tensors="pt")["input_ids"]
                        l2, cos = patch_and_measure(model, ids_c, ids_s, layer_idx)
                        l2_list.append(l2)
                        cos_list.append(cos)
                    except Exception as e:
                        print(f"  Error: {e}")
                if l2_list:
                    resid_results[layer_key][feat_name] = {
                        "mean_l2": float(np.mean(l2_list)),
                        "median_l2": float(np.median(l2_list)),
                        "std_l2": float(np.std(l2_list)),
                        "mean_cos": float(np.mean(cos_list)),
                        "n": len(l2_list),
                    }
                    print(f"  L{layer_idx} {feat_name}: mean={np.mean(l2_list):.1f}, median={np.median(l2_list):.1f}")
        all_results["resid"] = resid_results
    
    # S2: Attn patching
    elif step == "S2":
        print(f"--- S2: Attn patching ---")
        attn_results = {}
        for layer_idx in config["comp_layers"]:
            layer_key = f"L{layer_idx}"
            attn_results[layer_key] = {}
            for feat_name, pairs in SYNTAX_PAIRS.items():
                l2_list = []
                n_used = min(n_pairs, len(pairs))
                for i in range(n_used):
                    try:
                        ids_c = tokenizer(pairs[i][0], return_tensors="pt")["input_ids"]
                        ids_s = tokenizer(pairs[i][1], return_tensors="pt")["input_ids"]
                        l2, cos = patch_and_measure(model, ids_c, ids_s, layer_idx, component="attn")
                        l2_list.append(l2)
                    except:
                        pass
                if l2_list:
                    attn_results[layer_key][feat_name] = {
                        "mean_l2": float(np.mean(l2_list)),
                        "median_l2": float(np.median(l2_list)),
                        "n": len(l2_list),
                    }
                    print(f"  L{layer_idx} {feat_name}: attn median={np.median(l2_list):.1f}")
        all_results["attn"] = attn_results
    
    # S3: MLP patching
    elif step == "S3":
        print(f"--- S3: MLP patching ---")
        mlp_results = {}
        for layer_idx in config["comp_layers"]:
            layer_key = f"L{layer_idx}"
            mlp_results[layer_key] = {}
            for feat_name, pairs in SYNTAX_PAIRS.items():
                l2_list = []
                n_used = min(n_pairs, len(pairs))
                for i in range(n_used):
                    try:
                        ids_c = tokenizer(pairs[i][0], return_tensors="pt")["input_ids"]
                        ids_s = tokenizer(pairs[i][1], return_tensors="pt")["input_ids"]
                        l2, cos = patch_and_measure(model, ids_c, ids_s, layer_idx, component="mlp")
                        l2_list.append(l2)
                    except:
                        pass
                if l2_list:
                    mlp_results[layer_key][feat_name] = {
                        "mean_l2": float(np.mean(l2_list)),
                        "median_l2": float(np.median(l2_list)),
                        "n": len(l2_list),
                    }
                    print(f"  L{layer_idx} {feat_name}: mlp median={np.median(l2_list):.1f}")
        all_results["mlp"] = mlp_results
    
    # S4: Position-specific patching
    elif step == "S4":
        print(f"--- S4: Position-specific patching ---")
        pos_templates = [
            ("The cat did {} like the food.", "not", "really"),
            ("She was {} going to the party.", "not", "really"),
            ("They {} finish their homework.", "always", "never"),
            ("She {} remembers to call.", "always", "never"),
            ("The {} dog barked loudly.", "big", "small"),
        ]
        pos_results = {}
        for layer_idx in config["comp_layers"]:
            layer_key = f"L{layer_idx}"
            pos_results[layer_key] = {}
            for template, w1, w2 in pos_templates:
                try:
                    ids1 = tokenizer(template.format(w1), return_tensors="pt")["input_ids"]
                    ids2 = tokenizer(template.format(w2), return_tensors="pt")["input_ids"]
                    device = next(model.parameters()).device
                    
                    # Get source hiddens
                    layer = model.model.layers[layer_idx]
                    captured = {}
                    def src_hook(mod, inp, out):
                        captured["h"] = out[0].detach()
                    h = layer.register_forward_hook(src_hook)
                    with torch.no_grad():
                        _ = model(ids2.to(device))
                    h.remove()
                    source_h = captured["h"][0].cpu().float()
                    
                    # Get clean final
                    captured2 = {}
                    def cln_hook(mod, inp, out):
                        captured2["h"] = out[0][0, -1, :].detach()
                    h = layer.register_forward_hook(cln_hook)
                    with torch.no_grad():
                        _ = model(ids1.to(device))
                    h.remove()
                    clean_h = captured2["h"].cpu().float()
                    
                    # Patch each position
                    n_tok = ids1.shape[1]
                    pos_l2 = {}
                    for pidx in range(min(n_tok, 5)):
                        ph = source_h[pidx]
                        captured3 = {}
                        def make_hook(pi, pv):
                            def hk(mod, inp, out):
                                ho = out[0].clone()
                                ho[0, pi, :] = pv.to(ho.device)
                                captured3["h"] = ho[0, -1, :].detach()
                                return (ho,) + out[1:]
                            return hk
                        h = layer.register_forward_hook(make_hook(pidx, ph))
                        with torch.no_grad():
                            _ = model(ids1.to(device))
                        h.remove()
                        patched_h = captured3["h"].cpu().float()
                        l2 = torch.norm(patched_h - clean_h, p=2).item()
                        pos_l2[f"pos_{pidx}"] = float(l2)
                    
                    tname = template[:25].replace(" ", "_").replace("{}", "X")
                    pos_results[layer_key][tname] = pos_l2
                    print(f"  L{layer_idx} {tname}: {pos_l2}")
                except Exception as e:
                    print(f"  Error: {e}")
        all_results["position_specific"] = pos_results
    
    # S5: Semantic patching
    elif step == "S5":
        print(f"--- S5: Semantic patching ---")
        sem_results = {}
        n_sem = min(n_pairs, 25)
        for layer_idx in config["comp_layers"]:
            layer_key = f"L{layer_idx}"
            sem_results[layer_key] = {}
            for feat_name, pairs in SEMANTIC_PAIRS.items():
                l2_list = []
                for i in range(min(n_sem, len(pairs))):
                    try:
                        ids_c = tokenizer(pairs[i][0], return_tensors="pt")["input_ids"]
                        ids_s = tokenizer(pairs[i][1], return_tensors="pt")["input_ids"]
                        l2, cos = patch_and_measure(model, ids_c, ids_s, layer_idx)
                        l2_list.append(l2)
                    except:
                        pass
                if l2_list:
                    sem_results[layer_key][feat_name] = {
                        "mean_l2": float(np.mean(l2_list)),
                        "median_l2": float(np.median(l2_list)),
                        "n": len(l2_list),
                    }
                    print(f"  L{layer_idx} {feat_name}: mean={np.mean(l2_list):.1f}, median={np.median(l2_list):.1f}")
        all_results["semantic"] = sem_results
    
    # S6: Algebraic fits (no model needed)
    elif step == "S6":
        print(f"--- S6: Algebraic fits ---")
        if "resid" not in all_results:
            print("ERROR: No resid results. Run S1 first.")
            return
        
        fits = {}
        for feat_name in ["tense", "polarity", "number"]:
            layers = []
            l2_means = []
            l2_medians = []
            for layer_idx in config["resid_layers"]:
                key = f"L{layer_idx}"
                if key in all_results["resid"] and feat_name in all_results["resid"][key]:
                    layers.append(layer_idx)
                    l2_means.append(all_results["resid"][key][feat_name]["mean_l2"])
                    l2_medians.append(all_results["resid"][key][feat_name]["median_l2"])
            
            if len(layers) >= 3:
                fits[feat_name] = {
                    "mean": fit_algebraic_model(layers, l2_means),
                    "median": fit_algebraic_model(layers, l2_medians),
                }
                print(f"\n  {feat_name} (mean):")
                for model_type, vals in fits[feat_name]["mean"].items():
                    print(f"    {model_type}: R²={vals['r2']:.4f}, a={vals['a']:.2f}, b={vals['b']:.4f}")
                print(f"  {feat_name} (median):")
                for model_type, vals in fits[feat_name]["median"].items():
                    print(f"    {model_type}: R²={vals['r2']:.4f}, a={vals['a']:.2f}, b={vals['b']:.4f}")
        
        # Also fit semantic if available
        if "semantic" in all_results:
            for feat_name in ["sentiment", "semantic_topic"]:
                if any(feat_name in all_results["semantic"].get(f"L{l}", {}) for l in config["resid_layers"]):
                    layers = []
                    l2_medians = []
                    for layer_idx in config["resid_layers"]:
                        key = f"L{layer_idx}"
                        if key in all_results["semantic"] and feat_name in all_results["semantic"][key]:
                            layers.append(layer_idx)
                            l2_medians.append(all_results["semantic"][key][feat_name]["median_l2"])
                    if len(layers) >= 3:
                        fits[feat_name] = {"median": fit_algebraic_model(layers, l2_medians)}
                        print(f"\n  {feat_name} (median):")
                        for model_type, vals in fits[feat_name]["median"].items():
                            print(f"    {model_type}: R²={vals['r2']:.4f}")
        
        all_results["algebraic_fits"] = fits
    
    # Save
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved to {result_file}")
    
    if step in ["S1", "S2", "S3", "S4", "S5"]:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["deepseek7b", "qwen3", "glm4"])
    parser.add_argument("--step", type=str, required=True, choices=["S1", "S2", "S3", "S4", "S5", "S6"])
    parser.add_argument("--n_pairs", type=int, default=200)
    args = parser.parse_args()
    run_step(args.model, args.step, args.n_pairs)
