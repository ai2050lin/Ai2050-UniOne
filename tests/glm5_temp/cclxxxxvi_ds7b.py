"""CCLXXXXVI DS7B only runner - LN Weight + Cos Decomposition"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxxvi_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

import gc, traceback, json
from pathlib import Path
import numpy as np
from collections import defaultdict
from itertools import combinations

import torch
sys.path.insert(0, r"d:\Ai2050\TransformerLens-Project")
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

CATEGORIES_13 = {
    "animal": ["dog", "cat", "horse", "cow", "pig", "sheep", "goat", "donkey", "lion", "tiger", "bear", "wolf", "fox", "deer", "rabbit", "elephant", "giraffe", "zebra", "monkey", "camel"],
    "bird": ["eagle", "hawk", "owl", "crow", "swan", "goose", "duck", "penguin", "parrot", "robin", "sparrow", "pigeon", "seagull", "falcon", "vulture", "crane", "stork", "heron", "peacock", "flamingo"],
    "fish": ["shark", "whale", "dolphin", "salmon", "trout", "tuna", "cod", "bass", "carp", "catfish", "perch", "pike", "eel", "herring", "sardine", "anchovy", "flounder", "sole", "mackerel", "swordfish"],
    "insect": ["ant", "bee", "spider", "butterfly", "mosquito", "fly", "wasp", "beetle", "cockroach", "grasshopper", "cricket", "dragonfly", "ladybug", "moth", "flea", "tick", "mantis", "caterpillar", "worm", "snail"],
    "plant": ["tree", "flower", "grass", "bush", "shrub", "vine", "fern", "moss", "algae", "weed", "oak", "pine", "maple", "birch", "willow", "cactus", "bamboo", "palm", "rose", "lily"],
    "fruit": ["apple", "orange", "banana", "grape", "pear", "peach", "cherry", "plum", "mango", "lemon", "lime", "melon", "berry", "strawberry", "blueberry", "raspberry", "fig", "date", "coconut", "pineapple"],
    "vegetable": ["carrot", "potato", "tomato", "onion", "garlic", "cabbage", "lettuce", "spinach", "celery", "pea", "bean", "corn", "mushroom", "pepper", "cucumber", "pumpkin", "squash", "radish", "turnip", "broccoli"],
    "body_part": ["hand", "foot", "head", "heart", "brain", "eye", "ear", "nose", "mouth", "tooth", "neck", "shoulder", "arm", "finger", "knee", "chest", "back", "hip", "ankle", "wrist"],
    "tool": ["hammer", "knife", "scissors", "saw", "drill", "wrench", "screwdriver", "plier", "axe", "chisel", "ruler", "file", "clamp", "level", "shovel", "rake", "hoe", "trowel", "spade", "mallet"],
    "vehicle": ["car", "bus", "truck", "train", "bicycle", "motorcycle", "airplane", "helicopter", "boat", "ship", "submarine", "rocket", "tractor", "van", "taxi", "ambulance", "sled", "canoe", "wagon", "cart"],
    "clothing": ["shirt", "dress", "hat", "coat", "shoe", "belt", "scarf", "glove", "jacket", "sweater", "vest", "skirt", "pants", "jeans", "sock", "boot", "sandal", "tie", "uniform", "cape"],
    "weapon": ["sword", "spear", "bow", "arrow", "shield", "axe_w", "dagger", "mace", "pike_w", "lance", "crossbow", "catapult", "pistol", "rifle", "cannon", "grenade", "dynamite", "knife_w", "club", "whip"],
    "furniture": ["chair", "table", "desk", "bed", "sofa", "couch", "shelf", "cabinet", "drawer", "wardrobe", "dresser", "bench", "stool", "armchair", "bookcase", "mirror", "lamp", "rug", "curtain", "pillow"],
}

SUPERCLASS_MAP = {
    "animal": "animate", "bird": "animate", "fish": "animate", "insect": "animate",
    "plant": "plant", "fruit": "plant", "vegetable": "plant",
    "body_part": "body",
    "tool": "artifact", "vehicle": "artifact", "clothing": "artifact",
    "weapon": "artifact", "furniture": "artifact",
}

model_name = "deepseek7b"
log(f"=== Starting {model_name} ===")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

try:
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"Model: {info.model_class}, {info.n_layers} layers, d_model={info.d_model}")
    
    frac_layer = 7
    layers_list = get_layers(model)
    cat_names = sorted(CATEGORIES_13.keys())
    
    # Exp1: LN Weight analysis
    log(f"\n{'='*60}")
    log(f"Exp1: LN Weight Analysis ({model_name})")
    log(f"{'='*60}")
    
    for l in [max(0, frac_layer-2), frac_layer-1, frac_layer, frac_layer+1]:
        if l >= len(layers_list):
            continue
        layer = layers_list[l]
        marker = "FRACTURE" if l == frac_layer else f"L{l}"
        
        for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                if hasattr(ln, 'weight'):
                    w = ln.weight.detach().cpu().float().numpy()
                    top_dims = np.argsort(w)[-10:][::-1]
                    log(f"\n  {marker} input_ln: mean={w.mean():.4f}, std={w.std():.4f}")
                    log(f"    Top-10: {', '.join(f'D{d}={w[d]:.3f}' for d in top_dims[:5])}")
                break
        
        for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                if hasattr(ln, 'weight'):
                    w = ln.weight.detach().cpu().float().numpy()
                    top_dims = np.argsort(w)[-10:][::-1]
                    log(f"\n  {marker} post_attn_ln: mean={w.mean():.4f}, std={w.std():.4f}")
                    log(f"    Top-10: {', '.join(f'D{d}={w[d]:.3f}' for d in top_dims[:5])}")
                break
    
    # Compare pre vs fracture post_attn LN
    pre_ln = None
    frac_ln = None
    for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
        if hasattr(layers_list[frac_layer-1], ln_name):
            ln = getattr(layers_list[frac_layer-1], ln_name)
            if hasattr(ln, 'weight'):
                pre_ln = ln.weight.detach().cpu().float().numpy()
        if hasattr(layers_list[frac_layer], ln_name):
            ln = getattr(layers_list[frac_layer], ln_name)
            if hasattr(ln, 'weight'):
                frac_ln = ln.weight.detach().cpu().float().numpy()
    
    if pre_ln is not None and frac_ln is not None:
        diff = frac_ln - pre_ln
        log(f"\n  Post-attn LN comparison: diff_mean={diff.mean():.6f}, diff_std={diff.std():.6f}")
        log(f"  Correlation: {np.corrcoef(pre_ln, frac_ln)[0,1]:.6f}")
        top_increase = np.argsort(diff)[-10:][::-1]
        log(f"  Top-10 weight increase: {', '.join(f'D{d}={diff[d]:+.4f}' for d in top_increase)}")
    
    # Exp4: Cos decomposition
    log(f"\n{'='*60}")
    log(f"Exp4: Cos Decomposition ({model_name})")
    log(f"{'='*60}")
    
    key_layers = [max(0, frac_layer-1), frac_layer, min(len(layers_list)-1, frac_layer+1)]
    
    cache = defaultdict(lambda: defaultdict(list))
    hook_handles = []
    
    for l in key_layers:
        layer = layers_list[l]
        
        h = layer.register_forward_hook(lambda m, i, o, l=l:
            cache[f"L{l}_layer_out"]['layer_out'].append(
                (o[0] if isinstance(o, tuple) else o).detach().clone()[0, -1, :].cpu().float().numpy()
            ))
        hook_handles.append(h)
        
        for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                h = ln.register_forward_hook(lambda m, i, o, l=l:
                    cache[f"L{l}_input_ln_out"]['input_ln_out'].append(
                        (o[0] if isinstance(o, tuple) else o).detach().clone()[0, -1, :].cpu().float().numpy()
                    ))
                hook_handles.append(h)
                break
        
        for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                h = ln.register_forward_hook(lambda m, i, o, l=l:
                    cache[f"L{l}_post_attn_ln_out"]['post_attn_ln_out'].append(
                        (o[0] if isinstance(o, tuple) else o).detach().clone()[0, -1, :].cpu().float().numpy()
                    ))
                hook_handles.append(h)
                break
    
    for cat in cat_names:
        for word in CATEGORIES_13[cat]:
            inputs = tokenizer(word, return_tensors="pt", padding=False, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            with torch.no_grad():
                try:
                    model(input_ids)
                except:
                    continue
    
    for h in hook_handles:
        h.remove()
    
    for l in key_layers:
        marker = "FRACTURE" if l == frac_layer else f"L{l}"
        log(f"\n  --- L{l} ({marker}) ---")
        
        for rep_type in ['layer_out', 'input_ln_out', 'post_attn_ln_out']:
            key = f"L{l}_{rep_type}"
            if key not in cache or rep_type not in cache[key]:
                log(f"    {rep_type}: No data")
                continue
            
            vecs = cache[key][rep_type]
            cat_means = {}
            idx = 0
            for cat in cat_names:
                cat_vecs = []
                for word in CATEGORIES_13[cat]:
                    if idx < len(vecs):
                        cat_vecs.append(vecs[idx])
                        idx += 1
                if cat_vecs:
                    cat_means[cat] = np.mean(cat_vecs, axis=0)
            
            coses = []
            sup_cos = defaultdict(list)
            for c1, c2 in combinations(sorted(cat_means.keys()), 2):
                v1, v2 = cat_means[c1], cat_means[c2]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos = np.dot(v1, v2) / (n1 * n2)
                    coses.append(cos)
                    s1, s2 = SUPERCLASS_MAP[c1], SUPERCLASS_MAP[c2]
                    if s1 == s2:
                        sup_cos[f"within_{s1}"].append(cos)
                    else:
                        sup_cos["cross"].append(cos)
            
            if coses:
                log(f"    {rep_type}: avg_cos={np.mean(coses):.4f}")
                for ks in sorted(sup_cos.keys()):
                    if sup_cos[ks]:
                        log(f"      {ks}: avg_cos={np.mean(sup_cos[ks]):.4f}")
    
    release_model(model)
    log(f"=== Finished {model_name} ===\n")
    
except Exception as e:
    log(f"ERROR: {e}")
    traceback.print_exc()
