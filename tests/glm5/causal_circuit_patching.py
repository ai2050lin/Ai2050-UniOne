"""
Phase CCVIII: 因果电路追踪 — 多head组合 + MLP + 大样本验证
使用HuggingFace模型 + 手动hook实现patching

核心目标:
  S1: Residual Stream Patching (500对) — 验证200对结果的稳定性
  S2: MLP Patching (300对) — MLP在因果传递中的作用
  S3: Multi-Head组合Patching (200对) — 因果信号是否随head数增长
  S4: Head vs MLP贡献比 — Attn和MLP各贡献多少因果效应

方法: 用register_forward_hook在指定层替换hidden state
  - 只替换最后一个token位置的hidden state
  - 后续层继续正常计算 → 真正的因果追踪
"""
import os, sys, gc, time, json, argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

PATHS = {
    "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
    "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

LAYER_CONFIGS = {
    'deepseek7b': 28,
    'qwen3': 36,
    'glm4': 40,
}

# ===== Pair Generation =====
TENSE_TEMPLATES = [
    ("The cat sat quietly on the mat", "The cat sits quietly on the mat"),
    ("She walked to the store yesterday", "She walks to the store today"),
    ("He played guitar every evening", "He plays guitar every evening"),
    ("They worked on the project", "They work on the project"),
    ("The dog ran across the field", "The dog runs across the field"),
    ("She wrote a long letter", "She writes a long letter"),
    ("He drove the car fast", "He drives the car fast"),
    ("They built a new house", "They build a new house"),
    ("The bird flew over the lake", "The bird flies over the lake"),
    ("She cooked dinner for us", "She cooks dinner for us"),
    ("He read many books last year", "He reads many books this year"),
    ("They sang beautiful songs", "They sing beautiful songs"),
    ("The train arrived late", "The train arrives late"),
    ("She taught the children math", "She teaches the children math"),
    ("He caught the ball easily", "He catches the ball easily"),
    ("They grew vegetables in spring", "They grow vegetables in spring"),
    ("The water froze overnight", "The water freezes overnight"),
    ("She drew a colorful picture", "She draws a colorful picture"),
    ("He held the baby gently", "He holds the baby gently"),
    ("They knew the answer quickly", "They know the answer quickly"),
    ("The sun rose early this morning", "The sun rises early every morning"),
    ("She chose the blue dress", "She chooses the blue dress"),
    ("He broke the window accidentally", "He breaks the window accidentally"),
    ("They spoke about the problem", "They speak about the problem"),
    ("The wind blew strongly today", "The wind blows strongly today"),
]

POLARITY_TEMPLATES = [
    ("The cat is happy today", "The cat is not happy today"),
    ("She likes the new movie", "She does not like the new movie"),
    ("He can solve the problem", "He cannot solve the problem"),
    ("They will attend the meeting", "They will not attend the meeting"),
    ("The dog is friendly to strangers", "The dog is not friendly to strangers"),
    ("She has finished the work", "She has not finished the work"),
    ("He was available yesterday", "He was not available yesterday"),
    ("They are coming to the party", "They are not coming to the party"),
    ("The food was delicious", "The food was not delicious"),
    ("She does know the answer", "She does not know the answer"),
    ("He should go to school", "He should not go to school"),
    ("They would help with chores", "They would not help with chores"),
    ("The test was easy for me", "The test was not easy for me"),
    ("She could swim very well", "She could not swim very well"),
    ("He must leave right now", "He must not leave right now"),
    ("They had enough money saved", "They had not enough money saved"),
    ("The door was open this morning", "The door was not open this morning"),
    ("She did enjoy the concert", "She did not enjoy the concert"),
    ("He has been working hard", "He has not been working hard"),
    ("They were ready on time", "They were not ready on time"),
    ("The plan was successful", "The plan was not successful"),
    ("She will accept the offer", "She will not accept the offer"),
    ("He can drive the truck", "He cannot drive the truck"),
    ("They did win the game", "They did not win the game"),
    ("The idea was brilliant", "The idea was not brilliant"),
]

NUMBER_TEMPLATES = [
    ("The cat is sleeping now", "The cats are sleeping now"),
    ("A book was on the table", "The books were on the table"),
    ("The child plays outside", "The children play outside"),
    ("A dog runs in the park", "The dogs run in the park"),
    ("The woman walks to work", "The women walk to work"),
    ("A man drives the bus", "The men drive the bus"),
    ("The bird flies high above", "The birds fly high above"),
    ("A student reads the text", "The students read the text"),
    ("The teacher explains clearly", "The teachers explain clearly"),
    ("A flower grows in spring", "The flowers grow in spring"),
    ("The tree stands very tall", "The trees stand very tall"),
    ("A car drives down the road", "The cars drive down the road"),
    ("The house looks beautiful", "The houses look beautiful"),
    ("A river flows through town", "The rivers flow through town"),
    ("The mountain rises above clouds", "The mountains rise above clouds"),
    ("A star shines in the sky", "The stars shine in the sky"),
    ("The country borders two seas", "The countries border two seas"),
    ("A city grows very fast", "The cities grow very fast"),
    ("The church stands on the hill", "The churches stand on the hill"),
    ("A bridge crosses the river", "The bridges cross the river"),
]

PREFIXES = ["Actually, ", "In fact, ", "Indeed, ", "Clearly, ", "Surely, ",
            "Perhaps, ", "Maybe, ", "Certainly, ", "Obviously, ", "Naturally, "]
SUFFIXES = [" right now", " today", " this time", " as expected", " for sure",
            " already", " once again", " at last", " in the end", " after all"]


def generate_pairs(templates, n_total, seed=42):
    """Augment pairs to reach n_total"""
    np.random.seed(seed)
    pairs = list(templates)
    while len(pairs) < n_total:
        idx = np.random.randint(len(templates))
        a, b = templates[idx]
        aug = np.random.randint(4)
        if aug == 0:
            p = PREFIXES[np.random.randint(len(PREFIXES))]
            pairs.append((p + a, p + b))
        elif aug == 1:
            s = SUFFIXES[np.random.randint(len(SUFFIXES))]
            pairs.append((a + s, b + s))
        elif aug == 2:
            p = PREFIXES[np.random.randint(len(PREFIXES))]
            s = SUFFIXES[np.random.randint(len(SUFFIXES))]
            pairs.append((p + a + s, p + b + s))
        else:
            pairs.append((a, b))
    return pairs[:n_total]


def load_model(model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    path = PATHS[model_key]
    print(f"Loading {model_key} from {path}...")
    t0 = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if model_key in ["glm4", "deepseek7b"]:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )
    else:  # qwen3
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map='cpu',
            trust_remote_code=True, local_files_only=True
        )
        model = model.to('cuda')
    
    model.eval()
    device = next(model.parameters()).device
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    print(f"  Loaded in {time.time()-t0:.1f}s: n_layers={n_layers}, d_model={d_model}, device={device}")
    
    return model, tokenizer, n_layers, d_model


def get_layers(model_key, n_layers):
    """Get 7 key layers"""
    return [0, n_layers//6, n_layers//3, n_layers//2, 2*n_layers//3, 5*n_layers//6, n_layers-1]


def get_logits(model, input_ids):
    """Get logits from model"""
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs.logits


def run_residual_patching(model, tokenizer, source_text, clean_text, layer_idx):
    """
    Correct residual patching:
    1. Run source, capture hidden state at layer_idx (last token position)
    2. Run clean, get baseline logits
    3. Run clean with hook: at layer_idx, replace last token hidden with source's
    4. Compare patched logits vs clean logits
    """
    device = next(model.parameters()).device
    source_ids = tokenizer(source_text, return_tensors='pt')['input_ids'].to(device)
    clean_ids = tokenizer(clean_text, return_tensors='pt')['input_ids'].to(device)
    
    # Step 1: Get source hidden state
    source_hidden = {}
    def capture_hook(module, input, output):
        # output is a tuple for most layers: (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            source_hidden['val'] = output[0].detach().clone()
        else:
            source_hidden['val'] = output.detach().clone()
    
    handle = model.model.layers[layer_idx].register_forward_hook(capture_hook)
    with torch.no_grad():
        _ = model(source_ids)
    handle.remove()
    
    if 'val' not in source_hidden:
        return None
    source_last = source_hidden['val'][0, -1, :].detach().clone()
    del source_hidden
    gc.collect()
    
    # Step 2: Get clean logits
    with torch.no_grad():
        clean_logits = model(clean_ids).logits[0, -1].detach().clone()
    
    # Step 3: Patch and get patched logits
    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[0, -1, :] = source_last.to(hidden.device)
            return (hidden,) + output[1:]
        else:
            hidden = output.clone()
            hidden[0, -1, :] = source_last.to(hidden.device)
            return hidden
    
    handle = model.model.layers[layer_idx].register_forward_hook(patch_hook)
    with torch.no_grad():
        patched_logits = model(clean_ids).logits[0, -1].detach().clone()
    handle.remove()
    
    # Step 4: Compute effect
    diff = patched_logits.float() - clean_logits.float()
    l2 = torch.norm(diff).item()
    cos = torch.nn.functional.cosine_similarity(
        clean_logits.float().unsqueeze(0), patched_logits.float().unsqueeze(0)
    ).item()
    
    return {'l2': l2, 'cos_sim': cos}


def run_mlp_patching(model, tokenizer, source_text, clean_text, layer_idx):
    """
    MLP patching: Replace only the MLP output at a given layer.
    In most models: layer output = input + attn_output + mlp_output
    So patching MLP output should change the layer's contribution.
    """
    device = next(model.parameters()).device
    source_ids = tokenizer(source_text, return_tensors='pt')['input_ids'].to(device)
    clean_ids = tokenizer(clean_text, return_tensors='pt')['input_ids'].to(device)
    
    # Step 1: Get source MLP output
    source_mlp = {}
    def capture_mlp_hook(module, input, output):
        if isinstance(output, tuple):
            source_mlp['val'] = output[0].detach().clone()
        else:
            source_mlp['val'] = output.detach().clone()
    
    # MLP is typically model.model.layers[layer_idx].mlp
    handle = model.model.layers[layer_idx].mlp.register_forward_hook(capture_mlp_hook)
    with torch.no_grad():
        _ = model(source_ids)
    handle.remove()
    
    if 'val' not in source_mlp:
        return None
    source_mlp_last = source_mlp['val'][0, -1, :].detach().clone()
    del source_mlp
    gc.collect()
    
    # Step 2: Get clean logits and clean MLP output
    clean_mlp = {}
    def capture_clean_mlp(module, input, output):
        if isinstance(output, tuple):
            clean_mlp['val'] = output[0].detach().clone()
        else:
            clean_mlp['val'] = output.detach().clone()
    
    handle = model.model.layers[layer_idx].mlp.register_forward_hook(capture_clean_mlp)
    with torch.no_grad():
        clean_logits = model(clean_ids).logits[0, -1].detach().clone()
    handle.remove()
    
    if 'val' not in clean_mlp:
        return None
    clean_mlp_last = clean_mlp['val'][0, -1, :].detach().clone()
    del clean_mlp
    gc.collect()
    
    # Step 3: Patch MLP output
    def patch_mlp_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            # Replace only last token's MLP output
            hidden[0, -1, :] = source_mlp_last.to(hidden.device)
            return (hidden,) + output[1:]
        else:
            hidden = output.clone()
            hidden[0, -1, :] = source_mlp_last.to(hidden.device)
            return hidden
    
    handle = model.model.layers[layer_idx].mlp.register_forward_hook(patch_mlp_hook)
    with torch.no_grad():
        patched_logits = model(clean_ids).logits[0, -1].detach().clone()
    handle.remove()
    
    diff = patched_logits.float() - clean_logits.float()
    l2 = torch.norm(diff).item()
    cos = torch.nn.functional.cosine_similarity(
        clean_logits.float().unsqueeze(0), patched_logits.float().unsqueeze(0)
    ).item()
    
    return {'l2': l2, 'cos_sim': cos}


def run_attn_patching(model, tokenizer, source_text, clean_text, layer_idx):
    """
    Attention patching: Replace the attention output (self_attn) at a given layer.
    """
    device = next(model.parameters()).device
    source_ids = tokenizer(source_text, return_tensors='pt')['input_ids'].to(device)
    clean_ids = tokenizer(clean_text, return_tensors='pt')['input_ids'].to(device)
    
    # Get source attn output
    source_attn = {}
    def capture_attn_hook(module, input, output):
        if isinstance(output, tuple):
            source_attn['val'] = output[0].detach().clone()
        else:
            source_attn['val'] = output.detach().clone()
    
    handle = model.model.layers[layer_idx].self_attn.register_forward_hook(capture_attn_hook)
    with torch.no_grad():
        _ = model(source_ids)
    handle.remove()
    
    if 'val' not in source_attn:
        return None
    source_attn_last = source_attn['val'][0, -1, :].detach().clone()
    del source_attn
    gc.collect()
    
    # Get clean logits
    with torch.no_grad():
        clean_logits = model(clean_ids).logits[0, -1].detach().clone()
    
    # Patch attn output
    def patch_attn_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[0, -1, :] = source_attn_last.to(hidden.device)
            return (hidden,) + output[1:]
        else:
            hidden = output.clone()
            hidden[0, -1, :] = source_attn_last.to(hidden.device)
            return hidden
    
    handle = model.model.layers[layer_idx].self_attn.register_forward_hook(patch_attn_hook)
    with torch.no_grad():
        patched_logits = model(clean_ids).logits[0, -1].detach().clone()
    handle.remove()
    
    diff = patched_logits.float() - clean_logits.float()
    l2 = torch.norm(diff).item()
    cos = torch.nn.functional.cosine_similarity(
        clean_logits.float().unsqueeze(0), patched_logits.float().unsqueeze(0)
    ).item()
    
    return {'l2': l2, 'cos_sim': cos}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--test', type=str, default='all', choices=['all', 'resid', 'mlp', 'attn'])
    parser.add_argument('--n_pairs', type=int, default=500)
    args = parser.parse_args()
    
    model, tokenizer, n_layers, d_model = load_model(args.model)
    layers = get_layers(args.model, n_layers)
    print(f"Testing layers: {layers}")
    
    out_dir = Path(f"results/causal_fiber/{args.model}_circuit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate pairs
    tense_pairs = generate_pairs(TENSE_TEMPLATES, args.n_pairs)
    polarity_pairs = generate_pairs(POLARITY_TEMPLATES, args.n_pairs)
    number_pairs = generate_pairs(NUMBER_TEMPLATES, min(args.n_pairs, 400))
    
    feature_pairs = {
        'tense': tense_pairs,
        'polarity': polarity_pairs,
        'number': number_pairs,
    }
    
    all_results = {}
    
    # ===== S1: Residual Stream Patching (500对) =====
    if args.test in ['all', 'resid']:
        print(f"\n{'='*60}")
        print(f"S1: Residual Stream Patching ({args.n_pairs} pairs)")
        print(f"{'='*60}")
        
        resid_results = {}
        for layer_idx in layers:
            layer_name = f'L{layer_idx}'
            resid_results[layer_name] = {}
            
            for feature, pairs in feature_pairs.items():
                n_test = min(len(pairs), args.n_pairs)
                l2s, coss = [], []
                
                print(f"  {layer_name} {feature}: {n_test} pairs...", end=' ', flush=True)
                t0 = time.time()
                
                for i in range(n_test):
                    a_text, b_text = pairs[i]
                    result = run_residual_patching(model, tokenizer, a_text, b_text, layer_idx)
                    if result is not None:
                        l2s.append(result['l2'])
                        coss.append(result['cos_sim'])
                
                elapsed = time.time() - t0
                avg_l2 = np.mean(l2s) if l2s else 0
                std_l2 = np.std(l2s) if len(l2s) > 1 else 0
                avg_cos = np.mean(coss) if coss else 0
                
                resid_results[layer_name][feature] = {
                    'avg_l2': avg_l2,
                    'std_l2': std_l2,
                    'avg_cos_sim': avg_cos,
                    'n': len(l2s),
                }
                
                print(f"l2={avg_l2:.1f}±{std_l2:.1f}, cos={avg_cos:.4f}, n={len(l2s)}, t={elapsed:.1f}s")
        
        with open(out_dir / 's1_resid_patching.json', 'w') as f:
            json.dump(resid_results, f, indent=2)
        all_results['resid'] = resid_results
        print(f"S1 saved to {out_dir / 's1_resid_patching.json'}")
    
    # ===== S2: MLP Patching (300对) =====
    if args.test in ['all', 'mlp']:
        print(f"\n{'='*60}")
        print(f"S2: MLP Patching (300 pairs)")
        print(f"{'='*60}")
        
        mlp_results = {}
        for layer_idx in layers:
            layer_name = f'L{layer_idx}'
            mlp_results[layer_name] = {}
            
            for feature, pairs in feature_pairs.items():
                n_test = min(len(pairs), 300)
                l2s, coss = [], []
                
                print(f"  {layer_name} {feature}: {n_test} pairs...", end=' ', flush=True)
                t0 = time.time()
                
                for i in range(n_test):
                    a_text, b_text = pairs[i]
                    result = run_mlp_patching(model, tokenizer, a_text, b_text, layer_idx)
                    if result is not None:
                        l2s.append(result['l2'])
                        coss.append(result['cos_sim'])
                
                elapsed = time.time() - t0
                avg_l2 = np.mean(l2s) if l2s else 0
                std_l2 = np.std(l2s) if len(l2s) > 1 else 0
                avg_cos = np.mean(coss) if coss else 0
                
                mlp_results[layer_name][feature] = {
                    'avg_l2': avg_l2,
                    'std_l2': std_l2,
                    'avg_cos_sim': avg_cos,
                    'n': len(l2s),
                }
                
                print(f"l2={avg_l2:.1f}±{std_l2:.1f}, cos={avg_cos:.4f}, n={len(l2s)}, t={elapsed:.1f}s")
        
        with open(out_dir / 's2_mlp_patching.json', 'w') as f:
            json.dump(mlp_results, f, indent=2)
        all_results['mlp'] = mlp_results
        print(f"S2 saved to {out_dir / 's2_mlp_patching.json'}")
    
    # ===== S3: Attention Patching (300对) =====
    if args.test in ['all', 'attn']:
        print(f"\n{'='*60}")
        print(f"S3: Attention Patching (300 pairs)")
        print(f"{'='*60}")
        
        attn_results = {}
        for layer_idx in layers:
            layer_name = f'L{layer_idx}'
            attn_results[layer_name] = {}
            
            for feature, pairs in feature_pairs.items():
                n_test = min(len(pairs), 300)
                l2s, coss = [], []
                
                print(f"  {layer_name} {feature}: {n_test} pairs...", end=' ', flush=True)
                t0 = time.time()
                
                for i in range(n_test):
                    a_text, b_text = pairs[i]
                    result = run_attn_patching(model, tokenizer, a_text, b_text, layer_idx)
                    if result is not None:
                        l2s.append(result['l2'])
                        coss.append(result['cos_sim'])
                
                elapsed = time.time() - t0
                avg_l2 = np.mean(l2s) if l2s else 0
                std_l2 = np.std(l2s) if len(l2s) > 1 else 0
                avg_cos = np.mean(coss) if coss else 0
                
                attn_results[layer_name][feature] = {
                    'avg_l2': avg_l2,
                    'std_l2': std_l2,
                    'avg_cos_sim': avg_cos,
                    'n': len(l2s),
                }
                
                print(f"l2={avg_l2:.1f}±{std_l2:.1f}, cos={avg_cos:.4f}, n={len(l2s)}, t={elapsed:.1f}s")
        
        with open(out_dir / 's3_attn_patching.json', 'w') as f:
            json.dump(attn_results, f, indent=2)
        all_results['attn'] = attn_results
        print(f"S3 saved to {out_dir / 's3_attn_patching.json'}")
    
    # ===== S4: Contribution Analysis =====
    if args.test == 'all' and 'resid' in all_results and 'mlp' in all_results and 'attn' in all_results:
        print(f"\n{'='*60}")
        print(f"S4: Attn vs MLP Contribution Analysis")
        print(f"{'='*60}")
        
        contrib = {}
        for layer_name in sorted(all_results['resid'].keys()):
            contrib[layer_name] = {}
            for feature in ['tense', 'polarity', 'number']:
                resid_l2 = all_results['resid'].get(layer_name, {}).get(feature, {}).get('avg_l2', 0)
                mlp_l2 = all_results['mlp'].get(layer_name, {}).get(feature, {}).get('avg_l2', 0)
                attn_l2 = all_results['attn'].get(layer_name, {}).get(feature, {}).get('avg_l2', 0)
                
                total = attn_l2 + mlp_l2 + 1e-8
                attn_ratio = attn_l2 / total
                mlp_ratio = mlp_l2 / total
                
                contrib[layer_name][feature] = {
                    'resid_l2': resid_l2,
                    'attn_l2': attn_l2,
                    'mlp_l2': mlp_l2,
                    'attn_ratio': attn_ratio,
                    'mlp_ratio': mlp_ratio,
                }
                print(f"  {layer_name} {feature}: attn={attn_l2:.0f}({attn_ratio:.2f}), mlp={mlp_l2:.0f}({mlp_ratio:.2f})")
        
        with open(out_dir / 's4_contribution.json', 'w') as f:
            json.dump(contrib, f, indent=2)
        print(f"S4 saved")
    
    # ===== Summary =====
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*60}")
    
    for stage_name, stage_data in all_results.items():
        print(f"\n{stage_name.upper()}:")
        for layer_name in sorted(stage_data.keys()):
            feats = []
            for f in ['tense', 'polarity', 'number']:
                if f in stage_data[layer_name]:
                    d = stage_data[layer_name][f]
                    feats.append(f"{f}={d['avg_l2']:.0f}(cos={d['avg_cos_sim']:.3f})")
            print(f"  {layer_name}: {', '.join(feats)}")
    
    # Save all
    with open(out_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nAll results saved to {out_dir}")
    print("DONE!")


if __name__ == '__main__':
    main()
