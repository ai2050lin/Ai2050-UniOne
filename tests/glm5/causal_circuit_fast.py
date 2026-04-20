"""
Phase CCVIII-Fast: 因果电路追踪 (精简版)
S1: Residual Patching (500对, 7层) — 已有结果
S2: MLP Patching (200对, 5层)
S3: Attn Patching (200对, 5层)
S4: 贡献比分析
"""
import os, sys, gc, time, json, argparse
import numpy as np
import torch
from pathlib import Path

PATHS = {
    "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
    "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

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
    print(f"Loading {model_key}...")
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
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map='cpu',
            trust_remote_code=True, local_files_only=True
        )
        model = model.to('cuda')
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s: n_layers={model.config.num_hidden_layers}")
    return model, tokenizer


def patch_component(model, tokenizer, source_text, clean_text, layer_idx, component='resid'):
    """
    Patch a component at layer_idx.
    component: 'resid' (full layer output), 'mlp' (MLP output), 'attn' (self_attn output)
    """
    device = next(model.parameters()).device
    src_ids = tokenizer(source_text, return_tensors='pt')['input_ids'].to(device)
    clean_ids = tokenizer(clean_text, return_tensors='pt')['input_ids'].to(device)
    
    layer = model.model.layers[layer_idx]
    
    if component == 'resid':
        target_module = layer
    elif component == 'mlp':
        target_module = layer.mlp
    elif component == 'attn':
        target_module = layer.self_attn
    else:
        raise ValueError(f"Unknown component: {component}")
    
    # Get source hidden
    source_val = {}
    def capture(mod, inp, out):
        if isinstance(out, tuple):
            source_val['h'] = out[0][0, -1, :].detach().clone()
        else:
            source_val['h'] = out[0, -1, :].detach().clone()
    
    h = target_module.register_forward_hook(capture)
    with torch.no_grad():
        _ = model(src_ids)
    h.remove()
    
    if 'h' not in source_val:
        return None
    src_vec = source_val['h']
    
    # Clean logits
    with torch.no_grad():
        clean_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
    
    # Patched logits
    def patch_hook(mod, inp, out, sv=src_vec):
        if isinstance(out, tuple):
            out[0][0, -1, :] = sv.to(out[0].device)
        else:
            out[0, -1, :] = sv.to(out.device)
    
    h = target_module.register_forward_hook(patch_hook)
    with torch.no_grad():
        patched_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
    h.remove()
    
    diff = patched_logits - clean_logits
    l2 = torch.norm(diff).item()
    cos = torch.nn.functional.cosine_similarity(clean_logits.unsqueeze(0), patched_logits.unsqueeze(0)).item()
    return {'l2': l2, 'cos_sim': cos}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--component', type=str, required=True, choices=['resid', 'mlp', 'attn', 'all'])
    parser.add_argument('--n_pairs', type=int, default=200)
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model)
    n_layers = model.config.num_hidden_layers
    
    # 5 key layers
    layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    print(f"Layers: {layers}")
    
    out_dir = Path(f"results/causal_fiber/{args.model}_circuit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tense_pairs = generate_pairs(TENSE_TEMPLATES, args.n_pairs)
    polarity_pairs = generate_pairs(POLARITY_TEMPLATES, args.n_pairs)
    number_pairs = generate_pairs(NUMBER_TEMPLATES, min(args.n_pairs, 200))
    
    feature_pairs = {'tense': tense_pairs, 'polarity': polarity_pairs, 'number': number_pairs}
    
    components = ['resid', 'mlp', 'attn'] if args.component == 'all' else [args.component]
    
    for component in components:
        print(f"\n{'='*60}")
        print(f"Component: {component} ({args.n_pairs} pairs)")
        print(f"{'='*60}")
        
        comp_results = {}
        for layer_idx in layers:
            layer_name = f'L{layer_idx}'
            comp_results[layer_name] = {}
            
            for feature, pairs in feature_pairs.items():
                n_test = min(len(pairs), args.n_pairs)
                l2s, coss = [], []
                
                print(f"  {layer_name} {feature}: {n_test}...", end=' ', flush=True)
                t0 = time.time()
                
                for i in range(n_test):
                    a_text, b_text = pairs[i]
                    try:
                        result = patch_component(model, tokenizer, a_text, b_text, layer_idx, component)
                        if result is not None:
                            l2s.append(result['l2'])
                            coss.append(result['cos_sim'])
                    except Exception as e:
                        pass
                
                elapsed = time.time() - t0
                avg_l2 = np.mean(l2s) if l2s else 0
                std_l2 = np.std(l2s) if len(l2s) > 1 else 0
                avg_cos = np.mean(coss) if coss else 0
                
                comp_results[layer_name][feature] = {
                    'avg_l2': avg_l2, 'std_l2': std_l2,
                    'avg_cos_sim': avg_cos, 'n': len(l2s),
                }
                print(f"l2={avg_l2:.1f}±{std_l2:.1f}, cos={avg_cos:.4f}, t={elapsed:.1f}s")
        
        with open(out_dir / f'{component}_patching.json', 'w') as f:
            json.dump(comp_results, f, indent=2)
        print(f"Saved to {out_dir / f'{component}_patching.json'}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*60}")
    
    for component in components:
        fpath = out_dir / f'{component}_patching.json'
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            print(f"\n{component.upper()}:")
            for layer_name in sorted(data.keys()):
                feats = []
                for feat in ['tense', 'polarity', 'number']:
                    if feat in data[layer_name]:
                        d = data[layer_name][feat]
                        feats.append(f"{feat}={d['avg_l2']:.0f}")
                print(f"  {layer_name}: {', '.join(feats)}")
    
    print("\nDONE!")


if __name__ == '__main__':
    main()
