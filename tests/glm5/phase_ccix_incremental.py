"""
Phase CCIX: 因果效应代数结构 + Median统计 + 语义特征
增量保存版本: 每层完成后保存, 防止崩溃丢失数据
日志同时输出到文件
"""
import os, sys, gc, time, json, argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 日志输出到文件
class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'w', buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

PATHS = {
    "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
    "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

TENSE = [
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

POLARITY = [
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

NUMBER = [
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

SENTIMENT = [
    ("I love this amazing product", "I hate this terrible product"),
    ("The experience was wonderful", "The experience was awful"),
    ("She felt joyful and grateful", "She felt angry and resentful"),
    ("This is the best day ever", "This is the worst day ever"),
    ("The gift was thoughtful and kind", "The insult was cruel and mean"),
    ("We had a fantastic celebration", "We had a dreadful argument"),
    ("The performance was outstanding", "The performance was pathetic"),
    ("He spoke with warmth and care", "He spoke with coldness and spite"),
    ("The results exceeded expectations", "The results fell below expectations"),
    ("She smiled with genuine happiness", "She frowned with deep sorrow"),
    ("The music was uplifting and inspiring", "The noise was annoying and disturbing"),
    ("His advice was helpful and wise", "His advice was harmful and foolish"),
    ("The meal was a delightful treat", "The meal was a terrible disappointment"),
    ("They showed great courage and bravery", "They showed great fear and cowardice"),
    ("The story had a happy ending", "The story had a tragic ending"),
    ("I admire her dedication and effort", "I despise her laziness and neglect"),
    ("The garden was a paradise of beauty", "The dump was a wasteland of ugliness"),
    ("She embraced him with love", "She rejected him with hatred"),
    ("The news brought hope and relief", "The news brought despair and panic"),
    ("His actions showed generosity", "His actions showed greed"),
    ("The atmosphere was friendly and warm", "The atmosphere was hostile and cold"),
    ("She praised his excellent work", "She criticized his poor work"),
    ("The day was filled with laughter", "The day was filled with tears"),
    ("We celebrated our great victory", "We mourned our bitter defeat"),
    ("The future looks bright and promising", "The future looks bleak and hopeless"),
]

SEMANTIC_TOPIC = [
    ("The doctor examined the patient carefully", "The chef prepared the meal carefully"),
    ("Scientists discovered a new particle", "Artists created a new painting"),
    ("The engine roared to life", "The orchestra played to life"),
    ("She solved the complex equation", "She wrote the complex poem"),
    ("The rocket launched into orbit", "The ship sailed into harbor"),
    ("He programmed the computer algorithm", "He composed the musical symphony"),
    ("The experiment yielded interesting data", "The novel yielded interesting insights"),
    ("The telescope observed distant galaxies", "The microscope observed tiny organisms"),
    ("She calculated the trajectory precisely", "She choreographed the dance precisely"),
    ("The reactor generated enormous power", "The storm generated enormous waves"),
    ("The mathematician proved the theorem", "The philosopher proved the argument"),
    ("The satellite orbited the planet", "The moon orbited the earth"),
    ("He analyzed the chemical compound", "He analyzed the literary work"),
    ("The bridge spanned the wide river", "The rainbow spanned the wide sky"),
    ("The laser cut through the metal", "The scissors cut through the fabric"),
    ("She researched the historical period", "She explored the geographical region"),
    ("The microscope revealed cell structures", "The telescope revealed star patterns"),
    ("The algorithm sorted the database", "The librarian sorted the collection"),
    ("The vaccine prevented the disease", "The umbrella prevented the soaking"),
    ("The code compiled without errors", "The speech delivered without pauses"),
]

PREFIXES = ["Actually, ", "In fact, ", "Indeed, ", "Clearly, ", "Surely, ",
            "Perhaps, ", "Maybe, ", "Certainly, ", "Obviously, ", "Naturally, "]
SUFFIXES = [" right now", " today", " this time", " as expected", " for sure"]

def gen_pairs(templates, n, seed=42):
    np.random.seed(seed)
    pairs = list(templates)
    while len(pairs) < n:
        idx = np.random.randint(len(templates))
        a, b = templates[idx]
        aug = np.random.randint(4)
        if aug == 0: p = PREFIXES[np.random.randint(len(PREFIXES))]; pairs.append((p+a, p+b))
        elif aug == 1: s = SUFFIXES[np.random.randint(len(SUFFIXES))]; pairs.append((a+s, b+s))
        elif aug == 2: p = PREFIXES[np.random.randint(len(PREFIXES))]; s = SUFFIXES[np.random.randint(len(SUFFIXES))]; pairs.append((p+a+s, p+b+s))
        else: pairs.append((a, b))
    return pairs[:n]

def fit_algebraic_model(layers, l2_values):
    x = np.array(layers, dtype=float)
    y = np.array(l2_values, dtype=float)
    results = {}
    for name, fit_fn in [
        ("linear", lambda: _polyfit(x, y, 1)),
        ("quadratic", lambda: _polyfit(x, y, 2)),
        ("exponential", lambda: _exp_fit(x, y)),
        ("power_law", lambda: _power_fit(x, y)),
        ("logarithmic", lambda: _log_fit(x, y)),
    ]:
        try:
            results[name] = fit_fn()
        except:
            results[name] = {"r2": 0}
    return results

def _polyfit(x, y, deg):
    coeffs = np.polyfit(x, y, deg)
    y_pred = np.polyval(coeffs, x)
    r2 = 1 - np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2) if np.sum((y-np.mean(y))**2) > 0 else 0
    return {"coeffs": [float(c) for c in coeffs], "r2": float(r2)}

def _exp_fit(x, y):
    y_pos = np.maximum(y, 1e-6)
    log_y = np.log(y_pos)
    coeffs = np.polyfit(x, log_y, 1)
    y_pred = np.exp(coeffs[1]) * np.exp(coeffs[0] * x)
    r2 = 1 - np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2) if np.sum((y-np.mean(y))**2) > 0 else 0
    return {"a": float(np.exp(coeffs[1])), "b": float(coeffs[0]), "r2": float(r2)}

def _power_fit(x, y):
    x_pos = np.maximum(x, 1.0)
    y_pos = np.maximum(y, 1e-6)
    log_x = np.log(x_pos)
    log_y = np.log(y_pos)
    valid = np.isfinite(log_x) & np.isfinite(log_y)
    if valid.sum() < 2: return {"r2": 0}
    coeffs = np.polyfit(log_x[valid], log_y[valid], 1)
    y_pred = np.exp(coeffs[1]) * x_pos ** coeffs[0]
    r2 = 1 - np.sum((y[valid]-y_pred[valid])**2) / np.sum((y[valid]-np.mean(y[valid]))**2) if np.sum((y[valid]-np.mean(y[valid]))**2) > 0 else 0
    return {"a": float(np.exp(coeffs[1])), "b": float(coeffs[0]), "r2": float(r2)}

def _log_fit(x, y):
    log_x = np.log(x + 1)
    valid = np.isfinite(log_x) & np.isfinite(y)
    if valid.sum() < 2: return {"r2": 0}
    coeffs = np.polyfit(log_x[valid], y[valid], 1)
    y_pred = coeffs[0] * log_x + coeffs[1]
    r2 = 1 - np.sum((y[valid]-y_pred[valid])**2) / np.sum((y[valid]-np.mean(y[valid]))**2) if np.sum((y[valid]-np.mean(y[valid]))**2) > 0 else 0
    return {"a": float(coeffs[0]), "b": float(coeffs[1]), "r2": float(r2)}


def save_incremental(out_dir, all_results):
    """Incremental save after each layer."""
    with open(out_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=200)
    args = parser.parse_args()

    # Setup logging
    out_dir = Path(f"results/causal_fiber/{args.model}_ccix")
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(out_dir / 'run.log')

    path = PATHS[args.model]
    print(f"\n{'='*60}")
    print(f"Phase CCIX: {args.model}")
    print(f"Path: {path}")
    print(f"N_pairs: {args.n_pairs}")
    print(f"{'='*60}")

    # Load model (与成功脚本一致!)
    print(f"[{time.strftime('%H:%M:%S')}] Loading model...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model in ["glm4", "deepseek7b"]:
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
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    print(f"[{time.strftime('%H:%M:%S')}] Loaded in {time.time()-t0:.1f}s: n_layers={n_layers}, device={device}", flush=True)

    N = args.n_pairs
    resid_layers = [int(i * (n_layers - 1) / 7) for i in range(8)]
    comp_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]

    print(f"Resid layers: {resid_layers}")
    print(f"Comp layers: {comp_layers}", flush=True)

    tense_pairs = gen_pairs(TENSE, N)
    polarity_pairs = gen_pairs(POLARITY, N)
    number_pairs = gen_pairs(NUMBER, min(N, len(NUMBER)))
    sentiment_pairs = gen_pairs(SENTIMENT, min(N, len(SENTIMENT)))
    topic_pairs = gen_pairs(SEMANTIC_TOPIC, min(N, len(SEMANTIC_TOPIC)))

    feature_data = {
        'tense': tense_pairs, 'polarity': polarity_pairs,
        'number': number_pairs, 'sentiment': sentiment_pairs,
        'semantic_topic': topic_pairs,
    }

    all_results = {}

    # ===== S1: Residual全层扫描 =====
    print(f"\n{'='*60}\nS1: Residual全层扫描\n{'='*60}", flush=True)
    resid_results = {}

    for layer_idx in resid_layers:
        layer_name = f'L{layer_idx}'
        resid_results[layer_name] = {}

        for feature, pairs in feature_data.items():
            n_test = min(len(pairs), N)
            l2s = []

            print(f"  {layer_name} {feature}: {n_test}...", end=' ', flush=True)
            t_start = time.time()

            for i in range(n_test):
                a_text, b_text = pairs[i]
                try:
                    src_ids = tokenizer(a_text, return_tensors='pt')['input_ids'].to(device)
                    clean_ids = tokenizer(b_text, return_tensors='pt')['input_ids'].to(device)
                    layer = model.model.layers[layer_idx]

                    src_val = {}
                    def cap(mod, inp, out):
                        if isinstance(out, tuple):
                            src_val['h'] = out[0][0, -1, :].detach().clone()
                        else:
                            src_val['h'] = out[0, -1, :].detach().clone()

                    h = layer.register_forward_hook(cap)
                    with torch.no_grad(): _ = model(src_ids)
                    h.remove()
                    if 'h' not in src_val: continue
                    sv = src_val['h']

                    with torch.no_grad():
                        clean_logits = model(clean_ids).logits[0, -1].detach().cpu().float()

                    def phook(mod, inp, out, _sv=sv):
                        if isinstance(out, tuple):
                            out[0][0, -1, :] = _sv.to(out[0].device)
                        else:
                            out[0, -1, :] = _sv.to(out.device)

                    h = layer.register_forward_hook(phook)
                    with torch.no_grad():
                        patched_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
                    h.remove()

                    diff = patched_logits - clean_logits
                    l2s.append(torch.norm(diff).item())
                except Exception as e:
                    print(f"E({e})", end='', flush=True)

            elapsed = time.time() - t_start
            if l2s:
                resid_results[layer_name][feature] = {
                    'mean_l2': float(np.mean(l2s)), 'median_l2': float(np.median(l2s)),
                    'std_l2': float(np.std(l2s)), 'n': len(l2s),
                }
                print(f"mean={np.mean(l2s):.1f}, med={np.median(l2s):.1f}, t={elapsed:.1f}s", flush=True)
            else:
                print(f"FAILED", flush=True)

        # 增量保存
        all_results['resid'] = resid_results
        save_incremental(out_dir, all_results)
        print(f"  [Saved after {layer_name}]", flush=True)

    # ===== S2: Attn/MLP =====
    print(f"\n{'='*60}\nS2: Attn/MLP\n{'='*60}", flush=True)
    N_comp = min(N, 150)
    comp_results = {}

    for component in ['attn', 'mlp']:
        comp_results[component] = {}

        for layer_idx in comp_layers:
            layer_name = f'L{layer_idx}'
            comp_results[component][layer_name] = {}

            for feature in ['tense', 'polarity', 'number']:
                pairs = feature_data[feature]
                n_test = min(len(pairs), N_comp)
                l2s = []

                print(f"  {component} {layer_name} {feature}...", end=' ', flush=True)
                t_start = time.time()

                for i in range(n_test):
                    a_text, b_text = pairs[i]
                    try:
                        src_ids = tokenizer(a_text, return_tensors='pt')['input_ids'].to(device)
                        clean_ids = tokenizer(b_text, return_tensors='pt')['input_ids'].to(device)
                        layer = model.model.layers[layer_idx]
                        target = layer.self_attn if component == 'attn' else layer.mlp

                        src_val = {}
                        def cap(mod, inp, out):
                            if isinstance(out, tuple):
                                src_val['h'] = out[0][0, -1, :].detach().clone()
                            else:
                                src_val['h'] = out[0, -1, :].detach().clone()

                        h = target.register_forward_hook(cap)
                        with torch.no_grad(): _ = model(src_ids)
                        h.remove()
                        if 'h' not in src_val: continue
                        sv = src_val['h']

                        with torch.no_grad():
                            clean_logits = model(clean_ids).logits[0, -1].detach().cpu().float()

                        def phook(mod, inp, out, _sv=sv):
                            if isinstance(out, tuple):
                                out[0][0, -1, :] = _sv.to(out[0].device)
                            else:
                                out[0, -1, :] = _sv.to(out.device)

                        h = target.register_forward_hook(phook)
                        with torch.no_grad():
                            patched_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
                        h.remove()

                        diff = patched_logits - clean_logits
                        l2s.append(torch.norm(diff).item())
                    except:
                        pass

                elapsed = time.time() - t_start
                if l2s:
                    comp_results[component][layer_name][feature] = {
                        'mean_l2': float(np.mean(l2s)), 'median_l2': float(np.median(l2s)),
                        'n': len(l2s),
                    }
                    print(f"med={np.median(l2s):.1f}, t={elapsed:.1f}s", flush=True)
                else:
                    print(f"FAILED", flush=True)

            all_results['components'] = comp_results
            save_incremental(out_dir, all_results)

    # Contribution
    print(f"\n--- 贡献比 ---", flush=True)
    contrib = {}
    for layer_idx in comp_layers:
        ln = f'L{layer_idx}'
        contrib[ln] = {}
        for feat in ['tense', 'polarity', 'number']:
            a = comp_results['attn'].get(ln, {}).get(feat, {}).get('median_l2', 0)
            m = comp_results['mlp'].get(ln, {}).get(feat, {}).get('median_l2', 0)
            total = a + m if (a + m) > 0 else 1
            contrib[ln][feat] = {'attn_l2': a, 'mlp_l2': m,
                'attn_pct': round(a/total*100,1), 'mlp_pct': round(m/total*100,1)}
            print(f"  {ln} {feat}: attn={a:.1f}({a/total*100:.0f}%), mlp={m:.1f}({m/total*100:.0f}%)")
    all_results['contribution'] = contrib
    save_incremental(out_dir, all_results)

    # ===== S3: 语义特征 =====
    print(f"\n{'='*60}\nS3: 语义特征\n{'='*60}", flush=True)
    sem_results = {}
    for layer_idx in resid_layers:
        ln = f'L{layer_idx}'
        sem_results[ln] = {}
        for feature in ['sentiment', 'semantic_topic']:
            pairs = feature_data[feature]
            n_test = min(len(pairs), N)
            l2s = []
            print(f"  {ln} {feature}...", end=' ', flush=True)
            t_start = time.time()
            for i in range(n_test):
                a_text, b_text = pairs[i]
                try:
                    src_ids = tokenizer(a_text, return_tensors='pt')['input_ids'].to(device)
                    clean_ids = tokenizer(b_text, return_tensors='pt')['input_ids'].to(device)
                    layer = model.model.layers[layer_idx]
                    src_val = {}
                    def cap(mod, inp, out):
                        if isinstance(out, tuple):
                            src_val['h'] = out[0][0, -1, :].detach().clone()
                        else:
                            src_val['h'] = out[0, -1, :].detach().clone()
                    h = layer.register_forward_hook(cap)
                    with torch.no_grad(): _ = model(src_ids)
                    h.remove()
                    if 'h' not in src_val: continue
                    sv = src_val['h']
                    with torch.no_grad():
                        clean_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
                    def phook(mod, inp, out, _sv=sv):
                        if isinstance(out, tuple):
                            out[0][0, -1, :] = _sv.to(out[0].device)
                        else:
                            out[0, -1, :] = _sv.to(out.device)
                    h = layer.register_forward_hook(phook)
                    with torch.no_grad():
                        patched_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
                    h.remove()
                    diff = patched_logits - clean_logits
                    l2s.append(torch.norm(diff).item())
                except:
                    pass
            elapsed = time.time() - t_start
            if l2s:
                sem_results[ln][feature] = {
                    'mean_l2': float(np.mean(l2s)), 'median_l2': float(np.median(l2s)),
                    'n': len(l2s),
                }
                print(f"med={np.median(l2s):.1f}, t={elapsed:.1f}s", flush=True)
            else:
                print(f"FAILED", flush=True)
        all_results['semantic'] = sem_results
        save_incremental(out_dir, all_results)

    # ===== S4: 代数拟合 =====
    print(f"\n{'='*60}\nS4: 代数拟合\n{'='*60}", flush=True)
    algebraic_fits = {}
    for feature in list(feature_data.keys()):
        layers_x, l2_means, l2_medians = [], [], []
        for layer_idx in resid_layers:
            key = f'L{layer_idx}'
            if key in resid_results and feature in resid_results[key]:
                layers_x.append(float(layer_idx))
                l2_means.append(resid_results[key][feature]['mean_l2'])
                l2_medians.append(resid_results[key][feature]['median_l2'])
        if len(layers_x) >= 3:
            fits = {"mean": fit_algebraic_model(layers_x, l2_means),
                    "median": fit_algebraic_model(layers_x, l2_medians)}
            algebraic_fits[feature] = fits
            best_mean = max(fits["mean"].items(), key=lambda x: x[1].get("r2", 0))
            best_med = max(fits["median"].items(), key=lambda x: x[1].get("r2", 0))
            print(f"  {feature} best: mean={best_mean[0]} R²={best_mean[1].get('r2',0):.4f}, median={best_med[0]} R²={best_med[1].get('r2',0):.4f}")
            for mtype, vals in fits["median"].items():
                if vals.get("r2", 0) > 0:
                    print(f"    {mtype}: R²={vals['r2']:.4f}")

    all_results['algebraic_fits'] = algebraic_fits
    save_incremental(out_dir, all_results)

    # ===== FINAL SUMMARY =====
    print(f"\n{'='*60}\nFINAL: {args.model}\n{'='*60}", flush=True)
    print(f"\nResidual l2 (mean/median):")
    for ln in sorted(resid_results.keys()):
        parts = []
        for feat in ['tense', 'polarity', 'number', 'sentiment', 'semantic_topic']:
            if feat in resid_results[ln]:
                d = resid_results[ln][feat]
                parts.append(f"{feat}={d['mean_l2']:.0f}/{d['median_l2']:.0f}")
        print(f"  {ln}: {', '.join(parts)}")

    print(f"\nBest Algebraic Fit:")
    for feat, fits in algebraic_fits.items():
        if "median" in fits:
            best = max(fits["median"].items(), key=lambda x: x[1].get("r2", 0))
            print(f"  {feat}: {best[0]} R²={best[1].get('r2',0):.4f}")

    print(f"\nAttn vs MLP:")
    for ln in sorted(contrib.keys()):
        parts = []
        for feat in ['tense', 'polarity', 'number']:
            if feat in contrib[ln]:
                d = contrib[ln][feat]
                parts.append(f"{feat}:{d['attn_pct']:.0f}/{d['mlp_pct']:.0f}")
        print(f"  {ln}: {', '.join(parts)} (attn/mlp%)")

    print(f"\nDONE! Saved to {out_dir}", flush=True)


if __name__ == '__main__':
    main()
