"""
Phase CCIX: 因果效应的代数结构分析
========================================
关键实验:
1. 全层扫描 (every layer) — 精确拟合因果效应的层间增长曲线
2. 代数拟合 — 线性 vs 指数 vs 幂律 vs Sigmoid
3. Position-specific patching — 不同token位置的因果贡献
4. 中位数统计 — 避免l2长尾对均值的干扰
5. 语义特征 (semantic/sentiment) — 扩展特征类型

样本数: 300对/特征 (平衡速度和可靠性)
层数: 全层扫描 (每2层采样)
"""
import os, sys, gc, time, json, argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

PATHS = {
    "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
    "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

# ========== 句对模板 ==========
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

SEMANTIC = [
    ("The king ruled the kingdom", "The queen ruled the kingdom"),
    ("The boy ran to school", "The girl ran to school"),
    ("He fixed the broken car", "She fixed the broken car"),
    ("The man opened the door", "The woman opened the door"),
    ("His brother left the city", "Her sister left the city"),
    ("The father helped the child", "The mother helped the child"),
    ("The husband cooked dinner", "The wife cooked dinner"),
    ("The gentleman read the paper", "The lady read the paper"),
    ("The uncle visited the family", "The aunt visited the family"),
    ("The nephew played the game", "The niece played the game"),
    ("The actor performed on stage", "The actress performed on stage"),
    ("The hero saved the city", "The heroine saved the city"),
    ("The prince rode the horse", "The princess rode the horse"),
    ("The waiter served the food", "The waitress served the food"),
    ("The steward helped passengers", "The stewardess helped passengers"),
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
        if aug == 0:
            p = PREFIXES[np.random.randint(len(PREFIXES))]
            pairs.append((p+a, p+b))
        elif aug == 1:
            s = SUFFIXES[np.random.randint(len(SUFFIXES))]
            pairs.append((a+s, b+s))
        elif aug == 2:
            p = PREFIXES[np.random.randint(len(PREFIXES))]
            s = SUFFIXES[np.random.randint(len(SUFFIXES))]
            pairs.append((p+a+s, p+b+s))
        else:
            pairs.append((a, b))
    return pairs[:n]


# ========== 代数拟合函数 ==========
def fit_linear(x, a, b):
    return a * x + b

def fit_exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def fit_power(x, a, b, c):
    return a * np.power(x + 1, b) + c

def fit_sigmoid(x, L, k, x0, b):
    return L / (1 + np.exp(-k * (x - x0))) + b


def algebraic_fit(layers, l2_values):
    """对l2(layer)做4种代数拟合, 返回最优模型"""
    x = np.array(layers, dtype=float)
    y = np.array(l2_values, dtype=float)
    
    results = {}
    
    # 1. 线性拟合
    try:
        popt, _ = curve_fit(fit_linear, x, y, maxfev=5000)
        y_pred = fit_linear(x, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['linear'] = {'params': popt.tolist(), 'r2': r2, 'formula': f'{popt[0]:.1f}*x + {popt[1]:.1f}'}
    except:
        results['linear'] = {'r2': 0}
    
    # 2. 指数拟合
    try:
        popt, _ = curve_fit(fit_exponential, x, y, p0=[1, 0.05, 0], maxfev=10000)
        y_pred = fit_exponential(x, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['exponential'] = {'params': popt.tolist(), 'r2': r2, 'formula': f'{popt[0]:.1f}*exp({popt[1]:.4f}*x) + {popt[2]:.1f}'}
    except:
        results['exponential'] = {'r2': 0}
    
    # 3. 幂律拟合
    try:
        popt, _ = curve_fit(fit_power, x, y, p0=[1, 1, 0], maxfev=10000)
        y_pred = fit_power(x, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['power'] = {'params': popt.tolist(), 'r2': r2, 'formula': f'{popt[0]:.1f}*(x+1)^{popt[1]:.3f} + {popt[2]:.1f}'}
    except:
        results['power'] = {'r2': 0}
    
    # 4. Sigmoid拟合
    try:
        popt, _ = curve_fit(fit_sigmoid, x, y, p0=[max(y), 0.1, np.mean(x), min(y)], maxfev=10000)
        y_pred = fit_sigmoid(x, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['sigmoid'] = {'params': popt.tolist(), 'r2': r2, 'formula': f'{popt[0]:.1f}/(1+exp(-{popt[1]:.3f}*(x-{popt[2]:.1f}))) + {popt[3]:.1f}'}
    except:
        results['sigmoid'] = {'r2': 0}
    
    # 选择最优
    best_name = max(results, key=lambda k: results[k].get('r2', 0))
    results['best'] = best_name
    results['best_r2'] = results[best_name].get('r2', 0)
    
    return results


def patching_experiment(model, tokenizer, device, n_layers, pairs, layer_indices,
                        component='resid', pos_indices=None):
    """核心patching实验 — 支持position-specific"""
    l2_by_layer = {li: [] for li in layer_indices}
    cos_by_layer = {li: [] for li in layer_indices}
    pos_l2 = {}  # position-specific results
    
    for i, (a_text, b_text) in enumerate(pairs):
        if i % 50 == 0 and i > 0:
            print(f"    pair {i}/{len(pairs)}", flush=True)
        
        try:
            src_ids = tokenizer(a_text, return_tensors='pt')['input_ids'].to(device)
            clean_ids = tokenizer(b_text, return_tensors='pt')['input_ids'].to(device)
            
            for li in layer_indices:
                layer = model.model.layers[li]
                if component == 'resid':
                    target = layer
                elif component == 'attn':
                    target = layer.self_attn
                else:
                    target = layer.mlp
                
                # Capture source
                src_val = {}
                def cap(mod, inp, out, _sv=src_val):
                    if isinstance(out, tuple):
                        _sv['h'] = out[0].detach().clone()
                    else:
                        _sv['h'] = out.detach().clone()
                
                h = target.register_forward_hook(cap)
                with torch.no_grad():
                    _ = model(src_ids)
                h.remove()
                
                if 'h' not in src_val:
                    continue
                src_hidden = src_val['h']  # [1, seq_len, d_model]
                
                # Clean forward
                with torch.no_grad():
                    clean_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
                
                # Patched forward — patch last token position
                src_last = src_hidden[0, -1, :].to(device)
                def phook(mod, inp, out, _sv=src_last):
                    if isinstance(out, tuple):
                        out[0][0, -1, :] = _sv.to(out[0].device)
                    else:
                        out[0, -1, :] = _sv.to(out.device)
                
                h = target.register_forward_hook(phook)
                with torch.no_grad():
                    patched_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
                h.remove()
                
                diff = patched_logits - clean_logits
                l2_by_layer[li].append(torch.norm(diff).item())
                cos_by_layer[li].append(torch.nn.functional.cosine_similarity(
                    clean_logits.unsqueeze(0), patched_logits.unsqueeze(0)
                ).item())
            
            # Position-specific patching (only for last layer, only if requested)
            if pos_indices is not None and len(pos_indices) > 0:
                src_len = src_ids.shape[1]
                clean_len = clean_ids.shape[1]
                min_len = min(src_len, clean_len)
                
                for pos in pos_indices:
                    if pos >= min_len:
                        continue
                    li = n_layers - 1
                    layer = model.model.layers[li]
                    
                    # Capture source
                    src_val = {}
                    def cap_pos(mod, inp, out, _sv=src_val):
                        if isinstance(out, tuple):
                            _sv['h'] = out[0].detach().clone()
                        else:
                            _sv['h'] = out.detach().clone()
                    
                    h = layer.register_forward_hook(cap_pos)
                    with torch.no_grad():
                        _ = model(src_ids)
                    h.remove()
                    
                    if 'h' not in src_val:
                        continue
                    src_hidden = src_val['h']
                    src_pos_vec = src_hidden[0, pos, :].to(device)
                    
                    # Clean forward
                    with torch.no_grad():
                        clean_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
                    
                    # Patch at position pos
                    def phook_pos(mod, inp, out, _sv=src_pos_vec, _p=pos):
                        if isinstance(out, tuple):
                            out[0][0, _p, :] = _sv.to(out[0].device)
                        else:
                            out[0, _p, :] = _sv.to(out.device)
                    
                    h = layer.register_forward_hook(phook_pos)
                    with torch.no_grad():
                        patched_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
                    h.remove()
                    
                    diff = patched_logits - clean_logits
                    key = f'pos_{pos}'
                    if key not in pos_l2:
                        pos_l2[key] = []
                    pos_l2[key].append(torch.norm(diff).item())
                    
        except Exception as e:
            pass
    
    # 汇总统计 (mean, median, std)
    results = {}
    for li in layer_indices:
        vals = l2_by_layer[li]
        cos_vals = cos_by_layer[li]
        if vals:
            results[f'L{li}'] = {
                'avg_l2': float(np.mean(vals)),
                'median_l2': float(np.median(vals)),
                'std_l2': float(np.std(vals)),
                'avg_cos_sim': float(np.mean(cos_vals)),
                'median_cos_sim': float(np.median(cos_vals)),
                'n': len(vals),
                'all_l2': vals[:20],  # 保存前20个用于分析分布
            }
    
    # Position-specific results
    pos_results = {}
    for key, vals in pos_l2.items():
        if vals:
            pos_results[key] = {
                'avg_l2': float(np.mean(vals)),
                'median_l2': float(np.median(vals)),
                'n': len(vals),
            }
    
    return results, pos_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=300)
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'resid_full', 'pos_specific', 'semantic'])
    args = parser.parse_args()
    
    path = PATHS[args.model]
    print(f"{'='*70}")
    print(f"Phase CCIX: {args.model} — 因果效应代数结构分析")
    print(f"{'='*70}")
    
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
    print(f"  Loaded in {time.time()-t0:.1f}s: n_layers={n_layers}, device={device}")
    
    out_dir = Path(f"results/causal_fiber/{args.model}_circuit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    N = args.n_pairs
    
    # 全层采样 (每2层)
    layer_indices = list(range(0, n_layers, 2))
    if n_layers - 1 not in layer_indices:
        layer_indices.append(n_layers - 1)
    print(f"  Layer indices: {layer_indices}")
    
    tense_pairs = gen_pairs(TENSE, N)
    polarity_pairs = gen_pairs(POLARITY, N)
    number_pairs = gen_pairs(NUMBER, min(N, 150))
    semantic_pairs = gen_pairs(SEMANTIC, min(N, 150))
    
    all_results = {}
    algebraic_results = {}
    pos_results_all = {}
    
    # ========== S1: Residual Stream 全层扫描 ==========
    if args.test in ['all', 'resid_full']:
        print(f"\n{'='*60}")
        print(f"S1: Residual Stream Full-Layer Scan ({N} pairs, {len(layer_indices)} layers)")
        print(f"{'='*60}")
        
        feature_data = {'tense': tense_pairs, 'polarity': polarity_pairs, 'number': number_pairs}
        resid_full = {}
        
        for feature, pairs in feature_data.items():
            n_test = len(pairs)
            print(f"\n  Feature: {feature} ({n_test} pairs)")
            
            results, _ = patching_experiment(
                model, tokenizer, device, n_layers, pairs,
                layer_indices, component='resid'
            )
            resid_full[feature] = results
            
            # 打印即时结果
            for li in layer_indices:
                key = f'L{li}'
                if key in results:
                    d = results[key]
                    print(f"    {key}: mean_l2={d['avg_l2']:.1f}, median_l2={d['median_l2']:.1f}, cos={d['avg_cos_sim']:.4f}")
        
        all_results['resid_full'] = resid_full
        
        # 代数拟合
        print(f"\n  === ALGEBRAIC FITTING ===")
        for feature in ['tense', 'polarity', 'number']:
            if feature not in resid_full:
                continue
            layers_x = []
            l2_y = []
            median_y = []
            for li in layer_indices:
                key = f'L{li}'
                if key in resid_full[feature]:
                    layers_x.append(li)
                    l2_y.append(resid_full[feature][key]['avg_l2'])
                    median_y.append(resid_full[feature][key]['median_l2'])
            
            if len(layers_x) >= 4:
                fit_mean = algebraic_fit(layers_x, l2_y)
                fit_median = algebraic_fit(layers_x, median_y)
                algebraic_results[f'{feature}_mean'] = fit_mean
                algebraic_results[f'{feature}_median'] = fit_median
                
                print(f"\n  {feature} (mean): best={fit_mean['best']}, R2={fit_mean['best_r2']:.4f}")
                for name in ['linear', 'exponential', 'power', 'sigmoid']:
                    if name in fit_mean:
                        print(f"    {name}: R2={fit_mean[name].get('r2',0):.4f}, {fit_mean[name].get('formula','N/A')}")
                
                print(f"  {feature} (median): best={fit_median['best']}, R2={fit_median['best_r2']:.4f}")
                for name in ['linear', 'exponential', 'power', 'sigmoid']:
                    if name in fit_median:
                        print(f"    {name}: R2={fit_median[name].get('r2',0):.4f}, {fit_median[name].get('formula','N/A')}")
        
        # 保存
        with open(out_dir / 'resid_full_layer.json', 'w') as f:
            # 不保存all_l2到最终JSON (太大)
            save_data = {}
            for feat, layers_data in resid_full.items():
                save_data[feat] = {}
                for lk, ld in layers_data.items():
                    save_data[feat][lk] = {k: v for k, v in ld.items() if k != 'all_l2'}
            json.dump(save_data, f, indent=2)
        
        with open(out_dir / 'algebraic_fit.json', 'w') as f:
            json.dump(algebraic_results, f, indent=2)
        
        print(f"\n  Saved resid_full_layer.json and algebraic_fit.json")
    
    # ========== S2: Position-Specific Patching ==========
    if args.test in ['all', 'pos_specific']:
        print(f"\n{'='*60}")
        print(f"S2: Position-Specific Patching (last layer, {N} pairs)")
        print(f"{'='*60}")
        
        # 只测polarity特征 (最明显的因果效应)
        pos_indices = [0, 1, 2, 3, -3, -2, -1]  # 前4个 + 后3个位置
        
        # 先确定序列长度范围
        sample_lens = []
        for a, b in polarity_pairs[:20]:
            sample_lens.append(len(tokenizer.encode(a)))
            sample_lens.append(len(tokenizer.encode(b)))
        avg_len = int(np.mean(sample_lens))
        print(f"  Average sequence length: {avg_len}")
        
        # 补充中间位置
        mid_positions = list(range(4, max(5, avg_len-3), 2))
        all_pos = sorted(set([0, 1, 2, 3] + mid_positions + [avg_len-3, avg_len-2, avg_len-1]))
        print(f"  Testing positions: {all_pos}")
        
        pos_results, _ = patching_experiment(
            model, tokenizer, device, n_layers,
            polarity_pairs[:min(N, 200)],  # 只用200对节省时间
            layer_indices=[n_layers-1], component='resid',
            pos_indices=all_pos
        )
        
        # 上面只做了last-token patching; position-specific需要特殊处理
        # 重新做position-specific (每次patch不同位置)
        pos_l2_data = {}
        for pos in all_pos:
            pos_l2_data[f'pos_{pos}'] = []
        
        for i, (a_text, b_text) in enumerate(polarity_pairs[:min(N, 200)]):
            if i % 50 == 0 and i > 0:
                print(f"    pair {i}/{min(N,200)}", flush=True)
            
            try:
                src_ids = tokenizer(a_text, return_tensors='pt')['input_ids'].to(device)
                clean_ids = tokenizer(b_text, return_tensors='pt')['input_ids'].to(device)
                src_len = src_ids.shape[1]
                clean_len = clean_ids.shape[1]
                min_len = min(src_len, clean_len)
                
                last_layer = model.model.layers[n_layers-1]
                
                # Capture source hidden states at last layer
                src_val = {}
                def cap_all(mod, inp, out, _sv=src_val):
                    if isinstance(out, tuple):
                        _sv['h'] = out[0].detach().clone()
                    else:
                        _sv['h'] = out.detach().clone()
                
                h = last_layer.register_forward_hook(cap_all)
                with torch.no_grad():
                    _ = model(src_ids)
                h.remove()
                
                if 'h' not in src_val:
                    continue
                src_hidden = src_val['h']  # [1, seq_len, d_model]
                
                # Clean forward
                with torch.no_grad():
                    clean_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
                
                # Patch each position
                for pos in all_pos:
                    if pos >= min_len or pos < 0:
                        # 处理负数索引
                        actual_pos = pos
                        if pos < 0:
                            actual_pos = min_len + pos
                        if actual_pos < 0 or actual_pos >= min_len:
                            continue
                        pos_vec = src_hidden[0, actual_pos, :].to(device)
                    else:
                        pos_vec = src_hidden[0, pos, :].to(device)
                    
                    def phook_pos(mod, inp, out, _sv=pos_vec, _p=pos):
                        if isinstance(out, tuple):
                            actual = _p
                            if _p < 0:
                                actual = out[0].shape[1] + _p
                            if 0 <= actual < out[0].shape[1]:
                                out[0][0, actual, :] = _sv.to(out[0].device)
                        else:
                            actual = _p
                            if _p < 0:
                                actual = out.shape[1] + _p
                            if 0 <= actual < out.shape[1]:
                                out[0, actual, :] = _sv.to(out.device)
                    
                    h = last_layer.register_forward_hook(phook_pos)
                    with torch.no_grad():
                        patched_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
                    h.remove()
                    
                    diff = patched_logits - clean_logits
                    key = f'pos_{pos}'
                    if key in pos_l2_data:
                        pos_l2_data[key].append(torch.norm(diff).item())
                    
            except Exception as e:
                pass
        
        # 汇总
        pos_summary = {}
        for key, vals in pos_l2_data.items():
            if vals:
                pos_summary[key] = {
                    'avg_l2': float(np.mean(vals)),
                    'median_l2': float(np.median(vals)),
                    'n': len(vals),
                }
                print(f"  {key}: mean_l2={pos_summary[key]['avg_l2']:.1f}, median_l2={pos_summary[key]['median_l2']:.1f}")
        
        pos_results_all['polarity'] = pos_summary
        
        with open(out_dir / 'position_specific.json', 'w') as f:
            json.dump(pos_results_all, f, indent=2)
        
        print(f"\n  Saved position_specific.json")
    
    # ========== S3: 语义特征 Patching ==========
    if args.test in ['all', 'semantic']:
        print(f"\n{'='*60}")
        print(f"S3: Semantic Feature Patching ({min(N,150)} pairs)")
        print(f"{'='*60}")
        
        # 只测5层 (节省时间)
        sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
        results, _ = patching_experiment(
            model, tokenizer, device, n_layers,
            semantic_pairs[:min(N, 150)],
            sample_layers, component='resid'
        )
        
        all_results['semantic'] = results
        
        for lk in sorted(results.keys()):
            d = results[lk]
            print(f"  {lk}: mean_l2={d['avg_l2']:.1f}, median_l2={d['median_l2']:.1f}, cos={d['avg_cos_sim']:.4f}")
        
        # 代数拟合
        layers_x = []
        l2_y = []
        for li in sample_layers:
            key = f'L{li}'
            if key in results:
                layers_x.append(li)
                l2_y.append(results[key]['avg_l2'])
        
        if len(layers_x) >= 4:
            fit = algebraic_fit(layers_x, l2_y)
            algebraic_results['semantic_mean'] = fit
            print(f"\n  Semantic (mean): best={fit['best']}, R2={fit['best_r2']:.4f}")
            for name in ['linear', 'exponential', 'power', 'sigmoid']:
                if name in fit:
                    print(f"    {name}: R2={fit[name].get('r2',0):.4f}, {fit[name].get('formula','N/A')}")
        
        with open(out_dir / 'semantic_patching.json', 'w') as f:
            save_data = {}
            for lk, ld in results.items():
                save_data[lk] = {k: v for k, v in ld.items() if k != 'all_l2'}
            json.dump(save_data, f, indent=2)
        
        print(f"\n  Saved semantic_patching.json")
    
    # ========== 最终汇总 ==========
    print(f"\n{'='*70}")
    print(f"PHASE CCIX COMPLETE: {args.model}")
    print(f"{'='*70}")
    
    # 合并代数拟合结果
    if algebraic_results:
        with open(out_dir / 'algebraic_fit.json', 'w') as f:
            json.dump(algebraic_results, f, indent=2)
    
    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Model unloaded, GPU freed")


if __name__ == '__main__':
    main()
