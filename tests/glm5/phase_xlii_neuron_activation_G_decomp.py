"""
Phase XLII-P254/255/256: 神经元级激活图样 + G项非线性逆向工程
================================================================

P253发现: 苹果独有签名≈0, 属性通道不正交, G残差比70-94%
核心问题: "苹果性"如果不是方向偏移, 那是什么? G项的非线性本质是什么?

三大实验:
  P254: 逐层逐神经元激活分析 — apple vs 其他水果, 找"苹果特异激活"的神经元集合
  P255: 稀疏激活模式(k-sparse) — 苹果/香蕉各激活哪些top-k神经元? 重叠度?
  P256: G项SVD分解 + 逐层G累积 — G残差到底是什么? 在哪些层开始显著?

实验模型: qwen3 -> deepseek7b -> glm4 (串行, 避免OOM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, gc, time, json, argparse
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_model_path(model_name):
    paths = {
        "qwen3": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
        "deepseek7b": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "glm4": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
    }
    return paths.get(model_name)

def load_model(model_name):
    p = get_model_path(model_name)
    p_abs = os.path.abspath(p)
    tok = AutoTokenizer.from_pretrained(p_abs, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        p_abs, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", device_map="cpu"
    )
    mdl = mdl.to("cuda")
    mdl.eval()
    device = next(mdl.parameters()).device
    return mdl, tok, device

STIMULI = {
    "fruit_family": ["apple","banana","pear","orange","grape","mango","strawberry","watermelon","cherry","peach","lemon","lime"],
    "animal_family": ["cat","dog","rabbit","horse","lion","eagle","elephant","dolphin","tiger","bear","fox","wolf"],
    "vehicle_family": ["car","bus","train","plane","boat","bicycle","truck","helicopter","motorcycle","ship","subway","taxi"],
    "furniture_family": ["chair","table","desk","sofa","bed","cabinet","shelf","bench","stool","dresser","couch","armchair"],
    "color_attrs": ["red","green","yellow","orange","brown","white","blue","black","pink","purple","gray","gold"],
    "taste_attrs": ["sweet","sour","bitter","salty","crisp","soft","spicy","fresh","tart","savory","rich","mild"],
    "fruit_color_combos": ["red apple","green apple","yellow banana","orange orange","green pear","red grape","yellow mango","red cherry","pink peach","yellow lemon"],
    "fruit_taste_combos": ["sweet apple","sour apple","sweet banana","sour orange","sweet pear","bitter grape","sweet mango","sweet cherry","tart lemon","sweet peach"],
    "animal_color_combos": ["brown cat","white dog","brown rabbit","black horse","golden eagle","white cat","black dog","orange tiger","brown bear","red fox"],
}

PROMPT_TEMPLATES = ["The {word} is", "A {word} can be", "This {word} has", "I see a {word}", "The {word} looks"]

TEST_TRIPLES = [
    ("apple","red","red apple"), ("apple","sweet","sweet apple"),
    ("apple","green","green apple"), ("banana","yellow","yellow banana"),
    ("banana","sweet","sweet banana"), ("pear","green","green pear"),
    ("cat","brown","brown cat"), ("dog","white","white dog"),
    ("car","red","red car"), ("car","fast","fast car"),
    ("apple","fresh","fresh apple"), ("horse","black","black horse"),
]


def collect_all_data(mdl, tok, device):
    """一次性收集所有hidden states"""
    n_layers = mdl.config.num_hidden_layers
    
    all_single = sorted(set(
        STIMULI["fruit_family"] + STIMULI["animal_family"] + 
        STIMULI["vehicle_family"] + STIMULI["color_attrs"] + STIMULI["taste_attrs"]
    ))
    all_combos = sorted(set(
        STIMULI["fruit_color_combos"] + STIMULI["fruit_taste_combos"] + STIMULI["animal_color_combos"]
    ))
    
    print(f"  收集: {len(all_single)}单字 + {len(all_combos)}组合词")
    
    single_hs = {}
    for i, word in enumerate(all_single):
        if i % 20 == 0:
            print(f"    单字 [{i+1}/{len(all_single)}]...")
        avg_hs = None
        for template in PROMPT_TEMPLATES:
            prompt = template.replace("{word}", word)
            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl(**inputs, output_hidden_states=True)
            hs = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
            avg_hs = hs if avg_hs is None else avg_hs + hs
            del out
        single_hs[word] = avg_hs / len(PROMPT_TEMPLATES)
    
    combo_hs = {}
    for i, combo in enumerate(all_combos):
        if i % 10 == 0:
            print(f"    组合 [{i+1}/{len(all_combos)}]...")
        inputs = tok(f"The {combo}", return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**inputs, output_hidden_states=True)
        combo_hs[combo] = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
        del out
    
    gc.collect()
    return single_hs, combo_hs


def p254(single_hs, n_layers, d_model):
    """P254: 神经元级激活分析 — 找苹果特异神经元"""
    print(f"\n  P254: 神经元级激活分析")
    
    results = {"per_layer": [], "apple_stable_neurons": []}
    
    for li in range(n_layers + 1):
        fruit_hvs = torch.stack([single_hs[w][li] for w in STIMULI["fruit_family"] if w in single_hs])
        non_fruit_hvs = torch.stack([single_hs[w][li] for w in STIMULI["animal_family"]+STIMULI["vehicle_family"] if w in single_hs])
        
        if "apple" not in single_hs:
            continue
        apple_hv = single_hs["apple"][li]
        
        # 水果特异性: |fruit_mean - non_fruit_mean| / pooled_std
        fruit_mean = fruit_hvs.mean(0)
        non_fruit_mean = non_fruit_hvs.mean(0)
        pooled_std = (fruit_hvs.std(0) + non_fruit_hvs.std(0)) / 2 + 1e-8
        fruit_spec = (fruit_mean - non_fruit_mean).abs() / pooled_std
        
        # 苹果特异性: |apple - other_fruit_mean| / other_fruit_std
        other_fruit = torch.stack([single_hs[w][li] for w in STIMULI["fruit_family"] if w != "apple" and w in single_hs])
        apple_spec = (apple_hv - other_fruit.mean(0)).abs() / (other_fruit.std(0) + 1e-8)
        
        top_fruit = fruit_spec.topk(min(50, d_model))
        top_apple = apple_spec.topk(min(50, d_model))
        overlap = len(set(top_fruit.indices.tolist()) & set(top_apple.indices.tolist()))
        
        # 线性探针
        all_hvs = torch.cat([fruit_hvs, non_fruit_hvs], dim=0)
        labels = np.array([1]*len(fruit_hvs) + [0]*len(non_fruit_hvs))
        
        acc_top50, acc_rand50 = -1, -1
        try:
            X_top = all_hvs[:, top_fruit.indices[:50].tolist()].numpy()
            clf = LogisticRegression(max_iter=500).fit(X_top, labels)
            acc_top50 = round(clf.score(X_top, labels), 4)
        except: pass
        try:
            ri = np.random.choice(d_model, min(50, d_model), replace=False)
            X_rand = all_hvs[:, ri.tolist()].numpy()
            clf2 = LogisticRegression(max_iter=500).fit(X_rand, labels)
            acc_rand50 = round(clf2.score(X_rand, labels), 4)
        except: pass
        
        ld = {
            "layer": li,
            "fruit_spec_top10_idx": top_fruit.indices[:10].tolist(),
            "apple_spec_top10_idx": top_apple.indices[:10].tolist(),
            "top50_overlap": overlap,
            "avg_fruit_spec": round(fruit_spec.mean().item(), 4),
            "max_fruit_spec": round(fruit_spec.max().item(), 4),
            "avg_apple_spec": round(apple_spec.mean().item(), 4),
            "max_apple_spec": round(apple_spec.max().item(), 4),
            "probe_top50": acc_top50,
            "probe_rand50": acc_rand50,
        }
        results["per_layer"].append(ld)
    
    # 跨层稳定的苹果特异神经元
    neuron_counts = {}
    for ld in results["per_layer"]:
        for idx in ld["apple_spec_top10_idx"]:
            neuron_counts[idx] = neuron_counts.get(idx, 0) + 1
    stable = sorted(neuron_counts.items(), key=lambda x: -x[1])
    stable = [(idx, cnt) for idx, cnt in stable if cnt >= 3]
    results["apple_stable_neurons"] = stable[:20]
    
    print(f"    苹果特异(跨3+层稳定): {len(stable)}个, top5={stable[:5]}")
    return results


def p255(single_hs, n_layers, d_model):
    """P255: 稀疏激活模式 — top-k神经元重叠度"""
    print(f"\n  P255: 稀疏激活模式")
    
    key_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]))
    results = {"sparse_patterns": []}
    
    for li in key_layers:
        ld = {"layer": li, "k_data": []}
        word_hvs = {w: single_hs[w][li] for w in single_hs}
        
        for k in [10, 50, 100, 500]:
            kd = {"k": k}
            
            # 各水果的top-k
            fruit_topk = {}
            for w in STIMULI["fruit_family"]:
                if w in word_hvs:
                    fruit_topk[w] = set(word_hvs[w].abs().topk(min(k, d_model)).indices.tolist())
            
            # 水果内重叠
            fw = list(fruit_topk.keys())
            pw = [len(fruit_topk[fw[i]] & fruit_topk[fw[j]]) / k 
                  for i in range(len(fw)) for j in range(i+1, len(fw))]
            kd["fruit_overlap"] = round(np.mean(pw), 4) if pw else 0
            
            # 苹果-水果
            if "apple" in fruit_topk:
                ao = [len(fruit_topk["apple"] & fruit_topk[w]) / k for w in fw if w != "apple"]
                kd["apple_fruit_overlap"] = round(np.mean(ao), 4) if ao else 0
            else:
                kd["apple_fruit_overlap"] = 0
            
            # 动物top-k
            animal_topk = {}
            for w in STIMULI["animal_family"]:
                if w in word_hvs:
                    animal_topk[w] = set(word_hvs[w].abs().topk(min(k, d_model)).indices.tolist())
            
            # 跨家族重叠
            co = [len(fruit_topk[fw[i]] & animal_topk[aw]) / k
                  for i in range(min(4, len(fw))) for aw in list(animal_topk.keys())[:4]
                  if fw[i] in fruit_topk and aw in animal_topk]
            kd["cross_overlap"] = round(np.mean(co), 4) if co else 0
            
            # k-sparse重建
            if "apple" in word_hvs:
                hv = word_hvs["apple"]
                idx = hv.abs().topk(min(k, d_model)).indices
                sparse = torch.zeros_like(hv)
                sparse[idx] = hv[idx]
                kd["reconstruct_cos"] = round(F.cosine_similarity(hv.unsqueeze(0), sparse.unsqueeze(0)).item(), 4)
            
            ld["k_data"].append(kd)
        
        results["sparse_patterns"].append(ld)
        
        for kd in ld["k_data"]:
            if kd["k"] == 100:
                print(f"    L{li} k=100: 水果内={kd['fruit_overlap']:.3f}, "
                      f"苹果-水果={kd['apple_fruit_overlap']:.3f}, "
                      f"跨家族={kd['cross_overlap']:.3f}, "
                      f"重建cos={kd.get('reconstruct_cos','N/A')}")
    return results


def p256(single_hs, combo_hs, n_layers, d_model):
    """P256: G项SVD分解 + 逐层累积"""
    print(f"\n  P256: G项非线性逆向工程")
    
    available = [(n, a, c) for n, a, c in TEST_TRIPLES if c in combo_hs and n in single_hs and a in single_hs]
    results = {"G_per_layer": [], "G_SVD": [], "G_accumulation": []}
    
    for noun, attr, combo in available:
        h_noun = single_hs[noun]
        h_attr = single_hs[attr]
        h_combo = combo_hs[combo]
        
        td = {"noun": noun, "attr": attr, "combo": combo, "layers": []}
        G_norms, G_ratios = [], []
        
        for li in range(n_layers + 1):
            hn, ha, hc = h_noun[li], h_attr[li], h_combo[li]
            G = hc - hn
            G_norm = G.norm().item()
            ad = ha - hn
            cos_G_attr = F.cosine_similarity(G.unsqueeze(0), ad.unsqueeze(0)).item()
            proj = (G @ ad / (ad.norm()**2 + 1e-8)) * ad
            G_res = (G - proj).norm().item()
            G_ratio = G_res / (G_norm + 1e-8)
            add_cos = F.cosine_similarity(hc.unsqueeze(0), (hn + ad).unsqueeze(0)).item()
            
            td["layers"].append({
                "layer": li, "G_norm": round(G_norm, 4), "cos_G_attr": round(cos_G_attr, 4),
                "G_ratio": round(G_ratio, 4), "add_cos": round(add_cos, 4),
            })
            G_norms.append(G_norm)
            G_ratios.append(G_ratio)
        
        growth = [G_norms[i+1]-G_norms[i] for i in range(len(G_norms)-1)]
        td["max_growth_L"] = growth.index(max(growth)) if growth else 0
        td["max_nonlinear_L"] = G_ratios.index(max(G_ratios))
        td["final_G_ratio"] = round(G_ratios[-1], 4)
        
        results["G_per_layer"].append(td)
        print(f"    {combo}: G增长最快L{td['max_growth_L']}, 最非线性L{td['max_nonlinear_L']}, "
              f"最终残差比={td['final_G_ratio']:.4f}")
    
    # G的PCA分析
    key_layers = sorted(set([n_layers//4, n_layers//2, 3*n_layers//4, n_layers]))
    for li in key_layers:
        G_mat = np.stack([combo_hs[c][li].numpy() - single_hs[n][li].numpy() for n, a, c in available])
        labels = [c for n, a, c in available]
        
        nc = min(10, G_mat.shape[0]-1, G_mat.shape[1])
        pca = PCA(n_components=nc).fit(G_mat)
        
        sd = {
            "layer": li, "n": len(available),
            "var_ratio": [round(v, 6) for v in pca.explained_variance_ratio_[:5]],
            "cum_var5": round(sum(pca.explained_variance_ratio_[:5]), 6),
            "mean_G_norm": round(np.mean(np.linalg.norm(G_mat, axis=1)), 4),
        }
        results["G_SVD"].append(sd)
        print(f"    L{li} G-PCA: 5d累积方差={sd['cum_var5']:.4f}, 平均G范数={sd['mean_G_norm']:.4f}")
    
    # 汇总
    for td in results["G_per_layer"]:
        results["G_accumulation"].append({
            "combo": td["combo"], "max_growth_L": td["max_growth_L"],
            "max_nonlinear_L": td["max_nonlinear_L"], "final_G_ratio": td["final_G_ratio"],
        })
    
    return results


def sanitize(obj):
    if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list): return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.ndarray): return sanitize(obj.tolist())
    if isinstance(obj, float): return round(obj, 6)
    if isinstance(obj, torch.Tensor): return sanitize(obj.tolist())
    return obj


def run_model(model_name):
    print(f"\n{'='*70}")
    print(f"Phase XLII-P254/255/256: 神经元级激活图样 + G项逆向工程 ({model_name})")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    t0 = time.time()
    mdl, tok, device = load_model(model_name)
    d_model = mdl.config.hidden_size
    n_layers = mdl.config.num_hidden_layers
    print(f"  模型: {model_name}, d_model={d_model}, n_layers={n_layers}")
    
    single_hs, combo_hs = collect_all_data(mdl, tok, device)
    
    # 释放模型节省显存
    del mdl; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"  模型已释放, 开始分析...")
    
    r254 = p254(single_hs, n_layers, d_model)
    r255 = p255(single_hs, n_layers, d_model)
    r256 = p256(single_hs, combo_hs, n_layers, d_model)
    
    # 核心发现
    findings = []
    
    # P254
    stable = r254.get("apple_stable_neurons", [])
    findings.append(f"[P254] 苹果特异神经元(跨3+层): {len(stable)}个, top5={stable[:5]}")
    for ld in r254["per_layer"]:
        if ld["layer"] == n_layers // 2:
            findings.append(f"[P254] L{ld['layer']}: 水果探针top50={ld['probe_top50']}, rand50={ld['probe_rand50']}")
    
    # P255
    for sp in r255["sparse_patterns"]:
        for kd in sp["k_data"]:
            if kd["k"] == 100 and sp["layer"] == n_layers // 2:
                findings.append(f"[P255] L{sp['layer']} k=100: 水果内重叠={kd['fruit_overlap']:.3f}, "
                              f"跨家族={kd['cross_overlap']:.3f}, 重建cos={kd.get('reconstruct_cos','N/A')}")
    
    # P256
    for acc in r256["G_accumulation"]:
        if "apple" in acc["combo"]:
            findings.append(f"[P256] {acc['combo']}: G增长L{acc['max_growth_L']}, "
                          f"非线性L{acc['max_nonlinear_L']}, 残差比={acc['final_G_ratio']:.4f}")
    for sd in r256["G_SVD"]:
        findings.append(f"[P256] L{sd['layer']}: G-PCA 5d累积方差={sd['cum_var5']:.4f}")
    
    elapsed = time.time() - t0
    output = sanitize({
        "experiment_id": "phase_xlii_p254_256",
        "model_name": model_name, "d_model": d_model, "n_layers": n_layers,
        "timestamp": datetime.now().isoformat(), "elapsed": round(elapsed, 1),
        "p254": r254, "p255": r255, "p256": r256,
        "core_findings": findings,
    })
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = OUT_DIR / f"phase_xlii_p254_256_{model_name}_{ts}.json"
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"\n  ===== 核心发现 =====")
    for f in findings: print(f"    {f}")
    print(f"\n  保存: {out_file}")
    print(f"  耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3", choices=["qwen3","deepseek7b","glm4"])
    args = parser.parse_args()
    run_model(args.model)

if __name__ == "__main__":
    main()
