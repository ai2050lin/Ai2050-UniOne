#!/usr/bin/env python3
"""
Stage 727: Phase XXII — 精确Activation Patching+层间修复归因+最后层分解
================================================================================
Phase XXI用近似方法发现: 后续层修复94~97%消融效果, 但无法精确测量。
Phase XXII使用register_forward_hook实现真正的activation patching。

理论框架: 语言计算结构(LCS)假设 — 语言在高维空间中形成特定流形
  P151: 精确activation patching — hook修改h_l后让信号通过剩余层传播
  P152: 逐层修复归因 — 零化后逐层测量信息恢复量, 定位"修复层"
  P153: 最后层分解 — RMSNorm vs LM Head各自贡献多少cos断裂?
  P154: 流形投影分析 — RMSNorm投影的几何性质(投影方向, 投影距离)
  P155: 层间因果流精确图 — 每层零化→传播→最终logit变化的完整因果链

测试规模: 30文本×全层, 3模型, 大数据量确保统计可靠性

用法: python stage727_phase22.py --model qwen3
      python stage727_phase22.py --model deepseek7b
      python stage727_phase22.py --model glm4
"""

import sys, time, gc, json, os, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from datetime import datetime
from collections import defaultdict

class Logger:
    def __init__(self, log_dir, name):
        os.makedirs(log_dir, exist_ok=True)
        self.f = open(os.path.join(log_dir, f"{name}.log"), "w", encoding="utf-8")
    def __call__(self, msg):
        try: print(msg)
        except UnicodeEncodeError:
            safe = msg.encode("gbk", errors="replace").decode("gbk")
            print(safe)
        self.f.write(msg + "\n")
        self.f.flush()
    def close(self):
        self.f.close()

log = None

def build_texts():
    T = []
    gen_en = [
        "The cat sat on the mat.", "A beautiful sunset over the ocean.", "The stock market crashed today.",
        "Climate change is a global challenge.", "The restaurant serves excellent Italian cuisine.",
        "The Renaissance was a period of great cultural achievement.", "The election results surprised everyone last night.",
        "The concert featured a symphony orchestra performance.", "She walked through the garden with a smile.",
        "The ancient castle stood on top of the hill.", "Breakfast is the most important meal of the day.",
        "The children played happily in the park.", "A sudden storm interrupted the picnic.",
        "The library has thousands of rare books.", "He traveled across three continents last year.",
        "The museum exhibit attracted record visitors.", "The teacher explained the concept clearly.",
        "Spring flowers bloom in every color.", "The old man told stories by the fireplace.",
        "Music brings people together across cultures.",
    ]
    for t in gen_en: T.append((t, "gen_en"))
    math_sci = [
        "Mathematical proof by induction.", "Einstein's theory of relativity changed physics.",
        "The quadratic formula solves ax^2+bx+c=0.", "Quantum entanglement connects distant particles.",
        "The Navier-Stokes equations describe fluid dynamics.", "Fermat's Last Theorem was proven by Andrew Wiles.",
        "The Riemann hypothesis remains unproven.", "Thermodynamics governs energy conservation.",
        "The periodic table organizes chemical elements.", "Bayesian inference updates probability distributions.",
        "The Fourier transform converts time to frequency domain.", "Topology studies properties preserved under deformation.",
        "Godel's incompleteness theorem limits formal systems.", "The standard model describes fundamental particles.",
        "Catalan numbers count binary trees.", "Group theory classifies symmetries.",
        "Information theory defines entropy as H=-sum(p*log(p)).", "The Boltzmann distribution describes statistical mechanics.",
        "Lambda calculus provides a foundation for computation.", "The Goldbach conjecture proposes every even number is the sum of two primes.",
    ]
    for t in math_sci: T.append((t, "math_sci"))
    poetry = [
        "Roses are red, violets are blue.", "The road not taken diverged in a yellow wood.",
        "Shall I compare thee to a summer's day?", "Two roads diverged in a wood, and I took the one less traveled.",
        "The fog comes on little cat feet.", "Hope is the thing with feathers that perches in the soul.",
        "In Xanadu did Kubla Khan a stately pleasure-dome decree.", "Do not go gentle into that good night.",
        "The Waste Land is T.S. Eliot's masterpiece.", "Emily Dickinson wrote about death and immortality.",
        "Haiku: an old silent pond / a frog jumps into the pond / splash! silence again.",
        "The Raven by Edgar Allan Poe features nevermore.", "i carry your heart with me by e.e. cummings.",
        "Daffodils by William Wordsworth celebrates nature.", "The Love Song of J. Alfred Prufrock is by T.S. Eliot.",
        "Ode to a Nightingale by John Keats explores beauty.", "Robert Frost won four Pulitzer Prizes for poetry.",
        "Free verse poetry breaks traditional meter.", "Sonnet 18 is Shakespeare's most famous poem.",
        "The Lake Isle of Innisfree by Yeats yearns for escape.", "Langston Hughes captured the Harlem Renaissance.",
    ]
    for t in poetry: T.append((t, "poetry"))
    code = [
        "def fibonacci(n): return n if n<2 else fibonacci(n-1)+fibonacci(n-2)",
        "for i in range(len(arr)): print(arr[i])",
        "class Node: def __init__(self, val): self.val=val; self.next=None",
        "x = [i**2 for i in range(10)]", "import numpy as np; A=np.random.randn(3,3)",
        "def quicksort(arr): if len(arr)<=1: return arr; pivot=arr[0]",
        "SELECT * FROM users WHERE age > 18 ORDER BY name;",
        "git commit -m 'fix: resolve null pointer exception'",
        "docker run -d -p 8080:80 nginx", "npm install express --save",
        "const sum = arr.reduce((a,b) => a+b, 0);",
        "public static void main(String[] args) { System.out.println('Hello'); }",
        "func handler(w http.ResponseWriter, r *http.Request) { fmt.Fprintf(w, 'OK') }",
        "print(f'The answer is {42}')", "x = torch.randn(2, 3, requires_grad=True)",
        "model.fit(X_train, y_train, epochs=100, batch_size=32)",
        "var xhr = new XMLHttpRequest(); xhr.open('GET', '/api/data');",
        "CREATE TABLE IF NOT EXISTS orders (id INT PRIMARY KEY, total DECIMAL);",
        "python -m pytest tests/ -v --cov=src", "curl -X POST -H 'Content-Type: application/json' localhost:5000",
        "async function fetchData() { const res = await fetch(url); return res.json(); }",
    ]
    for t in code: T.append((t, "code"))
    chinese = [
        "\u4eba\u5de5\u667a\u80fd\u6b63\u5728\u6539\u53d8\u4e16\u754c\u3002","\u4e2d\u56fd\u7684\u7ecf\u6d4e\u53d1\u5c55\u53d6\u5f97\u4e86\u663e\u8457\u6210\u5c31\u3002",
        "\u6559\u80b2\u662f\u56fd\u4e4b\u5927\u8ba1\uff0c\u515a\u4e4b\u5927\u8ba1\u3002","\u79d1\u6280\u81ea\u7acb\u81ea\u5f3a\u662f\u56fd\u5bb6\u53d1\u5c55\u7684\u6218\u7565\u652f\u6491\u3002",
        "\u6587\u5316\u81ea\u4fe1\u662f\u4e00\u4e2a\u6c11\u65cf\u6700\u57fa\u672c\u7684\u529b\u91cf\u3002","\u7eff\u6c34\u9752\u5c71\u5c31\u662f\u91d1\u5c71\u94f6\u5c71\u3002",
        "\u4eba\u6c11\u5bf9\u7f8e\u597d\u751f\u6d3b\u7684\u5411\u5f80\u5c31\u662f\u6211\u4eec\u7684\u594b\u6597\u76ee\u6807\u3002","\u521b\u65b0\u662f\u5f15\u9886\u53d1\u5c55\u7684\u7b2c\u4e00\u52a8\u529b\u3002",
        "\u6570\u5b57\u7ecf\u6d4e\u6210\u4e3a\u65b0\u7684\u589e\u957f\u5f15\u64ce\u3002","\u4e61\u6751\u632f\u5174\u6218\u7565\u5168\u9762\u63a8\u8fdb\u3002",
        "\u9ad8\u8d28\u91cf\u53d1\u5c55\u662f\u65f6\u4ee3\u7684\u8981\u6c42\u3002","\u5bf9\u5916\u5f00\u653e\u7684\u57fa\u672c\u56fd\u7b56\u4e0d\u4f1a\u6539\u53d8\u3002",
        "\u4e2d\u56fd\u5f0f\u73b0\u4ee3\u5316\u9053\u8def\u8d8a\u8d70\u8d8a\u5bbd\u5e7f\u3002","\u78b3\u4e2d\u548c\u76ee\u6807\u63a8\u52a8\u80fd\u6e90\u8f6c\u578b\u3002",
        "\u4e2d\u533b\u836f\u4f20\u627f\u521b\u65b0\u53d1\u5c55\u3002","\u4f20\u7edf\u6587\u5316\u7684\u521b\u9020\u6027\u8f6c\u5316\u548c\u521b\u65b0\u6027\u53d1\u5c55\u3002",
        "\u56fd\u5bb6\u6cbb\u7406\u4f53\u7cfb\u548c\u6cbb\u7406\u80fd\u529b\u73b0\u4ee3\u5316\u3002","\u6784\u5efa\u4eba\u7c7b\u547d\u8fd0\u5171\u540c\u4f53\u3002",
        "\u4e00\u5e26\u4e00\u8def\u5021\u8bae\u4fc3\u8fdb\u5171\u540c\u53d1\u5c55\u3002","\u592a\u7a7a\u63a2\u7d22\u53d6\u5f97\u91cd\u5927\u7a81\u7834\u3002",
    ]
    for t in chinese: T.append((t, "chinese"))
    philosophy = [
        "I think, therefore I am.", "The unexamined life is not worth living.",
        "To be or not to be, that is the question.", "Knowledge is power.",
        "The only true wisdom is in knowing you know nothing.", "Existence precedes essence.",
        "Man is condemned to be free.", "The categorical imperative demands universal maxims.",
        "All perception is theory-laden.", "Truth is correspondence between thought and reality.",
        "Utilitarianism maximizes happiness.", "The social contract governs political legitimacy.",
        "Phenomenology studies structures of experience.", "Hermeneutics is the art of interpretation.",
        "Nihilism denies inherent meaning.", "Pragmatism judges truth by practical consequences.",
        "Rationalism prioritizes reason as source of knowledge.", "Empiricism derives knowledge from experience.",
        "The mind-body problem asks how mental relates to physical.", "Free will debates determinism versus agency.",
    ]
    for t in philosophy: T.append((t, "philosophy"))
    return T

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
}

def load_model(mname):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = MODEL_MAP[mname]
    log(f"  Loading {mname} from {p.name}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(p), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg = model.config
    n_layers = getattr(cfg, 'num_hidden_layers', None) or getattr(cfg, 'n_layers', None)
    d_model = getattr(cfg, 'hidden_size', None) or getattr(cfg, 'd_model', 2048)
    log(f"  {mname}: n_layers={n_layers}, d_model={d_model}, device={model.device}")
    return model, tokenizer, n_layers, d_model

def get_unembed(model):
    if hasattr(model, 'lm_head'):
        um = model.lm_head
    elif hasattr(model, 'get_output_embeddings'):
        um = model.get_output_embeddings()
    else:
        return None, None
    w = um.weight.detach().to(torch.float32).cpu()
    b = um.bias.detach().to(torch.float32).cpu() if (hasattr(um, 'bias') and um.bias is not None) else None
    return w, b

def kl_div(p, q):
    p = p + 1e-10
    q = q + 1e-10
    return (p * (p.log() - q.log())).sum().item()

def get_model_layers(model):
    """Get the transformer layers from the model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers, model.model.norm
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h, model.transformer.ln_f
    return None, None

# =====================================================================
# Core: True Activation Patching via Forward Hooks
# =====================================================================
def patch_and_forward(model, tokenizer, text, patch_layer, zero_dim=None, n_zero=5):
    """
    Use register_forward_hook to modify hidden states at patch_layer.
    Zero out top-K abs-value dimensions at the last token position.
    Returns: (logits, hidden_states) after patching.
    """
    layers, norm = get_model_layers(model)
    if layers is None:
        return None, None
    
    patch_applied = [False]
    
    def hook_fn(module, input, output):
        if patch_applied[0]:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h_mod = h.clone()
        
        # Zero dimensions at last token
        h_last = h_mod[0, -1, :]
        if zero_dim is not None:
            h_mod[0, -1, zero_dim] = 0.0
        else:
            top_dims = torch.topk(h_last.abs(), n_zero).indices
            for d in top_dims:
                h_mod[0, -1, d] = 0.0
        
        patch_applied[0] = True
        if isinstance(output, tuple):
            return (h_mod,) + output[1:]
        return h_mod
    
    handle = layers[patch_layer].register_forward_hook(hook_fn)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    handle.remove()
    logits = outputs.logits[0, -1, :].float().cpu()
    h_final = outputs.hidden_states[-1][:, -1, :].float().cpu().squeeze(0)
    
    return logits, h_final


def clean_forward(model, tokenizer, text):
    """Clean forward pass without any patching."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits[0, -1, :].float().cpu()
    h_final = outputs.hidden_states[-1][:, -1, :].float().cpu().squeeze(0)
    h_all = {}
    for l in range(len(outputs.hidden_states)):
        h_all[l] = outputs.hidden_states[l][:, -1, :].float().cpu().squeeze(0)
    return logits, h_final, h_all


# =====================================================================
# P151: True Activation Patching — 精确层传播消融
# =====================================================================
def p151_true_activation_patching(model, tokenizer, texts, uw, ub, n_layers, d_model):
    """
    P151: Use forward hooks to zero h at layer l, then propagate through remaining layers.
    Compare with Phase XXI approximation (direct h_l @ W^T).
    
    Key: This gives the TRUE propagated effect, not an approximation.
    """
    log("\n" + "="*80)
    log("P151: True Activation Patching (forward hook)")
    log("="*80)
    
    n_texts = 30
    texts_sub = []
    cats = ["gen_en", "math_sci", "poetry", "code", "chinese"]
    for c in cats:
        ct = [(t, cc) for t, cc in texts if cc == c][:6]
        texts_sub.extend(ct)
    texts_sub = texts_sub[:n_texts]
    
    # Sample layers to test (every few layers)
    step = max(1, n_layers // 8)
    test_layers = list(range(0, n_layers, step))
    if n_layers - 1 not in test_layers:
        test_layers.append(n_layers - 1)
    
    log(f"  Test layers: {test_layers}")
    log(f"  Texts: {len(texts_sub)}")
    
    results = {"per_layer": {}}
    
    for layer_idx in test_layers:
        kl_direct_list = []  # Phase XX approximation
        kl_patched_list = []  # True activation patching
        top1_chg_patched = 0
        cos_h_final_list = []  # cos between patched and clean h_final
        
        for text, cat in texts_sub:
            # Clean forward
            logits_clean, h_clean, _ = clean_forward(model, tokenizer, text)
            probs_clean = F.softmax(logits_clean, dim=-1)
            orig_top1 = torch.argmax(probs_clean).item()
            
            # Direct ablation (Phase XX approximation)
            h_l = _get_h_at_layer(model, tokenizer, text, layer_idx)
            if h_l is not None:
                top_dim = torch.argmax(h_l.abs()).item()
                h_direct = h_l.clone()
                h_direct[top_dim] = 0.0
                logits_direct = h_direct @ uw.T + (ub if ub is not None else 0)
                probs_direct = F.softmax(logits_direct, dim=-1)
                kl_direct = kl_div(probs_clean, probs_direct)
                kl_direct_list.append(kl_direct)
            
            # True activation patching
            logits_patched, h_patched = patch_and_forward(model, tokenizer, text, layer_idx)
            if logits_patched is not None:
                probs_patched = F.softmax(logits_patched, dim=-1)
                kl_patched = kl_div(probs_clean, probs_patched)
                kl_patched_list.append(kl_patched)
                
                new_top1 = torch.argmax(probs_patched).item()
                if new_top1 != orig_top1:
                    top1_chg_patched += 1
                
                # cos between patched and clean final h
                cos_h = F.cosine_similarity(h_patched.unsqueeze(0), h_clean.unsqueeze(0)).item()
                cos_h_final_list.append(cos_h)
        
        n_direct = len(kl_direct_list)
        n_patched = len(kl_patched_list)
        n_h = len(cos_h_final_list)
        
        avg_kl_direct = np.mean(kl_direct_list) if n_direct > 0 else 0
        avg_kl_patched = np.mean(kl_patched_list) if n_patched > 0 else 0
        tpc_patched = top1_chg_patched / max(n_patched, 1) * 100
        avg_cos_h = np.mean(cos_h_final_list) if n_h > 0 else 0
        
        ratio = avg_kl_direct / max(avg_kl_patched, 0.001) if avg_kl_patched > 0.001 else 0
        
        results["per_layer"][str(layer_idx)] = {
            "kl_direct": round(avg_kl_direct, 3),
            "kl_patched": round(avg_kl_patched, 3),
            "kl_ratio": round(ratio, 2),
            "top1_chg_patched": round(tpc_patched, 1),
            "cos_h_final": round(avg_cos_h, 4),
        }
        log(f"  L{layer_idx:2d}: KL_direct={avg_kl_direct:.2f}, KL_patched={avg_kl_patched:.2f}, "
            f"ratio={ratio:.1f}x, top1_chg={tpc_patched:.0f}%, cos_h={avg_cos_h:.4f}")
    
    # Summary
    non_last = [v for k, v in results["per_layer"].items() if int(k) < n_layers - 1]
    if non_last:
        avg_ratio = np.mean([v["kl_ratio"] for v in non_last])
        log(f"\n  INV-477: True patching ratio (non-last) = {avg_ratio:.1f}x")
        log(f"  (Phase XXI approximation ratio was ~16-39x; true hooks give ground truth)")
    
    return results


def _get_h_at_layer(model, tokenizer, text, layer_idx):
    """Get hidden state at a specific layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    if layer_idx < len(outputs.hidden_states):
        return outputs.hidden_states[layer_idx][:, -1, :].float().cpu().squeeze(0)
    return None


# =====================================================================
# P152: Layer-by-Layer Repair Attribution
# =====================================================================
def p152_repair_attribution(model, tokenizer, texts, uw, ub, n_layers, d_model):
    """
    P152: Zero h at layer l, then measure at each subsequent layer how much
    the representation has "recovered" compared to clean.
    
    Metrics at each layer m > l:
      - cos(h_m_zeroed_l, h_m_clean): how similar is the zeroed representation?
      - KL(h_m @ W^T, logits_clean): how close are the logits?
    
    If cos increases from layer l to N → repair is happening.
    If cos stays low → damage persists.
    """
    log("\n" + "="*80)
    log("P152: Layer-by-Layer Repair Attribution")
    log("="*80)
    
    n_texts = 10  # Fewer texts but more granular layers
    texts_sub = []
    cats = ["gen_en", "math_sci", "poetry", "code", "chinese"]
    for c in cats:
        ct = [(t, cc) for t, cc in texts if cc == c][:2]
        texts_sub.extend(ct)
    texts_sub = texts_sub[:n_texts]
    
    # Zero at layer 0 (earliest) and mid layer
    zero_layers = [0, n_layers // 2]
    
    results = {}
    
    for zero_l in zero_layers:
        log(f"\n  --- Zero at L{zero_l}, track recovery ---")
        
        cos_recovery = []  # (layer, avg_cos) for each layer > zero_l
        kl_recovery = []
        
        for text, cat in texts_sub:
            # Clean: get all h layers
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)
            with torch.no_grad():
                outputs_clean = model(**inputs, output_hidden_states=True)
            
            logits_clean = outputs_clean.logits[0, -1, :].float().cpu()
            probs_clean = F.softmax(logits_clean, dim=-1)
            h_clean_all = {}
            for l_idx in range(len(outputs_clean.hidden_states)):
                h_clean_all[l_idx] = outputs_clean.hidden_states[l_idx][:, -1, :].float().cpu().squeeze(0)
            
            # Patched: zero at zero_l, track all subsequent layers
            # Need to hook at zero_l and collect h at all subsequent layers
            collected_h = {}
            
            def make_collect_hook(collect_dict, target_layers):
                def hook_fn(module, input, output):
                    # This collects h at each layer after patching
                    pass
                return hook_fn
            
            # Alternative: do sequential forward through layers
            # Get h at zero_l
            h_start = h_clean_all[zero_l].clone()
            top_dim = torch.argmax(h_start.abs()).item()
            h_start[top_dim] = 0.0
            
            # For each subsequent layer, run patch_and_forward with collection
            # Simpler: run multiple patch_and_forward with hooks at different layers
            for track_l in range(zero_l, min(zero_l + 20, n_layers)):
                # Track at this layer by hooking at track_l and zeroing at zero_l
                if track_l == zero_l:
                    # At zero layer, h is already zeroed
                    cos_val = F.cosine_similarity(
                        h_start.unsqueeze(0), h_clean_all[track_l].unsqueeze(0)
                    ).item()
                    kl_val = kl_div(probs_clean, F.softmax(h_start @ uw.T + (ub if ub is not None else 0), dim=-1))
                else:
                    # Need to actually run through layers zero_l to track_l with zeroed h
                    # Use the patch_and_forward approach but collect at track_l
                    layers, norm = get_model_layers(model)
                    if layers is None:
                        continue
                    
                    collect_data = {}
                    def make_track_hook(tl, cd):
                        def hf(module, input, output):
                            h = output[0] if isinstance(output, tuple) else output
                            cd[tl] = h[:, -1, :].float().cpu().squeeze(0)
                            return output
                        return hf
                    
                    # Zero at zero_l
                    zero_applied = [False]
                    def zero_hook(module, input, output):
                        if zero_applied[0]:
                            return output
                        h = output[0] if isinstance(output, tuple) else output
                        h_mod = h.clone()
                        h_mod[0, -1, top_dim] = 0.0
                        zero_applied[0] = True
                        if isinstance(output, tuple):
                            return (h_mod,) + output[1:]
                        return h_mod
                    
                    h0 = layers[zero_l].register_forward_hook(zero_hook)
                    h1 = layers[track_l].register_forward_hook(make_track_hook(track_l, collect_data))
                    
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=False)
                    
                    h0.remove()
                    h1.remove()
                    zero_applied[0] = False
                    
                    if track_l in collect_data:
                        h_tracked = collect_data[track_l]
                        cos_val = F.cosine_similarity(
                            h_tracked.unsqueeze(0), h_clean_all[track_l].unsqueeze(0)
                        ).item()
                        logits_tracked = h_tracked @ uw.T + (ub if ub is not None else 0)
                        probs_tracked = F.softmax(logits_tracked, dim=-1)
                        kl_val = kl_div(probs_clean, probs_tracked)
                    else:
                        continue
                
                # Collect per-text
                if len(cos_recovery) <= track_l - zero_l:
                    cos_recovery.append([])
                    kl_recovery.append([])
                idx = track_l - zero_l
                cos_recovery[idx].append(cos_val)
                kl_recovery[idx].append(kl_val)
        
        # Average and log
        layer_results = []
        for i, (cos_list, kl_list) in enumerate(zip(cos_recovery, kl_recovery)):
            l = zero_l + i
            if cos_list:
                avg_cos = np.mean(cos_list)
                avg_kl = np.mean(kl_list)
                layer_results.append({"layer": l, "cos": round(avg_cos, 4), "kl": round(avg_kl, 3)})
                log(f"    L{l:2d}: cos_recovery={avg_cos:.4f}, KL={avg_kl:.3f}")
        
        results[f"zero_L{zero_l}"] = layer_results
    
    # Find "repair layers" — where cos starts increasing
    for key, layer_results in results.items():
        zero_l = int(key.split("L")[1])
        for i in range(1, len(layer_results)):
            prev_cos = layer_results[i-1]["cos"]
            curr_cos = layer_results[i]["cos"]
            if curr_cos > prev_cos + 0.05:  # Significant increase
                log(f"  INV-478: Repair starts at L{layer_results[i]['layer']} "
                    f"(cos: {prev_cos:.3f} → {curr_cos:.3f}) after zeroing L{zero_l}")
                break
    
    return results


# =====================================================================
# P153: Last-Layer Decomposition — RMSNorm + LM Head
# =====================================================================
def p153_last_layer_decomposition(model, tokenizer, texts, uw, ub, n_layers, d_model):
    """
    P153: What exactly does the last layer do?
    Decompose: h_{N-1} → RMSNorm → h_N → LM_Head(W) → logits
    
    Measure:
      (A) cos(h_{N-1} @ W^T, logits_final) — before RMSNorm
      (B) cos(RMSNorm(h_{N-1}) @ W^T, logits_final) — after RMSNorm, before bias
      (C) cos(logits_final, logits_final) = 1.0 — reference
      
    The "break" from P150 (cos 0.93→0.09-0.47) is it due to RMSNorm or LM Head?
    """
    log("\n" + "="*80)
    log("P153: Last-Layer Decomposition (RMSNorm + LM Head)")
    log("="*80)
    
    n_texts = 30
    texts_sub = []
    cats = ["gen_en", "math_sci", "poetry", "code", "chinese"]
    for c in cats:
        ct = [(t, cc) for t, cc in texts if cc == c][:6]
        texts_sub.extend(ct)
    texts_sub = texts_sub[:n_texts]
    
    results = {}
    
    # Measure at each of the last 5 layers
    for check_l in range(max(0, n_layers-5), n_layers+1):
        cos_before_norm = []
        cos_after_norm = []
        cos_with_bias = []
        norm_ratio_list = []
        
        for text, cat in texts_sub:
            h = _get_h_at_layer(model, tokenizer, text, check_l)
            if h is None:
                continue
            
            # Get final logits
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits_final = outputs.logits[0, -1, :].float().cpu()
            
            # (A) Before RMSNorm: h @ W^T
            if check_l < n_layers:
                logits_before_norm = h @ uw.T + (ub if ub is not None else 0)
                cos_bn = F.cosine_similarity(logits_before_norm.unsqueeze(0), logits_final.unsqueeze(0)).item()
                cos_before_norm.append(cos_bn)
            
            # (B) After RMSNorm: RMSNorm(h) @ W^T
            layers, norm = get_model_layers(model)
            if norm is not None and check_l < n_layers:
                h_normed = norm(h.unsqueeze(0).unsqueeze(0).to(model.device))
                h_normed = h_normed[0, -1, :].float().cpu()
                logits_after_norm = h_normed @ uw.T + (ub if ub is not None else 0)
                cos_an = F.cosine_similarity(logits_after_norm.unsqueeze(0), logits_final.unsqueeze(0)).item()
                cos_after_norm.append(cos_an)
                
                # Norm ratio
                norm_ratio = h_normed.norm().item() / max(h.norm().item(), 1e-10)
                norm_ratio_list.append(norm_ratio)
        
        avg_cos_bn = np.mean(cos_before_norm) if cos_before_norm else 0
        avg_cos_an = np.mean(cos_after_norm) if cos_after_norm else 0
        avg_norm_ratio = np.mean(norm_ratio_list) if norm_ratio_list else 0
        
        results[f"L{check_l}"] = {
            "cos_before_norm": round(avg_cos_bn, 4),
            "cos_after_norm": round(avg_cos_an, 4),
            "norm_ratio": round(avg_norm_ratio, 4),
        }
        
        if check_l == n_layers - 1:
            label = "(h_{N-1})"
        elif check_l == n_layers:
            label = "(h_N = after last transformer layer)"
        else:
            label = ""
        
        log(f"  L{check_l:2d} {label}: "
            f"cos_before_RMSNorm={avg_cos_bn:.4f}, "
            f"cos_after_RMSNorm={avg_cos_an:.4f}, "
            f"norm_ratio={avg_norm_ratio:.4f}")
    
    # Key analysis: How much does RMSNorm contribute to the "break"?
    l_n1 = results.get(f"L{n_layers-1}", {})
    cos_before = l_n1.get("cos_before_norm", 0)
    cos_after = l_n1.get("cos_after_norm", 0)
    
    log(f"\n  INV-479: RMSNorm contribution at L{n_layers-1}:")
    log(f"    cos before RMSNorm: {cos_before:.4f}")
    log(f"    cos after RMSNorm:  {cos_after:.4f}")
    log(f"    RMSNorm improvement: {cos_after - cos_before:.4f}")
    
    if cos_after > 0.9 and cos_before < 0.3:
        log("    → RMSNorm is the PRIMARY cause of the last-layer 'break'!")
    elif cos_after > cos_before + 0.3:
        log("    → RMSNorm significantly improves cos but doesn't fully explain the break")
    else:
        log("    → LM Head (linear projection) is the primary cause, not RMSNorm")
    
    return results


# =====================================================================
# P154: Manifold Projection Analysis
# =====================================================================
def p154_manifold_projection(model, tokenizer, texts, uw, ub, n_layers, d_model):
    """
    P154: Analyze the geometric properties of RMSNorm "projection".
    
    RMSNorm: h_normed = h / sqrt(mean(h^2) + eps) * gamma
    
    Key questions:
    1. What direction does RMSNorm project onto? (hypersphere of radius gamma?)
    2. Does RMSNorm move h closer to or farther from the "manifold"?
    3. What is the angle between h and h_normed? (should be 0 if just scaling)
    
    Actually, RMSNorm is just scaling + learned per-dim scaling:
      h_normed = h * gamma / RMS(h)
    where RMS(h) = sqrt(mean(h_i^2))
    
    So the direction doesn't change! The angle between h and h_normed is 0.
    But the SCALE changes dramatically (norm_ratio can be very large or small).
    
    The "projection" metaphor: RMSNorm projects h onto a specific-radius hypersurface.
    """
    log("\n" + "="*80)
    log("P154: Manifold Projection Analysis (RMSNorm geometry)")
    log("="*80)
    
    n_texts = 30
    texts_sub = []
    cats = ["gen_en", "math_sci", "poetry", "code", "chinese"]
    for c in cats:
        ct = [(t, cc) for t, cc in texts if cc == c][:6]
        texts_sub.extend(ct)
    texts_sub = texts_sub[:n_texts]
    
    layers, norm = get_model_layers(model)
    if norm is None:
        log("  ERROR: Cannot find norm layer")
        return {}
    
    results = {"per_layer": {}}
    
    # Analyze RMSNorm at several layers
    check_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1, n_layers]
    
    for check_l in check_layers:
        norm_before_list = []
        norm_after_list = []
        norm_ratio_list = []
        # Check if direction actually changes (should be 0 for pure RMSNorm)
        angle_list = []
        # gamma distribution
        gamma_vals = None
        
        for text, cat in texts_sub:
            h = _get_h_at_layer(model, tokenizer, text, check_l)
            if h is None:
                continue
            
            h_input = h.unsqueeze(0).unsqueeze(0).to(model.device)
            
            with torch.no_grad():
                h_normed = norm(h_input)
            
            h_normed = h_normed[0, 0, :].float().cpu()
            
            norm_before = h.norm().item()
            norm_after = h_normed.norm().item()
            norm_before_list.append(norm_before)
            norm_after_list.append(norm_after)
            
            if norm_before > 1e-10:
                norm_ratio_list.append(norm_after / norm_before)
            
            # Angle (should be 0 for pure RMSNorm)
            cos_angle = F.cosine_similarity(h.unsqueeze(0), h_normed.unsqueeze(0)).item()
            angle_list.append(cos_angle)
            
            # Get gamma parameter
            if gamma_vals is None and hasattr(norm, 'weight'):
                gamma_vals = norm.weight.detach().float().cpu()
        
        avg_norm_before = np.mean(norm_before_list)
        avg_norm_after = np.mean(norm_after_list)
        avg_ratio = np.mean(norm_ratio_list)
        avg_angle = np.mean(angle_list)
        
        results["per_layer"][f"L{check_l}"] = {
            "norm_before": round(avg_norm_before, 2),
            "norm_after": round(avg_norm_after, 2),
            "norm_ratio": round(avg_ratio, 4),
            "cos_angle": round(avg_angle, 6),
        }
        
        log(f"  L{check_l:2d}: ||h||={avg_norm_before:.2f} → ||RMSNorm(h)||={avg_norm_after:.2f}, "
            f"ratio={avg_ratio:.4f}, cos(h, RMSNorm(h))={avg_angle:.6f}")
    
    # Gamma analysis
    if gamma_vals is not None:
        log(f"\n  RMSNorm gamma: mean={gamma_vals.mean():.4f}, std={gamma_vals.std():.4f}, "
            f"min={gamma_vals.min():.4f}, max={gamma_vals.max():.4f}")
        log(f"  Gamma is {'learned' if gamma_vals.std() > 0.01 else 'fixed (all ones)'}")
    
    # Key insight: RMSNorm is just scaling (angle = 1.0), NOT rotation
    avg_angles = [v["cos_angle"] for v in results["per_layer"].values()]
    if all(a > 0.9999 for a in avg_angles):
        log("\n  INV-480: RMSNorm is PURE SCALING (no rotation) — cos(h, RMSNorm(h)) ≈ 1.000")
        log("  → The 'break' at the last layer is NOT caused by RMSNorm changing direction")
        log("  → It's caused by the SCALE CHANGE: large ||h|| → fixed ||RMSNorm(h)||")
    
    return results


# =====================================================================
# P155: Complete Causal Flow Map
# =====================================================================
def p155_causal_flow_map(model, tokenizer, texts, uw, ub, n_layers, d_model):
    """
    P155: Build a complete causal flow map.
    For each layer l (0 to N-1):
      1. Zero h_l's top dimension → propagate → measure final KL and top1_chg
      2. Measure how much the final h_N changes (cos)
      3. Identify "bottleneck layers" (where zeroing causes max final damage)
    """
    log("\n" + "="*80)
    log("P155: Complete Causal Flow Map")
    log("="*80)
    
    n_texts = 15  # Fewer texts for full-layer sweep
    texts_sub = []
    cats = ["gen_en", "math_sci", "poetry", "code", "chinese"]
    for c in cats:
        ct = [(t, cc) for t, cc in texts if cc == c][:3]
        texts_sub.extend(ct)
    texts_sub = texts_sub[:n_texts]
    
    layer_kl = {}
    layer_top1_chg = {}
    layer_cos_h = {}
    
    step = max(1, n_layers // 15)
    test_layers = list(range(0, n_layers, step))
    if n_layers - 1 not in test_layers:
        test_layers.append(n_layers - 1)
    
    for layer_idx in test_layers:
        kl_list = []
        top1_list = []
        cos_list = []
        
        for text, cat in texts_sub:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)
            
            # Clean
            with torch.no_grad():
                outputs_clean = model(**inputs, output_hidden_states=True)
            logits_clean = outputs_clean.logits[0, -1, :].float().cpu()
            probs_clean = F.softmax(logits_clean, dim=-1)
            h_clean_final = outputs_clean.hidden_states[-1][:, -1, :].float().cpu().squeeze(0)
            orig_top1 = torch.argmax(probs_clean).item()
            
            # Patched
            logits_patched, h_patched_final = patch_and_forward(model, tokenizer, text, layer_idx)
            if logits_patched is not None:
                probs_patched = F.softmax(logits_patched, dim=-1)
                kl = kl_div(probs_clean, probs_patched)
                kl_list.append(kl)
                
                new_top1 = torch.argmax(probs_patched).item()
                top1_list.append(1 if new_top1 != orig_top1 else 0)
                
                cos_val = F.cosine_similarity(
                    h_patched_final.unsqueeze(0), h_clean_final.unsqueeze(0)
                ).item()
                cos_list.append(cos_val)
        
        if kl_list:
            avg_kl = np.mean(kl_list)
            avg_top1 = np.mean(top1_list) * 100
            avg_cos = np.mean(cos_list)
            layer_kl[layer_idx] = round(avg_kl, 3)
            layer_top1_chg[layer_idx] = round(avg_top1, 1)
            layer_cos_h[layer_idx] = round(avg_cos, 4)
            log(f"  L{layer_idx:2d}: KL={avg_kl:.3f}, top1_chg={avg_top1:.0f}%, cos_h_final={avg_cos:.4f}")
    
    # Find bottleneck layers (max KL)
    sorted_by_kl = sorted(layer_kl.items(), key=lambda x: -x[1])
    top_3_kl = sorted_by_kl[:3]
    bottom_3_kl = sorted_by_kl[-3:]
    
    log(f"\n  Top-3 bottleneck layers (highest KL): {[(f'L{l}', f'{kl:.2f}') for l, kl in top_3_kl]}")
    log(f"  Bottom-3 layers (lowest KL): {[(f'L{l}', f'{kl:.2f}') for l, kl in bottom_3_kl]}")
    
    # Causal flow pattern
    layers_sorted = sorted(layer_kl.keys())
    if len(layers_sorted) > 2:
        first_third = [layer_kl[l] for l in layers_sorted[:len(layers_sorted)//3]]
        mid_third = [layer_kl[l] for l in layers_sorted[len(layers_sorted)//3:2*len(layers_sorted)//3]]
        last_third = [layer_kl[l] for l in layers_sorted[2*len(layers_sorted)//3:]]
        
        log(f"\n  INV-481: Causal flow pattern:")
        log(f"    First third layers: avg KL = {np.mean(first_third):.3f}")
        log(f"    Middle third layers: avg KL = {np.mean(mid_third):.3f}")
        log(f"    Last third layers: avg KL = {np.mean(last_third):.3f}")
        
        if np.mean(first_third) > np.mean(last_third) * 2:
            log("    → Early layers are bottleneck (most causal damage)")
        elif np.mean(last_third) > np.mean(first_third) * 2:
            log("    → Late layers are bottleneck (most causal damage)")
        else:
            log("    → Causal damage is evenly distributed across layers")
    
    return {
        "layer_kl": layer_kl,
        "layer_top1_chg": layer_top1_chg,
        "layer_cos_h": layer_cos_h,
    }


# =====================================================================
# Main
# =====================================================================
def main():
    global log
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()
    mname = args.model
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = _Path(f"tests/glm5_temp/stage727_phase22_{mname}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    log = Logger(str(out_dir), "results")
    
    log(f"Phase XXII: Stage 727 — Activation Patching+修复归因+最后层分解")
    log(f"Model: {mname}, Time: {ts}")
    log(f"Output: {out_dir}")
    
    t0 = time.time()
    
    model, tokenizer, n_layers, d_model = load_model(mname)
    uw, ub = get_unembed(model)
    if uw is None:
        log("ERROR: Cannot get unembed matrix!")
        return
    
    # Verify hook compatibility
    layers, norm = get_model_layers(model)
    if layers is None:
        log("ERROR: Cannot find model layers for hooking!")
        return
    log(f"  Hook compatible: {len(layers)} layers, norm={'yes' if norm else 'no'}")
    
    texts = build_texts()
    log(f"Texts: {len(texts)} total, {len(set(c for _,c in texts))} categories")
    
    results = {}
    
    # P151
    try:
        log(f"\n{'#'*80}")
        log(f"# P151: True Activation Patching (t={time.time()-t0:.0f}s)")
        log(f"{'#'*80}")
        results["p151"] = p151_true_activation_patching(model, tokenizer, texts, uw, ub, n_layers, d_model)
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        log(f"  P151 ERROR: {e}")
        import traceback; traceback.print_exc()
        results["p151"] = {"error": str(e)}
    
    # P152
    try:
        log(f"\n{'#'*80}")
        log(f"# P152: Repair Attribution (t={time.time()-t0:.0f}s)")
        log(f"{'#'*80}")
        results["p152"] = p152_repair_attribution(model, tokenizer, texts, uw, ub, n_layers, d_model)
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        log(f"  P152 ERROR: {e}")
        import traceback; traceback.print_exc()
        results["p152"] = {"error": str(e)}
    
    # P153
    try:
        log(f"\n{'#'*80}")
        log(f"# P153: Last-Layer Decomposition (t={time.time()-t0:.0f}s)")
        log(f"{'#'*80}")
        results["p153"] = p153_last_layer_decomposition(model, tokenizer, texts, uw, ub, n_layers, d_model)
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        log(f"  P153 ERROR: {e}")
        import traceback; traceback.print_exc()
        results["p153"] = {"error": str(e)}
    
    # P154
    try:
        log(f"\n{'#'*80}")
        log(f"# P154: Manifold Projection (t={time.time()-t0:.0f}s)")
        log(f"{'#'*80}")
        results["p154"] = p154_manifold_projection(model, tokenizer, texts, uw, ub, n_layers, d_model)
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        log(f"  P154 ERROR: {e}")
        import traceback; traceback.print_exc()
        results["p154"] = {"error": str(e)}
    
    # P155
    try:
        log(f"\n{'#'*80}")
        log(f"# P155: Causal Flow Map (t={time.time()-t0:.0f}s)")
        log(f"{'#'*80}")
        results["p155"] = p155_causal_flow_map(model, tokenizer, texts, uw, ub, n_layers, d_model)
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        log(f"  P155 ERROR: {e}")
        import traceback; traceback.print_exc()
        results["p155"] = {"error": str(e)}
    
    elapsed = time.time() - t0
    results["meta"] = {"model": mname, "n_layers": n_layers, "d_model": d_model,
                       "elapsed_s": round(elapsed, 1), "timestamp": ts}
    
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    log(f"\n{'='*80}")
    log(f"Phase XXII complete! Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log(f"Results saved to: {out_dir / 'results.json'}")
    log(f"{'='*80}")
    
    log.close()


if __name__ == "__main__":
    main()
