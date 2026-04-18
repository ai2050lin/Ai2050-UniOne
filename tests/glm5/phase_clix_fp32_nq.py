"""
Phase CLIX-FP32: 对抗平衡的数学结构分析 (全精度版, 无量化)
============================================================
核心改进:
1. 使用torch.float32 (全精度), 不使用float16/bfloat16量化
2. 使用Hook方式获取每层的Attention输出(A)和FFN输出(G)
3. 避免手动调用层(device_map="auto"时不安全)
4. 逐模型测试, 避免GPU内存溢出
5. 结果保存到独立目录, 便于与FP16结果对比
"""

import torch
import numpy as np
from pathlib import Path
import json
import time
import gc

MODELS = {
    "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
    "deepseek": "D:/develop/model/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60",
}
TEST_WORDS = ["apple", "banana", "cat", "dog", "run", "red", "the", "is", "beautiful", "mountain"]
TEMPLATE = "The word I am thinking of is"
RESULTS_DIR = Path("d:/Ai2050/TransformerLens-Project/results/clix_adversarial_balance_fp32")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def to_np(t):
    return t.detach().float().cpu().numpy()


def load_model_fp32(model_key):
    """全精度加载模型, GPU放不下则使用CPU"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = MODELS[model_key]
    print(f"[FP32] Loading {model_key} from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 先加载到CPU (全精度FP32, 无量化)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,              # 全精度!
        device_map="cpu",                # 先加载到CPU
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # 尝试整体移到CUDA
    target_device = "cpu"  # 默认CPU
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        model_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"[FP32] GPU显存: {gpu_mem/1e9:.1f}GB, 模型需要: {model_mem/1e9:.1f}GB")

        if model_mem < gpu_mem * 0.85:
            print(f"[FP32] 模型可以放入GPU, 尝试移到cuda...")
            try:
                model = model.to("cuda")
                target_device = "cuda"
                print(f"[FP32] 成功! 模型全部在GPU上")
            except RuntimeError as e:
                print(f"[FP32] GPU显存不足: {e}")
                print(f"[FP32] 使用CPU推理(较慢但可靠)")
        else:
            print(f"[FP32] 模型太大({model_mem/1e9:.1f}GB > GPU可用{gpu_mem*0.85/1e9:.1f}GB)")
            print(f"[FP32] 使用CPU推理(较慢但可靠)")

    model.eval()
    device = next(model.parameters()).device
    print(f"[FP32] {model_key} 最终设备: {device}")
    return model, tokenizer


def get_nlayers(model, model_key):
    c = model.config
    if model_key == "qwen3": return c.num_hidden_layers
    elif model_key == "glm4": return c.num_layers
    else: return c.num_hidden_layers


def get_W_U_np(model, model_key):
    if model_key == "qwen3": w = model.lm_head.weight
    elif model_key == "glm4": w = model.transformer.output_layer.weight
    else: w = model.lm_head.weight
    return to_np(w)


def get_mlp_and_attn_modules(model, model_key, layer_idx):
    """获取指定层的MLP和Attention模块"""
    if model_key == "qwen3":
        layer = model.model.layers[layer_idx]
        return layer.mlp, layer.self_attn
    elif model_key == "glm4":
        layer = model.transformer.encoder.layers[layer_idx]
        return layer.mlp, layer.self_attention
    else:  # deepseek
        layer = model.model.layers[layer_idx]
        return layer.mlp, layer.self_attn


def run_experiment_with_hooks(model, tokenizer, model_key):
    """
    P695+P696+P697合并实验 — 使用Hook方式获取G和A
    
    Hook策略:
    - 在每层MLP的输出上注册hook → 捕获G向量
    - 从delta - G推算A向量
    - 这样避免手动调用层, 完全兼容device_map="auto"
    """
    print(f"\n=== Phase CLIX-FP32 Experiment ({model_key}) ===")

    n_layers = get_nlayers(model, model_key)
    W_U = get_W_U_np(model, model_key)
    hidden_dim = W_U.shape[1]

    all_results = {}

    for word in TEST_WORDS:
        text = f"{TEMPLATE} {word}"
        inputs = tokenizer(text, return_tensors="pt")

        word_tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        if not word_tokens:
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
        word_tid = word_tokens[-1]

        # ===== 使用Hook捕获MLP输出 =====
        mlp_outputs = {}  # {layer_idx: output_tensor}
        hooks = []

        def make_mlp_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    mlp_outputs[layer_idx] = output[0].detach()
                else:
                    mlp_outputs[layer_idx] = output.detach()
            return hook_fn

        # 注册hooks
        for l in range(n_layers):
            mlp, _ = get_mlp_and_attn_modules(model, model_key, l)
            hooks.append(mlp.register_forward_hook(make_mlp_hook(l)))

        # Forward pass
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states

        # 移除hooks
        for h in hooks:
            h.remove()

        # 转numpy
        hs_np = [to_np(h[0, -1, :]) for h in hs]

        # 获取G向量 (MLP输出)
        G_list = []
        for l in range(n_layers):
            if l in mlp_outputs:
                G = to_np(mlp_outputs[l][0, -1, :])
            else:
                # fallback: 用delta近似 (不应该发生)
                G = hs_np[l+1] - hs_np[l]
                print(f"  [WARNING] Layer {l} MLP hook missed, using delta approximation")
            G_list.append(G)

        # A向量 = delta - G
        A_list = []
        for l in range(n_layers):
            delta = hs_np[l+1] - hs_np[l]
            A = delta - G_list[l]
            A_list.append(A)

        # === P695: 逐层G/A动力学 ===
        w_u = W_U[word_tid]
        layer_data = []
        cumG, cumA = 0.0, 0.0

        for l in range(n_layers):
            lg = float(w_u @ G_list[l])
            la = float(w_u @ A_list[l])
            cumG += lg
            cumA += la
            G_norm = float(np.linalg.norm(G_list[l]))
            A_norm = float(np.linalg.norm(A_list[l]))
            cos_GA = float(np.dot(G_list[l], A_list[l]) / (G_norm * A_norm + 1e-10))
            layer_data.append({
                "layer": l, "logit_G": lg, "logit_A": la,
                "logit_total": lg + la,
                "cum_G": cumG, "cum_A": cumA,
                "G_norm": G_norm, "A_norm": A_norm,
                "sign_G": int(np.sign(lg)), "sign_A": int(np.sign(la)),
                "balance_ratio": abs(lg + la) / (abs(lg) + abs(la) + 1e-10),
                "cos_GA": cos_GA,
            })

        signs_G = [d["sign_G"] for d in layer_data]
        signs_A = [d["sign_A"] for d in layer_data]
        same_sign = sum(1 for g,a in zip(signs_G, signs_A) if g==a and g!=0)
        opp_sign = sum(1 for g,a in zip(signs_G, signs_A) if g!=0 and a!=0 and g!=a)

        # === P696: 对抗平衡指标 ===
        total_G = sum(G_list)
        total_A = sum(A_list)
        global_balance = np.linalg.norm(total_G + total_A) / (np.linalg.norm(total_G) + np.linalg.norm(total_A) + 1e-10)
        global_cos_GA = np.dot(total_G, total_A) / (np.linalg.norm(total_G) * np.linalg.norm(total_A) + 1e-10)
        mean_balance = np.mean([d["balance_ratio"] for d in layer_data])
        low_balance = sum(1 for d in layer_data if d["balance_ratio"] < 0.3)
        mean_cos_GA = np.mean([d["cos_GA"] for d in layer_data])

        # Logit空间G/A相关性
        logits_G_all = W_U @ G_list[-1]
        logits_A_all = W_U @ A_list[-1]
        logit_corr = np.corrcoef(logits_G_all, logits_A_all)[0,1] if np.std(logits_G_all)>1e-6 and np.std(logits_A_all)>1e-6 else 0.0

        # === P697: 因果干预 ===
        h_final = hs_np[-1]
        logit_normal = float(w_u @ h_final)
        prob_normal_all = np.exp(W_U @ h_final)
        prob_normal_all = prob_normal_all / prob_normal_all.sum()
        prob_normal = float(prob_normal_all[word_tid])
        top5_normal = [tokenizer.decode([int(t)]) for t in np.argsort(W_U @ h_final)[-5:][::-1]]

        # 干预1: only_A (移除所有G)
        h_only_A = hs_np[0].copy()
        for l in range(n_layers): h_only_A += A_list[l]
        logits_A = W_U @ h_only_A
        prob_A = np.exp(logits_A - logits_A.max()); prob_A /= prob_A.sum()
        top5_A = [tokenizer.decode([int(t)]) for t in np.argsort(logits_A)[-5:][::-1]]

        # 干预2: only_G (移除所有A)
        h_only_G = hs_np[0].copy()
        for l in range(n_layers): h_only_G += G_list[l]
        logits_G = W_U @ h_only_G
        prob_G = np.exp(logits_G - logits_G.max()); prob_G /= prob_G.sum()
        top5_G = [tokenizer.decode([int(t)]) for t in np.argsort(logits_G)[-5:][::-1]]

        # 干预3: flip_G (-G + A)
        h_flip = hs_np[0].copy()
        for l in range(n_layers): h_flip += -G_list[l] + A_list[l]
        logits_flip = W_U @ h_flip
        prob_flip = np.exp(logits_flip - logits_flip.max()); prob_flip /= prob_flip.sum()
        top5_flip = [tokenizer.decode([int(t)]) for t in np.argsort(logits_flip)[-5:][::-1]]

        # 干预4: amplify_G (2*G + A)
        h_amp = hs_np[0].copy()
        for l in range(n_layers): h_amp += 2.0 * G_list[l] + A_list[l]
        logits_amp = W_U @ h_amp
        prob_amp = np.exp(logits_amp - logits_amp.max()); prob_amp /= prob_amp.sum()

        # 干预5: last3_only
        h_last3 = hs_np[0].copy()
        for l in range(n_layers-3, n_layers): h_last3 += G_list[l] + A_list[l]
        logits_last3 = W_U @ h_last3
        prob_last3 = np.exp(logits_last3 - logits_last3.max()); prob_last3 /= prob_last3.sum()

        all_results[word] = {
            "final_logit": logit_normal,
            "cum_G": cumG, "cum_A": cumA,
            "G_A_ratio": abs(cumG) / (abs(cumA) + 1e-10),
            "same_sign_count": same_sign,
            "opposite_sign_count": opp_sign,
            "global_balance": float(global_balance),
            "global_cos_GA": float(global_cos_GA),
            "mean_balance_ratio": float(mean_balance),
            "low_balance_layers": low_balance,
            "mean_cos_GA": float(mean_cos_GA),
            "last_layer_logit_corr": float(logit_corr),
            "interventions": {
                "normal": {"prob": prob_normal, "top5": top5_normal},
                "only_A": {"prob": float(prob_A[word_tid]), "top5": top5_A, "prob_delta": float(prob_A[word_tid] - prob_normal)},
                "only_G": {"prob": float(prob_G[word_tid]), "top5": top5_G, "prob_delta": float(prob_G[word_tid] - prob_normal)},
                "flip_G": {"prob": float(prob_flip[word_tid]), "top5": top5_flip, "prob_delta": float(prob_flip[word_tid] - prob_normal)},
                "amplify_G": {"prob": float(prob_amp[word_tid]), "prob_delta": float(prob_amp[word_tid] - prob_normal)},
                "last3_only": {"prob": float(prob_last3[word_tid]), "prob_delta": float(prob_last3[word_tid] - prob_normal)},
            },
            "layer_data": layer_data,
        }

        print(f"  {word}: final={logit_normal:.2f}, cumG={cumG:.1f}, cumA={cumA:.1f}, "
              f"G/A={abs(cumG)/(abs(cumA)+1e-10):.2f}, "
              f"global_bal={global_balance:.3f}, global_cos={global_cos_GA:.3f}, "
              f"same={same_sign}, opp={opp_sign}")
        print(f"    Δ_only_A={prob_A[word_tid]-prob_normal:+.6f}, "
              f"Δ_only_G={prob_G[word_tid]-prob_normal:+.6f}, "
              f"Δ_flip_G={prob_flip[word_tid]-prob_normal:+.6f}, "
              f"Δ_amp_G={prob_amp[word_tid]-prob_normal:+.6f}")

    return all_results


def compare_with_fp16(model_key, fp32_results):
    """与FP16结果对比"""
    fp16_path = Path(f"d:/Ai2050/TransformerLens-Project/results/clix_adversarial_balance/clix_{model_key}_results.json")
    if not fp16_path.exists():
        print(f"  [对比] {model_key} 无FP16结果, 跳过对比")
        return

    with open(fp16_path, "r", encoding="utf-8") as f:
        fp16_data = json.load(f)
    fp16_results = fp16_data["results"]

    print(f"\n--- {model_key} FP32 vs FP16 对比 ---")
    print(f"{'Word':<12} {'Metric':<22} {'FP16':>12} {'FP32':>12} {'Diff%':>10}")
    print("-" * 72)

    for word in TEST_WORDS:
        if word not in fp16_results or word not in fp32_results:
            continue
        f16 = fp16_results[word]
        f32 = fp32_results[word]

        for metric in ["final_logit", "global_balance", "global_cos_GA", "G_A_ratio", "mean_balance_ratio"]:
            v16 = f16.get(metric, 0)
            v32 = f32.get(metric, 0)
            diff_pct = (v32 - v16) / abs(v16) * 100 if abs(v16) > 1e-10 else 0
            print(f"{word:<12} {metric:<22} {v16:>12.6f} {v32:>12.6f} {diff_pct:>+10.2f}%")

        for iv_name in ["only_A", "only_G", "flip_G", "amplify_G"]:
            v16 = f16["interventions"][iv_name]["prob"]
            v32 = f32["interventions"][iv_name]["prob"]
            diff_pct = (v32 - v16) / abs(v16) * 100 if abs(v16) > 1e-15 else 0
            print(f"{word:<12} prob_{iv_name:<16} {v16:>12.2e} {v32:>12.2e} {diff_pct:>+10.1f}%")


def main():
    all_results = {}

    for model_key in ["glm4", "deepseek"]:
        print(f"\n{'='*80}")
        print(f"[FP32] 开始测试: {model_key}")
        print(f"{'='*80}")

        model, tokenizer = load_model_fp32(model_key)
        results = run_experiment_with_hooks(model, tokenizer, model_key)

        output = {
            "model": model_key,
            "precision": "float32",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }
        path = RESULTS_DIR / f"clix_fp32_{model_key}_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print(f"[FP32] 保存到: {path}")

        all_results[model_key] = output

        compare_with_fp16(model_key, results)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[FP32] {model_key} 已释放, 等待5秒...")
        time.sleep(5)

    print("\n" + "="*80 + "\n[FP32] 三模型对比总结\n" + "="*80)
    for mk in ["qwen3", "glm4", "deepseek"]:
        r = all_results[mk]["results"]
        mean_bal = np.mean([v["global_balance"] for v in r.values()])
        mean_cos = np.mean([v["global_cos_GA"] for v in r.values()])
        mean_GA = np.mean([v["G_A_ratio"] for v in r.values()])
        mean_dA = np.mean([v["interventions"]["only_A"]["prob_delta"] for v in r.values()])
        mean_dG = np.mean([v["interventions"]["only_G"]["prob_delta"] for v in r.values()])
        mean_dF = np.mean([v["interventions"]["flip_G"]["prob_delta"] for v in r.values()])
        print(f"\n{mk} [FP32]: global_balance={mean_bal:.4f}, cos_GA={mean_cos:.4f}, G/A_ratio={mean_GA:.3f}")
        print(f"  干预: Δ_only_A={mean_dA:+.6f}, Δ_only_G={mean_dG:+.6f}, Δ_flip_G={mean_dF:+.6f}")

    summary_path = RESULTS_DIR / "clix_fp32_all_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[FP32] 汇总保存到: {summary_path}")


if __name__ == "__main__":
    main()
