"""
Phase 5: 因果干预实验 + FFN旋转器精确分析
==========================================
Phase 5A: 真正的因果干预 - 在中间层零化/缩小维度差异方向，观察输出变化
Phase 5C: FFN旋转器方向偏好 - 用低秩近似解决维度问题

目标：
1. 确定维度信息是否因果地编码在残差流的特定方向上
2. 确定FFN旋转器的方向偏好是否跨层一致

使用模型: GPT-2 (768d, 12层, 12头) — 小模型适合因果干预实验
硬件: RTX 3080 8GB
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
from datetime import datetime
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("d:/ai2050/TransformerLens-Project/tests/glm5/phase1_results")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============ 测试句子 ============
TEST_PAIRS = {
    "syntax": [
        ("The cat chased the mouse across the field.", "The mouse was chased by the cat across the field."),
        ("A scientist discovered a new method.", "A new method was discovered by a scientist."),
        ("The teacher graded every paper carefully.", "Every paper was graded carefully by the teacher."),
        ("Students must complete all assignments on time.", "All assignments must be completed on time by students."),
    ],
    "logic": [
        ("If it rains, the ground gets wet. It is raining. Therefore, the ground is wet.",
         "If it rains, the ground gets wet. It is raining. Therefore, the ground is dry."),
        ("All birds can fly. Penguins are birds. Therefore, penguins can fly.",
         "All birds can fly. Penguins are birds. Therefore, penguins cannot fly."),
        ("If A implies B and B implies C, then A implies C.",
         "If A implies B and B implies C, then A does not imply C."),
        ("Every mammal is warm-blooded. Whales are mammals. Therefore, whales are warm-blooded.",
         "Every mammal is warm-blooded. Whales are mammals. Therefore, whales are cold-blooded."),
    ],
    "style": [
        ("The results indicate a significant improvement in performance.",
         "The results show things got way better."),
        ("We conducted an investigation into the matter.",
         "We looked into it."),
        ("The individual demonstrated considerable expertise.",
         "The person really knew their stuff."),
        ("Subsequent analysis revealed no anomalies.",
         "Later checks showed nothing weird."),
    ],
    "semantic": [
        ("The lion chased the gazelle across the savanna.", "The equation describes quantum entanglement."),
        ("She walked through the autumn forest.", "The algorithm processes binary data."),
        ("Dolphins communicate using complex sounds.", "Volcanoes form from tectonic activity."),
        ("The chef prepared a delicious meal.", "The satellite orbits the planet."),
    ],
}


def load_model(model_name="gpt2"):
    """加载模型和tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n{'='*60}")
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  Loaded: {model_name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  Layers: {len(model.transformer.h)}")
    return model, tokenizer


def get_residual_stream(model, tokenizer, text):
    """获取所有层的残差流"""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # (L+1) tensors of [1, seq_len, hidden]
    return [h.squeeze(0).float() for h in hidden_states]


def compute_diff_direction(res_a, res_b, layer_idx, token_pos=None):
    """计算两个句子在指定层的差异方向"""
    if token_pos is None:
        # 平均差异方向
        diff = res_a[layer_idx].mean(dim=0) - res_b[layer_idx].mean(dim=0)
    else:
        diff = res_a[layer_idx][token_pos] - res_b[layer_idx][token_pos]
    norm = diff.norm()
    if norm < 1e-8:
        return None, 0.0
    return diff / norm, norm.item()


# ============================================================
# Phase 5A: 因果干预实验
# ============================================================
def phase5a_causal_intervention(model, tokenizer):
    """
    核心实验: 在中间层干预残差流，观察输出变化

    实验设计:
    1. 取一对句子 (A, B)
    2. 计算在layer L的残差流差异方向 d = normalize(mean(res_A) - mean(res_B))
    3. 在A的前向传播中，在layer L之后减去 alpha * d 的投影
    4. 比较干预后的输出概率分布 vs 原始分布

    干预强度: alpha = 0.0 (无干预) 到 alpha = 2.0 (过度干预)
    """
    print("\n" + "="*60)
    print("Phase 5A: 因果干预实验")
    print("="*60)

    results = {}

    for dim_name, pairs in TEST_PAIRS.items():
        print(f"\n--- {dim_name} ---")
        dim_results = []

        for pair_idx, (text_a, text_b) in enumerate(pairs):
            print(f"\n  Pair {pair_idx+1}:")
            print(f"    A: {text_a[:60]}...")
            print(f"    B: {text_b[:60]}...")

            pair_result = {
                "text_a": text_a, "text_b": text_b,
                "interventions": []
            }

            # 获取token对齐信息
            toks_a = tokenizer(text_a, return_tensors="pt")
            toks_b = tokenizer(text_b, return_tensors="pt")
            seq_len_a = toks_a["input_ids"].shape[1]
            seq_len_b = toks_b["input_ids"].shape[1]
            min_len = min(seq_len_a, seq_len_b)

            # 预先获取A和B的所有残差流
            with torch.no_grad():
                orig_logits_a = model(**toks_a).logits.squeeze(0)
                orig_logits_b = model(**toks_b).logits.squeeze(0)
                res_a_all = get_residual_stream(model, tokenizer, text_a)
                res_b_all = get_residual_stream(model, tokenizer, text_b)

            # 原始分布差异 (baseline)
            orig_diffs = []
            for t in range(min_len - 1):
                p_a = F.softmax(orig_logits_a[t], dim=-1)
                p_b = F.softmax(orig_logits_b[t], dim=-1)
                js = 0.5 * (F.kl_div(p_a.log(), p_b, reduction='sum') +
                           F.kl_div(p_b.log(), p_a, reduction='sum')).item()
                orig_diffs.append(js)
            baseline_js = np.mean(orig_diffs)
            print(f"    Baseline JS divergence: {baseline_js:.4f}")

            # 预计算差异方向
            intervention_layers = [3, 6, 9]
            alphas = [0.5, 1.0, 2.0]
            diff_dirs = {}
            for il in intervention_layers:
                dd, dn = compute_diff_direction(res_a_all, res_b_all, il)
                diff_dirs[il] = (dd, dn)

            # 预计算各层hidden states (逐层前向, 用min_len)
            input_ids_a_trim = toks_a["input_ids"][:, :min_len]
            embeds = model.transformer.wte(input_ids_a_trim)
            position_ids = torch.arange(min_len).unsqueeze(0)
            pos_embeds = model.transformer.wpe(position_ids)
            hiddens_at_layers = {}
            hidden = model.transformer.drop(embeds + pos_embeds)
            hiddens_at_layers[0] = hidden.clone()
            for l in range(len(model.transformer.h)):
                block = model.transformer.h[l]
                ln_out = block.ln_1(hidden)
                attn_out = block.attn(ln_out, layer_past=None,
                                      attention_mask=None,
                                      head_mask=None)[0]
                hidden = hidden + attn_out
                ln_out2 = block.ln_2(hidden)
                ffn_out = block.mlp(ln_out2)
                hidden = hidden + ffn_out
                hiddens_at_layers[l+1] = hidden.clone()

            for int_layer in intervention_layers:
                diff_dir, diff_norm = diff_dirs[int_layer]
                if diff_dir is None:
                    continue

                hidden = hiddens_at_layers[int_layer + 1]

                for alpha in alphas:

                    # 干预: 减去 alpha * diff_dir 的投影
                    scalar_proj = (hidden @ diff_dir)  # [1, seq_len]
                    proj = scalar_proj.unsqueeze(-1) * diff_dir  # [1, seq_len, 768]
                    hidden_int = hidden - alpha * proj

                    # 通过剩余层
                    remaining_layers = len(model.transformer.h) - int_layer - 1
                    if remaining_layers > 0:
                        for l in range(int_layer + 1, len(model.transformer.h)):
                            block = model.transformer.h[l]
                            ln_out = block.ln_1(hidden_int)
                            attn_out = block.attn(ln_out, layer_past=None,
                                                  attention_mask=None,
                                                  head_mask=None)[0]
                            hidden_int = hidden_int + attn_out
                            ln_out2 = block.ln_2(hidden_int)
                            ffn_out = block.mlp(ln_out2)
                            hidden_int = hidden_int + ffn_out

                    # 最终LN + LM head
                    hidden_int = model.transformer.ln_f(hidden_int)
                    int_logits = model.lm_head(hidden_int).squeeze(0)

                # 计算干预后的JS散度
                int_js_diffs = []
                for t in range(min_len - 1):
                    p_int = F.softmax(int_logits[t], dim=-1)
                    p_orig = F.softmax(orig_logits_a[t], dim=-1)
                    js = 0.5 * (F.kl_div(p_int.log(), p_orig, reduction='sum') +
                               F.kl_div(p_orig.log(), p_int, reduction='sum')).item()
                    int_js_diffs.append(js)
                mean_int_js = np.mean(int_js_diffs)

                # 计算干预后与B的JS散度 (是否趋近B?)
                int_ab_diffs = []
                for t in range(min_len - 1):
                    p_int = F.softmax(int_logits[t], dim=-1)
                    p_b = F.softmax(orig_logits_b[t], dim=-1)
                    js = 0.5 * (F.kl_div(p_int.log(), p_b, reduction='sum') +
                               F.kl_div(p_b.log(), p_int, reduction='sum')).item()
                    int_ab_diffs.append(js)
                mean_int_ab = np.mean(int_ab_diffs)

                # Top-1 token变化
                top1_changes = 0
                for t in range(min_len - 1):
                    orig_top = orig_logits_a[t].argmax().item()
                    int_top = int_logits[t].argmax().item()
                    if orig_top != int_top:
                        top1_changes += 1

                pair_result["interventions"].append({
                    "layer": int_layer, "alpha": alpha,
                    "js_self_change": mean_int_js,
                    "js_to_b": mean_int_ab,
                    "top1_changes": top1_changes,
                    "total_positions": min_len - 1,
                    "diff_norm": diff_norm
                })

                print(f"    L{int_layer} α={alpha:.1f}: "
                      f"JS_self={mean_int_js:.4f}, JS_to_B={mean_int_ab:.4f}, "
                      f"top1_change={top1_changes}/{min_len-1}")

            # 找本pair的最佳干预参数
            if pair_result["interventions"]:
                best = min(pair_result["interventions"],
                           key=lambda x: x["js_to_b"])
                pair_result["best_intervention"] = best
                print(f"    >> Best: L{best['layer']} α={best['alpha']:.1f} "
                      f"JS_to_B={best['js_to_b']:.4f} (baseline={baseline_js:.4f})")

            dim_results.append(pair_result)
            torch.cuda.empty_cache()

        results[dim_name] = dim_results

    # 总结
    print("\n" + "="*60)
    print("Phase 5A Summary: 因果干预效果")
    print("="*60)

    for dim_name, dim_results in results.items():
        best_js_list = [r["best_intervention"]["js_to_b"] for r in dim_results if "best_intervention" in r]
        if best_js_list:
            print(f"  {dim_name}: mean_best_JS_to_B = {np.mean(best_js_list):.4f}, "
                  f"n_pairs = {len(best_js_list)}")

    return {"causal_intervention": results}


# ============================================================
# Phase 5C: FFN旋转器方向分析 (低秩近似)
# ============================================================
def phase5c_ffn_rotation_analysis(model, tokenizer):
    """
    Phase 5C: 用低秩近似解决FFN组合权重矩阵的维度问题

    方法:
    1. 对每层的FFN组合 W = W_down @ W_up，不做完整SVD
    2. 改用随机SVD (randomized SVD) 或只取Top-K奇异向量
    3. 分析旋转角度和跨层方向一致性
    """
    print("\n" + "="*60)
    print("Phase 5C: FFN旋转器方向分析 (低秩近似)")
    print("="*60)

    results = {}
    num_layers = len(model.transformer.h)
    hidden = model.config.n_embd

    # 随机测试方向
    torch.manual_seed(42)
    n_random = 50
    random_dirs = torch.randn(n_random, hidden)
    random_dirs = random_dirs / random_dirs.norm(dim=-1, keepdim=True)

    # 对每层分析FFN旋转
    layer_rotations = []

    for l in range(num_layers):
        block = model.transformer.h[l]
        W_up = block.mlp.c_fc.weight.detach().float()  # [intermediate, hidden]
        W_down = block.mlp.c_proj.weight.detach().float()  # [hidden, intermediate]

        # GPT-2使用Conv1D, 权重形状为 [in_features, out_features]
        # c_fc: [hidden, intermediate] = [768, 3072], 计算: x @ W_up
        # c_proj: [intermediate, hidden] = [3072, 768], 计算: y @ W_down
        # Combined: x @ W_up @ W_down
        W_combined = W_up @ W_down  # [hidden, hidden]

        if W_combined.shape != (hidden, hidden):
            continue

        # 低秩近似: 只取Top-K奇异向量
        try:
            # Randomized SVD: 只取前10个奇异向量
            U, S, Vt = torch.linalg.svd(W_combined, full_matrices=False)
        except Exception:
            continue

        top_k = min(10, len(S))
        top_singular_values = S[:top_k].tolist()
        top_directions = Vt[:top_k]  # [top_k, hidden]

        # 计算旋转角度 (Top-1方向)
        top_dir = Vt[0]
        rotation_angles = []
        for i in range(n_random):
            transformed = W_combined @ random_dirs[i]
            cos_sim = F.cosine_similarity(random_dirs[i].unsqueeze(0),
                                          transformed.unsqueeze(0))
            angle = torch.acos(cos_sim.clamp(-1, 1)).item()
            rotation_angles.append(angle)

        mean_rotation = np.mean(rotation_angles)
        std_rotation = np.std(rotation_angles)

        # 与前一层的Top-1方向相关性
        cross_layer_corr = None
        if layer_rotations:
            prev_dir = layer_rotations[-1]["top_direction"]
            cross_layer_corr = F.cosine_similarity(
                top_dir.unsqueeze(0), prev_dir.unsqueeze(0)).item()

        layer_data = {
            "layer": l,
            "top_singular_values": top_singular_values,
            "spectral_norm": S[0].item(),
            "effective_rank": float(torch.exp(-torch.sum(
                (S/S.sum()) * torch.log(S/S.sum() + 1e-10)))),
            "mean_rotation_deg": mean_rotation,
            "std_rotation_deg": std_rotation,
            "top_direction": top_dir,
            "cross_layer_top1_corr": cross_layer_corr
        }

        layer_rotations.append(layer_data)
        print(f"  L{l:2d}: rot={mean_rotation:.1f}±{std_rotation:.1f}°, "
              f"σ1={S[0]:.2f}, eff_rank={layer_data['effective_rank']:.1f}, "
              f"cross_corr={cross_layer_corr:.4f}" if cross_layer_corr else
              f"  L{l:2d}: rot={mean_rotation:.1f}±{std_rotation:.1f}°, "
              f"σ1={S[0]:.2f}, eff_rank={layer_data['effective_rank']:.1f}")

    # 分析跨层方向一致性
    print("\n  Cross-layer direction correlation matrix (Top-1):")
    n = len(layer_rotations)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if abs(i-j) <= 3:  # 只算邻近层
                corr = F.cosine_similarity(
                    layer_rotations[i]["top_direction"].unsqueeze(0),
                    layer_rotations[j]["top_direction"].unsqueeze(0)).item()
                corr_matrix[i][j] = corr
    # 打印三角部分
    for i in range(n):
        row = []
        for j in range(max(0,i-3), min(n,i+4)):
            row.append(f"{corr_matrix[i][j]:.3f}")
        print(f"    L{i}: {' '.join(row)}")

    # 分析Top-K方向的累积旋转
    print("\n  Cumulative rotation via Top-1 direction:")
    cumulative = 0
    for l, ld in enumerate(layer_rotations):
        cumulative += ld["mean_rotation_deg"]
        print(f"    L{l}: step={ld['mean_rotation_deg']:.1f}°, "
              f"cumulative={cumulative:.1f}°")

    # FFN旋转与实际残差流偏转的对比
    # 取一个测试句子计算实际偏转
    print("\n  FFN rotation vs actual residual stream rotation:")
    test_text = TEST_PAIRS["syntax"][0][0]
    res_stream = get_residual_stream(model, tokenizer, test_text)
    actual_angles = []
    for l in range(len(res_stream) - 1):
        dir_a = res_stream[l].mean(dim=0)
        dir_b = res_stream[l+1].mean(dim=0)
        cos = F.cosine_similarity(dir_a.unsqueeze(0), dir_b.unsqueeze(0)).item()
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        actual_angles.append(angle)

    print(f"    {'Layer':>6} {'FFN_rot':>10} {'Actual_rot':>12} {'Ratio':>8}")
    for l in range(min(len(layer_rotations), len(actual_angles))):
        ratio = actual_angles[l] / layer_rotations[l]["mean_rotation_deg"] if layer_rotations[l]["mean_rotation_deg"] > 0.1 else 0
        print(f"    L{l:>4d} {layer_rotations[l]['mean_rotation_deg']:>10.1f}° "
              f"{actual_angles[l]:>11.1f}° {ratio:>7.2f}")

    # 准备输出数据 (去除tensor)
    output_layers = []
    for ld in layer_rotations:
        d = {k: v for k, v in ld.items() if k != "top_direction"}
        d["top1_norm"] = ld["top_direction"].norm().item()
        output_layers.append(d)

    return {
        "ffn_rotation": {
            "per_layer": output_layers,
            "cross_layer_correlations": corr_matrix.tolist(),
            "ffn_vs_actual": [
                {"layer": l,
                 "ffn_rotation": layer_rotations[l]["mean_rotation_deg"],
                 "actual_rotation": actual_angles[l] if l < len(actual_angles) else None}
                for l in range(len(layer_rotations))
            ]
        }
    }


# ============================================================
# Main
# ============================================================
def main():
    print(f"\n{'#'*60}")
    print(f"# Phase 5: 因果干预 + FFN旋转器分析")
    print(f"# Time: {TIMESTAMP}")
    print(f"{'#'*60}")

    model, tokenizer = load_model("gpt2")
    all_results = {"models": {"gpt2": {}}}

    # Phase 5A
    skip_5a = "--skip-5a" in sys.argv

    if not skip_5a:
        p5a = phase5a_causal_intervention(model, tokenizer)
        all_results["models"]["gpt2"].update(p5a)
    else:
        print("\nSkipping Phase 5A (--skip-5a)")

    # Phase 5C
    p5c = phase5c_ffn_rotation_analysis(model, tokenizer)
    all_results["models"]["gpt2"].update(p5c)

    # 保存
    out_path = OUTPUT_DIR / f"phase5_causal_ffn_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")

    # 简要总结
    print("\n" + "="*60)
    print("Phase 5 Complete Summary")
    print("="*60)

    gpt2 = all_results["models"]["gpt2"]

    # 5A summary
    ci = gpt2.get("causal_intervention")
    if ci:
        print("\n5A Causal Intervention:")
        for dim, pairs in ci.items():
            best_js_list = [r["best_intervention"]["js_to_b"]
                           for r in pairs if "best_intervention" in r]
            if best_js_list:
                print(f"  {dim}: best_mean_JS_to_B = {np.mean(best_js_list):.4f}")

    # 5C summary
    fr = gpt2["ffn_rotation"]
    layers = fr["per_layer"]
    mean_rot = np.mean([l["mean_rotation_deg"] for l in layers])
    mean_eff_rank = np.mean([l["effective_rank"] for l in layers])
    cross_corrs = [l["cross_layer_top1_corr"] for l in layers if l.get("cross_layer_top1_corr") is not None]
    mean_cross = np.mean(cross_corrs)

    print(f"\n5C FFN Rotation:")
    print(f"  Mean rotation angle: {mean_rot:.1f}°")
    print(f"  Mean effective rank: {mean_eff_rank:.1f}")
    print(f"  Cross-layer Top-1 correlation: {mean_cross:.4f}")

    # FFN vs actual
    fva = fr["ffn_vs_actual"]
    ratios = [r["actual_rotation"]/r["ffn_rotation"]
              for r in fva if r["ffn_rotation"] > 0.1 and r.get("actual_rotation")]
    print(f"  FFN rotation / Actual residual rotation ratio: {np.mean(ratios):.2f}")

    del model
    torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
