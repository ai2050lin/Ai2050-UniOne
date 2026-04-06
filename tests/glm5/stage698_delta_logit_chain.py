#!/usr/bin/env python3
"""
P52: Fine-tuning Delta -> Logit Margin Causal Chain (Stage698)

P50 showed: delta-h directions are category-dependent (ANOVA F>3)
P51 showed: ablating layers changes logits (causal)
P52 asks: Can we QUANTIFY the mapping from fine-tuning delta-h to logit margin?

Method:
1. For each text, get all hidden states (L0 to Ln)
2. For each layer, compute delta-h and project through lm_head:
   delta_logits = (h_l - h_{l-1}) @ unembed.T
3. Compare delta_logits cumulative sum with actual logits
4. Key: does sum of per-layer delta_logits ≈ actual logit?

This directly tests whether the "linear accumulation" model works:
    logit(h_final) ≈ sum_{l=1}^{n} logit(delta_h_l)
    
If YES → we can predict logits by predicting each layer's delta
If NO → there are nonlinear interactions between layers

Also test: which layers contribute most to the margin (top-1 vs top-2)?
"""
import sys, math, time, gc
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

TEXTS = [
    "The cat sat on the mat.", "The dog chased the ball.",
    "Paris is the capital of France.", "The Amazon is a long river.",
    "She carefully folded the origami crane.", "The orchestra played beautifully.",
    "If it rains then the ground gets wet.", "She studied hard because she wanted to pass.",
    "Yesterday it rained heavily all day.", "She will finish the report by Friday.",
    "Two plus two equals four exactly.", "DNA contains genetic instructions for life.",
    "A red apple is a fruit.", "The bank by the river was flooded.",
    "The quick brown fox jumps over the lazy dog.", "The report was submitted on time.",
]

TIMESTAMP = time.strftime("%Y%m%d_%H%M")
OUTPUT_DIR = _Path(f"d:\\develop\\TransformerLens-main\\tests\\glm5_temp\\stage698_delta_logit_chain_{TIMESTAMP}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class Logger:
    def __init__(self, filepath):
        self.f = open(filepath, "w", encoding="utf-8")
    def __call__(self, msg="", end="\n"):
        try:
            safe_msg = msg.encode('gbk', errors='replace').decode('gbk')
        except:
            safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
        print(safe_msg, end=end)
        self.f.write(msg + end)
        if end == "\n":
            self.f.flush()
    def close(self):
        self.f.close()


def load_model(model_name):
    model_path = MODEL_MAP[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def analyze_delta_logit_chain(model, tokenizer, text):
    """Analyze the delta-h -> logit mapping for a single text.

    Returns:
        actual_logits: (vocab_size,) - actual logits from model
        delta_logits_per_layer: list of (vocab_size,) - contribution of each layer's delta
        cumulative_logits: list of (vocab_size,) - running sum of delta logits
        h_per_layer: list of (d_model,) - hidden states per layer
    """
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)

    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
        actual_logits = outputs.logits[0, -1, :].float().cpu()  # (vocab_size,)
        hidden_states = outputs.hidden_states  # tuple of (1, seq_len, d_model)

    # Get unembed matrix
    lm_head = model.lm_head
    unembed_weight = lm_head.weight.float().cpu()  # (vocab_size, d_model)
    unembed_bias = lm_head.bias.float().cpu() if hasattr(lm_head, 'bias') and lm_head.bias is not None else None

    n_layers = len(hidden_states) - 1
    h_per_layer = []
    for hs in hidden_states:
        h_per_layer.append(hs[0, -1, :].float().cpu())  # (d_model,)

    # Compute delta logits for each layer
    delta_logits_per_layer = []
    cumulative = torch.zeros_like(actual_logits)
    cumulative_logits = []

    for l in range(1, n_layers + 1):
        delta_h = h_per_layer[l] - h_per_layer[l - 1]
        # delta_logits = delta_h @ unembed.T (without bias, as bias is constant)
        delta_logits = F.linear(delta_h.unsqueeze(0), unembed_weight).squeeze(0)
        delta_logits_per_layer.append(delta_logits)
        cumulative = cumulative + delta_logits
        cumulative_logits.append(cumulative.clone())

    return {
        "actual_logits": actual_logits,
        "delta_logits_per_layer": delta_logits_per_layer,
        "cumulative_logits": cumulative_logits,
        "h_per_layer": h_per_layer,
        "n_layers": n_layers,
    }


def get_margin(logits):
    sorted_vals = torch.sort(logits, descending=True).values
    return (sorted_vals[0] - sorted_vals[1]).item()


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    log_path = OUTPUT_DIR / "results.log"
    log = Logger(log_path)

    log("=" * 80)
    log("P52: Fine-tuning Delta -> Logit Margin Causal Chain")
    log(f"Timestamp: {TIMESTAMP}")
    log("=" * 80)

    all_model_results = {}

    for model_name in MODEL_MAP:
        log(f"\n{'='*60}")
        log(f"Processing model: {model_name}")
        log(f"{'='*60}")
        t0 = time.time()

        model, tokenizer = load_model(model_name)

        # Test on first text to get number of layers
        test_result = analyze_delta_logit_chain(model, tokenizer, TEXTS[0])
        n_layers = test_result["n_layers"]
        log(f"Layers: {n_layers}")

        # 1. Test linear accumulation hypothesis
        log(f"\n  A. Linear Accumulation Test")
        log(f"  (Does sum of per-layer delta_logits approximate actual logits?)")

        actual_vs_cum_cos = []
        actual_vs_cum_margin_corr = []
        actual_vs_embed_cos = []

        for text in TEXTS:
            try:
                result = analyze_delta_logit_chain(model, tokenizer, text)
                actual = result["actual_logits"]
                cumulative = result["cumulative_logits"][-1]  # sum of all deltas

                # Cosine similarity between actual and cumulative logits
                cos_val = cosine_sim(actual, cumulative)
                actual_vs_cum_cos.append(cos_val)

                # Margin correlation
                actual_margin = get_margin(actual)
                cum_margin = get_margin(cumulative)
                actual_vs_cum_margin_corr.append((actual_margin, cum_margin))

                # Also compare with embed-only logits (h_L0 through lm_head)
                embed_logits = F.linear(
                    result["h_per_layer"][0].unsqueeze(0),
                    model.lm_head.weight.float().cpu()
                ).squeeze(0)
                embed_cos = cosine_sim(actual, embed_logits)
                actual_vs_embed_cos.append(embed_cos)
            except Exception as e:
                log(f"    Error: {e}")

        mean_cos = np.mean(actual_vs_cum_cos)
        std_cos = np.std(actual_vs_cum_cos)
        mean_embed_cos = np.mean(actual_vs_embed_cos) if actual_vs_embed_cos else 0

        log(f"    Actual vs Cumulative logits cos: {mean_cos:.4f} +/- {std_cos:.4f}")
        log(f"    Actual vs Embed-only logits cos: {mean_embed_cos:.4f}")
        log(f"    Linear accumulation R2 (margin): ", end="")
        if actual_vs_cum_margin_corr:
            actual_margins = [x[0] for x in actual_vs_cum_margin_corr]
            cum_margins = [x[1] for x in actual_vs_cum_margin_corr]
            r, p = np.corrcoef(actual_margins, cum_margins)[0, 1], 0
            log(f"r={r:.3f}")
        else:
            log("N/A")

        if mean_cos > 0.8:
            log(f"    >> LINEAR ACCUMULATION WORKS! We can predict logits by summing per-layer deltas.")
        elif mean_cos > 0.5:
            log(f"    >> Partial linear: deltas capture some structure, but nonlinear effects exist.")
        else:
            log(f"    >> NONLINEAR: per-layer delta logits do NOT sum to actual logits.")
            log(f"    >> This means inter-layer interactions (via attention) create nonlinear effects.")

        # 2. Per-layer contribution to margin
        log(f"\n  B. Per-Layer Margin Contribution")
        log(f"  (Which layers contribute most to the logit margin?)")

        layer_margin_contributions = np.zeros(n_layers)
        layer_margin_std = np.zeros(n_layers)
        n_valid = 0

        for text in TEXTS:
            try:
                result = analyze_delta_logit_chain(model, tokenizer, text)
                actual = result["actual_logits"]

                # For each layer, measure: how much does adding this layer's delta
                # change the margin of cumulative logits?
                prev_cum = torch.zeros_like(actual)
                for l, delta_logit in enumerate(result["delta_logits_per_layer"]):
                    new_cum = prev_cum + delta_logit
                    prev_margin = get_margin(prev_cum)
                    new_margin = get_margin(new_cum)
                    contribution = new_margin - prev_margin
                    layer_margin_contributions[l] += contribution
                    layer_margin_std[l] += contribution ** 2
                    prev_cum = new_cum
                n_valid += 1
            except:
                pass

        if n_valid > 0:
            layer_margin_contributions /= n_valid
            layer_margin_std = np.sqrt(layer_margin_std / n_valid - layer_margin_contributions ** 2)

            # Print every 5th layer + top contributors
            top_contrib_idx = np.argsort(layer_margin_contributions)[-5:]
            print_layers = sorted(set(list(range(0, n_layers, max(1, n_layers // 5))) + list(top_contrib_idx)))

            for l in print_layers:
                if l < n_layers:
                    log(f"    L{l:2d}: margin contrib = {layer_margin_contributions[l]:+.2f} "
                        f"+/- {layer_margin_std[l]:.2f}  "
                        f"{'***' if abs(layer_margin_contributions[l]) > np.mean(np.abs(layer_margin_contributions)) * 2 else ''}")

            # Total margin from accumulation
            total_accumulated = np.sum(layer_margin_contributions)
            log(f"\n    Total accumulated margin: {total_accumulated:+.2f}")
            log(f"    Max single-layer contribution: L{np.argmax(layer_margin_contributions)} "
                f"({layer_margin_contributions[np.argmax(layer_margin_contributions)]:+.2f})")
            log(f"    Min single-layer contribution: L{np.argmin(layer_margin_contributions)} "
                f"({layer_margin_contributions[np.argmin(layer_margin_contributions)]:+.2f})")

        # 3. Progressive prediction accuracy
        log(f"\n  C. Progressive Prediction (at what layer does cumulative logits match actual?)")

        layer_cos_values = np.zeros(n_layers)
        for text in TEXTS:
            try:
                result = analyze_delta_logit_chain(model, tokenizer, text)
                actual = result["actual_logits"]
                for l, cum in enumerate(result["cumulative_logits"]):
                    layer_cos_values[l] += cosine_sim(actual, cum)
            except:
                pass

        if n_valid > 0:
            layer_cos_values /= n_valid

            # Find layer where cos first exceeds 0.5
            threshold_layers = {}
            for thresh in [0.3, 0.5, 0.7, 0.9]:
                exceeding = [l for l in range(n_layers) if layer_cos_values[l] >= thresh]
                threshold_layers[thresh] = exceeding[0] if exceeding else -1

            for thresh, layer in threshold_layers.items():
                if layer >= 0:
                    log(f"    Cos>{thresh}: first achieved at L{layer} ({layer_cos_values[layer]:.3f})")
                else:
                    log(f"    Cos>{thresh}: NEVER achieved (max={np.max(layer_cos_values):.3f})")

            # Print a few key points
            for l in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
                if l < n_layers:
                    log(f"    L{l:2d}: cos(actual, cumulative) = {layer_cos_values[l]:.4f}")

        model_result = {
            "model": model_name,
            "n_layers": n_layers,
            "linear_accumulation_cos": float(mean_cos),
            "linear_accumulation_cos_std": float(std_cos),
            "embed_only_cos": float(mean_embed_cos),
            "layer_margin_contributions": layer_margin_contributions.tolist(),
            "threshold_layers": threshold_layers,
        }
        all_model_results[model_name] = model_result

        import json
        json_path = OUTPUT_DIR / f"results_{model_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(model_result, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

        elapsed = time.time() - t0
        log(f"  Completed in {elapsed:.1f}s")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Cross-model comparison
    log(f"\n{'='*80}")
    log("CROSS-MODEL COMPARISON")
    log(f"{'='*80}")

    log(f"\n  A. Linear Accumulation Accuracy")
    log(f"  {'Model':>12s}: {'CumCos':>7s}  {'EmbedCos':>9s}  {'Linear?':>7s}")
    for m, r in all_model_results.items():
        is_linear = "YES" if r["linear_accumulation_cos"] > 0.8 else ("PARTIAL" if r["linear_accumulation_cos"] > 0.5 else "NO")
        log(f"  {m:>12s}: {r['linear_accumulation_cos']:>7.4f}  {r['embed_only_cos']:>9.4f}  {is_linear:>7s}")

    log(f"\n  B. Layer at which cumulative logits match actual (cos>0.5)")
    for m, r in all_model_results.items():
        l50 = r["threshold_layers"].get(0.5, -1)
        l90 = r["threshold_layers"].get(0.9, -1)
        log(f"  {m:>12s}: cos>0.5 at L{l50}, cos>0.9 at L{l90}")

    log(f"\n  C. Key Question - Can we build a linear prediction chain?")
    all_linear = all(r["linear_accumulation_cos"] > 0.8 for r in all_model_results.values())
    any_linear = any(r["linear_accumulation_cos"] > 0.5 for r in all_model_results.values())

    if all_linear:
        log(f"  YES (all models): Linear accumulation of per-layer delta logits approximates actual logits.")
        log(f"  >> We CAN build a prediction chain: text -> delta-h per layer -> sum -> logits")
    elif any_linear:
        log(f"  PARTIAL: Some models show linear accumulation, others don't.")
        log(f"  >> Prediction chain may work for some architectures but not all.")
    else:
        log(f"  NO: Nonlinear interactions between layers prevent linear prediction.")
        log(f"  >> Need to model inter-layer attention effects for accurate prediction.")

    log(f"\n{'='*80}")
    log("P52 COMPLETE")
    log(f"{'='*80}")

    log.close()


if __name__ == "__main__":
    main()
