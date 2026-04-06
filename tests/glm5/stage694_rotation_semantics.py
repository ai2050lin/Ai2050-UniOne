#!/usr/bin/env python3
"""
P48: Rotation Semantics - Does rotation encode linguistic features? (Stage694)

Core question: Does the rotation angle/magnitude correlate with text properties?

Method:
1. Define text categories: concrete/abstract, short/long, simple/complex syntax
2. For each text, compute rotation angle and rotation vector
3. Test if categories predict rotation properties (ANOVA / correlation)
4. For each model, compute per-category rotation statistics
"""
import sys, math, time, gc
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from collections import defaultdict

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

# Texts with category labels
TEXTS_WITH_CATS = [
    # (text, category, subcategory)
    # Concrete entities
    ("The cat sat on the mat.", "concrete", "animals"),
    ("The dog chased the ball.", "concrete", "animals"),
    ("Birds fly south in winter.", "concrete", "animals"),
    ("A red apple is a fruit.", "concrete", "objects"),
    ("The seal swam in the cold ocean.", "concrete", "animals"),

    # Geographic/factual
    ("Paris is the capital of France.", "factual", "geography"),
    ("Tokyo is a large city.", "factual", "geography"),
    ("The Amazon is a long river.", "factual", "geography"),

    # Aesthetic/creative
    ("She carefully folded the origami crane.", "aesthetic", "art"),
    ("The orchestra played beautifully.", "aesthetic", "music"),
    ("His writing was elegant and precise.", "aesthetic", "writing"),
    ("The painting was incredibly detailed.", "aesthetic", "art"),

    # Logical/causal
    ("If it rains then the ground gets wet.", "logical", "causal"),
    ("She studied hard because she wanted to pass.", "logical", "causal"),
    ("Although tired she continued working.", "logical", "concessive"),
    ("The boy who was running fell down.", "logical", "relative"),

    # Temporal
    ("Yesterday it rained heavily all day.", "temporal", "past"),
    ("She will finish the report by Friday.", "temporal", "future"),
    ("The project was completed last month.", "temporal", "past"),
    ("He arrived before the ceremony began.", "temporal", "sequence"),

    # Mathematical/scientific
    ("Two plus two equals four exactly.", "math", "arithmetic"),
    ("The derivative of x squared is two x.", "math", "calculus"),
    ("DNA contains genetic instructions for life.", "science", "biology"),
    ("Gravity causes objects to fall.", "science", "physics"),
    ("The neural network learned patterns.", "science", "cs"),

    # Scientific method
    ("The equation can be solved step by step.", "science", "method"),
    ("The experiment yielded consistent results.", "science", "method"),
    ("The hypothesis was supported by data.", "science", "method"),

    # Ambiguous
    ("The bank by the river was flooded.", "ambiguous", "polysemy"),
    ("Spring flowers bloom in March.", "ambiguous", "polysemy"),
    ("The match was exciting to watch.", "ambiguous", "polysemy"),
    ("A fair decision was made by the judge.", "ambiguous", "polysemy"),

    # General/complex
    ("The quick brown fox jumps over the lazy dog.", "general", "proverb"),
    ("She has been working on this project.", "general", "progressive"),
    ("They went to the market when it started.", "general", "complex"),
    ("The report was submitted on time.", "general", "passive"),
]

OUTPUT_DIR = _Path(r"d:\develop\TransformerLens-main\tests\glm5_temp\stage694_rotation_semantics_20260406_2220")
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
    path = MODEL_MAP[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def measure_rotation(model, tokenizer, text):
    """Compute rotation from L0 to final layer"""
    model_device = next(model.parameters()).device
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64).to(model_device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    n_layers = len(outputs.hidden_states)
    h_l0 = outputs.hidden_states[0][0, -1, :].float().cpu()
    h_final = outputs.hidden_states[n_layers - 1][0, -1, :].float().cpu()

    # Compute rotation angle
    cos_theta = F.cosine_similarity(h_l0.unsqueeze(0), h_final.unsqueeze(0)).item()
    angle = math.degrees(math.acos(max(-1, min(1, cos_theta))))

    # Compute rotation vector (perpendicular component)
    l0_norm = torch.norm(h_l0)
    if l0_norm > 1e-10:
        l0_dir = h_l0 / l0_norm
        proj = cos_theta * torch.norm(h_final) * l0_dir
        rot_vec = h_final - proj
        rot_norm = torch.norm(rot_vec).item()
    else:
        rot_norm = 0.0

    # Also compute norm ratio (final/L0)
    f_norm = torch.norm(h_final).item()
    norm_ratio = f_norm / max(l0_norm, 1e-10)

    return {
        "angle": angle,
        "cos_theta": cos_theta,
        "rot_norm": rot_norm,
        "l0_norm": l0_norm.item(),
        "final_norm": f_norm,
        "norm_ratio": norm_ratio,
    }


def anova_test(group_values):
    """Simple one-way ANOVA: F-statistic"""
    groups = [np.array(v) for v in group_values.values() if len(v) > 1]
    if len(groups) < 2:
        return 0.0, 1.0

    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_within = sum(np.sum((g - g.mean())**2) for g in groups)

    df_between = len(groups) - 1
    df_within = len(all_vals) - len(groups)

    if df_within == 0 or ss_within < 1e-15:
        return float('inf'), 0.0

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    f_stat = ms_between / max(ms_within, 1e-15)
    return f_stat, 0.0  # Skip p-value for simplicity


def run_single_model(model_name, texts_with_cats, log):
    """Analyze rotation semantics for one model"""
    log(f"\n{'='*60}")
    log(f"  Processing: {model_name}")
    log(f"{'='*60}")

    t0 = time.time()
    model, tokenizer = load_model(model_name)

    # Measure rotation for each text
    measurements = []
    for text, cat, subcat in texts_with_cats:
        try:
            m = measure_rotation(model, tokenizer, text)
            m["category"] = cat
            m["subcategory"] = subcat
            m["text"] = text
            m["word_count"] = len(text.split())
            measurements.append(m)
        except Exception as e:
            log(f"    ERROR: {e}")

    log(f"  Measured {len(measurements)} texts")

    # Group by category
    cat_groups = defaultdict(list)
    for m in measurements:
        cat_groups[m["category"]].append(m)

    # A: Category-level rotation statistics
    log(f"\n  A: Rotation Angle by Category")
    log(f"  {'category':>12s}  {'n':>3s}  {'mean_angle':>11s}  {'std_angle':>10s}  {'mean_cos':>9s}  {'mean_norm_ratio':>15s}")
    cat_stats = {}
    for cat in sorted(cat_groups.keys()):
        group = cat_groups[cat]
        angles = [m["angle"] for m in group]
        cos_vals = [m["cos_theta"] for m in group]
        norms = [m["norm_ratio"] for m in group]
        stat = {
            "n": len(group),
            "mean_angle": np.mean(angles),
            "std_angle": np.std(angles),
            "mean_cos": np.mean(cos_vals),
            "mean_norm_ratio": np.mean(norms),
        }
        cat_stats[cat] = stat
        log(f"  {cat:>12s}  {stat['n']:>3d}  {stat['mean_angle']:>11.1f}  {stat['std_angle']:>10.1f}  {stat['mean_cos']:>9.4f}  {stat['mean_norm_ratio']:>15.4f}")

    # B: ANOVA - does category predict rotation angle?
    angles_by_cat = {cat: [m["angle"] for m in group] for cat, group in cat_groups.items()}
    f_angle, _ = anova_test(angles_by_cat)

    cos_by_cat = {cat: [m["cos_theta"] for m in group] for cat, group in cat_groups.items()}
    f_cos, _ = anova_test(cos_by_cat)

    norm_by_cat = {cat: [m["norm_ratio"] for m in group] for cat, group in cat_groups.items()}
    f_norm, _ = anova_test(norm_by_cat)

    log(f"\n  B: ANOVA F-statistics (category -> metric)")
    log(f"  angle:     F={f_angle:.3f}")
    log(f"  cos_theta: F={f_cos:.3f}")
    log(f"  norm_ratio:F={f_norm:.3f}")

    # C: Word count correlation
    word_counts = [m["word_count"] for m in measurements]
    angles = [m["angle"] for m in measurements]
    cos_vals = [m["cos_theta"] for m in measurements]

    if len(word_counts) > 2:
        r_angle_wc = np.corrcoef(word_counts, angles)[0, 1]
        r_cos_wc = np.corrcoef(word_counts, cos_vals)[0, 1]
    else:
        r_angle_wc = r_cos_wc = 0.0

    log(f"\n  C: Word Count Correlation")
    log(f"  angle vs word_count:  r={r_angle_wc:.4f}")
    log(f"  cos_theta vs word_count: r={r_cos_wc:.4f}")

    # D: Pairwise category differences
    log(f"\n  D: Largest Category Differences (angle)")
    cats = sorted(cat_stats.keys())
    max_diff = 0
    max_pair = ("", "")
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            diff = abs(cat_stats[cats[i]]["mean_angle"] - cat_stats[cats[j]]["mean_angle"])
            if diff > max_diff:
                max_diff = diff
                max_pair = (cats[i], cats[j])
    log(f"  Largest: {max_pair[0]} vs {max_pair[1]} = {max_diff:.1f} deg")

    # E: Interpretation
    log(f"\n  E: INV-351 Rotation-Semantics Coupling")
    if f_angle > 3.0:
        log(f"  -> INV-351 PARTIAL: Category explains rotation angle (F={f_angle:.2f})")
    else:
        log(f"  -> INV-351 REJECTED: Category does NOT predict rotation angle (F={f_angle:.2f})")

    if abs(r_angle_wc) > 0.3:
        log(f"  -> INV-352 CONFIRMED: Text length correlates with rotation (r={r_angle_wc:.3f})")
    else:
        log(f"  -> INV-352 REJECTED: Text length does NOT correlate with rotation (r={r_angle_wc:.3f})")

    elapsed = time.time() - t0
    log(f"  elapsed: {elapsed:.1f}s")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_name": model_name,
        "n_texts": len(measurements),
        "cat_stats": cat_stats,
        "anova": {"angle": f_angle, "cos_theta": f_cos, "norm_ratio": f_norm},
        "word_count_corr": {"angle": r_angle_wc, "cos_theta": r_cos_wc},
        "max_cat_diff": {"pair": list(max_pair), "diff": max_diff},
    }


def main():
    import json

    log_path = OUTPUT_DIR / "output.txt"
    log = Logger(log_path)

    log(f"\n{'='*70}")
    log(f"  P48: Rotation Semantics Analysis (Stage694)")
    log(f"{'='*70}")

    all_results = {}
    t_total = time.time()

    for model_name in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        try:
            result = run_single_model(model_name, TEXTS_WITH_CATS, log)
            all_results[model_name] = result
        except Exception as e:
            log(f"  ERROR processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-model summary
    log(f"\n{'='*70}")
    log(f"  CROSS-MODEL SUMMARY")
    log(f"{'='*70}")

    log(f"\n  A: ANOVA F-statistics across models")
    log(f"  {'model':>12s}  {'F_angle':>10s}  {'F_cos':>10s}  {'F_norm':>10s}  {'r_wc_angle':>12s}")
    for m, r in all_results.items():
        log(f"  {m:>12s}  {r['anova']['angle']:>10.3f}  {r['anova']['cos_theta']:>10.3f}  {r['anova']['norm_ratio']:>10.3f}  {r['word_count_corr']['angle']:>12.4f}")

    log(f"\n  B: Consistent category patterns across models")
    # Check which categories have highest/lowest angles consistently
    categories = ["concrete", "factual", "aesthetic", "logical", "temporal", "math", "science", "ambiguous", "general"]
    log(f"  {'category':>12s}", end="")
    for m in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        log(f"  {m:>10s}", end="")
    log()

    for cat in categories:
        log(f"  {cat:>12s}", end="")
        for m in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
            if m in all_results and cat in all_results[m]["cat_stats"]:
                val = all_results[m]["cat_stats"][cat]["mean_angle"]
                log(f"  {val:>10.1f}", end="")
            else:
                log(f"  {'N/A':>10s}", end="")
        log()

    elapsed = time.time() - t_total
    log(f"\n{'='*70}")
    log(f"  P48 COMPLETE - Total elapsed: {elapsed:.1f}s")
    log(f"{'='*70}")

    out_path = OUTPUT_DIR / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    log(f"\n  Results saved to: {out_path}")

    log.close()


if __name__ == "__main__":
    main()
