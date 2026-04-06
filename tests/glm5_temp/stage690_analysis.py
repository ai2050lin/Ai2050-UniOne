import json, math, numpy as np, pathlib

models = {
    "qwen3": {"d": 2560, "layers": 37, "l0_align": 1.0, "final_align": 0.9359,
              "l0_pca": [1.0, 0.0, 0.0], "final_pca": [0.2232, 0.0822, 0.0758],
              "l0_final_cos": 0.0217, "l0_ue_max": 1.0, "final_ue_max": 0.179},
    "deepseek7b": {"d": 4096, "layers": 29, "l0_align": 1.0, "final_align": 0.8965,
                   "l0_pca": [1.0, 0.0, 0.0], "final_pca": [0.5422, 0.1225, 0.0546],
                   "l0_final_cos": -0.006, "l0_ue_max": 0.067, "final_ue_max": 0.152},
    "glm4": {"d": 4096, "layers": 41, "l0_align": 1.0, "final_align": 0.7931,
             "l0_pca": [1.0, 0.0, 0.0], "final_pca": [0.2228, 0.1225, 0.0546],
             "l0_final_cos": -0.006, "l0_ue_max": 0.067, "final_ue_max": 0.152},
    "gemma4": {"d": 1536, "layers": 36, "l0_align": 1.0, "final_align": 0.8731,
               "l0_pca": [1.0, 0.0, 0.0], "final_pca": [0.1894, 0.0892, 0.0716],
               "l0_final_cos": -0.171, "l0_ue_max": 1.0, "final_ue_max": 0.067},
}

lines = []
def p(s=""):
    lines.append(s)
    print(s)

p("=" * 70)
p("  P44: Cross-Model Semantic Basis Alignment (Stage690) Summary")
p("=" * 70)

p("\n  1. L0 Basis: ALL models cos=1.0000 (100% variance)")
p("  " + "-" * 60)
for m, d in models.items():
    p(f"  {m:>12s}: d={d['d']}, L={d['layers']}, L0={d['l0_align']:.4f}, Final={d['final_align']:.4f}")

p("\n  2. L0 vs Final Basis Rotation")
p("  " + "-" * 60)
for m, d in models.items():
    p(f"  {m:>12s}: L0-Final cos = {d['l0_final_cos']:+.4f} (angle = {math.degrees(math.acos(min(abs(d['l0_final_cos']),1.0))):.1f} deg)")
p("\n  -> ALL models: L0 basis nearly orthogonal to final basis")
p("  -> Gemma4 extreme: cos=-0.171 (anti-aligned)")

p("\n  3. Final Alignment Ranking")
p("  " + "-" * 60)
ranked = sorted(models.items(), key=lambda x: -x[1]["final_align"])
for i, (m, d) in enumerate(ranked):
    p(f"  {i+1}. {m:>12s}: {d['final_align']:.4f} (PCA1={d['final_pca'][0]*100:.1f}%)")

p("\n  4. Basis vs Unembed (L0)")
p("  " + "-" * 60)
for m, d in models.items():
    p(f"  {m:>12s}: L0 max_cos_unembed = {d['l0_ue_max']:.3f}")

p("\n  5. Key Findings")
p("  " + "-" * 60)
p("  [F1] L0 alignment=1.0 is UNIVERSAL (4/4 models) -> INV-339 CONFIRMED")
p("  [F2] L0 basis = embedding output direction (Qwen3/Gemma4 cos=1.0 with unembed)")
p("  [F3] L0->Final rotation is NEARLY COMPLETE (cos 0.02 to -0.17)")
p("  [F4] Qwen3 preserves most directional structure (final align=0.936)")
p("  [F5] GLM4 scatters information most (final align=0.793)")
p("  [F6] DS7B strongest single-axis encoding (PCA1=54.2%)")

p("\n  6. Theoretical Correction")
p("  " + "-" * 60)
p("  The 'semantic basis direction' from P43 is NOT a learned structure.")
p("  It is simply the DIRECTION OF THE EMBEDDING OUTPUT.")
p("  L0 = embedding(h_last_token), and all texts share similar embedding norms")
p("  -> the 'mean direction' is just the embedding space center.")
p("  ")
p("  The REAL question should be:")
p("  (a) Why does embedding produce such aligned outputs?")
p("  (b) What information is in the ROTATION from L0 to final?")
p("  (c) Is the rotation axis shared across models?")

p("\n  7. INV-343 Verdict: CANNOT DIRECTLY TEST")
p("  " + "-" * 60)
p("  All 4 models have different hidden dims (1536/2560/4096)")
p("  Only DS7B-GLM4 share dim=4096")
p("  Indirect evidence: L0=1.0 universal, but basis is embedding-derived")

p("\n  8. Next: Need to test ROTATION AXIS instead of raw basis")

# Save
out = {
    "findings": {
        "F1_L0_universal": "All 4/4 L0 alignment=1.0",
        "F2_L0_embedding_derived": "L0 basis = embedding output direction",
        "F3_L0_rotation_nearly_complete": "L0-Final cos: 0.02 to -0.17",
        "F4_alignment_ranking": "Qwen3(0.936) > DS7B(0.897) > Gemma4(0.873) > GLM4(0.793)",
        "F5_PCA1_ranking": "DS7B(54.2%) >> others(~20%)",
    },
    "INV343": "CANNOT_DIRECTLY_TEST - different hidden dims",
    "correction": "P43 basis = embedding direction, NOT learned semantic structure",
    "models": {m: {k: v for k, v in d.items()} for m, d in models.items()},
}
out_path = pathlib.Path(r"d:\develop\TransformerLens-main\tests\glm5_temp\stage690_cross_model_basis_20260406_2145\summary.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

txt_path = out_path.parent / "output.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
p(f"\n  Saved to: {out_path}")
