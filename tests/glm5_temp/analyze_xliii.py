"""分析Phase XLIII结果"""
import json, sys
from pathlib import Path

f = sys.argv[1] if len(sys.argv) > 1 else None
if f is None:
    files = sorted(Path("tests/glm5_temp").glob("phase_xliii_p257_259_*.json"))
    f = str(files[-1]) if files else None

if f is None:
    print("No result file found")
    sys.exit(1)

print(f"File: {f}")
d = json.load(open(f, 'r', encoding='utf-8'))
model = d['model']
print(f"\n{'='*60}")
print(f"  {model} Phase XLIII Result Summary")
print(f"  {d['n_layers']}L, d={d['d_model']}, d_mlp={d['d_mlp']}")
print(f"{'='*60}")

# P257
print("\n=== P257: FFN Weight Analysis ===")
for l in d['p257_ffn_weight_analysis']['per_layer']:
    li = l['layer']
    print(f"  L{li:2d}: key_cos={l['key_pairwise_cos_mean']:.4f} "
          f"gate_in_cos={l['gate_in_cos_mean']:.4f} "
          f"W_gate_cum500={l['W_gate_cumvar_500']} "
          f"fruit_Wout_max={l.get('fruit_dir_Wout_cos_max','?')}")

gs = d['p257_ffn_weight_analysis'].get('global_summary', {})
if gs:
    print(f"  Global: key_cos={gs.get('key_cos_early_vs_late')}, "
          f"gate_in_cos={gs.get('gate_in_cos_early_vs_late')}")

# P258
print("\n=== P258: FFN Causal Chain ===")
for l in d['p258_ffn_causal_chain']['per_layer_summary']:
    li = l['layer']
    print(f"  L{li:2d}: ffn_G_cos={l['avg_ffn_G_cos']:+.4f} "
          f"ffn_ratio={l['avg_ffn_norm_ratio']:.2f} "
          f"gate_overlap={l['avg_gate_overlap']:.4f} "
          f"gate_sim={l.get('avg_gate_lin_sim','?')}")

# Per-triple summary
print("\n=== P258: Per-Triple Key Findings ===")
for td in d['p258_ffn_causal_chain']['per_triple'][:5]:
    noun, attr, combo = td['noun'], td['attr'], td['combo']
    # Find the layer with highest ffn_G_cos
    best_layer = max(td['per_layer'], key=lambda x: abs(x.get('ffn_G_cos', 0)))
    last_layer = [x for x in td['per_layer'] if x['layer'] == d['n_layers']-1]
    last = last_layer[0] if last_layer else None
    print(f"  {noun}+{attr}: best_L{best_layer['layer']}(cos={best_layer['ffn_G_cos']:+.4f}), "
          f"last_L{last['layer'] if last else '?'}(ratio={last['ffn_norm_ratio'] if last else '?'})")

# P259
print("\n=== P259: Weight-Level Source ===")
p259 = d['p259_weight_level_source']
gf = p259.get('global_findings', {})
print(f"  Stable fruit detectors: {len(gf.get('stable_fruit_detectors', []))}")
print(f"  Stable fruit outputs: {len(gf.get('stable_fruit_outputs', []))}")
print(f"  Detector-output overlap: {gf.get('detector_output_stable_overlap', '?')}")

for l in p259['per_layer']:
    li = l['layer']
    print(f"  L{li:2d}: fruit_det_in_cos={l.get('fruit_detector_in_fruit_cos_top10_mean','?')}, "
          f"fruit_det_out_cos={l.get('fruit_detector_out_fruit_cos_top10_mean','?')}, "
          f"gate_fruit_proj_max={l.get('gate_fruit_proj_max','?')}")

print("\nDone.")
