"""CCXXXVIII 汇总分析"""
import json
from pathlib import Path

TEMP = Path("tests/glm5_temp")

for model in ['qwen3', 'glm4', 'deepseek7b']:
    path = TEMP / f"ccxxxviii_prompt_syntax_{model}.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"  {model.upper()}")
    print(f"{'='*70}")
    
    # Part 1: 颜色prompt对比
    print("\n--- Part 1: Color Prompt n_sep (最大值) ---")
    cp = data.get("color_prompts", {})
    for pname in ["A_unnatural", "B_color_explicit", "C_color_attribute", "D_color_subjective"]:
        results = [v for v in cp.values() if v.get("prompt_name") == pname]
        if results:
            best = max(results, key=lambda x: x.get("n_separating_PCs", 0))
            print(f"  {pname:25s}: max n_sep={best['n_separating_PCs']}, L{best['layer']}")
    
    # Part 2: 情感可加性
    print("\n--- Part 2: Emotion Additivity ---")
    ea = data.get("emotion_additivity", {})
    for lkey in sorted(ea.keys()):
        ld = ea[lkey]
        deltas = [ld[k]["delta_n_sep"] for k in sorted(ld.keys())]
        matches = sum(1 for k in ld if ld[k]["match"])
        total = len(ld)
        # 找最佳regularity
        best_reg = max(ld.values(), key=lambda x: x.get("geometry", {}).get("regularity_score", 0) if x.get("geometry") else 0)
        reg_str = f", reg={best_reg['geometry']['regularity_score']:.3f}" if best_reg.get("geometry") else ""
        print(f"  {lkey}: Δ={deltas}, match={matches}/{total}{reg_str}")
    
    # Part 3: 跨领域最佳结果
    print("\n--- Part 3: Best per Domain ---")
    md = data.get("multi_domain", {})
    for domain in ["habitat", "emotion", "occupation", "color"]:
        domain_data = {k: v for k, v in md.items() if v.get("domain") == domain}
        if domain_data:
            best = max(domain_data.values(), key=lambda x: x.get("n_separating_PCs", 0))
            geom_str = ""
            if best.get("geometry"):
                geom_str = f", reg={best['geometry']['regularity_score']:.3f}"
            print(f"  {domain:12s}: n_sep={best['n_separating_PCs']}, "
                  f"prompt={best.get('prompt_name','?')}, L{best['layer']}, "
                  f"match={'✓' if best['match'] else '✗'}{geom_str}")
    
    # Part 4: 颜色可加性
    ca = data.get("color_additivity", {})
    if ca:
        print("\n--- Part 4: Color Additivity ---")
        for lkey in sorted(ca.keys()):
            ld = ca[lkey]
            deltas = [ld[k]["delta_n_sep"] for k in sorted(ld.keys())]
            matches = sum(1 for k in ld if ld[k]["match"])
            print(f"  {lkey}: Δ={deltas}, match={matches}/{len(ld)}")

# 跨模型关键对比
print(f"\n{'='*70}")
print("  ★★★★★ 跨模型关键对比 ★★★★★")
print(f"{'='*70}")

print("\n1. 情感可加性: 最佳层N=3→6, Δn_sep")
for model in ['qwen3', 'glm4', 'deepseek7b']:
    path = TEMP / f"ccxxxviii_prompt_syntax_{model}.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    ea = data.get("emotion_additivity", {})
    # 找匹配率最高的层
    best_layer = None
    best_rate = 0
    for lkey, ld in ea.items():
        rate = sum(1 for k in ld if ld[k]["match"]) / max(len(ld), 1)
        if rate > best_rate:
            best_rate = rate
            best_layer = lkey
    if best_layer and best_layer in ea:
        ld = ea[best_layer]
        deltas = [ld[k]["delta_n_sep"] for k in sorted(ld.keys())]
        matches = sum(1 for k in ld if ld[k]["match"])
        print(f"  {model:12s} {best_layer}: Δ={deltas}, {matches}/{len(ld)} match")

print("\n2. 颜色领域: 各prompt最大n_sep")
for model in ['qwen3', 'glm4', 'deepseek7b']:
    path = TEMP / f"ccxxxviii_prompt_syntax_{model}.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cp = data.get("color_prompts", {})
    max_nsep = 0
    best_prompt = "?"
    for v in cp.values():
        if v.get("n_separating_PCs", 0) > max_nsep:
            max_nsep = v["n_separating_PCs"]
            best_prompt = v["prompt_name"]
    print(f"  {model:12s}: max n_sep={max_nsep} (prompt={best_prompt})")

print("\n3. 各领域最佳n_sep (跨所有prompt)")
for domain in ["habitat", "emotion", "occupation", "color"]:
    vals = []
    for model in ['qwen3', 'glm4', 'deepseek7b']:
        path = TEMP / f"ccxxxviii_prompt_syntax_{model}.json"
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        md = data.get("multi_domain", {})
        domain_data = {k: v for k, v in md.items() if v.get("domain") == domain}
        if domain_data:
            best = max(domain_data.values(), key=lambda x: x.get("n_separating_PCs", 0))
            vals.append((model, best["n_separating_PCs"], best.get("prompt_name", "?")))
    print(f"  {domain:12s}: " + ", ".join(f"{m}={n}({p})" for m, n, p in vals))
