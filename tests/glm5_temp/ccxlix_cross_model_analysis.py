"""
CCXLIX 跨模型综合分析 — 修正假说
"""
import json, numpy as np
from pathlib import Path

TEMP = Path("tests/glm5_temp")
models = ["qwen3", "glm4", "deepseek7b"]
domains = ["emotion", "evaluation", "temperature", "animal", "food", "material"]

all_data = {}
for m in models:
    p = TEMP / f"ccxlix_opposition_{m}.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            all_data[m] = json.load(f)

print("=" * 80)
print("CCXLIX 跨模型综合分析 — 假说修正!")
print("=" * 80)

# 1. 核心结果表
print("\n### 1. 核心结果: 几何-语义Pearson相关系数")
print(f"  {'领域':<14} {'类型':<6} {'Qwen3':>8} {'GLM4':>8} {'DS7B':>8} {'均值':>8} {'预测':>8} {'正确?':>6}")

for d in domains:
    rvals = []
    for m in models:
        r = all_data[m]["results"].get(d, {}).get("pearson_r", None)
        rvals.append(r)
    
    dtype = all_data[models[0]]["results"].get(d, {}).get("domain_type", "?")
    pred = "neg" if dtype == "oppositional" else "pos"
    mean_r = np.mean([r for r in rvals if r is not None])
    
    # 检查预测是否正确
    signs = ["+" if r > 0 else "-" for r in rvals if r is not None]
    majority_sign = "+" if sum(1 for s in signs if s == "+") >= 2 else "-"
    correct = "Y" if (pred == "neg" and majority_sign == "-") or (pred == "pos" and majority_sign == "+") else "N"
    
    rstrs = [f"{r:+.3f}" if r is not None else "N/A" for r in rvals]
    print(f"  {d:<14} {dtype:<6} {rstrs[0]:>8} {rstrs[1]:>8} {rstrs[2]:>8} {mean_r:>+8.3f} {pred:>8} {correct:>6}")

# 2. 关键发现
print(f"\n### 2. 关键发现")

# Evaluation: 强正相关(对比假说预测负相关!)
print(f"\n  ★★★★★ evaluation领域: 强正相关 (r=+0.45~+0.67)")
print(f"    → 同价类别聚类: excellent-amazing最近, terrible-horrific次近")
print(f"    → 这是'相似组织', 不是'对比组织'!")
print(f"    → ★ 对比假说的预测完全错误!")

# Emotion: 混合
print(f"\n  ★★★★ emotion领域: 混合结果")
print(f"    Qwen3: +0.145 (弱正), GLM4: +0.035 (零), DS7B: -0.418 (负)")
print(f"    → 只在DS7B中出现负相关")
print(f"    → CCXLVIII的'对比组织'发现可能是N=4的小样本效应")

# Temperature: 正相关 + warm-cool最近
print(f"\n  ★★★★ temperature领域: 正相关 + 中等反义词最近")
print(f"    → warm-cool(反义词)在Qwen3/GLM4中几何最近(0.532/0.589×)")
print(f"    → 但总体是正相关: cold-freezing(相似)也很近")
print(f"    → 混合模式: 同极相似 + 中等程度对比接近")

# Material: 负相关(非对立领域!)
print(f"\n  ★★★ material领域: 负相关 (r=-0.06~-0.33)")
print(f"    → '非对立'领域却出现负相关!")
print(f"    → metal-plastic(语义最远)却几何较近(0.72-0.88×)")
print(f"    → 这与对比假说的预测相反")

# 3. 修正假说
print(f"\n### 3. ★★★★★ 假说修正: 分布相似性而非语义对立")
print("""
原始假说(已推翻):
  对立领域 → 负相关(对比组织)
  非对立领域 → 正相关(相似组织)

修正假说:
  几何距离 ∝ 分布距离(共现模式距离), 而非语义距离(valence-arousal)

  关键洞察:
  - 反义词(hot/cold, happy/sad)在分布空间中是近邻, 不是远亲!
  - 因为反义词出现在相同的上下文中(都描述温度/情感)
  - valence-arousal模型把它们放在对立面, 但模型学到的是共现模式
  - → 几何-语义负相关 = valence-arousal模型与分布结构的偏差

  验证:
  - evaluation: excellent-amazing共现在正面评价中 → 几何最近(0.33-0.65×)
  - temperature: warm-cool共现在温和天气中 → 几何最近(0.53-0.84×)
  - emotion: happy-sad共现在情感描述中 → 几何较近(0.89-0.94×)
  → 这些"反义对"在分布空间中是近邻!

  数学刻画:
  d_geo(i,j) ∝ d_dist(i,j) ≈ 1 - cos(context_i, context_j)
  d_sem(i,j) (valence-arousal) 与 d_dist(i,j) 不一定一致
  → 当d_sem与d_dist方向相反时, 出现"负相关"假象
""")

# 4. 温度领域的特殊模式
print(f"\n### 4. 温度领域: warm-cool效应")
print(f"  {'模型':<12} {'warm-cool':>10} {'hot-cold':>10} {'cold-freezing':>14}")
for m in models:
    res = all_data[m]["results"].get("temperature", {})
    if "antonym_ratio" in res:
        print(f"  {m:<12} {res.get('antonym_ratio', 'N/A'):>10}")

print(f"""
  温度领域的关键观察:
  - warm-cool(中等强度反义词) → 几何最近 (0.53-0.84×)
  - hot-cold(强反义词) → 几何接近均值 (0.95-1.05×)  
  - boiling-freezing(极强反义词) → 几何接近均值 (0.80-1.12×)
  → 只有中等程度的对立词对几何最近!
  → 太强的对立(hot/cold)不如中等对立(warm/cool)近
  → 因为warm/cool的共现频率高于hot/cold
""")

# 5. 统计显著性
print(f"### 5. 统计显著性")
sig_count = 0
total = 0
for m in models:
    for d in domains:
        r = all_data[m]["results"].get(d, {})
        p = r.get("pearson_p", 1.0)
        total += 1
        if p < 0.05:
            sig_count += 1
            print(f"  ★ {m} {d}: p={p:.3f} (r={r.get('pearson_r', 0):.3f})")

print(f"\n  显著结果: {sig_count}/{total} ({sig_count/total*100:.0f}%)")
print(f"  → 统计功效仍然不足(N=6只有15对)")

# 6. 对比CCXLVIII
print(f"\n### 6. 与CCXLVIII结果的对比")
print(f"  CCXLVIII emotion4 Pearson:")
print(f"    Qwen3: -0.410, GLM4: -0.290, DS7B: -0.074")
print(f"  CCXLIX emotion6 Pearson:")
print(f"    Qwen3: +0.145, GLM4: +0.035, DS7B: -0.418")
print(f"""
  关键差异:
  - CCXLVIII中Qwen3/GLM4显示负相关, CCXLIX中变为正相关/零相关
  - 原因1: N=4(6对) vs N=6(15对) — 小样本效应
  - 原因2: 不同的词集(12词 vs 10词) — 中心估计不稳定
  - 原因3: emotion4只有happy一个正面类, emotion6增加了surprise
  → "对比组织"效应在N=6时消失或减弱!
  → CCXLVIII的负相关可能是小样本假阳性
""")
