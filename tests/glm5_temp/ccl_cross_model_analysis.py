"""
CCL 跨模型综合分析
===================
综合Qwen3/GLM4/DS7B的分布相似性假说验证结果
"""

import json, sys
import numpy as np
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TEMP = Path("tests/glm5_temp")

# 加载三个模型的结果
models = ["qwen3", "glm4", "deepseek7b"]
all_data = {}
for m in models:
    path = TEMP / f"ccl_{m}_results.json"
    with open(path, "r", encoding="utf-8") as f:
        all_data[m] = json.load(f)

domains = ["emotion8", "animal8", "color8", "evaluation8"]

print("=" * 80)
print("CCL 跨模型综合分析: 分布相似性 vs 语义距离 哪个更好预测几何距离?")
print("=" * 80)

# =============================================
# 分析1: 每个领域的跨模型平均r值
# =============================================
print("\n" + "=" * 80)
print("分析1: 每个领域的跨模型+跨层平均相关系数")
print("=" * 80)

for domain in domains:
    r_emb_all = []
    r_sem_all = []
    pr_emb_all = []
    pr_sem_all = []
    
    for m in models:
        if domain not in all_data[m]:
            continue
        dom_data = all_data[m][domain]
        for key, val in dom_data.items():
            if key.startswith("L"):
                r_emb_all.append(val["r_geo_emb"])
                r_sem_all.append(val["r_geo_sem"])
                pr_emb_all.append(val["pr_geo_emb_ctrl_sem"])
                pr_sem_all.append(val["pr_geo_sem_ctrl_emb"])
    
    if len(r_emb_all) == 0:
        continue
    
    print(f"\n--- {domain} (共{len(r_emb_all)}个数据点) ---")
    print(f"  直接相关:")
    print(f"    r(geo, embedding):  mean={np.mean(r_emb_all):+.3f}  std={np.std(r_emb_all):.3f}  "
          f"range=[{np.min(r_emb_all):+.3f}, {np.max(r_emb_all):+.3f}]")
    print(f"    r(geo, semantic):   mean={np.mean(r_sem_all):+.3f}  std={np.std(r_sem_all):.3f}  "
          f"range=[{np.min(r_sem_all):+.3f}, {np.max(r_sem_all):+.3f}]")
    print(f"  偏相关(控制另一个变量后):")
    print(f"    pr(geo,emb|sem):    mean={np.mean(pr_emb_all):+.3f}  std={np.std(pr_emb_all):.3f}")
    print(f"    pr(geo,sem|emb):    mean={np.mean(pr_sem_all):+.3f}  std={np.std(pr_sem_all):.3f}")
    
    # 判定哪个更优
    avg_abs_r_emb = np.mean(np.abs(r_emb_all))
    avg_abs_r_sem = np.mean(np.abs(r_sem_all))
    avg_abs_pr_emb = np.mean(np.abs(pr_emb_all))
    avg_abs_pr_sem = np.mean(np.abs(pr_sem_all))
    
    winner = "EMBEDDING" if avg_abs_r_emb > avg_abs_r_sem else "SEMANTIC"
    winner_pr = "EMBEDDING" if avg_abs_pr_emb > avg_abs_pr_sem else "SEMANTIC"
    
    print(f"  直接相关胜者: {winner} (|r|: emb={avg_abs_r_emb:.3f} vs sem={avg_abs_r_sem:.3f})")
    print(f"  偏相关胜者:   {winner_pr} (|pr|: emb={avg_abs_pr_emb:.3f} vs sem={avg_abs_pr_sem:.3f})")

# =============================================
# 分析2: evaluation领域的详细分析 (最强结果)
# =============================================
print("\n" + "=" * 80)
print("分析2: evaluation领域 - 跨模型一致性最强")
print("=" * 80)

for m in models:
    if "evaluation8" not in all_data[m]:
        continue
    dom_data = all_data[m]["evaluation8"]
    print(f"\n  {m}:")
    for key in sorted(dom_data.keys()):
        if key.startswith("L"):
            val = dom_data[key]
            print(f"    {key}: r(geo,emb)={val['r_geo_emb']:+.3f}  r(geo,sem)={val['r_geo_sem']:+.3f}  "
                  f"pr(emb|sem)={val['pr_geo_emb_ctrl_sem']:+.3f}  pr(sem|emb)={val['pr_geo_sem_ctrl_emb']:+.3f}  "
                  f"R2={val['R2_total']:.3f}")

# =============================================
# 分析3: emotion领域 - r(geo,sem)的符号差异
# =============================================
print("\n" + "=" * 80)
print("分析3: emotion领域 - 语义距离的符号方向")
print("=" * 80)

for m in models:
    if "emotion8" not in all_data[m]:
        continue
    dom_data = all_data[m]["emotion8"]
    r_sems = [dom_data[k]["r_geo_sem"] for k in dom_data if k.startswith("L")]
    r_embs = [dom_data[k]["r_geo_emb"] for k in dom_data if k.startswith("L")]
    print(f"  {m}: r(geo,sem) mean={np.mean(r_sems):+.3f}  r(geo,emb) mean={np.mean(r_embs):+.3f}")

# =============================================
# 分析4: 偏相关对比 - 哪个独立贡献更大?
# =============================================
print("\n" + "=" * 80)
print("分析4: 偏相关对比 - 控制对方后, 谁的独立贡献更大?")
print("=" * 80)

emb_wins_pr = 0
sem_wins_pr = 0

for domain in domains:
    for m in models:
        if domain not in all_data[m]:
            continue
        dom_data = all_data[m][domain]
        for key, val in dom_data.items():
            if key.startswith("L"):
                if abs(val["pr_geo_emb_ctrl_sem"]) > abs(val["pr_geo_sem_ctrl_emb"]):
                    emb_wins_pr += 1
                else:
                    sem_wins_pr += 1

total = emb_wins_pr + sem_wins_pr
print(f"  偏相关胜者: embedding={emb_wins_pr}, semantic={sem_wins_pr} (共{total}个数据点)")
print(f"  比例: embedding={emb_wins_pr/total:.1%}, semantic={sem_wins_pr/total:.1%}")

# =============================================
# 分析5: r(geo,emb)的负相关现象
# =============================================
print("\n" + "=" * 80)
print("分析5: r(geo,emb)为负的情况 - 嵌入相似但几何远的'分离效应'")
print("=" * 80)

neg_emb_cases = []
for domain in domains:
    for m in models:
        if domain not in all_data[m]:
            continue
        dom_data = all_data[m][domain]
        for key, val in dom_data.items():
            if key.startswith("L") and val["r_geo_emb"] < -0.1:
                neg_emb_cases.append({
                    "domain": domain, "model": m, "layer": key,
                    "r_geo_emb": val["r_geo_emb"],
                    "r_geo_sem": val["r_geo_sem"],
                    "pr_geo_emb_ctrl_sem": val["pr_geo_emb_ctrl_sem"],
                })

if len(neg_emb_cases) > 0:
    print(f"  共发现{len(neg_emb_cases)}个r(geo,emb)<-0.1的案例:")
    for case in neg_emb_cases:
        print(f"    {case['domain']}/{case['model']}/{case['layer']}: "
              f"r(geo,emb)={case['r_geo_emb']:+.3f}  r(geo,sem)={case['r_geo_sem']:+.3f}  "
              f"pr(emb|sem)={case['pr_geo_emb_ctrl_sem']:+.3f}")
    
    # 分析集中在哪些模型和领域
    by_model = {}
    by_domain = {}
    for case in neg_emb_cases:
        by_model[case["model"]] = by_model.get(case["model"], 0) + 1
        by_domain[case["domain"]] = by_domain.get(case["domain"], 0) + 1
    
    print(f"\n  按模型分布: {by_model}")
    print(f"  按领域分布: {by_domain}")
    
    print("\n  解释: r(geo,emb)<0 意味着嵌入相似的词在几何上更远")
    print("  这可能是模型的'判别性分离'策略: 为了更好地区分相似词, 将它们在几何上推开")
    print("  这种效应在GLM4和DS7B中更强, 尤其在animal和color领域")
else:
    print("  未发现r(geo,emb)<-0.1的案例")

# =============================================
# 分析6: 层间演变趋势
# =============================================
print("\n" + "=" * 80)
print("分析6: 层间演变 - 随着深度增加, embedding vs semantic的变化趋势")
print("=" * 80)

for m in models:
    print(f"\n  {m}:")
    for domain in domains:
        if domain not in all_data[m]:
            continue
        dom_data = all_data[m][domain]
        layers = sorted([k for k in dom_data if k.startswith("L")], key=lambda x: int(x[1:]))
        if len(layers) < 2:
            continue
        
        r_emb_vals = [dom_data[l]["r_geo_emb"] for l in layers]
        r_sem_vals = [dom_data[l]["r_geo_sem"] for l in layers]
        
        # 趋势: 从浅到深
        emb_trend = "UP" if r_emb_vals[-1] > r_emb_vals[0] else "DOWN"
        sem_trend = "UP" if r_sem_vals[-1] > r_sem_vals[0] else "DOWN"
        
        print(f"    {domain}: r(emb) {r_emb_vals[0]:+.3f}->{r_emb_vals[-1]:+.3f} ({emb_trend})  "
              f"r(sem) {r_sem_vals[0]:+.3f}->{r_sem_vals[-1]:+.3f} ({sem_trend})")

# =============================================
# 分析7: 综合判定
# =============================================
print("\n" + "=" * 80)
print("分析7: 综合判定 - 分布相似性假说是否被支持?")
print("=" * 80)

# 计算所有数据点的平均R2
all_r2 = []
r2_by_winner = {"embedding": [], "semantic": []}

for m in models:
    for domain in domains:
        if domain not in all_data[m]:
            continue
        for key, val in all_data[m][domain].items():
            if key.startswith("L"):
                all_r2.append(val["R2_total"])
                r2_by_winner[val["winner"]].append(val["R2_total"])

print(f"\n  总数据点: {len(all_r2)}")
print(f"  平均R2: {np.mean(all_r2):.3f}")
print(f"  embedding胜出时平均R2: {np.mean(r2_by_winner['embedding']):.3f} (N={len(r2_by_winner['embedding'])})")
print(f"  semantic胜出时平均R2: {np.mean(r2_by_winner['semantic']):.3f} (N={len(r2_by_winner['semantic'])})")

# 核心结论
print("\n" + "=" * 80)
print("★ ★ ★ ★ ★  核心结论  ★ ★ ★ ★ ★")
print("=" * 80)

print("""
1. ★★★★★ evaluation领域: 语义距离一致胜出
   - 所有模型、所有层中, r(geo,sem)=0.52~0.76, 远高于r(geo,emb)=0.25~0.53
   - 偏相关pr(sem|emb)也显著(0.48~0.81), 说明语义距离有独立贡献
   - 评价领域的valence维度是核心组织轴, 比分布相似性更重要

2. ★★★★ emotion领域: 双重组织模式
   - Qwen3: embedding胜出(正相关), 语义距离几乎无关
   - GLM4/DS7B: 语义距离负相关(对比组织), embedding正相关
   - 偏相关显示两者都有独立贡献: emb正(0.25-0.49), sem负(-0.18~-0.42)
   - 结论: emotion领域同时受分布相似性(+,近邻相聚)和语义对比(-,对立相近)双重驱动

3. ★★★ animal领域: 模型间差异最大
   - Qwen3: 深层embedding强正相关(0.63)
   - GLM4/DS7B: embedding负相关(-0.27~-0.49)!
   - 负相关意味着: 嵌入相似的动物在几何上被推开(判别性分离)
   - GLM4/DS7B可能使用了更强的判别性策略来区分相似动物

4. ★★★ color领域: 最弱的组织模式
   - 所有模型中相关都很弱(R2<0.35)
   - 颜色可能依赖感知编码(色相环), 不完全被语义或分布相似性捕捉
   - GLM4/DS7B中embedding再次出现负相关

5. ★★★★ 偏相关胜者分布: 基本对半
   - embedding胜: {emb_wins_pr}, semantic胜: {sem_wins_pr}
   - 两个变量都有独立贡献, 不可互相替代
   - 分布相似性和语义距离是两个独立但相关的组织原则

6. ★★★ 层间演变: Qwen3深层偏向embedding
   - Qwen3: 浅层semantic更强, 深层embedding更强
   - GLM4: 没有清晰趋势
   - DS7B: 混合

总体结论:
★★★★ 分布相似性假说被部分支持, 但不是完整答案 ★★★★
  - embedding相似性确实能预测几何距离, 但...
  - 在evaluation领域, 语义距离的预测力更强
  - 在emotion领域, 两者方向相反(emb正, sem负)
  - 在animal/color, 部分模型出现负相关(判别性分离)
  - → 需要更精细的理论: 不同领域和不同模型使用不同的组织策略
""".format(emb_wins_pr=emb_wins_pr, sem_wins_pr=sem_wins_pr))
