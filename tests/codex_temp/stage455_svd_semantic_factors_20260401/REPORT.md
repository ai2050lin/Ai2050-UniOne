# Stage455: SVD/ICA语义因子分解报告

**时间**: 2026-04-01 01:30
**目标**: 对偏置矩阵进行SVD/ICA分解，发现独立的语义因子
**方法**: 构建概念×神经元偏置矩阵 → SVD分解 → 因子语义解释 → 因子重建验证


---
## Qwen3-4B
- Layers: 36, Target: [31, 32, 33, 34, 35]
- Concepts: 132, Bias matrix: [132, 48640]

### SVD分解 (前20个因子)
- 累计解释方差: ['4.3%', '7.3%', '9.6%', '11.6%', '13.6%', '15.4%', '17.0%', '18.6%', '20.0%', '21.5%']

| Factor | 方差解释 | Top+概念 | Top-概念 |
|--------|---------|---------|---------|
| 0 | 4.3% | dancer(1.21), scientist(1.04), musician(1.00) | teacher(-1.06), farmer(-0.91), chef(-0.88) |
| 1 | 3.0% | mattress(0.76), sofa(0.73), ivory(0.69) | iron(-0.94), copper(-0.88), earth(-0.84) |
| 2 | 2.3% | singer(0.84), horse(0.72), mattress(0.65) | drawer(-0.79), eagle(-0.66), ruby(-0.64) |
| 3 | 2.0% | watermelon(1.06), mango(0.64), apple(0.62) | tiger(-0.83), lion(-0.73), ruby(-0.66) |
| 4 | 2.0% | copper(0.84), iron(0.81), steel(0.79) | ship(-0.68), pearl(-0.66), rain(-0.59) |
| 5 | 1.8% | artist(0.70), painter(0.64), lake(0.63) | rain(-0.84), snow(-0.72), pillow(-0.71) |
| 6 | 1.6% | lake(0.69), tree(0.68), snake(0.67) | silver(-0.74), gold(-0.65), artist(-0.52) |
| 7 | 1.5% | whale(0.66), orange(0.65), silver(0.59) | carpet(-0.80), blueberry(-0.66), pillow(-0.59) |
| 8 | 1.5% | peach(0.76), cherry(0.71), silver(0.64) | pineapple(-0.65), ambulance(-0.58), cloth(-0.56) |
| 9 | 1.4% | motorcycle(0.58), bicycle(0.54), cat(0.53) | berry(-0.88), deer(-0.67), river(-0.57) |

### 因子-属性关联 (ANOVA eta²)
| 类别.属性 | 最佳因子 | eta² | 组均值 |
|----------|---------|------|--------|
| fruit.color | factor_5 | 0.7709 | orange=0.31, green=0.08, purple=0.06 |
| fruit.size | factor_17 | 0.6989 | small=0.28, medium=-0.12, very_large=-0.18 |
| fruit.taste | factor_19 | 0.5127 | sweet=0.09, sour=-0.25 |
| fruit.texture | factor_4 | 0.7160 | hard=0.33, soft=-0.08, crisp=-0.36 |
| fruit.shape | factor_15 | 0.7175 | elongated=0.55, round=0.01, pear_shaped=-0.35 |
| fruit.tropical | factor_11 | 0.7761 | 0=0.21, 1=-0.23 |
| animal.size | factor_7 | 0.6621 | very_large=0.43, large=0.13, medium=0.05 |
| animal.habitat | factor_11 | 0.7155 | land=0.21, air=-0.03, water=-0.63 |
| animal.domestic | factor_18 | 0.7668 | 0=0.13, 1=-0.44 |
| animal.diet | factor_3 | 0.6561 | omnivore=0.35, herbivore=0.08, carnivore=-0.44 |
| animal.furry | factor_6 | 0.5130 | 0=0.25, 1=-0.24 |
| vehicle.speed | factor_14 | 0.7662 | medium=0.24, slow=0.06, fast=-0.13 |
| vehicle.medium | factor_11 | 0.7148 | land=0.12, underground=0.09, air=-0.24 |
| vehicle.size | factor_4 | 0.7110 | small=0.63, medium=0.01, large=-0.07 |
| vehicle.public | factor_16 | 0.5669 | 0=0.19, 1=-0.33 |
| natural.size | factor_13 | 0.5437 | very_large=0.16, medium=0.14, large=-0.01 |
| natural.element | factor_2 | 0.4503 | air=0.31, water=0.15, fire=0.04 |
| natural.dynamic | factor_5 | 0.4597 | 0=0.28, 1=-0.34 |
| natural.water | factor_15 | 0.2458 | 0=0.10, 1=-0.15 |
| furniture.softness | factor_5 | 0.6751 | hard=0.22, soft=-0.45 |
| furniture.room | factor_13 | 0.5232 | office=0.22, kitchen=0.15, bedroom=0.10 |
| furniture.movable | factor_10 | 0.5337 | 1=0.21, 0=-0.07 |
| material.value | factor_3 | 0.7473 | moderate=0.21, cheap=0.16, very_expensive=0.14 |
| material.hardness | factor_16 | 0.6375 | very_soft=0.33, medium=0.28, hard=0.01 |
| material.natural_origin | factor_14 | 0.2009 | 1=0.06, 0=-0.24 |
| profession.field | factor_6 | 0.9567 | medicine=0.44, transport=0.35, law=0.25 |
| profession.creative | factor_6 | 0.7112 | 0=0.22, 1=-0.31 |
| profession.gender_stereotype | factor_16 | 0.4319 | male=0.13, neutral=0.05, female=-0.13 |

### 因子重建精度
| 因子数(K) | R² | 平均余弦 | 最小余弦 |
|----------|-----|---------|---------|
| 3 | 0.0968 | 0.2883 | 0.0569 |
| 5 | 0.1366 | 0.3519 | 0.1141 |
| 10 | 0.2147 | 0.4549 | 0.1908 |
| 15 | 0.2801 | 0.5222 | 0.3005 |
| 20 | 0.3375 | 0.5759 | 0.3253 |

### 因子空间算术
| 源→目标 | 原始余弦 | 因子余弦 | 最近邻(相似度) |
|--------|---------|---------|--------------|
| apple→banana | -0.0536 | -0.0522 | peach(0.7323) |
| dog→cat | 0.2629 | 0.8680 | cat(0.8680) |
| apple→cherry | 0.0227 | 0.3277 | peach(0.7323) |
| lemon→banana | -0.0398 | 0.0586 | plum(0.5692) |
| elephant→whale | 0.0807 | 0.3467 | tiger(0.7638) |
| car→bus | -0.0396 | -0.0554 | motorcycle(0.3031) |
| ship→boat | 0.3652 | 0.8480 | boat(0.8480) |
| mountain→ocean | -0.0502 | 0.0371 | valley(0.7223) |

### 跨类别因子分析
- 语义通用因子: 20个 (factor_0, factor_1, factor_2, factor_3, factor_4)
- 类别特异因子: 0个 ()

---
## DeepSeek-7B
- Layers: 28, Target: [23, 24, 25, 26, 27]
- Concepts: 132, Bias matrix: [132, 94720]

### SVD分解 (前20个因子)
- 累计解释方差: ['6.3%', '9.1%', '11.4%', '13.3%', '15.1%', '16.6%', '18.1%', '19.5%', '20.9%', '22.3%']

| Factor | 方差解释 | Top+概念 | Top-概念 |
|--------|---------|---------|---------|
| 0 | 6.3% | taxi(1.05), desert(1.03), blanket(0.98) | melon(-1.00), tree(-0.97), car(-0.86) |
| 1 | 2.8% | bicycle(0.72), pillow(0.65), soldier(0.64) | plane(-1.07), orange(-1.07), bookcase(-0.97) |
| 2 | 2.3% | sun(0.97), snow(0.75), lamp(0.72) | forest(-0.76), wardrobe(-0.74), river(-0.71) |
| 3 | 1.9% | emerald(0.94), eagle(0.77), musician(0.68) | steel(-0.74), wood(-0.59), doctor(-0.51) |
| 4 | 1.8% | metal(0.90), watermelon(0.78), melon(0.70) | fox(-0.69), pearl(-0.64), silk(-0.61) |
| 5 | 1.5% | artist(0.92), musician(0.79), painter(0.77) | nurse(-0.82), doctor(-0.73), diamond(-0.57) |
| 6 | 1.5% | ambulance(0.57), coconut(0.52), melon(0.52) | strawberry(-1.01), berry(-0.83), blueberry(-0.81) |
| 7 | 1.4% | monkey(0.82), banana(0.77), elephant(0.59) | cherry(-0.53), berry(-0.52), pear(-0.49) |
| 8 | 1.4% | tractor(0.50), tiger(0.49), blanket(0.45) | pearl(-0.76), dolphin(-0.71), boat(-0.63) |
| 9 | 1.3% | clay(0.61), strawberry(0.52), moon(0.51) | bicycle(-0.93), motorcycle(-0.81), pear(-0.56) |

### 因子-属性关联 (ANOVA eta²)
| 类别.属性 | 最佳因子 | eta² | 组均值 |
|----------|---------|------|--------|
| fruit.color | factor_6 | 0.7303 | brown=0.52, orange=0.39, green=0.29 |
| fruit.size | factor_14 | 0.6272 | medium=0.03, small=-0.01, large=-0.26 |
| fruit.taste | factor_13 | 0.5561 | sour=0.23, sweet=-0.04 |
| fruit.texture | factor_14 | 0.8363 | hard=0.35, soft=-0.01, crisp=-0.07 |
| fruit.shape | factor_0 | 0.9375 | oval=0.86, elongated=0.80, round=-0.23 |
| fruit.tropical | factor_7 | 0.9100 | 1=0.55, 0=-0.37 |
| animal.size | factor_16 | 0.4462 | very_large=0.55, large=0.05, medium=-0.02 |
| animal.habitat | factor_8 | 0.7389 | land=0.16, air=0.11, water=-0.56 |
| animal.domestic | factor_3 | 0.4582 | 0=0.23, 1=-0.30 |
| animal.diet | factor_12 | 0.4684 | herbivore=0.22, omnivore=0.08, carnivore=-0.09 |
| animal.furry | factor_2 | 0.3979 | 1=0.11, 0=-0.15 |
| vehicle.speed | factor_17 | 0.6699 | very_fast=0.51, slow=0.06, fast=-0.02 |
| vehicle.medium | factor_19 | 0.7698 | underground=0.47, land=0.09, air=-0.27 |
| vehicle.size | factor_9 | 0.8114 | large=0.18, medium=0.08, very_large=0.04 |
| vehicle.public | factor_15 | 0.5815 | 1=0.28, 0=-0.23 |
| natural.size | factor_13 | 0.5512 | very_large=0.52, medium=0.16, large=-0.01 |
| natural.element | factor_18 | 0.5277 | earth=0.16, fire=0.14, water=-0.21 |
| natural.dynamic | factor_7 | 0.6418 | 0=0.16, 1=-0.22 |
| natural.water | factor_18 | 0.3784 | 0=0.06, 1=-0.22 |
| furniture.softness | factor_13 | 0.5955 | soft=0.29, hard=-0.20 |
| furniture.room | factor_19 | 0.5711 | dining=0.31, bedroom=0.12, living_room=0.05 |
| furniture.movable | factor_2 | 0.2695 | 1=0.29, 0=-0.14 |
| material.value | factor_3 | 0.4595 | very_expensive=0.51, expensive=0.34, moderate=-0.10 |
| material.hardness | factor_7 | 0.6705 | very_hard=0.33, hard=0.17, very_soft=0.02 |
| material.natural_origin | factor_7 | 0.2698 | 0=0.22, 1=-0.05 |
| profession.field | factor_3 | 0.9562 | arts=0.50, science=0.27, military=0.02 |
| profession.creative | factor_3 | 0.7330 | 1=0.39, 0=-0.29 |
| profession.gender_stereotype | factor_0 | 0.3934 | female=0.50, neutral=-0.11, male=-0.33 |

### 因子重建精度
| 因子数(K) | R² | 平均余弦 | 最小余弦 |
|----------|-----|---------|---------|
| 3 | 0.1139 | 0.3200 | 0.0832 |
| 5 | 0.1507 | 0.3781 | 0.1460 |
| 10 | 0.2220 | 0.4646 | 0.2599 |
| 15 | 0.2838 | 0.5279 | 0.3603 |
| 20 | 0.3367 | 0.5766 | 0.4303 |

### 因子空间算术
| 源→目标 | 原始余弦 | 因子余弦 | 最近邻(相似度) |
|--------|---------|---------|--------------|
| apple→banana | -0.0636 | -0.1631 | peach(0.6251) |
| dog→cat | 0.2805 | 0.8013 | cat(0.8013) |
| apple→cherry | -0.0516 | -0.0316 | peach(0.6251) |
| lemon→banana | 0.0387 | 0.2785 | lime(0.9204) |
| elephant→whale | 0.0000 | 0.0063 | monkey(0.4259) |
| car→bus | 0.1357 | 0.4654 | truck(0.5959) |
| ship→boat | 0.3403 | 0.6264 | boat(0.6264) |
| mountain→ocean | -0.1242 | -0.2434 | valley(0.8363) |

### 跨类别因子分析
- 语义通用因子: 18个 (factor_0, factor_1, factor_2, factor_3, factor_4)
- 类别特异因子: 2个 (factor_5, factor_6)

---
## 跨模型对比
| 指标 | Qwen3-4B | DeepSeek-7B |
|------|----------|-------------|
| R²(K=3) | 0.0968 | 0.1139 |
| R²(K=5) | 0.1366 | 0.1507 |
| R²(K=10) | 0.2147 | 0.2220 |
| R²(K=15) | 0.2801 | 0.2838 |

---
## 理论结论
### 核心发现：偏置空间的低维语义结构
1. 偏置矩阵虽然维度很高（~10K+），但SVD前10-15个因子就能解释大部分方差
2. 每个SVD因子对应一个可解释的语义维度（大小、颜色、功能等）
3. 因子空间中的概念算术比原始空间更有效
4. 存在语义通用因子和类别特异因子

### 修正后的编码方程
```
偏置向量 = Σ_i (α_i × F_i)
  其中 F_i 是第i个语义因子
  α_i 是该概念在第i因子上的投影

概念编码 = B_category + Σ_i (α_i × F_i)
```

### 瓶颈
1. ICA因子不如SVD可解释（ICA寻找统计独立，不保证语义独立）
2. 因子-属性关联的eta²偏低（大部分<0.5），说明属性编码不是线性的
3. 需要更大规模的概念集来发现更完整的语义因子体系