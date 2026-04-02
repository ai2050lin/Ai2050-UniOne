# Stage456: 大规模概念验证 + 非线性语义因子分解

**时间**: 2026-04-01 01:50
**目标**: 200+词验证 + Autoencoder非线性分解 + 因子稳定性分析
**模型**: DeepSeek-7B (28层)
**概念数**: 220
**类别数**: 10

---

## 1. SVD分解（线性基线）
- Bias matrix shape: [220, 94720]

### 累计解释方差
| K | 累计方差 |
|---|---------|
| 5 | 13.1% |
| 10 | 18.8% |
| 15 | 23.4% |
| 20 | 27.4% |
| 25 | 31.0% |
| 30 | 34.3% |

### SVD因子语义（前15个）
| Factor | 方差 | Top+ | Top- |
|--------|------|------|------|
| factor_0 | 5.6% | desert(0.98), taxi(0.97), blanket(0.95) | tree(-0.99), melon(-0.97), frog(-0.94) |
| factor_1 | 2.3% | motorcycle(0.77), jeans(0.74), bicycle(0.72) | plane(-0.93), uniform(-0.80), orange(-0.79) |
| factor_2 | 2.1% | sun(0.74), snow(0.65), shirt(0.64) | sandals(-0.93), bookcase(-0.88), boots(-0.83) |
| factor_3 | 1.6% | apple(0.57), steel(0.55), rope(0.50) | apricot(-0.82), emerald(-0.79), amber(-0.75) |
| factor_4 | 1.4% | boots(0.95), sandals(0.84), gloves(0.82) | emerald(-0.63), hat(-0.63), diamond(-0.61) |
| factor_5 | 1.3% | metal(0.80), steel(0.63), driver(0.61) | wool(-0.71), gown(-0.66), silk(-0.64) |
| factor_6 | 1.2% | dress(0.76), cake(0.70), pizza(0.48) | beef(-0.85), pork(-0.74), blueberry(-0.60) |
| factor_7 | 1.1% | parrot(0.69), nurse(0.57), guava(0.54) | berry(-0.67), shoes(-0.58), strawberry(-0.58) |
| factor_8 | 1.1% | fish(0.63), bicycle(0.53), leather(0.51) | curtain(-0.60), ambulance(-0.52), glacier(-0.48) |
| factor_9 | 1.0% | scooter(0.44), monkey(0.42), ceramic(0.42) | pearl(-0.78), dress(-0.57), peach(-0.55) |
| factor_10 | 1.0% | shoes(0.65), socks(0.62), shorts(0.59) | coat(-0.62), jacket(-0.57), gown(-0.52) |
| factor_11 | 1.0% | scientist(0.65), wool(0.60), cloth(0.58) | motorcycle(-0.57), dancer(-0.54), iron(-0.53) |
| factor_12 | 0.9% | glue(0.49), pork(0.49), strawberry(0.45) | cave(-0.61), bench(-0.56), bicycle(-0.52) |
| factor_13 | 0.9% | cookie(0.46), apple(0.44), blanket(0.43) | dolphin(-0.75), ceramic(-0.59), dancer(-0.59) |
| factor_14 | 0.9% | watermelon(0.58), farmer(0.54), boat(0.53) | cherry(-0.50), blueberry(-0.49), berry(-0.46) |

## 2. Autoencoder非线性分解
| 潜在维度 | AE方差解释 | SVD基线 | AE提升 |
|---------|-----------|---------|--------|
| 10 | 9.9% | 15.0% | -5.1% |
| 20 | 10.2% | 22.7% | -12.4% |
| 30 | 10.9% | 29.3% | -18.4% |

### AE因子语义（latent_20, 前10个）
| Factor | Top+ | Top- |
|--------|------|------|
| ae_factor_0 | leather(2.59), carpet(2.57), apricot(2.17) | ice_cream(-3.79), eagle(-2.51), doctor(-2.39) |
| ae_factor_1 | eagle(2.42), owl(2.34), apricot(2.17) | bear(-2.09), sweater(-2.03), mango(-1.94) |
| ae_factor_2 | concrete(2.77), crocodile(2.74), storm(2.55) | apricot(-2.66), cart(-2.15), helicopter(-1.51) |
| ae_factor_3 | cloth(2.61), clay(1.98), rice(1.93) | desert(-3.24), plum(-2.47), cloud(-2.35) |
| ae_factor_4 | sofa(3.60), shark(2.23), doctor(2.23) | ice_cream(-4.16), helicopter(-2.74), eagle(-2.42) |
| ae_factor_5 | plum(3.17), doctor(2.93), owl(2.31) | fig(-2.55), pasta(-2.44), lemon(-1.82) |
| ae_factor_6 | architect(2.87), bench(2.34), owl(2.29) | storm(-2.72), painter(-2.29), pasta(-1.90) |
| ae_factor_7 | desk(2.00), doctor(1.75), desert(1.57) | wood(-3.02), taxi(-2.73), cloth(-2.42) |
| ae_factor_8 | eagle(2.11), carpet(2.05), dolphin(1.96) | owl(-2.83), banana(-2.64), elephant(-2.61) |
| ae_factor_9 | snake(2.33), bear(2.09), mango(2.03) | fox(-1.86), carpet(-1.83), doctor(-1.67) |

## 3. 因子-属性关联 (ANOVA eta²)

### SVD因子
| 属性 | 最佳因子 | eta² |
|------|---------|------|
| profession.field | factor_18 | 0.901 |
| clothing.warmth | factor_0 | 0.830 |
| clothing.body_part | factor_4 | 0.762 |
| profession.creative | factor_6 | 0.741 |
| furniture.softness | factor_16 | 0.734 |
| vehicle.medium | factor_18 | 0.716 |
| furniture.room | factor_9 | 0.666 |
| fruit.size | factor_14 | 0.652 |
| fruit.color | factor_7 | 0.619 |
| food.health | factor_15 | 0.599 |
| vehicle.size | factor_9 | 0.573 |
| material.value | factor_4 | 0.552 |
| tool.danger | factor_16 | 0.546 |
| animal.habitat | factor_2 | 0.545 |
| material.hardness | factor_14 | 0.534 |
| vehicle.speed | factor_3 | 0.500 |
| natural.size | factor_1 | 0.485 |
| tool.complexity | factor_10 | 0.451 |
| food.sweetness | factor_4 | 0.443 |
| natural.element | factor_4 | 0.415 |

### Autoencoder因子
| 属性 | 最佳因子 | eta² |
|------|---------|------|
| profession.field | ae_factor_17 | 0.829 |
| food.temperature | ae_factor_0 | 0.771 |
| food.sweetness | ae_factor_0 | 0.693 |
| clothing.body_part | ae_factor_18 | 0.675 |
| furniture.softness | ae_factor_2 | 0.648 |
| clothing.warmth | ae_factor_7 | 0.580 |
| furniture.room | ae_factor_2 | 0.552 |
| fruit.color | ae_factor_9 | 0.543 |
| tool.complexity | ae_factor_4 | 0.513 |
| vehicle.medium | ae_factor_18 | 0.492 |
| food.health | ae_factor_18 | 0.483 |
| animal.size | ae_factor_4 | 0.459 |
| vehicle.size | ae_factor_14 | 0.441 |
| natural.element | ae_factor_8 | 0.398 |
| fruit.size | ae_factor_4 | 0.376 |
| vehicle.speed | ae_factor_9 | 0.373 |
| profession.prestige | ae_factor_6 | 0.365 |
| clothing.formality | ae_factor_6 | 0.350 |
| profession.creative | ae_factor_19 | 0.349 |
| material.value | ae_factor_0 | 0.344 |

## 4. 概念算术测试

| 空间 | 同类平均 | 跨类平均 |
|------|---------|---------|
| 原始空间 | 0.000 | 0.114 |
| SVD空间 | 0.395 | - |
| AE空间 | 0.045 | - |

### 同类算术详情
| 对 | 原始 | SVD | AE |
|----|------|-----|----|
| dog→cat | 0.278 | 0.745 | 0.455 |
| car→bus | 0.165 | 0.474 | 0.150 |
| apple→banana | -0.019 | 0.051 | 0.152 |
| mountain→valley | 0.175 | 0.741 | -0.386 |
| chair→table | 0.001 | 0.081 | -0.055 |
| doctor→nurse | 0.202 | 0.839 | 0.201 |
| bread→cake | 0.114 | 0.428 | -0.140 |
| shirt→dress | 0.192 | 0.680 | -0.027 |

## 5. 结论
