# Stage459: 超大规模概念验证 + 本征维度估算

**时间**: 2026-04-01 02:20
**模型**: DeepSeek-7B (28层)
**概念数**: 447
**类别数**: 15

---

## 1. 本征维度估算

### PCA方差解释 vs 维度
| 目标方差 | 所需维度 |
|---------|---------|
| 50% | 100 |
| 60% | 144 |
| 70% | 200 |

- Participation Ratio: **80.9** (有效维度数)
- TwoNN本征维度: **-0.8**
- MLE本征维度: k=5=37.0, k=10=35.3, k=15=35.2

## 2. SVD方差解释 vs 概念数（Scaling Law）

| 概念数 | Top-5 | Top-10 | Top-20 |
|--------|-------|--------|--------|
| 50 | 20.9% | 33.4% | 53.8% |
| 100 | 15.8% | 24.0% | 36.7% |
| 150 | 13.8% | 20.7% | 30.9% |
| 200 | 12.6% | 18.5% | 27.5% |
| 250 | 12.5% | 17.9% | 25.9% |
| 300 | 12.2% | 17.2% | 24.6% |
| 350 | 12.0% | 16.8% | 23.7% |
| 400 | 11.6% | 16.2% | 22.8% |
| 447 | 11.6% | 16.0% | 22.3% |
| 447 | 11.6% | 16.0% | 22.3% |

## 3. 完整SVD分解（500+词）

| K | 累计方差 |
|---|---------|
| 5 | 11.6% |
| 10 | 16.0% |
| 20 | 22.4% |
| 30 | 27.4% |
| 50 | 35.2% |

### 因子语义（前20个）
| Factor | 方差 | Top+ | Top- |
|--------|------|------|------|
| factor_0 | 5.3% | olive(0.98), desert(0.97), velvet(0.97) | red(-1.08), tree(-0.98), yellow(-0.98) |
| factor_1 | 2.0% | shirt(0.75), watermelon(0.68), pineapple(0.66) | file(-1.00), bookcase(-0.94), plane(-0.87) |
| factor_2 | 1.7% | hat(0.77), tie(0.70), pen(0.68) | pliers(-0.70), bookcase(-0.70), sandals(-0.69) |
| factor_3 | 1.3% | blazer(0.80), apricot(0.68), amber(0.67) | river(-0.68), fish(-0.67), orange(-0.67) |
| factor_4 | 1.2% | javelin(0.63), shame(0.60), dew(0.59) | sunny(-0.73), uniform(-0.67), cloudy(-0.64) |
| factor_5 | 1.1% | loneliness(0.72), drought(0.71), shoes(0.69) | pride(-0.59), cloudy(-0.58), rugby(-0.54) |
| factor_6 | 0.9% | brass(0.76), pork(0.66), copper(0.66) | cloth(-0.65), dress(-0.55), cake(-0.48) |
| factor_7 | 0.9% | ferry(0.57), cruiser(0.48), mirror(0.47) | bicycle(-0.75), motorcycle(-0.72), gold(-0.53) |
| factor_8 | 0.8% | boots(0.83), shoes(0.80), jeans(0.77) | apron(-0.60), coral(-0.53), hurricane(-0.52) |
| factor_9 | 0.8% | navy(0.62), boat(0.60), fish(0.60) | orange(-0.68), yellow(-0.56), bronze(-0.52) |
| factor_10 | 0.7% | strawberry(0.66), lavender(0.64), dragonfruit(0.52) | plum(-0.57), orange(-0.54), cave(-0.52) |
| factor_11 | 0.7% | diving(0.57), beige(0.56), warm(0.55) | axe(-0.54), football(-0.52), boots(-0.50) |
| factor_12 | 0.7% | bronze(0.65), silver(0.54), beef(0.49) | empathy(-0.46), envy(-0.43), waffle(-0.43) |
| factor_13 | 0.7% | lime(0.63), javelin(0.53), painter(0.51) | amber(-0.54), ambulance(-0.50), key(-0.48) |
| factor_14 | 0.6% | soccer(0.67), football(0.54), rugby(0.49) | javelin(-0.44), wood(-0.43), muffin(-0.43) |
| factor_15 | 0.6% | skin(0.52), rhino(0.49), socks(0.49) | brass(-0.55), wrist(-0.49), blackberry(-0.47) |
| factor_16 | 0.6% | silver(0.53), surfing(0.50), gold(0.47) | basketball(-0.51), soccer(-0.48), mist(-0.46) |
| factor_17 | 0.6% | shorts(0.50), beef(0.47), doctor(0.46) | musician(-0.68), cruiser(-0.51), silver(-0.46) |
| factor_18 | 0.6% | mittens(0.53), slippers(0.50), raspberry(0.50) | ankle(-0.50), island(-0.46), pear(-0.39) |
| factor_19 | 0.6% | panda(0.55), chef(0.41), tornado(0.40) | calm(-0.52), mulberry(-0.50), muffin(-0.48) |

## 4. 结论
