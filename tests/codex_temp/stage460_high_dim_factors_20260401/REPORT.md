# Stage460: 80维高阶SVD因子分析 — 双模型交叉验证

**时间**: 2026-04-01 08:50
**模型**: Qwen3-4B (36层) + DeepSeek-7B (28层)
**概念数**: 405
**SVD维度**: 80

---

## 1. 方差解释里程碑（双模型对比）

| 里程碑 | Qwen3-4B | DeepSeek-7B |
|--------|--------|--------|
| 25% | 34维(25.2%) | 42维(25.3%) |
| 30% | 44维(30.0%) | 54维(30.2%) |
| 40% | 69维(40.2%) | N/A |
| 50% | N/A | N/A |
| 60% | N/A | N/A |
| 70% | N/A | N/A |
| 80% | N/A | N/A |

### 累计方差解释

| 维度 | Qwen3-4B | DeepSeek-7B |
|------|------|------|
| Top-5 | 6.4% | 5.4% |
| Top-10 | 10.4% | 9.0% |
| Top-20 | 17.3% | 14.8% |
| Top-30 | 23.1% | 19.9% |
| Top-50 | 32.7% | 28.7% |
| Top-80 | 44.1% | 39.5% |

## 2. 高阶因子语义发现（Factor 21-80）

### Qwen3-4B: 60/60有跨类别语义

| Factor 20 | 0.64% | plane(112.07), cheetah(107.24), bagel(95.93) | milk(-130.59), leg(-114.78), pomegranate(-66.52) |
| Factor 21 | 0.63% | teacher(93.73), leg(91.92), bagel(70.50) | milk(-213.66), knee(-56.72), tangerine(-55.12) |
| Factor 22 | 0.61% | tangerine(106.02), chicken(87.52), bronze(55.39) | cheetah(-92.89), dew(-87.66), leg(-81.39) |
| Factor 23 | 0.60% | brick(92.92), leg(69.30), concrete(60.58) | bronze(-99.53), cheetah(-83.07), bronze(-82.46) |
| Factor 24 | 0.58% | chicken(121.68), cheetah(106.83), running(55.37) | bagel(-187.70), milk(-82.70), rowing(-78.63) |
| Factor 25 | 0.58% | tangerine(138.26), plane(105.79), plumber(60.61) | file(-77.37), rowing(-71.76), bagel(-60.26) |
| Factor 26 | 0.57% | bagel(73.01), hand(62.23), ear(56.96) | leg(-173.55), tangerine(-70.79), ankle(-68.59) |
| Factor 27 | 0.55% | milk(69.49), boysenberry(49.93), happy(40.71) | rowing(-97.93), bus(-91.06), taxi(-82.55) |
| Factor 28 | 0.55% | leg(71.00), bagel(44.92), rowing(38.65) | happy(-101.86), sad(-88.10), angry(-72.37) |
| Factor 29 | 0.54% | rowing(172.75), plumber(43.37), brick(42.14) | prairie(-82.15), bagel(-71.40), running(-67.22) |
| Factor 30 | 0.53% | dew(84.16), tangerine(81.28), shirt(69.94) | bagel(-120.93), plum(-78.90), milk(-72.30) |
| Factor 31 | 0.52% | prairie(156.12), chicken(100.68), bagel(89.33) | bronze(-62.12), bronze(-59.72), bus(-52.66) |
| Factor 32 | 0.52% | bagel(74.02), farmer(69.39), plumber(65.38) | boysenberry(-72.54), rowing(-71.23), bicycle(-65.66) |
| Factor 33 | 0.51% | dew(88.24), taxi(63.84), plumber(60.55) | running(-58.34), milk(-51.04), prairie(-49.05) |
| Factor 34 | 0.50% | date(59.86), watermelon(58.21), bronze(43.39) | koala(-123.61), plum(-75.51), kangaroo(-55.16) |
| Factor 35 | 0.50% | prairie(108.94), file(54.25), cookie(53.44) | chicken(-63.65), black(-54.83), bagel(-51.48) |
| Factor 36 | 0.49% | prairie(94.75), hat(72.59), wind(53.10) | dew(-110.92), chicken(-68.42), tangerine(-60.65) |
| Factor 37 | 0.49% | prairie(128.27), dew(114.48), plum(109.23) | hat(-63.11), monkey(-62.79), tangerine(-50.43) |
| Factor 38 | 0.48% | chicken(86.84), boysenberry(72.13), taxi(63.77) | nurse(-60.48), plum(-60.37), pharmacist(-58.75) |
| Factor 39 | 0.48% | car(65.45), boysenberry(53.60), ivory(49.25) | bronze(-98.95), bronze(-92.55), shorts(-75.89) |

### DeepSeek-7B: 60/60有跨类别语义

| Factor 20 | 0.54% | doctor(119.01), veterinarian(110.69), chicken(90.37) | cheetah(-110.36), coral(-76.28), firefighter(-73.46) |
| Factor 21 | 0.53% | knee(133.53), cheetah(120.69), ankle(116.98) | taco(-139.02), tuxedo(-121.68), taxi(-94.49) |
| Factor 22 | 0.52% | knee(92.94), ankle(89.89), chicken(86.27) | firefighter(-182.19), butcher(-88.16), police(-80.81) |
| Factor 23 | 0.51% | doctor(93.97), veterinarian(80.22), tuxedo(69.55) | ottoman(-126.06), firefighter(-118.24), fog(-108.08) |
| Factor 24 | 0.51% | taxi(147.80), train(100.28), taco(80.09) | chicken(-95.28), prairie(-86.96), firefighter(-81.94) |
| Factor 25 | 0.50% | ear(84.80), eye(81.64), concrete(69.80) | butcher(-155.63), chicken(-103.29), tangerine(-62.25) |
| Factor 26 | 0.50% | cheetah(158.59), volcano(100.90), tornado(71.13) | firefighter(-136.04), tangerine(-97.29), desert(-96.96) |
| Factor 27 | 0.49% | tuxedo(126.32), prairie(123.33), happy(93.83) | chicken(-90.37), mango(-81.89), van(-69.25) |
| Factor 28 | 0.48% | fig(108.42), olive(95.44), ear(86.12) | tuxedo(-123.38), fog(-111.96), gray(-82.08) |
| Factor 29 | 0.48% | cabinet(88.89), horse(83.79), desert(76.72) | eye(-126.91), ear(-109.03), knee(-103.39) |
| Factor 30 | 0.48% | pasta(121.17), rice(102.61), prairie(87.12) | desert(-125.86), lightning(-103.18), cookie(-92.93) |
| Factor 31 | 0.47% | tuxedo(170.70), cheetah(102.85), chicken(102.56) | ottoman(-159.71), lemon(-82.55), taco(-73.44) |
| Factor 32 | 0.47% | compass(84.46), sweater(83.30), submarine(78.65) | tuxedo(-85.82), ottoman(-84.82), chicken(-72.35) |
| Factor 33 | 0.46% | butcher(140.23), calm(123.74), desert(78.40) | tuxedo(-108.76), lemon(-104.57), lightning(-82.81) |
| Factor 34 | 0.46% | gray(83.58), paper(76.29), firefighter(70.72) | storm(-148.70), canoe(-84.37), fog(-70.30) |
| Factor 35 | 0.45% | prairie(161.70), bear(105.25), cabinet(90.73) | ottoman(-119.38), cheetah(-116.85), calm(-101.59) |
| Factor 36 | 0.45% | tuxedo(110.79), firefighter(92.47), calm(91.16) | taxi(-139.57), ottoman(-98.27), prairie(-64.46) |
| Factor 37 | 0.45% | fog(140.52), cheetah(110.90), calm(77.57) | concrete(-127.20), ottoman(-102.87), butcher(-91.42) |
| Factor 38 | 0.44% | ottoman(113.01), plateau(96.75), horse(87.34) | lemon(-96.98), taco(-86.80), chicken(-72.82) |
| Factor 39 | 0.44% | calm(116.69), chicken(98.73), rabbit(95.09) | desert(-85.35), ottoman(-81.42), mango(-77.15) |

## 3. 因子-类别关联（eta² > 0.3）

### Qwen3-4B

| Factor | eta² | Top类别 | Bottom类别 |
|--------|-----|---------|------------|

### DeepSeek-7B

| Factor | eta² | Top类别 | Bottom类别 |
|--------|-----|---------|------------|

## 4. 重建精度

| 维度 | Qwen3-4B | DeepSeek-7B |
|------|------|------|
| 10 | var=10.3%, cos=0.271 | var=8.9%, cos=0.274 |
| 20 | var=17.1%, cos=0.343 | var=14.6%, cos=0.350 |
| 30 | var=22.9%, cos=0.396 | var=19.6%, cos=0.400 |
| 50 | var=32.3%, cos=0.483 | var=28.3%, cos=0.479 |
| 80 | var=44.1%, cos=0.576 | var=39.5%, cos=0.557 |

## 5. 类内分散度（概念多样性指标）

| 类别 | Qwen3-4B | DeepSeek-7B |
|------|------|------|
| animal | avg_std=16.268 | avg_std=21.369 |
| body | avg_std=16.389 | avg_std=24.032 |
| clothing | avg_std=17.820 | avg_std=19.705 |
| color | avg_std=14.368 | avg_std=20.820 |
| emotion | avg_std=9.251 | avg_std=24.138 |
| food | avg_std=18.738 | avg_std=23.753 |
| fruit | avg_std=18.712 | avg_std=23.646 |
| furniture | avg_std=11.639 | avg_std=20.122 |
| material | avg_std=15.652 | avg_std=20.158 |
| natural | avg_std=11.768 | avg_std=24.227 |
| profession | avg_std=16.695 | avg_std=17.921 |
| sport | avg_std=18.304 | avg_std=21.795 |
| tool | avg_std=12.558 | avg_std=18.133 |
| vehicle | avg_std=17.773 | avg_std=22.680 |
| weather | avg_std=17.659 | avg_std=28.270 |

## 7. 核心结论

### Qwen3-4B
- 80维SVD解释方差: 44.1%
- 压缩比: 730:1 (58368→80维)
- 50%方差所需维度: N/A
- 高阶有意义因子: 60/60

### DeepSeek-7B
- 80维SVD解释方差: 39.5%
- 压缩比: 1421:1 (113664→80维)
- 50%方差所需维度: N/A
- 高阶有意义因子: 60/60
