# Stage130: Noun 多句法上下文探针

## 核心结果
- 句法簇数量: 6
- 早层主导簇数: 6
- L1 主导簇数: 3
- 句法稳定率: 1.0000
- 句法均值得分: 0.5506
- 反复出现的早层神经元平均命中数: 2.6667
- 多句法名词上下文分数: 0.9265

## 各句法簇
- subject_copula: dominant=L1, score=0.5603
- object_transitive: dominant=L0, score=0.5417
- preposition_about: dominant=L0, score=0.5383
- relative_clause: dominant=L1, score=0.5603
- possessive_frame: dominant=L1, score=0.5631
- evaluation_frame: dominant=L0, score=0.5399

## 反复出现的早层神经元
- L2 N385: hits=6, rule=0.5898, group=micro_material
- L0 N1344: hits=3, rule=0.6371, group=meso_fruit
- L1 N2693: hits=3, rule=0.6151, group=meso_fruit
- L1 N541: hits=3, rule=0.6368, group=meso_food
- L2 N2380: hits=3, rule=0.7145, group=micro_material
- L1 N2519: hits=3, rule=0.6707, group=meso_fruit
- L1 N2256: hits=3, rule=0.6333, group=micro_temperature
- L0 N144: hits=3, rule=0.6044, group=meso_animal
- L1 N759: hits=2, rule=0.5734, group=meso_animal
- L1 N2397: hits=1, rule=0.5860, group=meso_animal
- L1 N2304: hits=1, rule=0.5768, group=micro_temperature
- L2 N2224: hits=1, rule=0.5930, group=meso_human

## 理论提示
- 如果多个句法簇都把名词主导层压到早层，说明名词进入句子后会先做快速定锚，再由更深层做后续聚合。
- 如果早层神经元跨句法反复出现，说明这里更接近稳定编码规则，而不是模板偶然性。
