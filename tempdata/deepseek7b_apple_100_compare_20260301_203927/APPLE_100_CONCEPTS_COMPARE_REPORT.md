# Apple vs 100 Concepts 编码结构比较报告

## 1) Apple 签名结构
- top-k 签名神经元: 120
- 层分布: {0: 2, 1: 3, 2: 1, 3: 13, 4: 2, 5: 9, 6: 3, 7: 6, 9: 1, 10: 1, 14: 1, 15: 1, 17: 1, 19: 2, 20: 1, 21: 2, 22: 3, 23: 24, 24: 20, 25: 14, 26: 7, 27: 3}

## 2) 与 Apple 最相似概念 (Top-10)
- mango (fruit): layer-jaccard=0.5171, cosine=0.9913, layer-shared=1199
- orange (fruit): layer-jaccard=0.5096, cosine=0.9907, layer-shared=1180
- peach (fruit): layer-jaccard=0.4972, cosine=0.9899, layer-shared=1160
- watermelon (fruit): layer-jaccard=0.4962, cosine=0.9912, layer-shared=1157
- lemon (fruit): layer-jaccard=0.4960, cosine=0.9891, layer-shared=1156
- strawberry (fruit): layer-jaccard=0.4921, cosine=0.9901, layer-shared=1150
- flower (nature): layer-jaccard=0.4747, cosine=0.9888, layer-shared=1122
- horse (animal): layer-jaccard=0.4477, cosine=0.9873, layer-shared=1079
- pizza (food): layer-jaccard=0.4475, cosine=0.9883, layer-shared=1063
- rice (food): layer-jaccard=0.4471, cosine=0.9865, layer-shared=1067

## 3) 与 Apple 最不相似概念 (Bottom-10 by layer-jaccard)
- hate (abstract): layer-jaccard=0.2831, cosine=0.9676, layer-shared=745
- peace (abstract): layer-jaccard=0.2900, cosine=0.9685, layer-shared=749
- banana (fruit): layer-jaccard=0.2947, cosine=0.9651, layer-shared=759
- grape (fruit): layer-jaccard=0.2947, cosine=0.9651, layer-shared=759
- pear (fruit): layer-jaccard=0.2947, cosine=0.9651, layer-shared=759
- rabbit (animal): layer-jaccard=0.2947, cosine=0.9651, layer-shared=759
- tiger (animal): layer-jaccard=0.2947, cosine=0.9651, layer-shared=759
- lion (animal): layer-jaccard=0.2947, cosine=0.9651, layer-shared=759
- elephant (animal): layer-jaccard=0.2947, cosine=0.9651, layer-shared=759
- monkey (animal): layer-jaccard=0.2947, cosine=0.9651, layer-shared=759

## 4) 重点对比
- apple vs banana: layer-jaccard=0.2947, layer-shared=759, strict-jaccard=0.0000
- apple vs rabbit: layer-jaccard=0.2947, layer-shared=759, strict-jaccard=0.0000
- apple vs sun: layer-jaccard=0.3525, layer-shared=880, strict-jaccard=0.0000

## 5) 因果消融 (apple signature top-40)
| group | baseline | ablated | delta |
|---|---:|---:|---:|
| apple_attr | 0.00030165 | 0.00026477 | -0.00003688 |
| banana_attr | 0.00004167 | 0.00004181 | +0.00000014 |
| rabbit_attr | 0.00015002 | 0.00015014 | +0.00000012 |
| sun_attr | 0.08352250 | 0.08488836 | +0.00136586 |