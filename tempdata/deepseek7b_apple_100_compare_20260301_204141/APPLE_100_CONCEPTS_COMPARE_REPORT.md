# Apple vs 100 Concepts 编码结构比较报告

## 1) Apple 签名结构
- top-k 签名神经元: 120
- 层分布: {0: 5, 1: 11, 2: 4, 3: 3, 4: 4, 5: 7, 6: 8, 7: 2, 8: 1, 9: 2, 11: 2, 12: 1, 14: 3, 16: 7, 17: 2, 18: 4, 19: 3, 20: 4, 21: 6, 22: 6, 23: 12, 24: 14, 25: 5, 26: 2, 27: 2}

## 2) 与 Apple 最相似概念 (Top-10)
- orange (fruit): layer-jaccard=0.3946, cosine=0.9748, layer-shared=988
- tree (nature): layer-jaccard=0.3458, cosine=0.9722, layer-shared=894
- flower (nature): layer-jaccard=0.3451, cosine=0.9714, layer-shared=888
- peach (fruit): layer-jaccard=0.3411, cosine=0.9697, layer-shared=883
- leaf (nature): layer-jaccard=0.3375, cosine=0.9707, layer-shared=871
- fish (animal): layer-jaccard=0.3347, cosine=0.9699, layer-shared=870
- cake (food): layer-jaccard=0.3341, cosine=0.9703, layer-shared=866
- window (object): layer-jaccard=0.3327, cosine=0.9695, layer-shared=866
- cheese (food): layer-jaccard=0.3309, cosine=0.9677, layer-shared=863
- bird (animal): layer-jaccard=0.3304, cosine=0.9670, layer-shared=865

## 3) 与 Apple 最不相似概念 (Bottom-10 by layer-jaccard)
- banana (fruit): layer-jaccard=0.1804, cosine=0.9492, layer-shared=510
- grape (fruit): layer-jaccard=0.1804, cosine=0.9492, layer-shared=510
- pear (fruit): layer-jaccard=0.1804, cosine=0.9492, layer-shared=510
- rabbit (animal): layer-jaccard=0.1804, cosine=0.9492, layer-shared=510
- tiger (animal): layer-jaccard=0.1804, cosine=0.9492, layer-shared=510
- lion (animal): layer-jaccard=0.1804, cosine=0.9492, layer-shared=510
- elephant (animal): layer-jaccard=0.1804, cosine=0.9492, layer-shared=510
- monkey (animal): layer-jaccard=0.1804, cosine=0.9492, layer-shared=510
- airplane (vehicle): layer-jaccard=0.1804, cosine=0.9492, layer-shared=510
- subway (vehicle): layer-jaccard=0.1804, cosine=0.9492, layer-shared=510

## 4) 重点对比
- apple vs banana: layer-jaccard=0.1804, layer-shared=510, strict-jaccard=0.0000
- apple vs rabbit: layer-jaccard=0.1804, layer-shared=510, strict-jaccard=0.0000
- apple vs sun: layer-jaccard=0.3107, layer-shared=820, strict-jaccard=0.0000

## 5) 因果消融 (apple signature top-40)
| group | baseline | ablated | delta |
|---|---:|---:|---:|
| apple_attr | 0.00030165 | 0.00010136 | -0.00020030 |
| banana_attr | 0.00004167 | 0.00004223 | +0.00000055 |
| rabbit_attr | 0.00015002 | 0.00015150 | +0.00000148 |
| sun_attr | 0.08352250 | 0.08469212 | +0.00116962 |