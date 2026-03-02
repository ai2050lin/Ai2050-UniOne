# Apple vs 100 Concepts 编码结构比较报告

## 1) Apple 签名结构
- top-k 签名神经元: 120
- 层分布: {0: 2, 1: 3, 2: 1, 3: 13, 4: 2, 5: 9, 6: 3, 7: 6, 9: 1, 10: 1, 14: 1, 15: 1, 17: 1, 19: 2, 20: 1, 21: 2, 22: 3, 23: 24, 24: 20, 25: 14, 26: 7, 27: 3}

## 2) 与 Apple 最相似概念 (Top-10)
- mango (fruit): jaccard=0.0000, cosine=0.9913, shared=0
- watermelon (fruit): jaccard=0.0000, cosine=0.9912, shared=0
- orange (fruit): jaccard=0.0000, cosine=0.9907, shared=0
- strawberry (fruit): jaccard=0.0000, cosine=0.9901, shared=0
- peach (fruit): jaccard=0.0000, cosine=0.9899, shared=0
- lemon (fruit): jaccard=0.0000, cosine=0.9891, shared=0
- flower (nature): jaccard=0.0000, cosine=0.9888, shared=0
- pizza (food): jaccard=0.0000, cosine=0.9883, shared=0
- cake (food): jaccard=0.0000, cosine=0.9878, shared=0
- horse (animal): jaccard=0.0000, cosine=0.9873, shared=0

## 3) 与 Apple 最不相似概念 (Bottom-10 by Jaccard)
- mango (fruit): jaccard=0.0000, cosine=0.9913, shared=0
- watermelon (fruit): jaccard=0.0000, cosine=0.9912, shared=0
- orange (fruit): jaccard=0.0000, cosine=0.9907, shared=0
- strawberry (fruit): jaccard=0.0000, cosine=0.9901, shared=0
- peach (fruit): jaccard=0.0000, cosine=0.9899, shared=0
- lemon (fruit): jaccard=0.0000, cosine=0.9891, shared=0
- flower (nature): jaccard=0.0000, cosine=0.9888, shared=0
- pizza (food): jaccard=0.0000, cosine=0.9883, shared=0
- cake (food): jaccard=0.0000, cosine=0.9878, shared=0
- horse (animal): jaccard=0.0000, cosine=0.9873, shared=0

## 4) 重点对比
- apple vs banana: jaccard=0.0000, shared=0
- apple vs rabbit: jaccard=0.0000, shared=0
- apple vs sun: jaccard=0.0000, shared=0

## 5) 因果消融 (apple signature top-40)
| group | baseline | ablated | delta |
|---|---:|---:|---:|
| apple_attr | 0.00030165 | 0.00026477 | -0.00003688 |
| banana_attr | 0.00004167 | 0.00004181 | +0.00000014 |
| rabbit_attr | 0.00015002 | 0.00015014 | +0.00000012 |
| sun_attr | 0.08352250 | 0.08488836 | +0.00136586 |