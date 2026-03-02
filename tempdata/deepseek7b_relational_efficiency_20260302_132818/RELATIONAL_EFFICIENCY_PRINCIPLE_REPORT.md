# 关系网络高效编码原理测试报告 (REPT)

## 1) 图几何对齐
- Spearman(graph_dist, neural_dist): 0.460260

## 2) 压缩后对齐
- k=4: spearman=0.293266
- k=8: spearman=0.402870
- k=16: spearman=0.394255
- k=24: spearman=0.377996
- k=32: spearman=0.375289
- k=48: spearman=0.376584
- sample effective_rank: 33.540653
- sample k95: 51

## 3) 稀疏基效率
- k80 bridge mass: 336926
- k90 bridge mass: 409441
- eta_rel_mean: 0.33039236
- eta_cat_mean: 0.18376639

## 4) 因果桥接路由
- mean routing gain: 0.00000527
- std routing gain: 0.00001399
- top bridge neurons:
  - L2N10110: gain=0.00005126
  - L4N1844: gain=0.00000471
  - L9N11960: gain=0.00000398
  - L24N11239: gain=0.00000362
  - L21N10661: gain=0.00000175
  - L26N10814: gain=0.00000031
  - L0N7371: gain=0.00000000
  - L4N6796: gain=-0.00000011
  - L24N15939: gain=-0.00000024
  - L25N14354: gain=-0.00000046
  - L7N8230: gain=-0.00000069
  - L24N13233: gain=-0.00000088