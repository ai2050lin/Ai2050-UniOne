# 统一编码结构解码报告（离线汇总）

## 1. 数据来源
- 多维探针文件数: 8
- 因果消融文件数: 9
- 名词扫描文件数(过滤后): 23

## 2. 维度轴稳定性
- style: set_jaccard=0.1952±0.2532, profile_cos=0.9670±0.0358
- logic: set_jaccard=0.6100±0.2304, profile_cos=0.9987±0.0012
- syntax: set_jaccard=0.2519±0.2759, profile_cos=0.9530±0.0540

## 3. 维度轴因果可分离性（对角优势）
- style: mean=0.0157, std=0.0146, positive_ratio=77.78%
- logic: mean=0.0366, std=0.0088, positive_ratio=100.00%
- syntax: mean=0.0041, std=0.0059, positive_ratio=44.44%

## 4. 概念层级共享-特异结构
### 4.1 概念对重叠
- apple__banana: Jaccard=0.0000±0.0000, LayerCos=0.2545±0.0365
- apple__cat: Jaccard=0.0000±0.0000, LayerCos=0.6803±0.0444
- apple__king: Jaccard=0.0000±0.0000, LayerCos=0.7606±0.0000
- king__queen: Jaccard=0.0959±0.0000, LayerCos=0.8242±0.0000
- cat__dog: Jaccard=0.0548±0.0204, LayerCos=0.7916±0.0448
### 4.2 概念-类别共享比例
- apple__fruit: shared=0.0475, specific=0.9525, LayerCos=0.6745±0.0289
- apple__food: shared=0.0000, specific=1.0000, LayerCos=0.4937±0.2999
- cat__animal: shared=0.0721, specific=0.9279, LayerCos=0.4970±0.0685
- king__human: shared=0.0000, specific=1.0000, LayerCos=0.6377±0.0000
- queen__human: shared=0.0000, specific=1.0000, LayerCos=0.7662±0.0000

## 5. 有限基与复用效率信号
- 参与率(PR): 11.2714±2.2853
- 能量占比: top1=0.2721, top5=0.4710
- 复用神经元比例: 0.0002±0.0000

## 6. 假设检验（结构级）
- H1_finite_basis_plus_composition: PASS (PR>0 and top5_energy>top1_energy)
- H2_causal_axis_separable: PASS (all diagonal_advantage_mean > 0)
- H3_semantic_hierarchy_consistent: FAIL (LayerCos(apple,banana) > LayerCos(apple,cat))
- H4_axis_layer_profile_stable: PASS (all profile_cosine_mean > 0.4)
- 通过率: 75.00%

## 7. 解释
- 该报告支持“静态概念坐标 + 动态层级路由 + 有限基复用”的统一编码图景。
- 若要逼近微观数学原理，下一步应在同一概念集上增加跨seed同prompt的因果子回路追踪。
- 对 apple/king/queen 这类概念，建议在同一模板上做反事实最小编辑并比较最小子回路的可迁移性。
