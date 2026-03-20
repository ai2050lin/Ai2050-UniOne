# Stage56 权重更新几何摘要

- main_judgment: 权重更新改变图册、前沿、闭包边界的方式并不相同：图册更像身份稳定化，前沿更像高质量支撑重排，闭包边界更像后期选择性硬化。

{
  "atlas_shift": {
    "equation": "Delta_Atlas ~ + atlas_learning_drive",
    "magnitude": 0.09630133683739817,
    "meaning": "权重更新先抬高家族片区稳定性，再压低局部偏移噪声"
  },
  "frontier_shift": {
    "equation": "Delta_Frontier ~ + frontier_learning_drive",
    "magnitude": 0.309988849399917,
    "meaning": "权重更新通过基础负载和一般驱动共同塑造高质量前沿"
  },
  "boundary_shift": {
    "equation": "Delta_Boundary ~ + closure_learning_drive",
    "magnitude": 0.4498413321980581,
    "meaning": "权重更新在训练后段推动闭包边界逐步变硬"
  }
}
