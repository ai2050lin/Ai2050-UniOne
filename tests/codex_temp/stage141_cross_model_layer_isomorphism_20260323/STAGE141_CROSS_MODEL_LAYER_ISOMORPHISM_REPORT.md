# Stage141: 跨模型层同构块

## 核心结果
- 模型数量: 3
- 配对数量: 2
- 平均层同构分数: 0.6405
- 最佳对齐模型: DeepSeek-R1-Distill-Qwen-7B (0.7326)
- 最弱对齐模型: Qwen3-4B (0.5485)

## 模型层位坐标
- GPT-2: layers=12, early=1 (0.0909), route=3 (0.2727), late=11 (1.0000), support=(1.0000, 0.6386, 0.8626)
- Qwen3-4B: layers=36, early=3 (0.0857), route=26 (0.7429), late=19 (0.5429), support=(0.0000, 0.6353, 1.0000)
- DeepSeek-R1-Distill-Qwen-7B: layers=28, early=2 (0.0741), route=20 (0.7407), late=20 (0.7407), support=(0.5000, 0.3665, 0.9721)

## 相对 GPT-2 的同构结果
- Qwen3-4B: score=0.5485, slope=-0.5704, intercept=25.5489, mae=1.6302, norm_gap=0.3108, ratio_gap=1.2375, order_valid=False
- DeepSeek-R1-Distill-Qwen-7B: score=0.7326, slope=1.3042, intercept=6.3852, mae=3.8679, norm_gap=0.2480, ratio_gap=0.8000, order_valid=True