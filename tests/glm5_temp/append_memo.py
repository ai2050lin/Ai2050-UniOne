"""追加Phase CCVI总结到MEMO"""
import os
memo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'research', 'glm5', 'docs', 'AGI_GLM5_MEMO.md')

content = """
[2026-04-19 21:00] Phase CCVI: 五阶段进展总结与最严格审视

=================================================================
五阶段进展总结
=================================================================

P1 因果导航 (进展: 95%)
  已完成:
    - 确认因果空间: 差分向量在residual stream中的方向一致性
    - PCA分析: DS7B PC1=53.4%, GLM4 PC1=8.0%, Qwen3 PC1=14.8%
    - 8bit量化验证: 与4bit结果高度一致
    - Bootstrap 95%CI: 所有结果统计显著
    - 双方法验证: 直接head输出与W_o投影等价
  问题:
    - GLM4 L36-L39因CPU offload无法分析(末层!)
    - Qwen3是BF16加载到GPU, DS7B/GLM4是8bit, 量化精度差异可能影响比较
    - 差分向量定义(text_a - text_b)的方向性: 为什么不是text_b - text_a?

P2 几何定位 (进展: 70%)
  已完成:
    - Gram矩阵: DS7B L27有90%突变, GLM4无突变
    - W_o SVD: DS7B L27满秩(eff_rank_90=727), 不存在W_o坍缩
    - 层间趋势: 三模型best_single alignment在0.46-0.65范围
  ★★★重大修正★★★:
    - Phase CCIV的L27因果汇聚head(cos=0.508)是错误结论!
    - 实际L27跨特征alignment=0.308, 与其他层无明显差异!
    - L0的alignment反而最高(DS7B=0.624, Qwen3=0.634, GLM4=0.645)!
    - Gram矩阵90%突变需要重新解释(可能只是数值现象, 不是几何突变)
  问题:
    - 为什么L0的因果alignment最强? 是否因为L0做token embedding的初级特征提取?
    - Gram矩阵突变的真正含义是什么?
    - 层间alignment趋势(L0高->中间低->末层中等)如何解释?

P3 SAE逆向 (进展: 35%)
  已完成:
    - 特征分解alignment: 单特征alignment=0.5-0.9, 跨特征=0.2-0.4
    - 因果范数: DS7B L27 h12 norm=27583, 是其他head的10倍(但需验证)
    - Head-Feature特异性矩阵: DS7B h10=polarity, h12=tense, h13=number
  ★★★重大修正★★★:
    - 因果原子(DS7B L27 h10=polarity等)可能也是错误结论!
    - 因为L27并不特殊, 其他层也有类似head特异性
    - 需要在所有层做head-feature特异性分析才能确定L27是否特殊
  问题:
    - 因果范数(head级差分向量的L2范数)的定义是否合理?
    - SAE(Sparse Autoencoder)还没有训练, 这是因果原子发现的关键工具
    - Head-Feature特异性在不同层如何演化?

P4 电路逆向 (进展: 55%)
  已完成:
    - 双方法验证: 直接head输出与W_o投影等价
    - 三模型对比: GLM4 L0 cross=0.461, Qwen3 L0 cross=0.391, DS7B L0 cross=0.371
    - 单特征alignment: tense=0.8-0.99, polarity=0.6-0.9, number=0.4-0.8
    - 语义特征(semantic, sentiment)alignment低(0.15-0.4)
  ★★★新发现★★★:
    - 三模型都是tense alignment最高(0.8-0.99), polarity次之(0.6-0.9)
    - 这说明语法特征的因果信号比语义特征更强更一致!
    - 符合直觉: 语法特征(tense, polarity, number)是低维的, 语义特征是高维的
  问题:
    - 为什么L0的语法特征alignment最高? 后续层在做什么?
    - 跨层电路追踪: 一个tense特征从L0到末层的因果路径是什么?
    - Activation patching还没有做, 无法验证因果性(只有相关性)

P5 代数/拓扑 (进展: 25%)
  已完成:
    - 因果空间是5个特征方向的线性组合(跨特征alignment低因为方向正交)
    - 单特征alignment高->因果空间接近5个正交子空间
    - 不同特征的alignment排序: tense > polarity > number >> semantic > sentiment
  问题:
    - 5个正交子空间是否是因果空间的精确描述? 还是需要更细的分解?
    - 层间alignment变化的代数结构是什么?
    - 仿射联络理论还没有建立(层间Gram矩阵变化->微分几何)
    - 需要新的数学框架描述: 低维语法特征 vs 高维语义特征

=================================================================
最严格审视
=================================================================

硬伤1: Phase CCIV的因果汇聚结论是错的
  - L27 h12 cos=0.508 vs 实际0.308 -> 之前脚本有bug或样本不足
  - Gram矩阵90%突变可能只是数值噪声, 不是几何相变
  - DS7B L27因果坍缩可能是假象 -> 需要重新解释

硬伤2: 因果相关性 != 因果性
  - 所有alignment都是相关性度量, 不是因果性!
  - 要证明因果性, 需要activation patching/intervention
  - 目前所有结论都是观察性的, 不能排除混淆变量

硬伤3: 特征选择偏差
  - 只选了5个语法特征(tense, polarity, number, semantic, sentiment)
  - 这5个特征都是人为选择的, 不是从数据中发现的!
  - 真正的因果原子可能完全不同
  - 需要用SAE从数据中发现因果原子

硬伤4: 差分向量的统计问题
  - 差分向量(text_a - text_b)的方向取决于text_a和text_b的选择
  - 500对/特征的样本可能不够覆盖所有变化
  - 不同特征的差分方向可能相关(不是真正正交)

硬伤5: 模型规模和架构差异
  - Qwen3-4B(BF16), DS7B-7B(8bit), GLM4-9B(8bit+CPU offload)
  - 量化精度不同, 可能影响alignment比较
  - GLM4末4层无法分析, 可能丢失关键信息

=================================================================
第一性原理: 破解语言背后数学原理的下一步
=================================================================

核心洞察:
  1. 因果信号是全局的, 不是末层突变 -- 每一层都有因果alignment
  2. 语法特征(tense, polarity)的因果信号比语义特征更强
  3. 跨特征alignment低不是因为汇聚弱, 而是特征方向正交
  4. L0的alignment最高 -> 因果信息在embedding层就已经编码

下一步突破方向(按优先级):

1. Activation Patching因果干预 (P4核心)
  - 对DS7B做tense/polarity的activation patching
  - 方法: 把The cat sat的L0 h15输出patch到The cat sits上, 看tense是否改变
  - 这是证明因果性的唯一方法!

2. 数据驱动的因果原子发现 (P3核心)
  - 用SAE在差分向量空间训练, 发现真正的因果原子
  - 不再人为选择5个特征, 而是从数据中自动发现
  - SAE可能发现: tense是一个原子, 还是多个更细粒度原子的组合?

3. 层间因果电路追踪 (P4核心)
  - 追踪tense特征从L0到L27的因果路径
  - 每一层的哪个head传递了tense信息?
  - 是否存在tense电路?

4. 因果空间的代数结构 (P5)
  - 5个正交子空间的精确描述
  - 层间变换的矩阵分解
  - 建立仿射联络理论

5. 更大模型验证 (P1)
  - 在LLaMA-3-8B, Mistral-7B上验证
  - 确认结论的普遍性
"""

with open(memo_path, 'a', encoding='utf-8') as f:
    f.write(content)
print('done')
