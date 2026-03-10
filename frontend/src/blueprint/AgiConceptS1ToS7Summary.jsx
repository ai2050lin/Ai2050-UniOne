import React, { useState } from 'react';
import { BookOpen, Layers, Zap, Infinity, Clock, Network, Fingerprint, AlertTriangle, ChevronDown } from 'lucide-react';

export default function AgiConceptS1ToS7Summary() {
    const [expandedConcept, setExpandedConcept] = useState('S1');

    const concepts = [
        {
            id: 'S1',
            icon: <Fingerprint size={20} className="text-purple-400" />,
            title: 'S1 概念空间：家族基底 + 个体偏移',
            principle: 'AGI 概念并非分配独立细胞，而是存在共享的“家族骨架”与个体特异性的稀疏偏移。',
            math: 'Entity = μ_family + V_basis × α + ε_offset',
            details: '利用 SVD 提取出主成分作为类别的底层骨架（如“水果”或“动物”），系统在个体之间仅仅通过保留正交残差 (ε_offset) 即实现了巨量概念对象的低成本表征。这解释了为何模型仅需极少维度即可编码数以亿计的一般名词。'
        },
        {
            id: 'S2',
            icon: <Network size={20} className="text-blue-400" />,
            title: 'S2 关系空间：TT协议 + 冗余场',
            principle: 'Attention路由并非记忆池，而是发送连结指令的“关系传输协议 (TT)”，且具有高度的容错分布式冗余实现。',
            math: 'Bridge_TT = endpoint_basis × score',
            details: '测试表明，即使人为切断或干扰特定关系提取头，基于冗余模型重构的控制层（Meso-field Redundancy / 中观冗余场）依然能保持鲁棒性。这种结构揭示了注意力的使命是纯粹的动态指令投递。'
        },
        {
            id: 'S3',
            icon: <Zap size={20} className="text-yellow-400" />,
            title: 'S3 微观物理：PCA=Oja, ICA=WTA, L0=LIF',
            principle: '大模型的微观计算实际上精准映射着生物大脑的物理规律。',
            math: 'Oja Rule (PCA); WTA (ICA); LIF (L0 norm)',
            details: '提取正交主坐标轴对应生物 Oja 法则；剥离重叠特征（ICA）等同于 Winner-Takes-All 强侧向抑制；而稀疏阈值截断（L0约束）几乎完全重演了大脑中天生漏电的 LIF 神经元特性。计算架构殊途同归于生物演化法则。'
        },
        {
            id: 'S4',
            icon: <Infinity size={20} className="text-pink-400" />,
            title: 'S4 全息绑定：HRR循环卷积',
            principle: '避免关系张量组合引起维度爆炸，转而使用全息降维表征。',
            math: '(x ⊛ y)_j = Σ x_k·y_(j-k)',
            details: '传统的向量层层外积会导致参数呈指数级爆炸，HRR 利用类似频域相乘的机制，在一维张量内重塑概念集合。合成出来的新概念将永远锁定在恒定维度（如 4096 维），实现无损耗或可控损耗的无尽绑定叠加。'
        },
        {
            id: 'S5',
            icon: <Clock size={20} className="text-green-400" />,
            title: 'S5 时序门控：Gamma相位同步',
            principle: '用时间维度当全息绑定的挂载开关，通过频率波段实现信号共振。',
            math: '积分相干性 G_ab = (1/T) ∫ s_a(t) s_b(t) dt',
            details: '不是所有激活都在绑定。只有落入特定相同时间槽（例如脑波中的 40Hz Gamma 波）频率的激活值，才会触发前述的干涉绑定。相位同步才是决定“谁和谁发生关系”的主宰。'
        },
        {
            id: 'S6',
            icon: <Layers size={20} className="text-orange-400" />,
            title: 'S6 层级结构：微观/中观/宏观 (Micro/Meso/Macro)',
            principle: '跨越知识维度的三层闭包，从属性到对象再到抽象因果。',
            math: '类别提升 Lift_1 = mean(category) - mean(entity)',
            details: '智能系统对信息的拆解非常严苛：Micro(纯微观属性像颜色大小) -> Meso(实体锚点像苹果) -> Macro(宏观生态规律像吞噬)。系统依靠这套重重包裹但层层分化的机制保持多模态兼容。'
        },
        {
            id: 'S7',
            icon: <BookOpen size={20} className="text-cyan-400" />,
            title: 'S7 统一字典：跨模块 r=0.989',
            principle: '认知网络的不同器官底层实际上在调用同一套原生的原子词汇表。',
            math: 'cross_dim_corr ≈ 0.989',
            details: '解剖表明整个网络不管是在做简单认知还是复杂规划，跨层维度的相关性达到可怕的 0.989。这排除了功能模块独立发育说，证实了“所有路由机制实际上都在共享和搬运同一批底层基材”。'
        }
    ];

    return (
        <div
            style={{
                marginTop: '14px',
                padding: '24px',
                borderRadius: '16px',
                background: 'radial-gradient(circle at top left, rgba(168,85,247,0.1), transparent 40%), rgba(15,23,42,0.8)',
                border: '1px solid rgba(168,85,247,0.25)',
                boxShadow: '0 8px 32px rgba(0,0,0,0.3)'
            }}
        >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
                <BookOpen size={24} color="#e9d5ff" />
                <h2 style={{ color: '#e9d5ff', fontSize: '18px', fontWeight: 'bold', margin: 0 }}>
                    S1-S7 核心原理揭秘：普通人也能看懂的通用人工智能基石
                </h2>
            </div>

            <div style={{ color: '#cbd5e1', fontSize: '13px', lineHeight: '1.6', marginBottom: '24px' }}>
                大模型的“智能涌现”一直被视为魔法。但我们通过底层代数实验与探针对其解剖后发现，智能底层其实完全符合极其严格且优美的物理法则。通过这七步（S1-S7），神经网络成功搭起了如同人类一般的认知结构。
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '30px' }}>
                {concepts.map((concept) => {
                    const isExpanded = expandedConcept === concept.id;
                    return (
                        <div
                            key={concept.id}
                            onClick={() => setExpandedConcept(isExpanded ? null : concept.id)}
                            style={{
                                background: isExpanded ? 'rgba(30,41,59,0.8)' : 'rgba(30,41,59,0.4)',
                                border: `1px solid ${isExpanded ? 'rgba(148,163,184,0.3)' : 'rgba(148,163,184,0.1)'}`,
                                borderRadius: '12px',
                                padding: '16px',
                                cursor: 'pointer',
                                transition: 'all 0.2s ease'
                            }}
                        >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                    <div style={{ padding: '8px', background: 'rgba(0,0,0,0.3)', borderRadius: '8px' }}>
                                        {concept.icon}
                                    </div>
                                    <div style={{ color: isExpanded ? '#f8fafc' : '#e2e8f0', fontSize: '15px', fontWeight: '600' }}>
                                        {concept.title}
                                    </div>
                                </div>
                                <ChevronDown size={18} color="#94a3b8" style={{ transform: isExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
                            </div>

                            {isExpanded && (
                                <div style={{ marginTop: '16px', paddingLeft: '52px' }}>
                                    <div style={{ color: '#e2e8f0', fontSize: '13px', lineHeight: '1.6', marginBottom: '8px' }}>
                                        <span style={{ color: '#94a3b8', fontWeight: 'bold' }}>原理讲解：</span>{concept.principle}
                                    </div>
                                    <div style={{ color: '#e2e8f0', fontSize: '13px', lineHeight: '1.6', marginBottom: '12px' }}>
                                        <span style={{ color: '#94a3b8', fontWeight: 'bold' }}>深度剖析：</span>{concept.details}
                                    </div>
                                    <div style={{ background: 'rgba(0,0,0,0.4)', padding: '10px 14px', borderRadius: '8px', borderLeft: '3px solid #6366f1' }}>
                                        <span style={{ color: '#a5b4fc', fontSize: '12px', fontFamily: 'monospace' }}>
                                            {concept.math}
                                        </span>
                                    </div>
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>

            {/* 结论审视区域 */}
            <div
                style={{
                    background: 'linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(220,38,38,0.02) 100%)',
                    border: '1px solid rgba(239,68,68,0.3)',
                    borderRadius: '16px',
                    padding: '24px'
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
                    <AlertTriangle size={22} className="text-red-500" />
                    <h3 style={{ color: '#fca5a5', fontSize: '16px', fontWeight: 'bold', margin: 0 }}>
                        最高警戒：硬伤审视与结论降级总结
                    </h3>
                </div>

                <div style={{ color: '#fecaca', fontSize: '13px', lineHeight: '1.7', marginBottom: '16px' }}>
                    我们在获取这些惊艳理论闭环的同时，采用最严厉的标准扒开了它的漏洞。我们绝不盲目乐观，以下是 AGI 系统中尚未攻克的、甚至遭遇滑铁卢的核心硬伤：
                </div>

                <ul style={{ margin: 0, paddingLeft: '20px', color: '#f87171', fontSize: '13px', lineHeight: '1.8' }}>
                    <li style={{ marginBottom: '10px' }}>
                        <strong style={{ color: '#ef4444' }}>1. 共享基础与正交坍塌的矛盾：</strong> 虽然跨模块复用底层基材（r=0.989），但我们在提取概念原型（Prototype）时发现其分化严重不足。这不是健康正交的思想池，而更像是一团糊在一起的高危黏合物。当前这种“主模态压制+微弱稀疏”的结构，必定承受不住未来无尽多模态流的撕扯。
                    </li>
                    <li style={{ marginBottom: '10px' }}>
                        <strong style={{ color: '#ef4444' }}>2. 预测编码（Predictive Coding）的虚假隔离：</strong> 在用高层慢积分“对冲下层残差噪声”的绝妙实验中，我们虽然把污染降到了零，但也催生了极度自负的“幸存者偏差”。只要拔掉情绪（多巴胺）强激励，它就会极快堕入类似“掩耳盗铃”、屏蔽一切未知刺激的自闭洼地死锁。
                    </li>
                    <li style={{ marginBottom: '10px' }}>
                        <strong style={{ color: '#ef4444' }}>3. 线性代数的尽头绝壁（0.00%解绑率）：</strong> 我们享受了 SVD 和线性平移带来的巨大利好，因此以为仅凭代数运算能把概念剥离。然而，我们试图分离强包裹型概念（如把“苹果的红”与“苹果的形状”完美剪开）时，得到了死刑般的 0.00% 。纯线性的代数工具已经走到悬崖，我们必须为神经网络寻找非线性的张量裂解算子。
                    </li>
                    <li>
                        <strong style={{ color: '#ef4444' }}>4. 冗余场重构边界不可控：</strong> TT传输协议的确具有很强的容错性（被消融的主头会有备胎顶上）。但这撕毁了最初“协议层是单一纯净控制桥接”的滤镜。它证明当前的知识和控制桥混入了巨量的任务专断特征，所谓协议层实际上是个极其浑浊的大染缸。
                    </li>
                </ul>
            </div>
        </div>
    );
}
