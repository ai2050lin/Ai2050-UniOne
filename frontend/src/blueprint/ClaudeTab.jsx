import React, { useState } from 'react';
import { Brain, ChevronDown, ChevronRight, Activity } from 'lucide-react';

export const ClaudeTab = () => {
    const [expandedSteps, setExpandedSteps] = useState({});
    const [expandedTests, setExpandedTests] = useState({});

    const toggleStep = (idx) => {
        setExpandedSteps(prev => ({
            ...prev,
            [idx]: !prev[idx]
        }));
    };

    const toggleTest = (idx) => {
        setExpandedTests(prev => ({
            ...prev,
            [idx]: !prev[idx]
        }));
    };

    const testRecords = [
        {
            name: 'E1: MLP 稀疏激活解剖',
            goal: '探测神经网络中知识的隐式存储结构，验证大脑皮层激活稀疏性假说在 DNN 中的等价映射。',
            date: '2026-02-26',
            evidence: '神经元激活峰度攀升至 31.99；过滤微弱激活(|act|<0.1)后，依然保留41.4%特征。',
            result: 'MLP 层涌现极端尖峰重尾分布，证实知识存放于高特异化、极度稀疏的“专家神经元”中。领域内重叠率达 100%，跨领域仅 47%。',
            sig: '彻底打破知识在网络中是“密集混合存储”的传统预设，证实极致稀疏编码是高能效认知的前提。',
            summary: '实验一举揭示了基于微积分流形的稀疏编码本质，即庞大参数量并不运算所有信息，而是构建庞大字典，单次只激活极简路径。'
        },
        {
            name: 'E2: 逐层残差增量 SVD',
            goal: '量化信息在深浅层之间的传递动力学，寻找语义逐层抽象乃至重组的具体数学证据。',
            date: '2026-02-26',
            evidence: '层级方差深/浅比压缩至 0.18x；浅层 L0 增量达 3.25，深层极小，整体结构呈显著沙漏形（Hourglass）。',
            result: '浅层负责巨量方差的生成与粗颗粒特征的急剧扩张，深层则仅仅进行微小方差的平滑与收敛刻画。',
            sig: '揭示了 AGI 网络提取语义规律的“浅层大开大合，深层收敛定调”的拓扑雕刻过程。',
            summary: '对偶验证了理论中“特征在浅层生成干涉波谷，在深层滑向低维能量稳态流形”的拓扑抽象步骤。'
        },
        {
            name: 'E3: Attention 有效秩测量',
            goal: '剥离 Attention 机制的非必要参数，测定上下文绑定功能的实际所需理论维度。',
            date: '2026-02-26',
            evidence: '在维度 11 的空间中，全局平均有效秩仅为 4.58。高达 65.3% 的注意力头秩 < 5，且 88% 的权重死锁于首词(BOS)的默认沉入区。',
            result: 'Attention 并不像此前认为的在密集搬运知识，其注意力头极度低秩、任务极简、工作模式单调。',
            sig: '明确了上下文绑定（Systemicity）所需的算力极低，为我们在未来架构中大幅裁剪 Attention 组件找到了数学支撑。',
            summary: '结合 E1 中的 MLP 稀疏性定调，Attention 只做少量神经流形的切换与桥接。这坚定了存算分离（Logic与Memory解耦）设计的正确性。'
        },
        {
            name: 'E4: 权重矩阵低秩分析',
            goal: '区分参数矩阵中“知识容器”与“逻辑骨架”的数学身份，为下一代 FiberNet 的资源分配提供物理依据。',
            date: '2026-02-27',
            evidence: 'MLP 权重满秩维度高达 581（占比 75%），而 Attention QK 矩阵的满秩维度仅为 52（占比 6.8%）。',
            result: '出现了极具视觉冲击力的对比：MLP 是“极满秩”的信息黑洞，而 Attention 是极罕见的“极度低秩”拓扑算子。',
            sig: '清晰定义了 AGI 模型构建时的规模扩展法则（Scaling Law）：即用满秩的纤维存记忆，用低秩的底流形走逻辑。',
            summary: '实验终结了传统网络中“参数即是能力”的粗糙论断，指出网络包含结构化功能分工，即知识高度密集于局部，而逻辑操作则被压缩在极低维度。'
        },
        {
            name: 'E5: Z113 结构相变追踪',
            goal: '捕捉从“死记硬背”到“规律泛化”（Grokking）的临界点几何相变，破译泛化能力的涌现密码。',
            date: '2026-02-27',
            evidence: '训练至 14000 步时，网络空间表示拓扑圆度（Circularity）从 0.51 突升至 0.82，随后泛化准确率（22.9%）才开始出现。',
            result: 'Grokking 不是盲目的性能突变，而是系统底层的几何表征从混沌的记忆乱码重新组合成了完美圆环状（代数群结构）。',
            sig: '首次将“泛化理解能力”具象化为一个可测量的黎曼流形指标（圆度），奠定了用数学几何优化智能的基础。',
            summary: '说明 AGI 真正的掌握规律必经拓扑重组期。这验证了在模型离线演化（睡眠）时施加 Ricci Flow 平滑处理可以有效诱导物理结构的顿悟。'
        },
        {
            name: 'Emergence: 稀疏自发涌现',
            goal: '无 BP（反向传播）条件下，验证纯局部的物理竞争法则能否激发出类似人类大脑的底层稀疏编码格式。',
            date: '2026-02-27',
            evidence: '仅使用 5000 步的局部侧抑制结合 Hebbian 更新，特征激活峰度自发从随机的 2.7 几何级飙升至 19.7。',
            result: '成功利用极其轻量的纯底层局部物理竞争规则爆发了高度稀疏的神经网络，表现与大型化石大同小异。但特征专家化程度不足。',
            sig: '证实即使摒弃 BP 算法，只要引入物理的互斥抑制机制，极效系统的基础编码底座依然可以自发建立。',
            summary: '实验是一场惊险跳跃，成功证明稀疏结构不依赖全局梯度。但遗留的“专家化分工”缺陷直接触发了下一阶段 P0 级双向预测编码（Predictive Coding）机制的探索。'
        }
    ];

    const roadmapSteps = [
        {
            title: "H1 阶段",
            status: "已完成",
            desc: "理论奠基与小规模实证、极效三定律、可视化并网",
            details: "在此阶段，我们彻底验证了抛弃BP反向传播黑盒的可行性。建立了基于微积分几何和神经纤维丛理论的基础模型原型。完成了极效三定律的论证，并通过第一阶段的前端交互式可视化面板（Glass Matrix），成功对小规模纯代数引力场引擎（Mother Engine）进行了观察并网。"
        },
        {
            title: "H2 阶段 (当前)",
            status: "攻坚期",
            desc: "深度解剖化石与局部学习机制攻坚，信用分配突围",
            details: "直面最严峻的'信用分配'危机。我们正在开发能与BP匹敌，但保持极高局部约束的新一代信用下放机制。通过持续解刨现存大规模DNN化石（如GPT-2），尝试从中提取自发涌现的专家化聚类及对偶关联机制，以构建完整的分层预测编码（Predictive Coding）体系，目标是突破基础泛化能力的门槛。"
        },
        {
            title: "H3 阶段",
            status: "中期",
            desc: "跨模态统一，大规模语义涌现，百万Token连贯性",
            details: "重点突破高维拓扑空间的局限，解决跨模态（视觉、听觉与语言）信号的统一流形关联，实现真正的符号接地（Symbol Grounding）。同时克服目前引擎频繁陷入局部几何势能洼地的问题，实现长程（百万级Token）时序逻辑的自发顺滑推演。"
        },
        {
            title: "H4 阶段",
            status: "远期",
            desc: "脱离冯·诺依曼架构，神经形态芯片，可控 AGI 原型",
            details: "剥离高度依赖算力和传统冯·诺依曼结构的软件模拟，向具备'存算分离'及极致分布式计算特征的神经形态芯片（Neuromorphic Chip）移植。最终落地可进行物理干预、具有超高能效比、具备自发探索行为和安全可控协议的全尺寸 AGI 原型系统。"
        },
    ];

    return (
        <div style={{ display: 'grid', gap: '24px' }}>
            <div
                style={{
                    padding: '30px',
                    borderRadius: '24px',
                    border: '1px solid rgba(168,85,247,0.28)',
                    background: 'linear-gradient(135deg, rgba(168,85,247,0.10) 0%, rgba(168,85,247,0.03) 100%)',
                    marginBottom: '10px',
                }}
            >
                <div style={{ color: '#a855f7', fontWeight: 'bold', fontSize: '20px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Brain size={24} /> Project Genesis: 第一性原理 AGI 研究全景报告
                </div>

                {/* 1. 整体研究框架与进展 */}
                <div style={{ marginBottom: '28px' }}>
                    <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#e9d5ff', marginBottom: '12px', borderBottom: '1px solid rgba(168,85,247,0.3)', paddingBottom: '8px' }}>一、分析框架</div>
                    <div style={{ color: '#d1d5db', fontSize: '13px', lineHeight: '1.7' }}>
                        构建基于微分几何、神经纤维丛拓扑（NFBT）和纯代数演化的智能引擎（Mother Engine），抛弃传统 BP 黑盒与堆叠算力路线。<br />
                        <span style={{ color: '#a855f7', fontWeight: 'bold' }}>进展突破: </span>建立“极效三定律”（侧抑制正交、引力雕刻、能量坍塌）；通过解剖 DNN 证实大脑的激活稀疏性编码方式；发现 Attention 的极低秩关联拓扑；在无 BP 下利用局部规则实现空白网络自发涌现稀疏特征（峰度激增至 19.7）。
                    </div>
                </div>

                {/* 2. 完整路线图 */}
                <div style={{ marginBottom: '28px' }}>
                    <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#e9d5ff', marginBottom: '12px', borderBottom: '1px solid rgba(168,85,247,0.3)', paddingBottom: '8px' }}>二、路线图 (Roadmap)</div>
                    <div style={{ display: 'grid', gap: '12px' }}>
                        {roadmapSteps.map((step, idx) => (
                            <div
                                key={idx}
                                onClick={() => toggleStep(idx)}
                                style={{
                                    padding: '16px',
                                    background: 'rgba(0,0,0,0.4)',
                                    borderRadius: '10px',
                                    borderLeft: step.status === '已完成' ? '3px solid #10b981' : '3px solid #a855f7',
                                    cursor: 'pointer',
                                    transition: 'all 0.2s ease',
                                    userSelect: 'none'
                                }}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                        <div style={{ color: '#fff', fontSize: '15px', fontWeight: 'bold' }}>{step.title}</div>
                                        <div style={{
                                            padding: '2px 8px',
                                            borderRadius: '12px',
                                            background: step.status === '已完成' ? 'rgba(16,185,129,0.1)' : 'rgba(168,85,247,0.1)',
                                            color: step.status === '已完成' ? '#10b981' : '#e9d5ff',
                                            fontSize: '11px',
                                            border: step.status === '已完成' ? '1px solid rgba(16,185,129,0.3)' : '1px solid rgba(168,85,247,0.3)'
                                        }}>
                                            {step.status}
                                        </div>
                                    </div>
                                    {expandedSteps[idx] ? <ChevronDown size={18} color="#9ca3af" /> : <ChevronRight size={18} color="#9ca3af" />}
                                </div>
                                <div style={{ color: '#d1d5db', fontSize: '13px', marginTop: '8px' }}>{step.desc}</div>

                                {expandedSteps[idx] && (
                                    <div style={{
                                        marginTop: '16px',
                                        paddingTop: '16px',
                                        borderTop: '1px dashed rgba(255,255,255,0.1)',
                                        color: '#a1a1aa',
                                        fontSize: '13px',
                                        lineHeight: '1.6'
                                    }}>
                                        {step.details}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                {/* 3. 测试记录 (E1~E5) */}
                <div style={{ marginBottom: '28px' }}>
                    <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#e9d5ff', marginBottom: '12px', borderBottom: '1px solid rgba(168,85,247,0.3)', paddingBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Activity size={18} /> 三、测试记录 (Phase XXXVI)
                    </div>
                    <div style={{ display: 'grid', gap: '16px' }}>
                        {testRecords.map((exp, idx) => (
                            <div
                                key={idx}
                                onClick={() => toggleTest(idx)}
                                style={{
                                    padding: '16px',
                                    background: 'rgba(0,0,0,0.3)',
                                    borderRadius: '12px',
                                    borderLeft: '3px solid #00d2ff',
                                    cursor: 'pointer',
                                    transition: 'all 0.2s ease',
                                    userSelect: 'none'
                                }}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <div style={{ color: '#fff', fontSize: '14px', fontWeight: 'bold' }}>{exp.name}</div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <div style={{ fontSize: '12px', color: '#9ca3af' }}>{exp.date}</div>
                                        {expandedTests[idx] ? <ChevronDown size={18} color="#00d2ff" /> : <ChevronRight size={18} color="#00d2ff" />}
                                    </div>
                                </div>

                                {!expandedTests[idx] && (
                                    <div style={{ color: '#9ca3af', fontSize: '12px', marginTop: '6px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                        {exp.goal}
                                    </div>
                                )}

                                {expandedTests[idx] && (
                                    <div style={{ marginTop: '14px', paddingTop: '12px', borderTop: '1px solid rgba(0,210,255,0.15)', display: 'grid', gap: '10px', fontSize: '12px', color: '#d1d5db', lineHeight: '1.6' }}>
                                        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(80px, auto) 1fr', gap: '12px' }}>
                                            <div style={{ color: '#00d2ff', fontWeight: 'bold' }}>测试目标:</div><div>{exp.goal}</div>
                                            <div style={{ color: '#00d2ff', fontWeight: 'bold' }}>测试结果:</div><div>{exp.result}</div>
                                            <div style={{ color: '#00d2ff', fontWeight: 'bold' }}>关键证据:</div><div>{exp.evidence}</div>
                                            <div style={{ color: '#00d2ff', fontWeight: 'bold' }}>对AGI意义:</div><div style={{ color: '#e5e7eb', fontWeight: '500' }}>{exp.sig}</div>
                                            <div style={{ color: '#00d2ff', fontWeight: 'bold' }}>分析总结:</div><div>{exp.summary}</div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                {/* 4. 问题与硬伤 */}
                <div style={{ marginBottom: '28px' }}>
                    <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#ef4444', marginBottom: '12px', borderBottom: '1px solid rgba(239,68,68,0.3)', paddingBottom: '8px' }}>四、存在问题</div>
                    <div style={{ display: 'grid', gap: '10px' }}>
                        <div style={{ padding: '12px', background: 'rgba(239,68,68,0.05)', borderRadius: '8px', borderLeft: '3px solid #ef4444' }}>
                            <div style={{ color: '#ef4444', fontSize: '13px', fontWeight: 'bold', marginBottom: '4px' }}>🔴 致命硬伤: 信用分配 (Credit Assignment) 危机</div>
                            <div style={{ color: '#d1d5db', fontSize: '12px', lineHeight: '1.6' }}>在摒弃全局 BP 后，局部规则虽能长出稀疏神经元，却无法实现“专家化”分工。系统不知如何精准把宏观误差分摊给底层突触。SCRC 测试中 MNIST 仅21%准确率，这卡死了模型智能规模底线。</div>
                        </div>
                        <div style={{ padding: '12px', background: 'rgba(245,158,11,0.05)', borderRadius: '8px', borderLeft: '3px solid #f59e0b' }}>
                            <div style={{ color: '#f59e0b', fontSize: '13px', fontWeight: 'bold', marginBottom: '4px' }}>🟠 严重瓶颈: 语义连贯与符号接地断层</div>
                            <div style={{ color: '#d1d5db', fontSize: '12px', lineHeight: '1.6' }}>模型极易跌入局部势能洼地（循环输出 "the", "of"），无法展开长程深度逻辑。此外，纯代数引擎仍需外部解析器辅助，缺乏直接从感官像素流自发形成通用概念“符号接地”的能力。</div>
                        </div>
                    </div>
                </div>

                {/* 5. 接下来的工作 */}
                <div>
                    <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#10b981', marginBottom: '12px', borderBottom: '1px solid rgba(16,185,129,0.3)', paddingBottom: '8px' }}>五、接下来的核心工作 (Next Steps)</div>
                    <div style={{ padding: '16px', borderRadius: '12px', background: 'linear-gradient(90deg, rgba(16,185,129,0.1) 0%, rgba(0,0,0,0) 100%)', borderLeft: '4px solid #10b981' }}>
                        <div style={{ color: '#fff', fontSize: '14px', fontWeight: 'bold', marginBottom: '8px' }}>P0最高优: 完整分层预测编码 (Predictive Coding) 体系</div>
                        <div style={{ color: '#d1d5db', fontSize: '13px', lineHeight: '1.7' }}>
                            假说全面升级为 <strong>"竞争稀疏 + 预测编码" </strong> 的双层耦合架构。浅层竞争产生“高维稀疏特征聚类”，高层必须引入基于 Rao & Ballard 大一统框架的<b>完整版独立分层预测与误差逐层回传机制</b>充当教师，突破 85% MNIST 准确率指标。这是取代传统 BP 的关键战役。
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
};
