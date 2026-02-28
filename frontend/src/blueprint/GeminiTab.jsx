import React, { useState } from 'react';
import { Brain, ChevronDown, ChevronRight, Activity } from 'lucide-react';

export const GeminiTab = () => {
    const [expandedSteps, setExpandedSteps] = useState({});
    const [expandedPhase, setExpandedPhase] = useState(null);
    const [expandedTestItem, setExpandedTestItem] = useState(null);

    const toggleStep = (idx) => {
        setExpandedSteps(prev => ({
            ...prev,
            [idx]: !prev[idx]
        }));
    };

    const phasedTestRecords = [
        {
            id: 'phase_1',
            title: '阶段一：拓扑结构与稀疏性解剖',
            status: 'done',
            objective: '探测神经网络中知识的隐式存储结构与信息层级传递机制。',
            summary: '揭示了 MLP 层的极度稀疏激活特性（专家神经元）以及残差连接的浅层大开大合、深层收敛平滑拓扑。',
            tests: [
                {
                    id: 'E1',
                    name: 'MLP 稀疏激活解剖',
                    target: '探测神经网络中知识的隐式存储结构，验证大脑皮层激活稀疏性假说在 DNN 中的等价映射。',
                    testDate: '2026-02-26',
                    evidence_chain: ['神经元激活峰度攀升至 31.99', '过滤微弱激活(|act|<0.1)后依然保留41.4%特征', '专家神经元领域重叠率高达 100%'],
                    result: 'MLP 层涌现极端尖峰重尾分布，证实知识存放于高特异化、极度稀疏的专家神经元中。',
                    agi_significance: '彻底打破知识密集混合存储的预设，证实极致稀疏编码是高能效认知的前提。',
                    analysis: '证实庞大参数量并不运算所有信息，而是构建庞大字典，单次只激活极简路径。',
                    current_gap: '已完成理论验证，现需将此稀疏结构提取到专属芯片形态中。',
                    params: { focus: "MLP Activation", threshold: "|act| < 0.1", metrics: "Kurtosis, Overlap" },
                    details: { kurtosis: 31.99, feature_retention: "41.43%", overlap_in_domain: "100%", cross_domain: "47%" }
                },
                {
                    id: 'E2',
                    name: '逐层残差增量 SVD',
                    target: '量化信息在深浅层之间的传递动力学，寻找语义逐层抽象的数学证据。',
                    testDate: '2026-02-26',
                    evidence_chain: ['层级方差深/浅比压缩至 0.18x', '浅层 L0 增量达 3.25，深层增量约 0.60', '整体结构呈显著沙漏形（Hourglass）'],
                    result: '浅层负责巨量方差生成与特征扩张，深层进行微小方差平滑与收敛刻画。',
                    agi_significance: '揭示了 AGI 网络提取语义规律的“浅层大开大合，深层收敛定调”的拓扑雕刻过程。',
                    analysis: '对偶验证了理论中“特征在浅层生成干涉波谷，在深层滑向低维能量稳态流形”的抽象步骤。',
                    current_gap: '尚未在超大规模网络（如千亿参数）上独立证实此流形收敛轨迹，需要算力进一步验证。',
                    params: { method: "SVD", target: "Residual Stream", metric: "L0 Norm Increment, Variance Ratio" },
                    details: { deep_shallow_ratio: 0.18, shallow_delta_mean: 3.25, deep_delta_mean: 0.60, shape: "Hourglass" }
                }
            ]
        },
        {
            id: 'phase_2',
            title: '阶段二：功能组件算力维度压缩与角色分离',
            status: 'done',
            objective: '剥离注意力机制非必要参数，区分不同网络结构的“知识容器”与“逻辑骨架”定位。',
            summary: '确认 Attention 组件工作在极低维度空间且行为单调，确立了“满秩纤维存记忆，低秩流形走逻辑”的资源分配准则。',
            tests: [
                {
                    id: 'E3',
                    name: 'Attention 有效秩测量',
                    target: '剥离 Attention 机制的非必要参数，测定上下文绑定功能的实际所需理论维度。',
                    testDate: '2026-02-26',
                    evidence_chain: ['维度 11 空间中全局平均有效秩仅为 4.58', '高达 65.3% 的注意力头秩 < 5', '87.5% 的权重死锁于首词(BOS)的默认沉入区'],
                    result: 'Attention 并非密集搬运知识，其注意力头极度低秩、任务极简、工作模式单调。',
                    agi_significance: '明确了上下文绑定所需算力极低，为未来架构大幅裁剪 Attention 找到了数学支撑。',
                    analysis: '证实 Attention 只做少量神经流形的切换与桥接。此成果坚定了存算分离（Logic与Memory解耦）设计的正确性。',
                    current_gap: '需在纯代数代入层面试验裁剪后的低秩 Attention 是否依然能维持百万 Tokens 上下文长距关联。',
                    params: { target: "Attention Heads", metric: "Effective Rank", embedding_dim: 11 },
                    details: { avg_rank: 4.58, low_rank_ratio: "65.28%", bos_deadlock: "87.5%" }
                },
                {
                    id: 'E4',
                    name: '权重矩阵低秩分析',
                    target: '区分参数矩阵中“知识容器”与“逻辑骨架”的数学身份，为下一代 FiberNet 资源分配提供物理依据。',
                    testDate: '2026-02-27',
                    evidence_chain: ['MLP 权重满秩维度(Rank 95)均值约 600（占比约 75%）', 'Attention QK 矩阵满秩维度均值仅 52（占比 6.8%）'],
                    result: '出现了极具视觉冲击力的对比：MLP 是“极满秩”的信息黑洞，而 Attention 是极罕见的“极度低秩”拓扑算子。',
                    agi_significance: '清晰定义了 AGI 模型规模扩展法则（Scaling Law）：用满秩的纤维存记忆，用低秩的底流形走逻辑。',
                    analysis: '终结了传统网络“参数即能力”的粗糙论断，指出网络必然包含着更细分、结构化的高效功能分工。',
                    current_gap: '逻辑算子和记忆容器的物理硬件隔离尚处于图纸阶段，软件抽象模拟会带来一定开销。',
                    params: { target_1: "MLP Weights", target_2: "Attention QK", metric: "Rank at 95% Variance" },
                    details: { mlp_avg_rank_95: "~600", mlp_ratio: "~75%", qk_avg_rank_95: 52.0, qk_ratio: "6.8%" }
                }
            ]
        },
        {
            id: 'phase_3',
            title: '阶段三：泛化相变与局部涌现机制追踪',
            status: 'in_progress',
            objective: '捕捉模型从硬记忆到规律泛化（Grokking）的临界点，在无 BP 网络中重现底层稀疏编码格式。',
            summary: '首次将泛化具象为流形圆度（Circularity），并成功用局部侧抑制引发极其轻量的纯底层物理稀疏激活，但触发了新的信用分配挑战。',
            tests: [
                {
                    id: 'E5',
                    name: 'Z113 结构相变追踪',
                    target: '捕捉从“死记硬背”到“规律泛化”（Grokking）的临界点几何相变，破译泛化能力的涌现密码。',
                    testDate: '2026-02-27',
                    evidence_chain: ['网络空间表示拓扑圆度（Circularity）从 0.515 稳定跃升至 0.536', '随后泛化准确率开始从 0 突变并爬升'],
                    result: 'Grokking 不是盲目的性能突变，而是底层几何表征从记忆乱码重排形成了完美圆环状（代数群结构）。',
                    agi_significance: '首次将“泛化理解能力”具象化为一个可测量的黎曼流形指标，奠定几何优化智能实验的基础。',
                    analysis: '说明掌握规律必经拓扑重组期。验证了离线演化（睡眠）时施加 Ricci Flow 平滑处理可诱导网络物理结构的顿悟。',
                    current_gap: '主动诱导 Grokking 提前发生的方法论由于收敛机制非常微妙，尚未实现 100% 稳定再现。',
                    params: { dataset: "Z113 Modulo", tracking_metric: "Topological Circularity" },
                    details: { circularity_init: 0.515, circularity_stable: 0.536, gen_accuracy_emerging: true }
                },
                {
                    id: 'E6',
                    name: 'Emergence: 稀疏自发涌现',
                    target: '无 BP 条件下，验证纯局部的物理竞争法则能否激发出类似人类大脑的底层稀疏编码格式。',
                    testDate: '2026-02-27',
                    evidence_chain: ['仅用 5000 步的纯局部侧抑制结合 Hebbian 更新', '特征激活峰度激增，自发从随机的 2.70 爆升至 19.75'],
                    result: '无全局梯度下，成功爆发高度稀疏系统底座，但在更复杂维度下“专家化分工”程度稍显不足。',
                    agi_significance: '证实摈弃 BP 后只要引入物理互斥抑制机制，极效系统的基础编码底座同样也会自发稳健建立。',
                    analysis: '实验是一场惊险跳跃，成功证明稀疏结构不依赖全局梯度。但“专家化分工”的缺陷确立了下阶段构建预测编码（Predictive Coding）机制的高优地位。',
                    current_gap: '缺乏全局监督和精确向下的误差分摊机制（信用分配危机），导致在复杂数据集上聚类粒度太粗糙。',
                    params: { algorithm: "Lateral Inhibition + Hebbian", steps: 5000, constraint: "No BP Gradient" },
                    details: { initial_kurtosis: 2.70, final_kurtosis: 19.75, expert_specialization: "Insufficient for hard tasks" }
                }
            ]
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

                {/* 3. 测试记录 (E1~E6) - 多层阶段化展示 */}
                <div
                    style={{
                        padding: '16px',
                        borderRadius: '14px',
                        border: '1px solid rgba(168,85,247,0.24)',
                        background: 'linear-gradient(135deg, rgba(168,85,247,0.08) 0%, rgba(168,85,247,0.02) 100%)',
                        marginBottom: '18px',
                    }}
                >
                    <div style={{ color: '#a855f7', fontWeight: 'bold', fontSize: '15px', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Activity size={18} /> 三、测试记录
                    </div>
                    <div style={{ color: '#9ca3af', fontSize: '12px', lineHeight: '1.7', marginBottom: '16px' }}>
                        按核心探索阶段展开，查看解剖底层机理、捕捉数学结构到规律泛化的一系列历史完整试验点对。
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
                        {phasedTestRecords.map((phase) => {
                            const isPhaseExpanded = expandedPhase === phase.id;
                            const phaseStatusColor =
                                phase.status === 'done' ? '#10b981' : phase.status === 'in_progress' ? '#f59e0b' : '#94a3b8';
                            const phaseTestCount = (phase.tests || []).length;
                            return (
                                <div
                                    key={phase.id}
                                    style={{
                                        padding: '14px 16px',
                                        borderRadius: '12px',
                                        border: `1px solid ${isPhaseExpanded ? 'rgba(168,85,247,0.45)' : 'rgba(255,255,255,0.08)'}`,
                                        background: isPhaseExpanded ? 'rgba(168,85,247,0.08)' : 'rgba(255,255,255,0.02)',
                                    }}
                                >
                                    <button
                                        onClick={() => {
                                            const nextPhase = isPhaseExpanded ? null : phase.id;
                                            setExpandedPhase(nextPhase);
                                            setExpandedTestItem(null); // 收起阶段时同时重置内部测试项的展开状态
                                        }}
                                        style={{
                                            width: '100%',
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            alignItems: 'center',
                                            gap: '12px',
                                            marginBottom: isPhaseExpanded ? '10px' : 0,
                                            background: 'transparent',
                                            border: 'none',
                                            cursor: 'pointer',
                                            padding: 0,
                                            textAlign: 'left',
                                        }}
                                    >
                                        <div>
                                            <div style={{ color: '#f3e8ff', fontWeight: 'bold', fontSize: '14px' }}>{phase.title}</div>
                                            <div style={{ color: '#c084fc', fontSize: '11px', marginTop: '2px' }}>
                                                累计核心实验点：{phaseTestCount} 个
                                            </div>
                                        </div>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                            <div style={{ fontSize: '10px', color: phaseStatusColor }}>{String(phase.status).toUpperCase()}</div>
                                            <div style={{ fontSize: '11px', color: '#d8b4fe' }}>{isPhaseExpanded ? '收起' : '展开'}</div>
                                        </div>
                                    </button>

                                    {isPhaseExpanded && (
                                        <div>
                                            <div style={{ color: '#e9d5ff', fontSize: '12px', marginBottom: '6px' }}>阶段长线目标：{phase.objective}</div>
                                            <div style={{ color: '#d8b4fe', fontSize: '12px', lineHeight: '1.6', marginBottom: '12px' }}>
                                                核心验证总结：{phase.summary}
                                            </div>
                                            <div style={{ color: '#a855f7', fontSize: '12px', fontWeight: 'bold', marginBottom: '10px', display: 'flex', alignItems: 'center' }}>
                                                <ChevronDown size={14} style={{ marginRight: '4px' }} /> 实验探针列表（点击每块查看详细参数数据）
                                            </div>

                                            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
                                                {(phase.tests || []).map((test) => {
                                                    const testKey = `${phase.id}:${test.id}`;
                                                    const isTestExpanded = expandedTestItem === testKey;
                                                    const evidenceChain = Array.isArray(test.evidence_chain) ? test.evidence_chain : [];
                                                    const keyEvidenceText = evidenceChain.length > 0 ? evidenceChain.join('； ') : test.result;
                                                    return (
                                                        <div
                                                            key={test.id}
                                                            style={{
                                                                borderRadius: '10px',
                                                                border: `1px solid ${isTestExpanded ? 'rgba(192,132,252,0.5)' : 'rgba(255,255,255,0.08)'}`,
                                                                background: isTestExpanded ? 'rgba(88,28,135,0.2)' : 'rgba(0,0,0,0.18)',
                                                                padding: '12px',
                                                            }}
                                                        >
                                                            <button
                                                                onClick={() => setExpandedTestItem(isTestExpanded ? null : testKey)}
                                                                style={{
                                                                    width: '100%',
                                                                    background: 'transparent',
                                                                    border: 'none',
                                                                    cursor: 'pointer',
                                                                    padding: 0,
                                                                    textAlign: 'left',
                                                                    display: 'flex',
                                                                    justifyContent: 'space-between',
                                                                    alignItems: 'center',
                                                                    gap: '10px',
                                                                }}
                                                            >
                                                                <div style={{ color: '#e9d5ff', fontSize: '13px', fontWeight: 'bold' }}>
                                                                    {test.name}
                                                                </div>
                                                                <div style={{ color: '#c084fc', fontSize: '11px' }}>{isTestExpanded ? '收起详情' : '检视探针详情'}</div>
                                                            </button>

                                                            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '10px' }}>
                                                                <div style={{ color: '#d1d5db', fontSize: '12px', lineHeight: '1.6' }}>
                                                                    <span style={{ color: '#d8b4fe', fontWeight: 'bold' }}>测试目标：</span>{test.target}
                                                                </div>
                                                                <div style={{ color: '#d1d5db', fontSize: '12px', lineHeight: '1.6' }}>
                                                                    <span style={{ color: '#a7f3d0', fontWeight: 'bold' }}>关键证据：</span>{keyEvidenceText}
                                                                </div>
                                                                <div style={{ color: '#d1d5db', fontSize: '12px', lineHeight: '1.6' }}>
                                                                    <span style={{ color: '#67e8f9', fontWeight: 'bold' }}>深远意义：</span>{test.agi_significance}
                                                                </div>
                                                                <div style={{ color: '#d1d5db', fontSize: '12px', lineHeight: '1.6' }}>
                                                                    <span style={{ color: '#fca5a5', fontWeight: 'bold' }}>当前边界盲区：</span>{test.current_gap}
                                                                </div>
                                                            </div>

                                                            {isTestExpanded && (
                                                                <div
                                                                    style={{
                                                                        marginTop: '12px',
                                                                        borderRadius: '8px',
                                                                        border: '1px solid rgba(168,85,247,0.3)',
                                                                        background: 'rgba(2,6,23,0.55)',
                                                                        padding: '12px',
                                                                    }}
                                                                >
                                                                    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(80px, auto) 1fr', gap: '8px', fontSize: '11px', lineHeight: '1.5' }}>
                                                                        <div style={{ color: '#c084fc', fontWeight: 'bold' }}>执行日期:</div><div style={{ color: '#9ca3af' }}>{test.testDate}</div>
                                                                        <div style={{ color: '#c084fc', fontWeight: 'bold' }}>实验结论:</div><div style={{ color: '#e2e8f0' }}>{test.result}</div>
                                                                        <div style={{ color: '#c084fc', fontWeight: 'bold' }}>推演总结:</div><div style={{ color: '#e2e8f0' }}>{test.analysis}</div>
                                                                    </div>

                                                                    <div style={{ color: '#d8b4fe', fontSize: '11px', fontWeight: 'bold', marginTop: '14px', marginBottom: '6px' }}>
                                                                        调控参数矩阵 (Params)
                                                                    </div>
                                                                    <pre style={{ margin: 0, color: '#a1a1aa', fontSize: '11px', lineHeight: '1.5', whiteSpace: 'pre-wrap', background: 'rgba(0,0,0,0.3)', padding: '6px', borderRadius: '4px' }}>
                                                                        {JSON.stringify(test.params, null, 2)}
                                                                    </pre>

                                                                    <div style={{ color: '#d8b4fe', fontSize: '11px', fontWeight: 'bold', marginTop: '14px', marginBottom: '6px' }}>
                                                                        输出细粒度游标 (Details)
                                                                    </div>
                                                                    <pre style={{ margin: 0, color: '#a1a1aa', fontSize: '11px', lineHeight: '1.5', whiteSpace: 'pre-wrap', background: 'rgba(0,0,0,0.3)', padding: '6px', borderRadius: '4px' }}>
                                                                        {JSON.stringify(test.details, null, 2)}
                                                                    </pre>
                                                                </div>
                                                            )}
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            );
                        })}
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
