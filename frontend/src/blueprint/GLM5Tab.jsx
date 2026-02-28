import React, { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

export const GLM5Tab = () => {
    const [expandedSteps, setExpandedSteps] = useState({});
    const [expandedTest, setExpandedTest] = useState(null);

    const toggleStep = (idx) => {
        setExpandedSteps(prev => ({
            ...prev,
            [idx]: !prev[idx]
        }));
    };

    const toggleTest = (testId) => {
        setExpandedTest(expandedTest === testId ? null : testId);
    };

    // 分析框架 - 核心问题与方法论
    const coreQuestion = '神经网络如何从信号流中提取特征、形成编码？';
    const coreInsight = '这是一切能力的基石。知识层次网络结构、高效读写、意识统一处理，都是建立在这个编码机制之上。';

    const researchPrinciples = [
        '不预设理论正确：可能的所有错误预设（稀疏编码是关键、纤维丛是数学基础等）都可能只是现象而非原因',
        '聚焦核心问题：一切围绕"特征如何提取、编码如何形成"这一核心问题',
        '方法论转变：从"理论假设→实验验证"转向"观察→记录→模式识别→假说→再观察"',
        '让结构自然浮现：先收集足够多的观察，完成拼图，再尝试还原',
    ];

    const problemChain = [
        '一切都是基石：神经网络如何从信号流中提取特征、形成编码？',
        '在此之上形成：特征如何形成层级结构？',
        '在此之上形成：层级结构如何支持抽象与精确并存？',
        '在此之上形成：不同模态如何统一编码？',
        '最终涌现：AGI能力',
    ];

    // 线路图 - 五个研究阶段
    const roadmapPhases = [
        {
            id: 'Phase 1',
            name: '特征涌现追踪',
            status: 'in_progress',
            time: '1-2月',
            objective: '观察"特征如何从无到有形成"',
            details: '从随机初始化开始训练模型，每100步记录激活分布变化、稀疏度变化、特征方向变化、输出质量变化。识别关键转变点，分析训练过程中的特征涌现时间线。关键问题：特征是什么时候出现的？第一个"有意义"的特征是什么？特征如何分化、组合？涌现是否有阶段性？预期获得：特征涌现的时间线、涌现的临界条件、涌现的顺序规律。'
        },
        {
            id: 'Phase 2',
            name: '编码基本单位分析',
            status: 'pending',
            time: '2-3月',
            objective: '找到编码的"原子"',
            details: '分析单个神经元 vs 神经元群体，分析最小可解释单元，分析特征组合规则。关键问题：一个特征对应多少神经元？特征是连续还是离散？特征之间如何组合？有没有"基本特征集"？通过实验确定编码的最小单位，理解特征如何组合形成复杂概念。'
        },
        {
            id: 'Phase 3',
            name: '稀疏性与正交性机制',
            status: 'pending',
            time: '2-3月',
            objective: '理解"为什么是这些值"',
            details: '深入分析稀疏度和正交性的决定因素。关键问题：稀疏度由什么决定？为什么DNN稳定在78%而不是50%或90%？正交性有什么功能？能否强制改变这些值？通过干预实验，理解这些数值背后的机制，探索它们对系统性能的影响。'
        },
        {
            id: 'Phase 4',
            name: '层级结构形成',
            status: 'pending',
            time: '3-4月',
            objective: '理解特征层级如何形成',
            details: '研究特征如何从低级到高级形成层级结构。分析不同层级的特征表示，理解抽象概念如何在更高层级涌现。关键问题：层级结构是预设的还是涌现的？不同层级之间如何交互？高层特征如何组合低层特征？这与大脑皮层的层级结构有何相似之处？'
        },
        {
            id: 'Phase 5',
            name: '大脑对比验证',
            status: 'pending',
            time: '持续',
            objective: '验证机制是否存在于大脑',
            details: '将DNN中发现的各种机制与大脑神经科学数据进行对比验证。需要神经科学数据（fMRI、单细胞记录等）支持。关键对比：稀疏度（大脑~2% vs DNN 78%）、能效（大脑20W vs GPU 300W+）、编码方式、层级结构。验证DNN机制是否真的存在于大脑，还是只是数学巧合。'
        },
    ];

    // 测试记录 - 本机实际测试数据
    const testRecords = [
        {
            id: 'test-000',
            name: 'AGI研究进展报告 - 方向调整',
            testDate: '2026-02-28',
            status: 'completed',
            objective: '总结当前AGI研究进展、存在问题，并制定下一步核心工作计划',
            result: '完成GLM5路线方向调整，确立"特征涌现追踪"为核心任务，明确方法论从"假设驱动"转向"观察驱动"',
            keyEvidence: [
                '总体进度: Gemini 50%, GPT5 10%, GLM5 5%',
                '关键发现: 78%稀疏度、97%正交性、Grokking现象',
                '核心问题: 特征如何从信号流中提取、编码如何形成',
                '方法论转变: 从"理论假设→实验验证"转向"观察→记录→模式识别→假说→再观察"',
                '优先级: P0 特征涌现追踪 → P1 编码基本单位 → P2 稀疏度机制 → P3 大脑数据获取'
            ],
            agiSignificance: '明确了AGI研究的核心瓶颈：不知道特征如何涌现，就无法理解智能的本质。确立了"不预设理论、先观察后假说"的研究方法论。预计乐观情况下2-3年可实现初步AGI原型。',
            analysisSummary: '报告系统梳理了三条路线的研究进展，识别出三大致命问题（核心机制未知、特征涌现过程缺失、编码基本单位未定义），并提出以"特征涌现追踪"为第一优先级的行动方案。关键洞察：大脑是自下而上的系统，需要通过海量数据冲刷逐步形成稳定系统。',
            params: {
                research_lines: 'Gemini(DNN分析) + GPT5(大脑机制) + GLM5(特征涌现)',
                current_phase: 'Phase 1 特征涌现追踪',
                key_milestones: '特征涌现时间线 → 编码基本单位 → 稀疏度机制',
                estimated_timeline: '乐观2-3年，保守5年以上'
            },
            details: {
                key_problems: [
                    '核心机制完全未知（知道是什么，不知道为什么）',
                    '特征涌现过程完全缺失（只知道结果，不知道过程）',
                    '编码基本单位未定义（特征对应多少神经元？）',
                    '理论框架可能完全错误',
                    '神经科学数据严重缺失'
                ],
                action_items: [
                    '开发特征涌现追踪工具',
                    '从随机初始化开始训练，每100步记录',
                    '识别关键转变点',
                    '绘制特征涌现时间线'
                ],
                brain_dnn_gap: {
                    energy: '大脑20W vs GPU 300W+ (15倍)',
                    sparsity: '大脑~2% vs DNN~78% (40倍)',
                    neurons: '大脑1e11 vs DNN 1e9 (100倍)',
                    learning: '在线学习 vs 离线训练'
                }
            }
        },
        {
            id: 'test-000b',
            name: '特征涌现追踪实验 - 3000步训练观察',
            testDate: '2026-02-28',
            status: 'completed',
            objective: '从随机初始化开始训练，观察特征如何涌现',
            result: '发现"压缩-重组"模式：深层有效秩在训练初期急剧下降（压缩），然后逐渐恢复（重组）。稀疏度稳定在6.3%，远低于GPT-2的78%。',
            keyEvidence: [
                '有效秩压缩: Layer 3 从54→6→42 (压缩-重组模式)',
                '稀疏度稳定: 6.2%-6.4% (与GPT-2的78%差异巨大)',
                '激活范数增长: 深层增长18%，特征表示更强',
                '涌现顺序: 浅层→深层（符合预期）',
                '训练阶段: Step 0-500压缩期 → Step 500-1500重组期 → Step 1500+稳定期'
            ],
            agiSignificance: '首次观察到特征涌现的"压缩-重组"模式。这可能是特征形成的核心机制：训练初期信息被"浓缩"，然后在新结构中"重组"。但稀疏度差异(6% vs 78%)提示：模型规模和数据类型可能显著影响涌现模式。',
            analysisSummary: '4层Transformer、79万参数、3000步训练，运行时间19.1秒。观察到深层有效秩剧烈波动（压缩-重组），而浅层保持稳定。稀疏度在整个训练过程中保持稳定。核心问题：稀疏度为何与GPT-2差异如此大？有效秩压缩是涌现的必要条件吗？',
            params: {
                model: 'SimpleTransformer',
                layers: 4,
                dimensions: 128,
                parameters: '790,272',
                training_steps: 3000,
                tracking_interval: 100,
                device: 'CUDA (GPU)',
                runtime: '19.1秒'
            },
            details: {
                effective_rank_pattern: {
                    layer_0: '57.7 → 57.5 → 59.5 (稳定)',
                    layer_1: '56.8 → 36.1 → 45.0 (压缩36%)',
                    layer_2: '55.8 → 9.7 → 40.4 (压缩83%)',
                    layer_3: '54.2 → 6.1 → 42.4 (压缩89%)'
                },
                sparsity_observation: '稳定在6.2%-6.4%，远低于GPT-2的78%',
                norm_growth: '深层增长更多 (Layer 3: +18%)',
                emergence_timeline: 'Step 0: 所有层同时检测到特征维度涌现',
                open_questions: [
                    '为什么稀疏度(6%)与GPT-2(78%)差异如此大？',
                    '有效秩压缩是涌现的必要条件吗？',
                    '小模型结果能否推广到大模型？'
                ],
                next_steps: [
                    '使用真实文本数据重新运行',
                    '扩大模型规模观察涌现模式变化',
                    '追踪特征方向而非仅追踪统计量'
                ]
            }
        },
        {
            id: 'test-001',
            name: 'GPT-2特征提取与稀疏编码分析',
            testDate: '2026-02-21',
            status: 'completed',
            objective: '分析GPT-2模型不同层的特征编码特性，测量稀疏度、正交性等关键指标',
            result: '成功识别出78%的L0稀疏度和97%的正交性，Layer 6表现最佳（抽象能力ratio=1.07，精确度80%）',
            keyEvidence: [
                '特征稀疏度稳定在78.2-78.3%之间（Layer 0-11）',
                '正交性得分稳定在97.0-97.1%之间',
                'Layer 6: 抽象能力ratio=1.07（类别区分度最佳）',
                'Layer 11: 抽象能力ratio=1.11（最高层抽象）',
                '大脑机制推断：稀疏编码的神经元阈值机制、高维神经空间的天然正交性'
            ],
            agiSignificance: '发现了DNN编码的基本特性（78%稀疏、97%正交），为理解神经网络如何编码信息提供了关键拼图。这些数值可能是神经网络涌现能力的必要条件，对AGI架构设计有重要指导意义。如果大脑也采用类似编码方式，则可解释其高效的能效比。',
            analysisSummary: '实验验证了DNN特征编码的稳定特性：高稀疏度（78%）和高正交性（97%）。这些特性在不同层保持稳定，暗示这是神经网络训练的自然收敛结果。关键问题是：为什么是这个值？这需要更深入的理论分析和对比实验。大脑推断提出了5个可验证的神经科学假设。',
            params: {
                model: 'gpt2-small',
                num_samples: 250,
                sae_latent_dim: 2048,
                sparsity_penalty: 0.05,
                layers_analyzed: [0, 3, 6, 9, 11]
            },
            details: {
                sparsity_range: '78.18% - 78.28%',
                orthogonality_range: '97.05% - 97.10%',
                best_layer: 6,
                abstraction_improvement: '10% from layer 0 to 11',
                precision_improvement: '40% from layer 0 to 11'
            }
        },
        {
            id: 'test-002',
            name: 'Z_113模加法训练与Grokking现象',
            testDate: '2026-02-21',
            status: 'completed',
            objective: '验证神经网络在已知结构任务（模加法）上的学习过程，观察能力涌现的临界点',
            result: '训练精度达到100%，测试精度最终达到99.48%。观察到明显的Grokking现象：训练早期测试精度接近0%，在训练后期突然涌现能力。',
            keyEvidence: [
                '训练精度: 0.8% → 100%（Epoch 0-14）',
                '测试精度: 0.01% → 99.48%（Epoch 0-2000）',
                '关键转折点: Epoch 1300附近，测试精度从0.14%突增',
                '完全泛化: Epoch 1700后测试精度超过90%',
                '涌现模式: 先记忆训练集（Epoch 14达100%训练精度），后泛化到测试集（Epoch 2000达99.48%测试精度）'
            ],
            agiSignificance: '证明了神经网络存在"先记忆、后泛化"的学习阶段。这种延迟涌现的能力暗示：智能系统需要先建立内部表示，才能实现真正的理解。这对AGI训练策略有重要启示：可能需要足够的训练时间和数据量，才能触发能力的涌现。大脑发育也可能存在类似的"关键期"。',
            analysisSummary: 'Grokking现象验证了神经网络训练的两个阶段假说：第一阶段快速拟合训练数据，第二阶段逐步形成泛化能力。关键发现：即使训练精度达到100%，模型仍可能在测试集上表现不佳，直到训练足够长时间。这挑战了"过拟合"的传统观念，暗示神经网络在学习更深层的结构。',
            params: {
                task: 'Z_113 modular addition',
                model: 'Transformer (2 layers, 64 dim)',
                training_epochs: 2000,
                dataset_size: '12769 samples (70% train, 30% test)',
                optimizer: 'AdamW',
                learning_rate: 0.001
            },
            details: {
                train_acc_start: 0.0083,
                train_acc_peak: 1.0,
                test_acc_start: 0.0067,
                test_acc_final: 0.9948,
                grokking_start_epoch: 1300,
                full_generalization_epoch: 1700
            }
        },
        {
            id: 'test-003',
            name: 'Adaptive FiberNet对比实验框架',
            testDate: '2026-02-21',
            status: 'prepared',
            objective: '对比三种架构（标准Transformer、FiberNet固定几何、FiberNet可学习几何）在模运算任务上的表现，验证几何先验的作用',
            result: '实验代码已准备就绪，待运行。预计可学习几何的FiberNet应更快收敛。',
            keyEvidence: [
                '实验框架: 对比Standard Transformer vs FiberNet (circle) vs Adaptive FiberNet (learnable)',
                '数据集: Z_113模加法 + 多位数加法',
                '评估指标: 最终精度、训练时间、收敛速度、结构信息',
                '预期结果: 几何先验应加速收敛，可学习几何应达到最优性能'
            ],
            agiSignificance: '验证"几何结构是否是智能的必要条件"这一核心假设。如果几何先验确实有效，则说明大脑可能利用了类似的几何结构来实现智能。这将为AGI架构设计提供理论基础：不是盲目增加参数，而是设计合适的几何先验。',
            analysisSummary: '实验设计完成，代码框架已就绪。核心假设：如果几何结构是智能的关键，那么引入几何先验应该显著提升学习效率。下一步：运行完整实验，对比三种架构在收敛速度、最终性能、泛化能力上的差异。特别关注Adaptive FiberNet是否能自动学习到有意义的几何结构。',
            params: {
                architectures: ['Standard Transformer', 'FiberNet (circle)', 'Adaptive FiberNet (learnable)'],
                d_model: 64,
                n_heads: 4,
                n_layers: 2,
                epochs: 100,
                batch_size: 64
            },
            details: {
                dataset_1: 'ModularArithmeticDataset (Z_113)',
                dataset_2: 'MultiDigitAdditionDataset (3-digit)',
                comparison_metrics: ['final_accuracy', 'training_time', 'convergence_speed'],
                structure_tracking: '每10 epoch记录结构信息'
            }
        }
    ];

    // 存在问题 - 缺失的拼图块
    const keyProblems = [
        '特征如何在训练中涌现？只知道结果，不知道过程',
        '为什么稀疏度是78%而不是其他值？没有理论解释',
        '编码的"基本单位"是什么？单个神经元还是神经元群体？',
        '大脑的编码与DNN有何本质不同？缺乏神经科学数据',
        '局部可塑性如何产生全局稳态？不知道自组织机制',
        '特异性是如何实现的？四大特性之一缺失解释',
        '系统性是如何实现的？四大特性之一缺失解释',
    ];

    // 接下来的核心工作 - Phase 1详细计划
    const nextSteps = [
        {
            priority: 'P0',
            task: '开发特征涌现追踪工具',
            detail: '从随机初始化开始训练，每100步记录激活分布、稀疏度、特征方向、输出质量变化',
        },
        {
            priority: 'P0',
            task: '识别关键转变点',
            detail: '分析训练过程中的特征涌现时间线，找出特征出现、分化、组合的关键节点',
        },
        {
            priority: 'P0',
            task: '回答关键问题',
            detail: '特征是什么时候出现的？第一个"有意义"的特征是什么？涌现是否有阶段性？',
        },
        {
            priority: 'P1',
            task: '绘制特征涌现时间线',
            detail: '可视化特征从无到有的形成过程，建立涌现的临界条件和顺序规律',
        },
        {
            priority: 'P1',
            task: '准备Phase 2实验',
            detail: '设计编码基本单位分析实验，确定最小可解释单元的分析方法',
        },
    ];

    const statusTextMap = {
        done: '已完成',
        in_progress: '进行中',
        pending: '待开始',
        completed: '已完成',
        prepared: '已准备'
    };

    const statusColorMap = {
        done: '#10b981',
        in_progress: '#f59e0b',
        pending: '#94a3b8',
        completed: '#10b981',
        prepared: '#3b82f6'
    };

    return (
        <div style={{ display: 'grid', gap: '20px' }}>
            <div
                style={{
                    padding: '30px',
                    borderRadius: '24px',
                    border: '1px solid rgba(168,85,247,0.28)',
                    background: 'linear-gradient(135deg, rgba(168,85,247,0.10) 0%, rgba(168,85,247,0.03) 100%)',
                    marginBottom: '28px',
                }}
            >
                {/* 标题区 */}
                <div style={{ color: '#a855f7', fontWeight: 'bold', fontSize: '18px', marginBottom: '8px' }}>
                    GLM5 路线：特征涌现与编码机制
                </div>
                <div style={{ color: '#e9d5ff', fontSize: '13px', lineHeight: '1.7', marginBottom: '20px' }}>
                    当前进度：5% | 状态：方向调整
                </div>

                {/* 一、分析框架 */}
                <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#f3e8ff', marginBottom: '10px', borderBottom: '1px solid rgba(168,85,247,0.35)', paddingBottom: '8px' }}>
                    一、分析框架
                </div>

                <div style={{ color: '#e9d5ff', fontSize: '13px', fontWeight: 'bold', marginBottom: '8px', marginTop: '12px' }}>
                    核心问题
                </div>
                <div style={{ padding: '12px', background: 'rgba(0,0,0,0.22)', borderRadius: '10px', marginBottom: '12px' }}>
                    <div style={{ color: '#c084fc', fontSize: '13px', fontWeight: 'bold', marginBottom: '6px' }}>{coreQuestion}</div>
                    <div style={{ color: '#d8b4fe', fontSize: '12px', lineHeight: '1.6' }}>{coreInsight}</div>
                </div>

                <div style={{ color: '#e9d5ff', fontSize: '13px', fontWeight: 'bold', marginBottom: '8px' }}>
                    研究原则
                </div>
                <div style={{ display: 'grid', gap: '6px', marginBottom: '12px' }}>
                    {researchPrinciples.map((item, idx) => (
                        <div key={idx} style={{ color: '#e9d5ff', fontSize: '12px', lineHeight: '1.6' }}>
                            {idx + 1}. {item}
                        </div>
                    ))}
                </div>

                <div style={{ color: '#e9d5ff', fontSize: '13px', fontWeight: 'bold', marginBottom: '8px' }}>
                    问题链
                </div>
                <div style={{ display: 'grid', gap: '6px', marginBottom: '18px' }}>
                    {problemChain.map((item, idx) => (
                        <div key={idx} style={{ color: '#d8b4fe', fontSize: '12px', lineHeight: '1.6' }}>
                            {idx + 1}. {item}
                        </div>
                    ))}
                </div>

                {/* 二、线路图 */}
                <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#f3e8ff', marginBottom: '10px', borderBottom: '1px solid rgba(168,85,247,0.35)', paddingBottom: '8px' }}>
                    二、线路图
                </div>
                <div style={{ display: 'grid', gap: '12px', marginBottom: '18px' }}>
                    {roadmapPhases.map((item, idx) => (
                        <div
                            key={item.id}
                            onClick={() => toggleStep(idx)}
                            style={{
                                padding: '16px',
                                background: 'rgba(0,0,0,0.4)',
                                borderRadius: '10px',
                                borderLeft: `3px solid ${statusColorMap[item.status] || '#94a3b8'}`,
                                cursor: 'pointer',
                                transition: 'all 0.2s ease',
                                userSelect: 'none'
                            }}
                        >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                    <div style={{ color: '#fff', fontSize: '15px', fontWeight: 'bold' }}>{item.id}</div>
                                    <div style={{
                                        padding: '2px 8px',
                                        borderRadius: '12px',
                                        background: statusColorMap[item.status] === '#10b981' ? 'rgba(16,185,129,0.1)' : statusColorMap[item.status] === '#f59e0b' ? 'rgba(245,158,11,0.1)' : 'rgba(148,163,184,0.1)',
                                        color: statusColorMap[item.status] || '#94a3b8',
                                        fontSize: '11px',
                                        border: statusColorMap[item.status] === '#10b981' ? '1px solid rgba(16,185,129,0.3)' : statusColorMap[item.status] === '#f59e0b' ? '1px solid rgba(245,158,11,0.3)' : '1px solid rgba(148,163,184,0.3)'
                                    }}>
                                        {statusTextMap[item.status] || '待开始'}
                                    </div>
                                    <div style={{ color: '#c084fc', fontSize: '11px' }}>{item.time}</div>
                                </div>
                                {expandedSteps[idx] ? <ChevronDown size={18} color="#9ca3af" /> : <ChevronRight size={18} color="#9ca3af" />}
                            </div>
                            <div style={{ color: '#d8b4fe', fontSize: '13px', marginTop: '8px' }}>{item.name}</div>

                            {expandedSteps[idx] && (
                                <div style={{
                                    marginTop: '16px',
                                    paddingTop: '16px',
                                    borderTop: '1px dashed rgba(255,255,255,0.1)',
                                    color: '#a1a1aa',
                                    fontSize: '13px',
                                    lineHeight: '1.6'
                                }}>
                                    <div style={{ color: '#c084fc', fontWeight: 'bold', marginBottom: '8px' }}>阶段目标：{item.objective}</div>
                                    <div style={{ color: '#d1d5db' }}>{item.details}</div>
                                </div>
                            )}
                        </div>
                    ))}
                </div>

                {/* 三、测试记录 */}
                <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#f3e8ff', marginBottom: '10px', borderBottom: '1px solid rgba(168,85,247,0.35)', paddingBottom: '8px' }}>
                    三、测试记录
                </div>
                <div
                    style={{
                        padding: '16px',
                        borderRadius: '14px',
                        border: '1px solid rgba(16,185,129,0.24)',
                        background: 'linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.02) 100%)',
                        marginBottom: '18px',
                    }}
                >
                    <div style={{ color: '#10b981', fontWeight: 'bold', fontSize: '14px', marginBottom: '6px' }}>本机测试记录</div>
                    <div style={{ color: '#9ca3af', fontSize: '12px', lineHeight: '1.7', marginBottom: '12px' }}>
                        点击查看详细测试数据、关键证据和对AGI的意义分析。
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
                        {testRecords.map((test, idx) => (
                            <div
                                key={test.id}
                                style={{
                                    borderRadius: '10px',
                                    border: `1px solid ${expandedTest === test.id ? 'rgba(96,165,250,0.5)' : 'rgba(255,255,255,0.08)'}`,
                                    background: expandedTest === test.id ? 'rgba(30,64,175,0.12)' : 'rgba(0,0,0,0.18)',
                                    padding: '10px 12px',
                                }}
                            >
                                <button
                                    onClick={() => toggleTest(test.id)}
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
                                    <div style={{ color: '#dbeafe', fontSize: '12px', fontWeight: 'bold' }}>
                                        T{idx + 1}. {test.name}
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                        <div style={{ fontSize: '10px', color: statusColorMap[test.status] || '#94a3b8' }}>
                                            {statusTextMap[test.status] || test.status}
                                        </div>
                                        <div style={{ color: '#93c5fd', fontSize: '11px' }}>
                                            {expandedTest === test.id ? '收起详情' : '查看详情'}
                                        </div>
                                    </div>
                                </button>

                                <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.6', marginTop: '6px' }}>
                                    测试日期：{test.testDate}
                                </div>
                                <div style={{ color: '#93c5fd', fontSize: '12px', lineHeight: '1.6', marginTop: '4px' }}>
                                    测试目标：{test.objective}
                                </div>
                                <div style={{ color: '#a7f3d0', fontSize: '12px', lineHeight: '1.6', marginTop: '4px' }}>
                                    测试结果：{test.result}
                                </div>

                                {expandedTest === test.id && (
                                    <div
                                        style={{
                                            marginTop: '8px',
                                            borderRadius: '8px',
                                            border: '1px solid rgba(148,163,184,0.35)',
                                            background: 'rgba(2,6,23,0.55)',
                                            padding: '10px',
                                        }}
                                    >
                                        <div style={{ color: '#bfdbfe', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>
                                            关键证据
                                        </div>
                                        <div style={{ color: '#dbeafe', fontSize: '11px', lineHeight: '1.6', marginBottom: '10px' }}>
                                            {test.keyEvidence.map((evidence, i) => (
                                                <div key={i} style={{ marginBottom: '4px' }}>• {evidence}</div>
                                            ))}
                                        </div>

                                        <div style={{ color: '#bfdbfe', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>
                                            对AGI的意义
                                        </div>
                                        <div style={{ color: '#d1d5db', fontSize: '11px', lineHeight: '1.7', marginBottom: '10px' }}>
                                            {test.agiSignificance}
                                        </div>

                                        <div style={{ color: '#bfdbfe', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>
                                            分析总结
                                        </div>
                                        <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.7', marginBottom: '10px' }}>
                                            {test.analysisSummary}
                                        </div>

                                        <div style={{ color: '#bfdbfe', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>
                                            测试参数
                                        </div>
                                        <pre
                                            style={{
                                                margin: 0,
                                                color: '#dbeafe',
                                                fontSize: '11px',
                                                lineHeight: '1.6',
                                                whiteSpace: 'pre-wrap',
                                                marginBottom: '10px',
                                            }}
                                        >
                                            {JSON.stringify(test.params, null, 2)}
                                        </pre>

                                        <div style={{ color: '#bfdbfe', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>
                                            详细测试数据
                                        </div>
                                        <pre
                                            style={{
                                                margin: 0,
                                                color: '#cbd5e1',
                                                fontSize: '11px',
                                                lineHeight: '1.6',
                                                whiteSpace: 'pre-wrap',
                                            }}
                                        >
                                            {JSON.stringify(test.details, null, 2)}
                                        </pre>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                {/* 四、存在问题 */}
                <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#fca5a5', marginBottom: '10px', borderBottom: '1px solid rgba(248,113,113,0.35)', paddingBottom: '8px' }}>
                    四、存在问题
                </div>
                <div style={{ display: 'grid', gap: '6px', marginBottom: '18px' }}>
                    {keyProblems.map((item, idx) => (
                        <div key={idx} style={{ color: '#fecaca', fontSize: '12px', lineHeight: '1.6' }}>
                            {idx + 1}. {item}
                        </div>
                    ))}
                </div>

                {/* 五、接下来的核心工作 */}
                <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#86efac', marginBottom: '10px', borderBottom: '1px solid rgba(74,222,128,0.35)', paddingBottom: '8px' }}>
                    五、接下来的核心工作
                </div>
                <div style={{ display: 'grid', gap: '8px' }}>
                    {nextSteps.map((item, idx) => (
                        <div
                            key={idx}
                            style={{
                                padding: '12px',
                                borderRadius: '10px',
                                border: '1px solid rgba(255,255,255,0.08)',
                                background: 'rgba(0,0,0,0.18)',
                            }}
                        >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                                <div style={{ color: '#dcfce7', fontWeight: 'bold', fontSize: '13px' }}>{item.task}</div>
                                <div style={{ fontSize: '10px', color: item.priority === 'P0' ? '#f59e0b' : '#94a3b8' }}>
                                    {item.priority}
                                </div>
                            </div>
                            <div style={{ color: '#a7f3d0', fontSize: '12px', lineHeight: '1.6' }}>
                                {item.detail}
                            </div>
                        </div>
                    ))}
                </div>

                {/* 核心洞察 */}
                <div
                    style={{
                        marginTop: '24px',
                        padding: '16px',
                        borderRadius: '12px',
                        border: '1px solid rgba(168,85,247,0.35)',
                        background: 'linear-gradient(135deg, rgba(168,85,247,0.12) 0%, rgba(168,85,247,0.04) 100%)',
                    }}
                >
                    <div style={{ color: '#c084fc', fontWeight: 'bold', fontSize: '14px', marginBottom: '8px' }}>核心洞察</div>
                    <div style={{ color: '#e9d5ff', fontSize: '12px', lineHeight: '1.7' }}>
                        大脑是自下而上的系统。每个神经元只根据前级信号进行充电和放电，没有中央设计者。通过海量数据冲刷 + 神经可塑性，逐步形成稳定系统。
                    </div>
                    <div style={{ color: '#d8b4fe', fontSize: '12px', lineHeight: '1.7', marginTop: '8px' }}>
                        关键问题：神经网络是如何从信号流中提取特征并形成编码的？这是一切能力的基石。不要预设答案，先观察，再假说，完成拼图，让结构自然浮现。
                    </div>
                </div>
            </div>
        </div>
    );
};
