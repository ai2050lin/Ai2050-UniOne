import React from 'react';
import { Brain } from 'lucide-react';

export const ClaudeTab = () => {
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
                    <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#e9d5ff', marginBottom: '12px', borderBottom: '1px solid rgba(168,85,247,0.3)', paddingBottom: '8px' }}>一、整体研究框架与核心进展</div>
                    <div style={{ color: '#d1d5db', fontSize: '13px', lineHeight: '1.7' }}>
                        构建基于微分几何、神经纤维丛拓扑（NFBT）和纯代数演化的智能引擎（Mother Engine），抛弃传统 BP 黑盒与堆叠算力路线。<br />
                        <span style={{ color: '#a855f7', fontWeight: 'bold' }}>进展突破: </span>建立“极效三定律”（侧抑制正交、引力雕刻、能量坍塌）；通过解剖 DNN 证实大脑的激活稀疏性编码方式；发现 Attention 的极低秩关联拓扑；在无 BP 下利用局部规则实现空白网络自发涌现稀疏特征（峰度激增至 19.7）。
                    </div>
                </div>

                {/* 2. 完整路线图 */}
                <div style={{ marginBottom: '28px' }}>
                    <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#e9d5ff', marginBottom: '12px', borderBottom: '1px solid rgba(168,85,247,0.3)', paddingBottom: '8px' }}>二、路线图全景 </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                        {[
                            { title: "H1", status: "已完成", desc: "理论奠基与小规模实证、极效三定律、可视化并网" },
                            { title: "H2", status: "攻坚期", desc: "深度解剖化石与局部学习机制攻坚，信用分配突围" },
                            { title: "H3", status: "中期", desc: "跨模态统一，大规模语义涌现，百万Token连贯性" },
                            { title: "H4", status: "远期", desc: "脱离冯·诺依曼架构，神经形态芯片，可控 AGI 原型" },
                        ].map((step, idx) => (
                            <div key={idx} style={{ padding: '12px', background: 'rgba(0,0,0,0.4)', borderRadius: '10px', borderTop: step.status === '已完成' ? '2px solid #10b981' : '2px solid #a855f7' }}>
                                <div style={{ color: '#fff', fontSize: '13px', fontWeight: 'bold' }}>{step.title}</div>
                                <div style={{ color: step.status === '已完成' ? '#10b981' : '#a855f7', fontSize: '11px', marginBottom: '6px' }}>[{step.status}]</div>
                                <div style={{ color: '#9ca3af', fontSize: '11px', lineHeight: '1.5' }}>{step.desc}</div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* 3. 测试记录 (E1~E5) */}
                <div style={{ marginBottom: '28px' }}>
                    <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#e9d5ff', marginBottom: '12px', borderBottom: '1px solid rgba(168,85,247,0.3)', paddingBottom: '8px' }}>三、化石解剖与自发涌现测试记录 (Phase XXXVI)</div>
                    <div style={{ display: 'grid', gap: '16px' }}>
                        {[
                            { name: 'E1: MLP 稀疏激活解剖', metrics: '峰度 31.99, |act|<0.1 占 41.4%', result: '发现极端尖峰重尾分布，专家神经元领域重叠100%，跨领域仅47%', sig: '证实大脑"知识"存放在高特异化、极度稀疏的专家神经元中，而非密集分布。' },
                            { name: 'E2: 逐层残差增量 SVD', metrics: '深/浅比 = 0.18x，浅层增量3.25，深层0.6', result: 'L0增量庞大，L8极小，呈沙漏模式；浅层方差集中于Top1', sig: '验证“浅层做粗颗粒特征抽象重构，深层做低维精确微调”。' },
                            { name: 'E3: Attention 有效秩测量', metrics: '全局平均有效秩 4.58 (满秩11)', result: '65.3% Head 秩<5，88%主导首词注意(BOS)', sig: '证实Attention只做少数模式切换，不负责海量知识存储，深层极其聚光。' },
                            { name: 'E4: 权重矩阵低秩分析', metrics: 'MLP 满秩维 581(75%) vs QK满秩 52(6.8%)', result: 'MLP 知识极其密集，Attention 极低秩', sig: '进一步证实表示容量（知识）全在满秩的 MLP，Attention 只关联骨架。' },
                            { name: 'E5: Z113 结构相变追踪', metrics: 'Epoch 14000: 准确率 22.9%, 圆度 0.822', result: '泛化不仅出现，且圆环拓扑系数从 0.51 爬升至 0.82', sig: '证实 Grokking 并非黑魔法，而是系统从记忆混沌向代数几何圆环空间的重构。' },
                            { name: 'Emergence: 稀疏自发涌现', metrics: '首末峰度 2.7 -> 19.7', result: '无 BP 局部侧抑制+Hebbian冲刷 5k 步，成功爆发稀疏网络', sig: '证实无需反向传播，纯底层局部物理竞争即可自发产生接近真实的稀疏编码。' },
                        ].map((exp, idx) => (
                            <div key={idx} style={{ padding: '14px', background: 'rgba(0,0,0,0.3)', borderRadius: '12px', borderLeft: '3px solid #00d2ff' }}>
                                <div style={{ color: '#fff', fontSize: '14px', fontWeight: 'bold', marginBottom: '6px' }}>{exp.name}</div>
                                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(120px, auto) 1fr', gap: '6px', fontSize: '12px', color: '#d1d5db', lineHeight: '1.5' }}>
                                    <div style={{ color: '#00d2ff' }}>测试数据:</div><div>{exp.metrics}</div>
                                    <div style={{ color: '#00d2ff' }}>测试结果:</div><div>{exp.result}</div>
                                    <div style={{ color: '#00d2ff' }}>AGI突破意义:</div><div style={{ color: '#e5e7eb', fontWeight: '500' }}>{exp.sig}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* 4. 问题与硬伤 */}
                <div style={{ marginBottom: '28px' }}>
                    <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#ef4444', marginBottom: '12px', borderBottom: '1px solid rgba(239,68,68,0.3)', paddingBottom: '8px' }}>四、当前问题与核心硬伤</div>
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
