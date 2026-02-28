import React from 'react';

export const GPT5Tab = ({
    evidenceDrivenPlan,
    improvements,
    expandedImprovementPhase,
    setExpandedImprovementPhase,
    expandedImprovementTest,
    setExpandedImprovementTest,
}) => {
    const progressHighlights = [
        '阶段1（done）：不变量发现与基线校准完成，已形成候选结构库并通过跨模型稳定性初验。',
        '阶段2（in_progress）：因果验证已跑通，特征级干预显著强于整层干预，但整层证据仍偏弱。',
        '阶段3（in_progress）：最小生成模型已在低复杂度下保留主要能力，跨模态联合重建已可运行。',
        '阶段4（in_progress）：局部-全局一致性链路可运行，规模化点位在 tuned 配置下可稳定收敛。',
        '阶段5（in_progress）：反证机制已建立，失败主因聚焦在跨模态联络漂移与记忆冲突。',
    ];

    const keyProblems = [
        '因果证据强度不均：整层干预信号弱，特征级干预强，证据链仍需统一收敛。',
        '跨架构一致性仍是硬点：不同模型族对同一参数/协议的敏感区间不一致。',
        '协议敏感性依然存在：legacy 与 expanded 会改变结论强度，需长期桥接验证。',
        '真实数据证据不足：跨模态验证仍偏 synthetic，外推到真实场景的可信度不足。',
        '失败模式已定位但未闭环：联络漂移与记忆冲突仍是主要失效源。',
    ];

    const nextSteps = [
        'P0：统一 A-E 自动汇总管线，强制单源 JSON 产出阶段指标，清除口径不一致。',
        'P0：冻结协议 + 新 blind seed block 持续复验，严格按 gate 条件升级结论。',
        'P1：分架构做 layer/top_k/alpha 剂量-响应细扫，先把 falsify 压到 0 再追求 support 提升。',
        'P1：扩展真实任务族与真实多模态数据，降低 synthetic 偏置并验证外推能力。',
        'P2：把“分数提升”升级为“结构能力验证”，固化失效边界与可证伪结论集。',
    ];

    const roadmapItems = (evidenceDrivenPlan?.phases || []).map((phase, idx) => {
        const mappedPhase = (improvements || [])[idx];
        return {
            ...phase,
            status: mappedPhase?.status || 'pending',
            summary: mappedPhase?.summary || phase?.desc || '',
        };
    });

    const statusTextMap = {
        done: '已完成',
        in_progress: '进行中',
        pending: '待开始',
    };

    const statusColorMap = {
        done: '#10b981',
        in_progress: '#f59e0b',
        pending: '#94a3b8',
    };

    return (
        <div style={{ display: 'grid', gap: '20px' }}>
            <div
                style={{
                    padding: '30px',
                    borderRadius: '24px',
                    border: '1px solid rgba(56,189,248,0.28)',
                    background: 'linear-gradient(135deg, rgba(56,189,248,0.10) 0%, rgba(56,189,248,0.03) 100%)',
                    marginBottom: '28px',
                }}
            >
                <div style={{ color: '#38bdf8', fontWeight: 'bold', fontSize: '18px', marginBottom: '8px' }}>
                    {evidenceDrivenPlan?.title}（GPT5）
                </div>
                <div style={{ color: '#bae6fd', fontSize: '13px', lineHeight: '1.7', marginBottom: '20px' }}>{evidenceDrivenPlan?.core}</div>

                <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#e0f2fe', marginBottom: '10px', borderBottom: '1px solid rgba(56,189,248,0.35)', paddingBottom: '8px' }}>
                    一、分析框架
                </div>
                <div style={{ color: '#dbeafe', fontSize: '12px', lineHeight: '1.6', marginBottom: '10px' }}>
                    {evidenceDrivenPlan?.core}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '6px', marginBottom: '18px' }}>
                    {(evidenceDrivenPlan?.overview || []).map((line, idx) => (
                        <div key={idx} style={{ color: '#dbeafe', fontSize: '12px', lineHeight: '1.6' }}>
                            {idx + 1}. {line}
                        </div>
                    ))}
                </div>
                <div style={{ color: '#67e8f9', fontWeight: 'bold', fontSize: '13px', marginBottom: '8px' }}>当前研究进展</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '6px', marginBottom: '18px' }}>
                    {progressHighlights.map((line, idx) => (
                        <div key={idx} style={{ color: '#bae6fd', fontSize: '12px', lineHeight: '1.6' }}>
                            {idx + 1}. {line}
                        </div>
                    ))}
                </div>

                <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#e0f2fe', marginBottom: '10px', borderBottom: '1px solid rgba(56,189,248,0.35)', paddingBottom: '8px' }}>
                    二、线路图
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, minmax(120px, 1fr))', gap: '10px', marginBottom: '18px' }}>
                    {roadmapItems.map((item) => (
                        <div key={item.id} style={{ padding: '10px', background: 'rgba(0,0,0,0.28)', borderRadius: '10px', borderTop: `2px solid ${statusColorMap[item.status] || '#94a3b8'}` }}>
                            <div style={{ color: '#dbeafe', fontSize: '12px', fontWeight: 'bold' }}>{item.id}</div>
                            <div style={{ color: statusColorMap[item.status] || '#94a3b8', fontSize: '11px', marginBottom: '4px' }}>[{statusTextMap[item.status] || '待开始'}]</div>
                            <div style={{ color: '#bfdbfe', fontSize: '11px', lineHeight: '1.5' }}>{item.name}</div>
                        </div>
                    ))}
                </div>

                <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#e0f2fe', marginBottom: '10px', borderBottom: '1px solid rgba(56,189,248,0.35)', paddingBottom: '8px' }}>
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
                    <div style={{ color: '#10b981', fontWeight: 'bold', fontSize: '14px', marginBottom: '6px' }}>分阶段详情与完整测试数据</div>
                    <div style={{ color: '#9ca3af', fontSize: '12px', lineHeight: '1.7', marginBottom: '12px' }}>
                        按阶段展开查看目标、测试参数、详细数据与分析结论。
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
                        {(improvements || []).map((phase) => {
                            const isPhaseExpanded = expandedImprovementPhase === phase.id;
                            const phaseStatusColor =
                                phase.status === 'done' ? '#10b981' : phase.status === 'in_progress' ? '#f59e0b' : '#94a3b8';
                            const phaseTestCount = (phase.tests || []).length;
                            return (
                                <div
                                    key={phase.id}
                                    style={{
                                        padding: '14px 16px',
                                        borderRadius: '12px',
                                        border: `1px solid ${isPhaseExpanded ? 'rgba(16,185,129,0.45)' : 'rgba(255,255,255,0.08)'}`,
                                        background: isPhaseExpanded ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.02)',
                                    }}
                                >
                                    <button
                                        onClick={() => {
                                            const nextPhase = isPhaseExpanded ? null : phase.id;
                                            setExpandedImprovementPhase(nextPhase);
                                            setExpandedImprovementTest(null);
                                        }}
                                        style={{
                                            width: '100%',
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            alignItems: 'center',
                                            gap: '12px',
                                            marginBottom: isPhaseExpanded ? '8px' : 0,
                                            background: 'transparent',
                                            border: 'none',
                                            cursor: 'pointer',
                                            padding: 0,
                                            textAlign: 'left',
                                        }}
                                    >
                                        <div>
                                            <div style={{ color: '#dcfce7', fontWeight: 'bold', fontSize: '14px' }}>{phase.title}</div>
                                            <div style={{ color: '#86efac', fontSize: '11px', marginTop: '2px' }}>
                                                累计测试：{phaseTestCount} 条
                                            </div>
                                        </div>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                            <div style={{ fontSize: '10px', color: phaseStatusColor }}>{String(phase.status).toUpperCase()}</div>
                                            <div style={{ fontSize: '11px', color: '#86efac' }}>{isPhaseExpanded ? '收起' : '展开'}</div>
                                        </div>
                                    </button>

                                    {isPhaseExpanded && (
                                        <div>
                                        <div style={{ color: '#9fe8c7', fontSize: '12px', marginBottom: '6px' }}>阶段目标：{phase.objective}</div>
                                        <div style={{ color: '#a7f3d0', fontSize: '12px', lineHeight: '1.6', marginBottom: '10px' }}>
                                            阶段总结：{phase.summary}
                                        </div>
                                        <div style={{ color: '#86efac', fontSize: '12px', marginBottom: '8px' }}>
                                            累计测试：{phaseTestCount} 条
                                        </div>
                                        <div style={{ color: '#d1fae5', fontSize: '12px', fontWeight: 'bold', marginBottom: '8px' }}>
                                            测试列表（点击查看详细数据）
                                        </div>

                                            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
                                                {(phase.tests || []).map((test, testIdx) => {
                                                    const testKey = `${phase.id}:${test.id}`;
                                                    const isTestExpanded = expandedImprovementTest === testKey;
                                                    const evidenceChain = Array.isArray(test?.details?.evidence_chain) ? test.details.evidence_chain : [];
                                                    const keyEvidenceText = evidenceChain.length > 0 ? evidenceChain.join('；') : (test.result || '未补充');
                                                    const agiSignificance = test?.details?.agi_significance || test.analysis || '未补充';
                                                    const currentGap = test?.details?.current_gap || '未补充';
                                                    return (
                                                        <div
                                                            key={test.id}
                                                            style={{
                                                                borderRadius: '10px',
                                                                border: `1px solid ${isTestExpanded ? 'rgba(96,165,250,0.5)' : 'rgba(255,255,255,0.08)'}`,
                                                                background: isTestExpanded ? 'rgba(30,64,175,0.12)' : 'rgba(0,0,0,0.18)',
                                                                padding: '10px 12px',
                                                            }}
                                                        >
                                                            <button
                                                                onClick={() => setExpandedImprovementTest(isTestExpanded ? null : testKey)}
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
                                                                    T{testIdx + 1}. {test.name}
                                                                </div>
                                                                <div style={{ color: '#93c5fd', fontSize: '11px' }}>{isTestExpanded ? '收起详情' : '查看详情'}</div>
                                                            </button>

                                                            <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.6', marginTop: '6px' }}>
                                                                关键测试：{test.target}
                                                            </div>
                                                            <div style={{ color: '#93c5fd', fontSize: '12px', lineHeight: '1.6', marginTop: '4px' }}>
                                                                测试日期：{test.testDate || '未记录'}
                                                            </div>
                                                            <div style={{ color: '#94a3b8', fontSize: '12px', lineHeight: '1.6', marginTop: '4px' }}>
                                                                关键证据：{keyEvidenceText}
                                                            </div>
                                                            <div style={{ color: '#a7f3d0', fontSize: '12px', lineHeight: '1.6', marginTop: '4px' }}>
                                                                对 AGI 的意义：{agiSignificance}
                                                            </div>
                                                            <div style={{ color: '#fca5a5', fontSize: '12px', lineHeight: '1.6', marginTop: '4px' }}>
                                                                当前不足：{currentGap}
                                                            </div>
                                                            <div style={{ color: '#94a3b8', fontSize: '12px', lineHeight: '1.6', marginTop: '4px' }}>
                                                                测试结果：{test.result}
                                                            </div>
                                                            <div style={{ color: '#a7f3d0', fontSize: '12px', lineHeight: '1.6', marginTop: '4px' }}>
                                                                分析总结：{test.analysis}
                                                            </div>

                                                            {isTestExpanded && (
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
                                                                        测试参数
                                                                    </div>
                                                                    <pre
                                                                        style={{
                                                                            margin: 0,
                                                                            color: '#dbeafe',
                                                                            fontSize: '11px',
                                                                            lineHeight: '1.6',
                                                                            whiteSpace: 'pre-wrap',
                                                                        }}
                                                                    >
                                                                        {JSON.stringify(test.params, null, 2)}
                                                                    </pre>
                                                                    <div
                                                                        style={{
                                                                            color: '#bfdbfe',
                                                                            fontSize: '11px',
                                                                            fontWeight: 'bold',
                                                                            marginTop: '10px',
                                                                            marginBottom: '6px',
                                                                        }}
                                                                    >
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

                <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#86efac', marginBottom: '10px', borderBottom: '1px solid rgba(74,222,128,0.35)', paddingBottom: '8px' }}>
                    五、接下来的核心工作
                </div>
                <div style={{ display: 'grid', gap: '6px' }}>
                    {nextSteps.map((item, idx) => (
                        <div key={idx} style={{ color: '#dcfce7', fontSize: '12px', lineHeight: '1.6' }}>
                            {idx + 1}. {item}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};
