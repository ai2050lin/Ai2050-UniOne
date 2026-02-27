import React from 'react';

export const GPT5Tab = ({
    evidenceDrivenPlan,
    improvements,
    expandedImprovementPhase,
    setExpandedImprovementPhase,
    expandedImprovementTest,
    setExpandedImprovementTest,
}) => {
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
                    {evidenceDrivenPlan?.title}
                </div>
                <div style={{ color: '#bae6fd', fontSize: '13px', lineHeight: '1.7', marginBottom: '12px' }}>{evidenceDrivenPlan?.core}</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '6px', marginBottom: '10px' }}>
                    {(evidenceDrivenPlan?.overview || []).map((line, idx) => (
                        <div key={idx} style={{ color: '#dbeafe', fontSize: '12px', lineHeight: '1.6' }}>
                            {idx + 1}. {line}
                        </div>
                    ))}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
                    {(evidenceDrivenPlan?.phases || []).map((item) => (
                        <details
                            key={item.id}
                            style={{
                                padding: '12px 14px',
                                borderRadius: '12px',
                                border: '1px solid rgba(255,255,255,0.10)',
                                background: 'rgba(0,0,0,0.18)',
                            }}
                        >
                            <summary style={{ cursor: 'pointer', listStyle: 'none' }}>
                                <div style={{ fontSize: '12px', color: '#7dd3fc', fontWeight: 'bold', marginBottom: '4px' }}>
                                    {item.id} 路 {item.name}
                                </div>
                                <div style={{ fontSize: '12px', color: '#e0f2fe', lineHeight: '1.6' }}>
                                    原理说明：{item.desc}
                                </div>
                                <div style={{ color: '#93c5fd', fontSize: '11px', marginTop: '4px' }}>
                                    点击展开详细说明
                                </div>
                            </summary>
                            <div style={{ marginTop: '8px' }}>
                                <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.55' }}>目标：{item.goal}</div>
                                <div style={{ color: '#bfdbfe', fontSize: '11px', lineHeight: '1.55' }}>方法：{item.method}</div>
                                <div style={{ color: '#a7f3d0', fontSize: '11px', lineHeight: '1.55' }}>证据：{item.evidence}</div>
                                <div style={{ color: '#ddd6fe', fontSize: '11px', lineHeight: '1.55' }}>产出：{item.outputs}</div>
                                <div style={{ color: '#fcd34d', fontSize: '11px', lineHeight: '1.55' }}>准出：{item.gate}</div>
                            </div>
                        </details>
                    ))}
                </div>
            </div>

            <div
                style={{
                    padding: '30px',
                    borderRadius: '24px',
                    border: '1px solid rgba(16,185,129,0.24)',
                    background: 'linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.02) 100%)',
                }}
            >
                <div style={{ color: '#10b981', fontWeight: 'bold', fontSize: '18px', marginBottom: '8px' }}>分析进展</div>
                <div style={{ color: '#9ca3af', fontSize: '13px', lineHeight: '1.7', marginBottom: '16px' }}>
                    通过五个阶段，尝试完成深度神经网络中数学结构的研究
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
                    {(improvements || []).map((phase) => {
                        const isPhaseExpanded = expandedImprovementPhase === phase.id;
                        const phaseStatusColor =
                            phase.status === 'done' ? '#10b981' : phase.status === 'in_progress' ? '#f59e0b' : '#94a3b8';
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
                                    <div style={{ color: '#dcfce7', fontWeight: 'bold', fontSize: '14px' }}>{phase.title}</div>
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
                                        <div style={{ color: '#d1fae5', fontSize: '12px', fontWeight: 'bold', marginBottom: '8px' }}>
                                            测试列表（点击查看详细数据）
                                        </div>

                                        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
                                            {(phase.tests || []).map((test, testIdx) => {
                                                const testKey = `${phase.id}:${test.id}`;
                                                const isTestExpanded = expandedImprovementTest === testKey;
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
                                                            测试目标：{test.target}
                                                        </div>
                                                        <div style={{ color: '#93c5fd', fontSize: '12px', lineHeight: '1.6', marginTop: '4px' }}>
                                                            测试日期：{test.testDate || '未记录'}
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
        </div>
    );
};
