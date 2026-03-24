import React, { Suspense, lazy, useMemo, useState } from 'react';

const GeminiTab = lazy(() => import('./GeminiTab').then((module) => ({ default: module.GeminiTab })));
const GPT5Tab = lazy(() => import('./GPT5Tab').then((module) => ({ default: module.GPT5Tab })));
const GLM5Tab = lazy(() => import('./GLM5Tab').then((module) => ({ default: module.GLM5Tab })));

export const DeepAnalysisTab = ({
    evidenceDrivenPlan,
    improvements,
    expandedImprovementPhase,
    setExpandedImprovementPhase,
    expandedImprovementTest,
    setExpandedImprovementTest,
}) => {
    const [activeModelTab, setActiveModelTab] = useState('Gemini');
    const modelTabs = ['Gemini', 'GPT5', 'GLM5'];
    const ActiveTabComponent = useMemo(() => {
        if (activeModelTab === 'Gemini') return GeminiTab;
        if (activeModelTab === 'GPT5') return GPT5Tab;
        return GLM5Tab;
    }, [activeModelTab]);

    const activeTabProps = activeModelTab === 'GPT5'
        ? {
            evidenceDrivenPlan,
            improvements,
            expandedImprovementPhase,
            setExpandedImprovementPhase,
            expandedImprovementTest,
            setExpandedImprovementTest,
        }
        : {};

    return (
        <div style={{ animation: 'roadmapFade 0.6s ease-out', maxWidth: '1000px', margin: '0 auto' }}>
            <div
                style={{
                    padding: '30px',
                    borderRadius: '24px',
                    border: '1px solid rgba(244,114,182,0.28)',
                    background: 'linear-gradient(135deg, rgba(244,114,182,0.10) 0%, rgba(168,85,247,0.06) 100%)',
                    marginBottom: '28px',
                }}
            >
                <div style={{ color: '#f472b6', fontWeight: 'bold', fontSize: '24px', marginBottom: '20px' }}>
                    深度分析与模型结构对比
                </div>
                <div style={{ display: 'flex', gap: '0', marginBottom: '24px', borderBottom: '1px solid rgba(255,255,255,0.12)' }}>
                    {modelTabs.map((tab) => (
                        <button
                            key={tab}
                            onClick={() => setActiveModelTab(tab)}
                            style={{
                                padding: '12px 32px',
                                background: activeModelTab === tab
                                    ? 'linear-gradient(135deg, rgba(244,114,182,0.22) 0%, rgba(168,85,247,0.18) 100%)'
                                    : 'transparent',
                                border: 'none',
                                borderBottom: activeModelTab === tab ? '3px solid #f472b6' : '3px solid transparent',
                                color: activeModelTab === tab ? '#f9a8d4' : '#9ca3af',
                                fontSize: '15px',
                                fontWeight: activeModelTab === tab ? 'bold' : 'normal',
                                cursor: 'pointer',
                                transition: 'all 0.2s ease',
                            }}
                        >
                            {tab}
                        </button>
                    ))}
                </div>

                {/* Tab 内容区 */}
                <Suspense
                    fallback={(
                        <div
                            style={{
                                padding: '22px',
                                borderRadius: '16px',
                                border: '1px solid rgba(255,255,255,0.08)',
                                background: 'rgba(255,255,255,0.03)',
                                color: '#cbd5e1',
                                fontSize: '13px',
                                lineHeight: '1.7',
                            }}
                        >
                            正在分批加载 {activeModelTab} 深度分析内容...
                        </div>
                    )}
                >
                    <ActiveTabComponent {...activeTabProps} />
                </Suspense>
            </div>
        </div>
    );
};
