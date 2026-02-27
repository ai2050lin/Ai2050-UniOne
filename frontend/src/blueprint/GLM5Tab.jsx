import React from 'react';

export const GLM5Tab = ({ activeModelTab }) => {
    return (
        <div
            style={{
                minHeight: '200px',
                padding: '20px',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.08)',
                background: 'rgba(0,0,0,0.22)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
            }}
        >
            <div style={{ textAlign: 'center' }}>
                <div style={{ color: '#d1d5db', fontSize: '14px', marginBottom: '8px' }}>
                    {activeModelTab} 模型分析
                </div>
                <div style={{ color: '#6b7280', fontSize: '12px' }}>
                    即将添加 {activeModelTab} 的结构分析数据
                </div>
            </div>
        </div>
    );
};
