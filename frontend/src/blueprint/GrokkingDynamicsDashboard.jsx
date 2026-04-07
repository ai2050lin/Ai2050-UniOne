import React, { useState, useEffect } from 'react';
import { Activity, Zap, TrendingUp, Circle, Share2, Layers, Cpu } from 'lucide-react';

const GrokkingDynamicsDashboard = () => {
    const [progress, setProgress] = useState(0);
    const [isGrokking, setIsGrokking] = useState(false);

    useEffect(() => {
        const interval = setInterval(() => {
            setProgress(prev => {
                if (prev >= 100) {
                    setIsGrokking(true);
                    return 100;
                }
                return prev + 1;
            });
        }, 100);
        return () => clearInterval(interval);
    }, []);

    const trainingData = [
        { label: "0-3000 Epoch", status: "Memorization (Overfitting)", acc: "0.0%", rank: "35.4", color: "#64748b" },
        { label: "3000-5000 Epoch", status: "Phase Transition (Breaking)", acc: "42.0%", rank: "15.2", color: "#f59e0b" },
        { label: "5000-10000 Epoch", status: "Grokking (Final Generalization)", acc: "100.0%", rank: "6.07", color: "#10b981" }
    ];

    return (
        <div style={{
            padding: '32px',
            background: 'rgba(15, 23, 42, 0.4)',
            borderRadius: '28px',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            backdropFilter: 'blur(30px)',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            color: '#fff'
        }}>
            <div style={{ marginBottom: '32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h2 style={{ margin: 0, fontSize: '24px', fontWeight: '800', display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <Zap color="#f59e0b" size={28} /> Stage 86：符号接地与顿悟动力学 (P=113)
                    </h2>
                    <p style={{ color: '#94a3b8', fontSize: '13px', marginTop: '4px' }}>
                        观察几何流形（Family Patch）从随机初始化的弥散状态中自发演化与对齐的过程。
                    </p>
                </div>
                <div style={{ padding: '8px 16px', background: isGrokking ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)', borderRadius: '12px', border: `1px solid ${isGrokking ? '#10b981' : '#f59e0b'}`, color: isGrokking ? '#10b981' : '#f59e0b', fontSize: '12px', fontWeight: 'bold' }}>
                    {isGrokking ? "顿悟已完成 (GROKKED)" : "正在对齐中 (ALIGNING...)"}
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px', marginBottom: '40px' }}>
                {trainingData.map((d, i) => (
                    <div key={i} style={{ 
                        background: 'rgba(255, 255, 255, 0.03)', 
                        padding: '20px', 
                        borderRadius: '20px', 
                        border: '1px solid rgba(255,255,255,0.05)',
                        transition: 'all 0.3s ease'
                    }}>
                        <div style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '8px' }}>{d.label}</div>
                        <div style={{ color: d.color, fontWeight: 'bold', fontSize: '14px', marginBottom: '16px' }}>{d.status}</div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
                            <div>
                                <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase' }}>Test Acc</div>
                                <div style={{ fontSize: '20px', fontWeight: '800' }}>{d.acc}</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase' }}>Eff. Rank</div>
                                <div style={{ fontSize: '20px', fontWeight: '800', opacity: 0.8 }}>{d.rank}</div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '24px', borderRadius: '24px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px', color: '#818cf8' }}>
                        <Circle size={20} />
                        <span style={{ fontWeight: 'bold', fontSize: '15px' }}>W_E 几何流形：PCA 投影</span>
                    </div>
                    <div style={{ height: '240px', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        {/* Simulated PCA Circle */}
                        <div style={{ 
                            width: '180px', height: '180px', 
                            borderRadius: '50%', 
                            border: `2px dashed ${isGrokking ? '#10b981' : '#64748b'}`,
                            boxShadow: isGrokking ? '0 0 40px rgba(16, 185, 129, 0.1)' : 'none',
                            animation: 'pulse 4s infinite linear',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                        }}>
                             {[...Array(12)].map((_, i) => (
                                <div key={i} style={{
                                    position: 'absolute',
                                    width: '8px', height: '8px',
                                    background: isGrokking ? `hsl(${i * 30}, 70%, 60%)` : '#64748b',
                                    borderRadius: '50%',
                                    transform: `rotate(${i * 30}deg) translateY(-90px)`,
                                    boxShadow: isGrokking ? `0 0 10px hsl(${i * 30}, 70%, 60%)` : 'none',
                                    transition: 'all 2s ease'
                                }} />
                             ))}
                             <div style={{ textAlign: 'center' }}>
                                 <div style={{ fontSize: '24px', fontWeight: '900', color: isGrokking ? '#fff' : '#475569' }}>Circular</div>
                                 <div style={{ fontSize: '10px', color: '#64748b' }}>Axiom 1: Geometry</div>
                             </div>
                        </div>
                    </div>
                    <p style={{ color: '#94a3b8', fontSize: '12px', lineHeight: '1.6', textAlign: 'center', marginTop: '16px' }}>
                        {isGrokking 
                            ? "检测到周期性圆周流形：模型已成功将离散符号接地为 2D 正交旋转不变量。" 
                            : "几何尚未对齐：符号在 128 维空间中呈现漫反射状态，能量尚未收敛。"}
                    </p>
                </div>

                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '24px', borderRadius: '24px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px', color: '#34d399' }}>
                        <Share2 size={20} />
                        <span style={{ fontWeight: 'bold', fontSize: '15px' }}>信号聚焦：傅里叶模态 (Top 5)</span>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', height: '240px', justifyContent: 'center' }}>
                        {isGrokking ? (
                            [5, 51, 35, 13, 26].map((k, i) => (
                                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                    <div style={{ fontSize: '10px', color: '#64748b', width: '30px' }}>k={k}</div>
                                    <div style={{ flex: 1, height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                                        <div style={{ 
                                            width: `${90 - i * 15}%`, 
                                            height: '100%', 
                                            background: 'linear-gradient(90deg, #34d399, #10b981)',
                                            animation: 'slideIn 1.5s ease-out forwards'
                                        }} />
                                    </div>
                                    <div style={{ fontSize: '10px', color: '#34d399', fontWeight: 'bold' }}>{90 - i * 15}%</div>
                                </div>
                            ))
                        ) : (
                            <div style={{ textAlign: 'center', color: '#64748b', fontSize: '13px' }}>
                                正在扫描潜在的周期性模态...
                            </div>
                        )}
                    </div>
                    <p style={{ color: '#94a3b8', fontSize: '12px', lineHeight: '1.6', textAlign: 'center', marginTop: '16px' }}>
                        {isGrokking 
                            ? "信号聚焦公理验证：128 维乱序特征已折叠为 6 个关键谐波频率。" 
                            : "缺乏主导模态：权重倾向于记忆特定样本，而非学习全局逻辑轨道。"}
                    </p>
                </div>
            </div>

            <style>{`
                @keyframes pulse {
                    0% { transform: scale(1); opacity: 0.8; }
                    50% { transform: scale(1.05); opacity: 1; }
                    100% { transform: scale(1); opacity: 0.8; }
                }
                @keyframes slideIn {
                    from { width: 0; }
                }
            `}</style>
        </div>
    );
};

export default GrokkingDynamicsDashboard;
