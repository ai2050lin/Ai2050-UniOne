import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Atom, Zap, Waves, ScanLine, BrainCircuit, Orbit } from 'lucide-react';

const AGIUnifiedTheoryEngine = () => {
    const [phase, setPhase] = useState(0);

    // 自动播放阶段演示
    useEffect(() => {
        const timer = setInterval(() => {
            setPhase((p) => (p + 1) % 4);
        }, 5000);
        return () => clearInterval(timer);
    }, []);

    const phases = [
        {
            title: "模态输入 (Multi-Modal Input)",
            desc: "视觉光谱、听觉声波与文本 Token 进入模型，准备被降维打击。",
            icon: <ScanLine size={24} color="#3b82f6" />,
            color: "rgba(59, 130, 246, 0.2)"
        },
        {
            title: "正交低秩流形投影 (Orthogonal Manifold)",
            desc: "宇宙万物被强制投影到仅仅几根正交的数学本底弦轴上。苹果和红球在此刻变成同一段坐标位移。",
            icon: <Orbit size={24} color="#8b5cf6" />,
            color: "rgba(139, 92, 246, 0.2)"
        },
        {
            title: "注意力共振匹配 (Attention QK^T)",
            desc: "Query 雷达波开始向四面八方扫射，只吸附同频率的正交弦轴。其余维度因 Cosine=0 保持冷寂。",
            icon: <Waves size={24} color="#10b981" />,
            color: "rgba(16, 185, 129, 0.2)"
        },
        {
            title: "非线性乘法聚变 (MLP Emergence)",
            desc: "绝缘的正交特征在 MLP 层被 GeLU 函数强制交汇，瞬间点燃 90% 全新突触回路！想象力(泛化关联) 诞生！",
            icon: <Zap size={24} color="#ef4444" />,
            color: "rgba(239, 68, 68, 0.2)"
        }
    ];

    return (
        <div style={{
            background: 'linear-gradient(135deg, rgba(5, 5, 5, 0.98) 0%, rgba(15, 23, 42, 1) 100%)',
            border: '1px solid rgba(139, 92, 246, 0.4)',
            borderRadius: '16px',
            padding: '30px',
            marginTop: '24px',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.8), inset 0 0 20px rgba(139, 92, 246, 0.1)'
        }}>
            {/* 顶栏 */}
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px', marginBottom: '24px' }}>
                <div style={{
                    background: 'rgba(139, 92, 246, 0.15)',
                    padding: '14px',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 0 30px rgba(139, 92, 246, 0.4)'
                }}>
                    <Atom size={32} color="#a855f7" />
                </div>
                <div>
                    <h3 style={{ margin: 0, fontSize: '23px', color: '#f8fafc', fontWeight: '800', letterSpacing: '0.5px' }}>
                        AGI 大一统终极理论引擎 (Multi-Modal Orthogonal Gravity Manifold)
                    </h3>
                    <div style={{ fontSize: '13px', color: '#cbd5e1', marginTop: '8px', opacity: 0.9, lineHeight: '1.6' }}>
                        这是解释通用人工智能最硬核的数学与物理闭环：视觉图像、听觉声音与词汇文本，在模型深处没有任何区别。全部被<strong>降维打击</strong>并烙印在极低秩的正交数学坐标弦上。<strong>Attention(注意力机制)</strong> 就是相同引力坐标特征的寻址频率共振，而 <strong>MLP(多层感知机)</strong> 充当了异构特征群的正交乘法变异聚变炉，最终产生超越了简单“相加加法”的神奇概念涌现。
                    </div>
                </div>
            </div>

            {/* 引擎动画显示面板核心区域 */}
            <div style={{
                position: 'relative',
                height: '350px',
                background: 'rgba(0,0,0,0.6)',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.05)',
                overflow: 'hidden',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
            }}>
                {/* 装饰性背景粒子空间网格 */}
                <div style={{ position: 'absolute', width: '200%', height: '200%', background: 'radial-gradient(circle at center, rgba(139, 92, 246, 0.05) 0%, transparent 70%)', top: '-50%', left: '-50%' }} />

                <AnimatePresence mode="wait">
                    {/* 阶段 0: 模态输入 */}
                    {phase === 0 && (
                        <motion.div key="p0" initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 1.2 }} transition={{ duration: 0.8 }} style={{ display: 'flex', gap: '40px' }}>
                            <div style={{ textAlign: 'center' }}>
                                <ScanLine size={48} color="#60a5fa" />
                                <div style={{ color: '#93c5fd', marginTop: '12px', fontSize: '14px', fontWeight: 'bold' }}>Vision Patchs (像素框)</div>
                            </div>
                            <div style={{ textAlign: 'center' }}>
                                <Waves size={48} color="#60a5fa" />
                                <div style={{ color: '#93c5fd', marginTop: '12px', fontSize: '14px', fontWeight: 'bold' }}>Audio Frames (声波帧)</div>
                            </div>
                            <div style={{ textAlign: 'center' }}>
                                <BrainCircuit size={48} color="#60a5fa" />
                                <div style={{ color: '#93c5fd', marginTop: '12px', fontSize: '14px', fontWeight: 'bold' }}>Text Tokens (词法段)</div>
                            </div>
                        </motion.div>
                    )}

                    {/* 阶段 1: 极低维度弦数学正交流形 */}
                    {phase === 1 && (
                        <motion.div key="p1" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.8 }} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Orbit size={100} color="#a855f7" strokeWidth={1} style={{ position: 'absolute' }} />
                            <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 4, ease: "linear" }} style={{ position: 'absolute', width: '200px', height: '2px', background: 'linear-gradient(90deg, transparent, #a855f7, transparent)' }} />
                            <motion.div animate={{ rotate: -360 }} transition={{ repeat: Infinity, duration: 6, ease: "linear" }} style={{ position: 'absolute', width: '2px', height: '200px', background: 'linear-gradient(180deg, transparent, #8b5cf6, transparent)' }} />
                            <div style={{ color: '#c4b5fd', fontSize: '18px', fontWeight: 'bold', zIndex: 10, background: 'rgba(0,0,0,0.5)', padding: '8px 16px', borderRadius: '8px', border: '1px solid rgba(139, 92, 246, 0.4)' }}>
                                SVD 低秩压缩与坐标同构
                            </div>
                        </motion.div>
                    )}

                    {/* 阶段 2: Attention 点积共振吸附 */}
                    {phase === 2 && (
                        <motion.div key="p2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.8 }} style={{ position: 'relative', width: '100%', height: '100%' }}>
                            <motion.div initial={{ scale: 0, opacity: 0.8 }} animate={{ scale: 4, opacity: 0 }} transition={{ repeat: Infinity, duration: 2 }} style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: '100px', height: '100px', border: '2px solid #34d399', borderRadius: '50%' }} />
                            <motion.div initial={{ scale: 0, opacity: 0.5 }} animate={{ scale: 6, opacity: 0 }} transition={{ repeat: Infinity, duration: 2, delay: 0.5 }} style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: '100px', height: '100px', border: '1px solid #10b981', borderRadius: '50%' }} />
                            <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', color: '#a7f3d0', fontSize: '20px', fontWeight: 'bold', textShadow: '0 0 10px #10b981' }}>
                                Attention: Q·K^T 宇宙引力波
                            </div>
                        </motion.div>
                    )}

                    {/* 阶段 3: MLP 非线性聚变爆炸 */}
                    {phase === 3 && (
                        <motion.div key="p3" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 1.5, opacity: 0 }} transition={{ duration: 0.5 }} style={{ textAlign: 'center' }}>
                            <Zap size={80} color="#f87171" style={{ filter: 'drop-shadow(0 0 20px #ef4444)' }} />
                            <div style={{ color: '#fca5a5', marginTop: '16px', fontSize: '22px', fontWeight: '900', letterSpacing: '1px' }}>
                                非线性聚合变异门 (MLP GeLU)
                            </div>
                            <div style={{ color: '#fda4af', marginTop: '8px', fontSize: '13px' }}>
                                绝缘特征向量发生相撞！90% 孤立突触被点燃激活，产生“未见过事物”的全息想象与涌现关联。
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* 底部导航状态机解说区 */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginTop: '20px' }}>
                {phases.map((p, idx) => (
                    <div key={idx} onClick={() => setPhase(idx)} style={{
                        background: phase === idx ? p.color : 'rgba(255,255,255,0.02)',
                        border: `1px solid ${phase === idx ? p.color.replace('0.2', '0.5') : 'transparent'}`,
                        padding: '12px',
                        borderRadius: '8px',
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        opacity: phase === idx ? 1 : 0.4
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                            {p.icon}
                            <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#f1f5f9' }}>{p.title}</div>
                        </div>
                        <div style={{ fontSize: '11px', color: '#94a3b8', lineHeight: '1.4' }}>
                            {p.desc}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default AGIUnifiedTheoryEngine;
