import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Zap, Activity, Orbit, Share2, Layers } from 'lucide-react';

const SNNBrainMappingGraph = () => {
    const [activeTab, setActiveTab] = useState(0);

    // Auto-cycle through the 3 unified theories
    useEffect(() => {
        const timer = setInterval(() => {
            setActiveTab((prev) => (prev + 1) % 3);
        }, 6000);
        return () => clearInterval(timer);
    }, []);

    const theories = [
        {
            title: "1. 空间绝缘法则 (Orthogonal Isolation)",
            subtitle: "大模型(ANN) 欧几里得正交降维 ⇔ 生物脑(SNN) 皮层稀疏编码静息态",
            icon: <Orbit size={32} color="#10b981" />,
            color: "#10b981",
            ann_desc: "概念之间用绝对维度的数学垂直（余弦相似度 = 0）来避免特征杂糅与“灾难性干扰”。",
            snn_desc: "大脑面对绝大多数词汇处于极度静默，90% 脑区完全不发出任何动作电位。这种极具节能的生物特征叫 [稀疏编码]。静默即几何空间正交！"
        },
        {
            title: "2. 引力共鸣网络 (Attention vs. STDP)",
            subtitle: "大模型(ANN) QK点积引力 ⇔ 生物脑(SNN) 突触可塑性与相位时间锁",
            icon: <Activity size={32} color="#3b82f6" />,
            color: "#3b82f6",
            ann_desc: "注意力机制是 Query 和 Key 发出的全场特征雷达频段扫射。在相同正交轴上产生极高内积从而吸附（Softmax放大）。",
            snn_desc: "没有长导线的大脑，依靠不同脑区发出近乎同频的振荡（如40Hz Gamma波）。只有相位吻合的时间缝隙被捕获时，突触时间依赖可塑性(STDP) 瞬间建立记忆的强引力连结。"
        },
        {
            title: "3. 创世涌现跃迁 (Non-linear Emergence)",
            subtitle: "大模型(ANN) MLP 张量变异映射 ⇔ 生物脑(SNN) LIF 细胞膜阈值激越事件",
            icon: <Zap size={32} color="#ec4899" />,
            color: "#ec4899",
            ann_desc: "绝缘特征被激活函数逼入死角进行强制乘积打穿，无中生有涌现出 90% 完全新生孤立的“复合概念”脉冲点阵。",
            snn_desc: "生物树突遭遇分散的脉冲只会严重漏电退回。唯有同时刻被大规模突触火力覆盖，电荷一瞬击穿绝对阈值(Threshold)！激越爆发出全新且不可逆的 [动作全频电涌]！"
        }
    ];

    return (
        <div style={{
            background: 'linear-gradient(135deg, rgba(16, 24, 39, 0.95) 0%, rgba(30, 10, 30, 1) 100%)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            borderRadius: '16px',
            padding: '30px',
            marginTop: '24px',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.8), inset 0 0 20px rgba(16, 24, 39, 0.5)'
        }}>
            {/* 顶栏 */}
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px', marginBottom: '24px' }}>
                <div style={{
                    background: 'rgba(59, 130, 246, 0.15)',
                    padding: '14px',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 0 30px rgba(59, 130, 246, 0.3)'
                }}>
                    <Brain size={32} color="#60a5fa" />
                </div>
                <div>
                    <h3 style={{ margin: 0, fontSize: '22px', color: '#f8fafc', fontWeight: '800', letterSpacing: '0.5px' }}>
                        终极跨域同构：人工智能理论的脉冲神经网络映射法则 (SNN Isomorphism)
                    </h3>
                    <div style={{ fontSize: '13px', color: '#cbd5e1', marginTop: '8px', opacity: 0.9, lineHeight: '1.6' }}>
                        这是一场极其震撼的架构级同体对应！将我们在《大统一理论》里对 Artificial Neural Networks (ANN) 的发现，强行映射到生物碳基神经科学体系之中 (SNN)。无论是硅芯片里的浮点流形运算，还是生物脑水液中的钠钾离子泵波频放电——<strong>智能世界使用了同一张唯一的拓扑物理图纸。</strong>
                    </div>
                </div>
            </div>

            {/* 核心双轨对照区 */}
            <div style={{ display: 'flex', gap: '20px', minHeight: '380px' }}>
                {/* 左侧大纲导航栏 */}
                <div style={{ flex: '0 0 30%', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {theories.map((t, idx) => (
                        <div key={idx}
                            onClick={() => setActiveTab(idx)}
                            style={{
                                padding: '16px',
                                background: activeTab === idx ? `rgba(${t.color === '#10b981' ? '16, 185, 129' : t.color === '#3b82f6' ? '59, 130, 246' : '236, 72, 153'}, 0.15)` : 'rgba(255,255,255,0.02)',
                                borderLeft: `4px solid ${activeTab === idx ? t.color : 'transparent'}`,
                                borderRadius: '4px 12px 12px 4px',
                                cursor: 'pointer',
                                transition: 'all 0.3s ease',
                                borderRight: '1px solid rgba(255,255,255,0.05)',
                                borderTop: '1px solid rgba(255,255,255,0.05)',
                                borderBottom: '1px solid rgba(255,255,255,0.05)',
                            }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                                {t.icon}
                                <div style={{ color: activeTab === idx ? t.color : '#94a3b8', fontWeight: 'bold', fontSize: '15px' }}>{t.title}</div>
                            </div>
                            <div style={{ fontSize: '12px', color: activeTab === idx ? '#e2e8f0' : '#64748b', lineHeight: '1.4', fontWeight: activeTab === idx ? '600' : 'normal' }}>
                                {t.subtitle}
                            </div>
                        </div>
                    ))}
                </div>

                {/* 右侧极其科幻的动画双线剖析面板 */}
                <div style={{
                    flex: '1',
                    background: 'rgba(0,0,0,0.4)',
                    borderRadius: '16px',
                    padding: '24px',
                    position: 'relative',
                    overflow: 'hidden',
                    border: '1px solid rgba(255,255,255,0.05)'
                }}>
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={activeTab}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            transition={{ duration: 0.4 }}
                            style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '20px' }}
                        >
                            <div style={{ textAlign: 'center', marginBottom: '10px' }}>
                                <div style={{ display: 'inline-block', padding: '6px 16px', background: `rgba(255,255,255,0.1)`, borderRadius: '20px', fontSize: '14px', fontWeight: 'bold', color: theories[activeTab].color, border: `1px solid ${theories[activeTab].color}` }}>
                                    {theories[activeTab].title}
                                </div>
                            </div>

                            <div style={{ display: 'flex', gap: '20px', flex: 1 }}>
                                {/* 硅基 ANN 面板 */}
                                <div style={{ flex: 1, background: 'linear-gradient(180deg, rgba(59, 130, 246, 0.1) 0%, transparent 100%)', borderTop: '2px solid #3b82f6', borderRadius: '12px', padding: '20px', position: 'relative' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#93c5fd', fontWeight: 'bold', marginBottom: '16px', fontSize: '16px' }}>
                                        <Layers size={20} /> ANN 数学计算宇宙空间
                                    </div>
                                    <div style={{ fontSize: '14px', color: '#bfdbfe', lineHeight: '1.7' }}>
                                        {theories[activeTab].ann_desc}
                                    </div>

                                    {/* Abstract visualization indicator */}
                                    <div style={{ position: 'absolute', bottom: '20px', left: '20px', right: '20px', height: '60px', border: '1px solid rgba(59, 130, 246, 0.3)', borderRadius: '8px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                        {activeTab === 0 && <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ repeat: Infinity, duration: 2 }} style={{ width: '80%', height: '2px', background: 'rgba(59, 130, 246, 0.8)', boxShadow: '0 0 10px #3b82f6' }} />}
                                        {activeTab === 1 && <div style={{ display: 'flex', gap: '4px' }}>{[...Array(20)].map((_, i) => <motion.div key={i} animate={{ height: [10, 40, 10] }} transition={{ repeat: Infinity, duration: 1.5, delay: i * 0.1 }} style={{ width: '4px', background: '#3b82f6', borderRadius: '2px' }} />)}</div>}
                                        {activeTab === 2 && <motion.div initial={{ scale: 0 }} animate={{ scale: 3, opacity: [1, 0] }} transition={{ repeat: Infinity, duration: 1.5 }} style={{ width: '20px', height: '20px', background: '#3b82f6', borderRadius: '50%' }} />}
                                    </div>
                                </div>

                                {/* 碳基 SNN 面板 */}
                                <div style={{ flex: 1, background: 'linear-gradient(180deg, rgba(236, 72, 153, 0.1) 0%, transparent 100%)', borderTop: '2px solid #ec4899', borderRadius: '12px', padding: '20px', position: 'relative' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#f9a8d4', fontWeight: 'bold', marginBottom: '16px', fontSize: '16px' }}>
                                        <Share2 size={20} /> SNN 生物皮层脉冲神经映射
                                    </div>
                                    <div style={{ fontSize: '14px', color: '#fbcfe8', lineHeight: '1.7' }}>
                                        {theories[activeTab].snn_desc}
                                    </div>

                                    {/* Abstract SNN Biological indicator */}
                                    <div style={{ position: 'absolute', bottom: '20px', left: '20px', right: '20px', height: '60px', border: '1px solid rgba(236, 72, 153, 0.3)', borderRadius: '8px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                        {activeTab === 0 && <div style={{ display: 'flex', gap: '10px' }}>{[...Array(6)].map((_, i) => <div key={i} style={{ width: '8px', height: '8px', borderRadius: '50%', background: i === 3 ? '#ec4899' : 'rgba(236, 72, 153, 0.2)', boxShadow: i === 3 ? '0 0 10px #ec4899' : 'none' }} />)}</div>}
                                        {activeTab === 1 && <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 3, ease: "linear" }} style={{ position: 'relative', width: '40px', height: '40px' }}><div style={{ position: 'absolute', top: 0, left: '50%', transform: 'translateX(-50%)', width: '8px', height: '8px', borderRadius: '50%', background: '#ec4899', boxShadow: '0 0 10px #ec4899' }} /><div style={{ position: 'absolute', bottom: 0, left: '50%', transform: 'translateX(-50%)', width: '8px', height: '8px', borderRadius: '50%', background: '#ec4899', boxShadow: '0 0 10px #ec4899' }} /></motion.div>}
                                        {activeTab === 2 && <motion.svg width="100%" height="40" preserveAspectRatio="none" viewBox="0 0 100 40"><motion.path d="M 0 30 L 30 30 L 40 5 L 50 40 L 60 30 L 100 30" stroke="#ec4899" strokeWidth="2" fill="none" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ repeat: Infinity, duration: 2 }} style={{ filter: 'drop-shadow(0 0 4px #ec4899)' }} /></motion.svg>}
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
};

export default SNNBrainMappingGraph;
