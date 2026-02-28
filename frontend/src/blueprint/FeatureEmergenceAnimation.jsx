import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Network, Search, Zap, Binary, Activity, Fingerprint } from 'lucide-react';

export const FeatureEmergenceAnimation = () => {
    const [phase, setPhase] = useState(0); // 0: Init, 1: Compression, 2: Reorganization

    // Auto cycle through phases
    useEffect(() => {
        const timer = setInterval(() => {
            setPhase((p) => (p + 1) % 3);
        }, 5000);
        return () => clearInterval(timer);
    }, []);

    const phasesInfo = [
        {
            title: "1. 初始浑沌态 (Step 0-500)",
            desc: "所有神经元对信号杂乱无章地做出反应。特征网络还未结晶，系统表现为高有效维度（Rank~57）和无序的信号漫射，像是一团没有焦点的光束。",
            icon: <Activity color="#f472b6" size={24} />,
            color: "#f472b6"
        },
        {
            title: "2. 信息压缩相变 (Step 500-1500)",
            desc: "反向传播与突触规则迫使网络进行急剧的“降维打击”。无关噪声被阻断，有效特征秩产生断崖式暴跌（Rank跌至6），系统抛弃了冗杂，只留下极其精华的主干骨架。",
            icon: <Binary color="#38bdf8" size={24} />,
            color: "#38bdf8"
        },
        {
            title: "3. 结晶重组涌现 (Step 1500+)",
            desc: "通过极其有限的基础骨架，系统开始进行精确的多维度拼装和结构分化。稀疏化（~6%）正式确立，神经元完成“专家级化分工”，形成高维泛化和低维精确并存的特征域。",
            icon: <Fingerprint color="#a7f3d0" size={24} />,
            color: "#a7f3d0"
        }
    ];

    return (
        <div style={{
            background: 'rgba(0,0,0,0.3)',
            borderRadius: '16px',
            border: '1px solid rgba(168,85,247,0.4)',
            padding: '24px',
            marginBottom: '20px',
            position: 'relative',
            overflow: 'hidden'
        }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div style={{ color: '#d8b4fe', fontWeight: 'bold', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Search size={20} color="#c084fc" />
                    特征涌现动力学演示 (最新实验实录)
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                    {[0, 1, 2].map((idx) => (
                        <div key={idx}
                            onClick={() => setPhase(idx)}
                            style={{
                                cursor: 'pointer',
                                width: '12px', height: '12px', borderRadius: '50%',
                                background: phase === idx ? phasesInfo[idx].color : 'rgba(255,255,255,0.2)',
                                transition: 'all 0.3s'
                            }}
                        />
                    ))}
                </div>
            </div>

            <div style={{ color: '#e2e8f0', fontSize: '13px', lineHeight: '1.6', marginBottom: '20px' }}>
                <span style={{ color: '#fcd34d' }}>原理解说：</span>
                通过我们刚刚在后台运行的 `train_from_scratch.py` 模型压测实录，我们揭示了神经网络不是线性的“渐渐变聪明”，而是经历了猛烈的 <strong>几何相变（Phase Transition）</strong>。该过程称为“压缩与重组”。
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                {/* 左侧：原理动画 */}
                <div style={{
                    height: '240px',
                    background: 'rgba(15,23,42,0.6)',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.05)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    position: 'relative',
                    overflow: 'hidden'
                }}>
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={phase}
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 1.1 }}
                            transition={{ duration: 0.5 }}
                            style={{ position: 'absolute', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                        >
                            {/* SVG Animations depending on phase */}
                            {phase === 0 && (
                                <div style={{ position: 'relative', width: '150px', height: '150px' }}>
                                    {[...Array(20)].map((_, i) => (
                                        <motion.div key={i}
                                            animate={{
                                                x: [Math.random() * 150, Math.random() * 150],
                                                y: [Math.random() * 150, Math.random() * 150],
                                                opacity: [0.3, 0.8, 0.3]
                                            }}
                                            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                                            style={{ position: 'absolute', width: 6, height: 6, borderRadius: '50%', background: '#f472b6' }}
                                        />
                                    ))}
                                    <div style={{ color: '#f472b6', position: 'absolute', bottom: -30, left: 30, fontSize: '12px', fontWeight: 'bold' }}>Rank: 57.5 (高漫射)</div>
                                </div>
                            )}

                            {phase === 1 && (
                                <div style={{ position: 'relative', width: '150px', height: '150px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <motion.div
                                        animate={{ scale: [1, 0.2] }}
                                        transition={{ duration: 1.5, ease: "easeInOut" }}
                                        style={{ width: '100%', height: '100%', borderRadius: '50%', border: '4px solid #38bdf8', borderStyle: 'dashed' }}
                                    />
                                    <motion.div
                                        animate={{ rotate: 360 }}
                                        transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                                        style={{ position: 'absolute', width: 40, height: 40, background: '#38bdf8', borderRadius: '8px', boxShadow: '0 0 20px #38bdf8' }}
                                    />
                                    <div style={{ color: '#38bdf8', position: 'absolute', bottom: -30, left: 25, fontSize: '12px', fontWeight: 'bold' }}>Rank: 6.1 (极致压缩)</div>
                                </div>
                            )}

                            {phase === 2 && (
                                <div style={{ position: 'relative', width: '150px', height: '150px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <Network color="#a7f3d0" size={100} strokeWidth={1} />
                                    {[...Array(5)].map((_, i) => (
                                        <motion.div key={i}
                                            animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] }}
                                            transition={{ duration: 1.5, delay: i * 0.2, repeat: Infinity }}
                                            style={{
                                                position: 'absolute', width: 12, height: 12, background: '#a7f3d0', borderRadius: '50%', boxShadow: '0 0 10px #a7f3d0',
                                                left: 40 + Math.cos(i) * 40, top: 40 + Math.sin(i) * 40
                                            }}
                                        />
                                    ))}
                                    <div style={{ color: '#a7f3d0', position: 'absolute', bottom: -30, left: 15, fontSize: '12px', fontWeight: 'bold' }}>Sparsity: 6.2% (高阶分化)</div>
                                </div>
                            )}
                        </motion.div>
                    </AnimatePresence>
                </div>

                {/* 右侧：解说文本 */}
                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <AnimatePresence mode="popLayout">
                        <motion.div
                            key={phase}
                            initial={{ x: 20, opacity: 0 }}
                            animate={{ x: 0, opacity: 1 }}
                            exit={{ x: -20, opacity: 0 }}
                            transition={{ duration: 0.3 }}
                        >
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                                {phasesInfo[phase].icon}
                                <div style={{ color: phasesInfo[phase].color, fontWeight: 'bold', fontSize: '16px' }}>
                                    {phasesInfo[phase].title}
                                </div>
                            </div>
                            <div style={{ color: '#cbd5e1', fontSize: '13px', lineHeight: '1.8' }}>
                                {phasesInfo[phase].desc}
                            </div>

                            <div style={{ marginTop: '20px', padding: '10px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', borderLeft: `3px solid ${phasesInfo[phase].color}` }}>
                                <div style={{ color: '#94a3b8', fontSize: '11px', marginBottom: '4px' }}>物理学隐喻</div>
                                <div style={{ color: '#e2e8f0', fontSize: '12px', fontStyle: 'italic' }}>
                                    {phase === 0 && "就像宇宙大爆炸初期的高温等离子体，能量到处乱窜无规律。"}
                                    {phase === 1 && "像恒星由于引力突然坍结，核心密度急剧上升，抛离多余废料。"}
                                    {phase === 2 && "降温固化，形成有着极高硬度（特异性）的完美晶体结构（正交化）。"}
                                </div>
                            </div>
                        </motion.div>
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
};
