import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, Target, Zap, ShieldAlert, Cpu } from 'lucide-react';

const TrajectoryCodebookGraph = () => {
    // 真实提取出的跨界绝对主宰神经元重收敛率 (Hit Rate %)
    // 分别为: Biology生物界, Science科学界, Emotions情感界
    const layersToTrack = [4, 12, 18, 26, 30, 35];

    // 我们追踪在第 35 层达到 100% 收敛的 "神经元 Index: 4" 在不同深度的命中率
    const kernelTrace = {
        Biology: [0, 0, 5, 20, 80, 100],
        Science: [0, 0, 0, 10, 65, 100],
        Emotions: [0, 0, 2, 35, 90, 100]
    };

    return (
        <div style={{
            background: 'linear-gradient(to bottom right, rgba(15, 23, 42, 0.95) 0%, rgba(0, 0, 0, 0.95) 100%)',
            border: '1px solid rgba(168, 85, 247, 0.4)',
            borderRadius: '16px',
            padding: '30px',
            marginTop: '24px',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.7)'
        }}>
            {/* 顶栏 */}
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px', marginBottom: '36px' }}>
                <div style={{
                    background: 'rgba(168, 85, 247, 0.2)',
                    padding: '14px',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 0 20px rgba(168, 85, 247, 0.5)'
                }}>
                    <Target size={30} color="#d8b4fe" />
                </div>
                <div>
                    <h3 style={{ margin: 0, fontSize: '22px', color: '#f3e8ff', fontWeight: '800', letterSpacing: '0.5px' }}>
                        万法归一：跨类别绝对基底提取 (Universal Base Kernels)
                    </h3>
                    <div style={{ fontSize: '14px', color: '#d8b4fe', marginTop: '8px', opacity: 0.9, lineHeight: '1.5' }}>
                        测定跨入深层网络后，截然不同的知识界别（生物、科学、情感等）是否会不可避免地触发 <strong style={{ color: '#f8fafc' }}>同一组“奇点神经突触”</strong>？
                    </div>
                </div>
            </div>

            {/* 核心现象预警图 */}
            <div style={{
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px dashed rgba(239, 68, 68, 0.4)',
                borderRadius: '8px',
                padding: '16px 20px',
                marginBottom: '32px',
                display: 'flex', gap: '12px', alignItems: 'center'
            }}>
                <ShieldAlert size={24} color="#fca5a5" />
                <span style={{ fontSize: '14px', color: '#fecaca' }}>
                    <strong>深孔异常警告：</strong> 在通过模型第 30 层后，原本相互独立的各大知识领域发生严重收束坍缩。追踪显示特定的神经脉冲（如 <span style={{ color: '#fff', fontWeight: 'bold', background: '#ef4444', padding: '2px 6px', borderRadius: '4px' }}>Index 4</span>）在末端被所有范畴 100% 绝对调用。
                </span>
            </div>

            {/* 动态追踪表格 */}
            <div style={{ position: 'relative', marginTop: '40px', paddingBottom: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px', paddingLeft: '90px' }}>
                    {layersToTrack.map(l => (
                        <div key={l} style={{ fontSize: '13px', color: '#94a3b8', fontWeight: 'bold' }}>Layer {l}</div>
                    ))}
                </div>

                {Object.keys(kernelTrace).map((category, idx) => {
                    const colors = ["#4ade80", "#60a5fa", "#f472b6"];
                    const color = colors[idx];

                    return (
                        <div key={category} style={{ display: 'flex', alignItems: 'center', marginBottom: '32px', position: 'relative' }}>
                            <div style={{ width: '80px', fontSize: '14px', fontWeight: '700', color: color, textAlign: 'right', paddingRight: '15px' }}>
                                {category}
                            </div>

                            <div style={{ flex: 1, position: 'relative', height: '12px', background: 'rgba(255,255,255,0.05)', borderRadius: '6px' }}>
                                {/* 连接线 */}
                                <div style={{ position: 'absolute', top: '50%', left: 0, right: 0, height: '2px', background: `linear-gradient(90deg, transparent, ${color}40)`, transform: 'translateY(-50%)' }} />

                                {/* 命中率测定节点 */}
                                {kernelTrace[category].map((rate, i) => {
                                    const leftPct = (i / (layersToTrack.length - 1)) * 100;
                                    const dotSize = Math.max(8, rate / 4); // 根据唤醒率放大突触视觉
                                    const isActive = rate > 0;

                                    return (
                                        <div key={i} style={{ position: 'absolute', left: `${leftPct}%`, top: '50%', transform: 'translate(-50%, -50%)', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                                            <motion.div
                                                initial={{ scale: 0 }}
                                                animate={{ scale: 1 }}
                                                transition={{ delay: i * 0.15 + idx * 0.2 }}
                                                style={{
                                                    width: `${dotSize}px`, height: `${dotSize}px`, borderRadius: '50%',
                                                    background: isActive ? color : '#334155',
                                                    boxShadow: isActive ? `0 0 ${dotSize}px ${color}` : 'none',
                                                    border: `2px solid ${isActive ? '#fff' : '#64748b'}`,
                                                    zIndex: 2
                                                }}
                                            />
                                            {isActive && (
                                                <motion.div
                                                    initial={{ opacity: 0, y: 10 }}
                                                    animate={{ opacity: 1, y: 20 }}
                                                    transition={{ delay: i * 0.15 + idx * 0.2 + 0.3 }}
                                                    style={{ position: 'absolute', fontSize: '12px', fontWeight: 'bold', color: '#fff', background: 'rgba(0,0,0,0.8)', padding: '2px 6px', borderRadius: '4px', border: `1px solid ${color}40` }}
                                                >
                                                    {rate}%
                                                </motion.div>
                                            )}
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    )
                })}
            </div>

            {/* 结论框 */}
            <div style={{
                marginTop: '30px',
                padding: '20px',
                background: 'rgba(168, 85, 247, 0.1)',
                border: '1px solid rgba(168, 85, 247, 0.3)',
                borderRadius: '12px',
                display: 'flex',
                gap: '16px',
                alignItems: 'center'
            }}>
                <Cpu size={32} color="#d8b4fe" />
                <div style={{ fontSize: '13px', color: '#e2e8f0', lineHeight: '1.7' }}>
                    <strong>极点证明推论：</strong> 在浅层至中层（Layer 4 到 18），不同概念领域拥有各自不重叠的隔离计算流形，特定主神经元（如第4轴）对它们的调用接近 0。然而一旦潜入核心判断深水区（Layer 26 之后），物理现象急剧生变——所有迥异的流形特征被暴力重叠到极少数的几何膜面上，最终在 35 层奇点，它们 <strong style={{ color: '#fca5a5' }}>100% 毫无例外地指向并绝对垄断了完全一致的底层突触密码</strong>。<br />没有任何一类知识可以逃脱底层的宏大数学同一性。
                </div>
            </div>
        </div>
    );
};

export default TrajectoryCodebookGraph;
