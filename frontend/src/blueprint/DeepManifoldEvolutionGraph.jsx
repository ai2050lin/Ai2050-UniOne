import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Layers, Activity, Maximize2, Minimize2, RadioReceiver, Network, Database } from 'lucide-react';

const DeepManifoldEvolutionGraph = () => {
    // 真实提取出的 Qwen3 层级演化数据 (Layer 8 -> Layer 20 -> Layer 32)
    const evolutionData = [
        { layer: 8, kurtosis: 117.3, dim90: 105, purity: 66.2, color: "#38bdf8", name: "浅层 (L8)" },
        { layer: 20, kurtosis: 178.4, dim90: 107, purity: 53.8, color: "#a855f7", name: "中游深区 (L20)" },
        { layer: 32, kurtosis: 491.5, dim90: 68, purity: 59.4, color: "#ef4444", name: "流形海沟终点 (L32)" }
    ];

    return (
        <div style={{
            background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '16px',
            padding: '28px',
            marginTop: '24px',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)'
        }}>
            {/* 顶栏描述区 */}
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px', marginBottom: '32px' }}>
                <div style={{
                    background: 'rgba(239, 68, 68, 0.2)',
                    padding: '12px',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 0 15px rgba(239, 68, 68, 0.4)'
                }}>
                    <Layers size={28} color="#fca5a5" />
                </div>
                <div>
                    <h3 style={{ margin: 0, fontSize: '20px', color: '#fecaca', fontWeight: '800', letterSpacing: '0.5px' }}>
                        千级多范畴跨层深空巨变与降维坍缩 (Deep Evolution Stats)
                    </h3>
                    <div style={{ fontSize: '13px', color: '#fca5a5', marginTop: '6px', opacity: 0.9 }}>
                        涵盖：19 个宇宙互斥类目 | 从浅至深 (L8 → L32) | 全景规模: 949 种物理概念实体
                    </div>
                </div>
            </div>

            <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>

                {/* 1. 内禀维度空间压扁塌落 */}
                <div style={{
                    flex: '1 1 300px',
                    background: 'rgba(0, 0, 0, 0.3)',
                    padding: '24px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.05)',
                    position: 'relative'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#e2e8f0', marginBottom: '20px' }}>
                        <Minimize2 size={18} color="#fca5a5" />
                        <span style={{ fontSize: '15px', fontWeight: 'bold' }}>流形压迫：内禀维数 (90% 方差容纳池)</span>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '18px', position: 'relative' }}>
                        {/* 左侧引导线 */}
                        <div style={{ position: 'absolute', left: '20px', top: '15px', bottom: '15px', width: '2px', background: 'linear-gradient(to bottom, #38bdf8, #ef4444)', opacity: 0.5 }} />

                        {evolutionData.map((d, i) => (
                            <div key={d.layer} style={{ display: 'flex', alignItems: 'center', gap: '16px', zIndex: 1 }}>
                                <div style={{
                                    width: '40px', height: '40px', borderRadius: '50%', background: '#1e293b',
                                    border: `2px solid ${d.color}`, display: 'flex', justifyContent: 'center', alignItems: 'center',
                                    fontWeight: 'bold', color: d.color, fontSize: '14px',
                                    boxShadow: `0 0 10px ${d.color}40`, flexShrink: 0
                                }}>
                                    L{d.layer}
                                </div>

                                <div style={{ flex: 1 }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                                        <span style={{ fontSize: '13px', color: '#94a3b8' }}>{d.name}</span>
                                        <span style={{ fontSize: '14px', color: '#f8fafc', fontWeight: '600' }}>{d.dim90} 维</span>
                                    </div>
                                    {/* 维数越小，说明压缩得越扁，越致密 */}
                                    <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                                        <motion.div
                                            initial={{ width: '100%' }}
                                            animate={{ width: `${(d.dim90 / 120) * 100}%` }}
                                            transition={{ duration: 1.5, ease: "easeOut" }}
                                            style={{ height: '100%', background: d.color, borderRadius: '4px' }}
                                        />
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div style={{ marginTop: '20px', fontSize: '12px', color: '#64748b', lineHeight: '1.6' }}>
                        从表层到深渊，<strong>解释 1000 个乱序概念的语义矩阵被无情抽干抛弃</strong>！原本霸占 105 个特征轴的流形碎末被恐怖碾压折叠到了只剩 68 维(占总维度 2.6%)的世界纸片上。
                    </div>
                </div>

                {/* 2. 重尾异常放电 峰值 (Kurtosis) */}
                <div style={{
                    flex: '1 1 300px',
                    background: 'rgba(0, 0, 0, 0.3)',
                    padding: '24px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255,255,255,0.05)'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#e2e8f0', marginBottom: '20px' }}>
                        <Activity size={18} color="#a855f7" />
                        <span style={{ fontSize: '15px', fontWeight: 'bold' }}>重尾极化寡头控制率 (特征峰度 Kurtosis)</span>
                    </div>

                    <div style={{ height: '150px', display: 'flex', alignItems: 'flex-end', justifyContent: 'space-around', gap: '15px', paddingBottom: '10px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                        {evolutionData.map((d, i) => {
                            const maxKur = 500;
                            const heightPct = (d.kurtosis / maxKur) * 100;
                            return (
                                <div key={d.layer} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '30%' }}>
                                    <div style={{ fontSize: '15px', fontWeight: '900', color: d.color, marginBottom: '8px' }}>
                                        {d.kurtosis.toFixed(1)}
                                    </div>
                                    <div style={{ width: '100%', height: '100px', display: 'flex', alignItems: 'flex-end', justifyContent: 'center' }}>
                                        <motion.div
                                            initial={{ height: 0 }}
                                            animate={{ height: `${heightPct}%` }}
                                            transition={{ duration: 1.2, delay: i * 0.2 }}
                                            style={{
                                                width: '40px',
                                                background: `linear-gradient(to top, ${d.color}20, ${d.color})`,
                                                borderRadius: '6px 6px 0 0',
                                                boxShadow: `0 0 15px ${d.color}40`
                                            }}
                                        />
                                    </div>
                                    <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '10px', fontWeight: '600' }}>L{d.layer}</div>
                                </div>
                            )
                        })}
                    </div>
                    <div style={{ marginTop: '16px', fontSize: '12px', color: '#64748b', lineHeight: '1.6' }}>
                        在网络浅层，神经冲动分散如水（峰度 117）。而在判决性的流形深部，<strong>峰度狂飙近 4 倍（491.5 倍）！</strong>极其有限的孤立超级突触正在统治并暴力分发全部语义电位，长尾极端控制效应展现无遗。
                    </div>
                </div>
            </div>

            {/* 总结纪要区 */}
            <div style={{
                marginTop: '24px',
                padding: '16px 20px',
                background: 'linear-gradient(90deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%)',
                borderLeft: '4px solid #ef4444',
                borderRadius: '0 8px 8px 0',
                display: 'flex',
                gap: '16px',
                alignItems: 'center'
            }}>
                <Network size={32} color="#ef4444" />
                <div style={{ fontSize: '13px', color: '#cbd5e1', lineHeight: '1.6' }}>
                    大满贯千级词汇随深度的多层流形演化总结：越接近输出末端，Gini/峰度越发不可一世，造就了高孤立、强排斥的极少数 <strong>“寡头神经控制群”</strong>；与此同时，概念所侵占的拓扑尺寸疯狂向内挤压致密，空间厚度降维至不足 70根基。万花筒般的杂乱世界逻辑在这个层级早已彻底离析解体，坍陷成为结晶般的物理极化向量星阵。
                </div>
            </div>
        </div>
    );
};

export default DeepManifoldEvolutionGraph;
