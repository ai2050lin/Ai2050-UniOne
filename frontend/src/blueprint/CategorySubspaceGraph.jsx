import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Network, CircleDot, Zap, BarChart2, Box, Eye, Combine } from 'lucide-react';

const CategorySubspaceGraph = () => {
    const [animationPhase, setAnimationPhase] = useState(0);

    // Qwen3 真机提炼的数据指纹
    const varianceData = [
        { pc: 1, val: 11.37 },
        { pc: 2, val: 8.86 },
        { pc: 3, val: 5.49 },
        { pc: 4, val: 4.28 },
        { pc: 5, val: 4.14 }
    ];

    // 覆盖率 >= 90% 的顶级骨干纤维
    const topNeurons = [
        { id: 4, ratio: 1.0 }, { id: 0, ratio: 1.0 }, { id: 2, ratio: 1.0 },
        { id: 56, ratio: 1.0 }, { id: 24, ratio: 1.0 }, { id: 3, ratio: 1.0 },
        { id: 66, ratio: 1.0 }, { id: 12, ratio: 1.0 }, { id: 46, ratio: 1.0 },
        { id: 37, ratio: 1.0 }, { id: 29, ratio: 1.0 }, { id: 1, ratio: 1.0 },
        { id: 25, ratio: 0.98 }, { id: 355, ratio: 0.95 }, { id: 117, ratio: 0.93 }
    ];

    // 自动播放帧
    useEffect(() => {
        const timer = setInterval(() => {
            setAnimationPhase((prev) => (prev + 1) % 3);
        }, 5000);
        return () => clearInterval(timer);
    }, []);

    return (
        <div style={{
            background: 'rgba(10, 20, 30, 0.4)',
            border: '1px solid rgba(234, 179, 8, 0.3)',
            borderRadius: '12px',
            padding: '24px',
            marginTop: '20px',
            fontFamily: 'system-ui, -apple-system, sans-serif'
        }}>
            {/* 标题区 */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                <div style={{
                    background: 'rgba(234, 179, 8, 0.2)',
                    padding: '10px',
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                }}>
                    <Combine size={24} color="#facc15" />
                </div>
                <div>
                    <h3 style={{ margin: 0, fontSize: '18px', color: '#fef08a', fontWeight: 'bold' }}>
                        单类别海量子空间坍缩实测 (Category Motif Subspace)
                    </h3>
                    <div style={{ fontSize: '12px', color: '#eab308', marginTop: '4px' }}>
                        Qwen3-4B | 海量 60种不同动物概念提取 | 第 20 隐层残差流
                    </div>
                </div>
            </div>

            <div style={{ color: '#d1d5db', fontSize: '13px', lineHeight: '1.6', marginBottom: '24px' }}>
                当我们将“狮子”、“水母”、“章鱼”、“蜻蜓”等形态差异极大的词汇灌入大模型深处时，它是否会混乱到动用全网络几千维去记忆？通过真机物理探测，我们发现 <strong>所有的特征全被无情地引力坍缩挤压在了“同一张亚级纸片流形”上，并被迫共用了十几条主干光缆。</strong>
            </div>

            <div style={{ display: 'flex', gap: '24px' }}>
                {/* 左侧：PCA 降维坍缩解释图 (Scree Plot) */}
                <div style={{
                    flex: 1,
                    background: 'rgba(0,0,0,0.3)',
                    padding: '20px',
                    borderRadius: '8px',
                    border: '1px solid rgba(255,255,255,0.05)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '16px'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#9ca3af' }}>
                        <BarChart2 size={16} />
                        <span style={{ fontSize: '13px', fontWeight: 'bold' }}>PCA 流形降维极度扁平化</span>
                    </div>
                    <div style={{ fontSize: '11px', color: '#6b7280', marginBottom: '8px' }}>
                        (高达 3584 维度的广袤散乱空间中，仅仅利用前几个主成分即压榨出解释全部 60 种动物超过 34% 的特征方差。证明子概念全部被引力场扁平化汇聚。)
                    </div>

                    <div style={{ display: 'flex', alignItems: 'flex-end', gap: '12px', height: '120px', paddingBottom: '20px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                        {varianceData.map((d, i) => (
                            <div key={d.pc} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1, gap: '8px' }}>
                                <motion.div
                                    initial={{ height: 0 }}
                                    animate={{ height: `${(d.val / 15) * 100}%` }}
                                    transition={{ duration: 1.5, delay: i * 0.2 }}
                                    style={{
                                        width: '100%',
                                        background: i === 0 ? 'rgba(234, 179, 8, 0.8)' : 'rgba(234, 179, 8, 0.4)',
                                        borderRadius: '4px 4px 0 0',
                                        position: 'relative'
                                    }}
                                >
                                    <div style={{ position: 'absolute', top: '-18px', width: '100%', textAlign: 'center', fontSize: '10px', color: '#fef08a' }}>
                                        {d.val}%
                                    </div>
                                </motion.div>
                                <div style={{ fontSize: '11px', color: '#9ca3af' }}>PC{d.pc}</div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* 右侧：原子纤维满额点亮图 */}
                <div style={{
                    flex: 1.2,
                    background: 'rgba(0,0,0,0.3)',
                    padding: '20px',
                    borderRadius: '8px',
                    border: '1px solid rgba(255,255,255,0.05)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '16px'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#9ca3af' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Network size={16} />
                            <span style={{ fontSize: '13px', fontWeight: 'bold' }}>寻找“同类概念”的绝对通用主干线</span>
                        </div>
                        <div style={{ fontSize: '12px', color: '#eab308' }}>
                            离群独立抽取 60 种实体样本
                        </div>
                    </div>

                    <div style={{ fontSize: '11px', color: '#6b7280' }}>
                        下方的神经主通道即使提示词任意千秋变换，它们也在任何动物名词下保持了 <strong>90%~100% 的满载极强激活出勤率</strong>。这宣告了动物概念专有“物理传输特快车道”的真实存在。
                    </div>

                    <div style={{
                        display: 'flex',
                        flexWrap: 'wrap',
                        gap: '8px',
                        marginTop: '12px'
                    }}>
                        {topNeurons.map((neuron, i) => {
                            const isFull = neuron.ratio === 1.0;
                            return (
                                <motion.div
                                    key={neuron.id}
                                    animate={{
                                        boxShadow: animationPhase === 1
                                            ? `0 0 10px ${isFull ? 'rgba(234,179,8,0.8)' : 'rgba(234,179,8,0.4)'}`
                                            : 'none',
                                        borderColor: isFull ? 'rgba(234,179,8,0.8)' : 'rgba(255,255,255,0.2)'
                                    }}
                                    transition={{ duration: 1 }}
                                    style={{
                                        background: isFull ? 'rgba(234,179,8,0.1)' : 'rgba(255,255,255,0.02)',
                                        border: '1px solid',
                                        borderRadius: '6px',
                                        padding: '6px 10px',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        minWidth: '50px'
                                    }}
                                >
                                    <div style={{ fontSize: '10px', color: isFull ? '#facc15' : '#9ca3af' }}>
                                        #{neuron.id}
                                    </div>
                                    <div style={{ fontSize: '11px', fontWeight: 'bold', color: '#fff', marginTop: '4px' }}>
                                        {neuron.ratio * 100}%
                                    </div>
                                </motion.div>
                            )
                        })}
                    </div>
                    <div style={{ textAlign: 'center', fontSize: '12px', color: animationPhase === 1 ? '#facc15' : '#6b7280', transition: '0.5s', marginTop: '10px' }}>
                        {animationPhase === 1 ? '⚡ 60种不同动物传入 —— 主干原子被物理满额激活点爆' : '等待实体传入扫描 (Idle)'}
                    </div>
                </div>
            </div>

            {/* 引力总结提取 */}
            <div style={{
                marginTop: '16px',
                padding: '12px 16px',
                background: 'rgba(234, 179, 8, 0.1)',
                border: '1px solid rgba(234, 179, 8, 0.2)',
                borderRadius: '6px',
                display: 'flex',
                gap: '12px',
                alignItems: 'flex-start'
            }}>
                <Eye size={20} color="#facc15" style={{ flexShrink: 0, marginTop: '2px' }} />
                <div style={{ fontSize: '12px', color: '#d1d5db', lineHeight: '1.6' }}>
                    <span style={{ color: '#facc15', fontWeight: 'bold' }}>极端的类别物理引力场。</span>
                    实验测定，所有独立的相异很大之动物概念，到“抽象纯动物中心”点位的平均引力聚拢相似度高达 <strong style={{ color: '#fef08a' }}>0.8733</strong>。这意味着只要词属于该类，模型就通过这十几根 #4, #0, #56 号的共享特供突触桥接将它们瞬移压缩进极低维度扁平孤岛。只要阻断它们，大模型连水母和雄狮都无法区分。
                </div>
            </div>
        </div>
    );
};

export default CategorySubspaceGraph;
