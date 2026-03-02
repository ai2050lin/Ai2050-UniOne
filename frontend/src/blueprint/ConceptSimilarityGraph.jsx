import React from 'react';
import { motion } from 'framer-motion';
import { Network, Activity, GitMerge, Combine } from 'lucide-react';

const ConceptSimilarityGraph = () => {
    // 提取出的真实类内/类间 Jaccard 交并比极化度演变数据
    // intra_similarity_pct & inter_similarity_pct
    const evolutionData = [
        { layer: 0, intra: 95.57, inter: 95.07, ratio: 1.01 },
        { layer: 5, intra: 85.71, inter: 82.28, ratio: 1.04 },
        { layer: 10, intra: 77.83, inter: 70.23, ratio: 1.11 },
        { layer: 15, intra: 76.72, inter: 68.80, ratio: 1.12 },
        { layer: 20, intra: 75.46, inter: 67.30, ratio: 1.12 },
        { layer: 25, intra: 78.37, inter: 70.14, ratio: 1.12 },
        { layer: 30, intra: 66.85, inter: 57.06, ratio: 1.17 },
        { layer: 32, intra: 67.86, inter: 57.26, ratio: 1.19 },
        { layer: 34, intra: 66.04, inter: 55.43, ratio: 1.19 }, // 极化排斥点
        { layer: 35, intra: 81.37, inter: 76.59, ratio: 1.06 }  // 万法归一坍缩点
    ];

    const chartHeight = 220;
    const paddingY = 40;
    const paddingX = 40;

    // Y轴的物理极值映射 (50% ~ 100%)
    const minY = 50;
    const maxY = 100;

    // 计算坐标系缩放
    const scaleY = (val) => chartHeight - ((val - minY) / (maxY - minY)) * chartHeight + paddingY;
    const scaleX = (index) => paddingX + (index / (evolutionData.length - 1)) * (850 - paddingX * 2);

    // 构建 SVG 路径串
    const generatePath = (key) => {
        let path = '';
        evolutionData.forEach((d, i) => {
            const x = scaleX(i);
            const y = scaleY(d[key]);
            if (i === 0) path += `M ${x},${y}`;
            else path += ` L ${x},${y}`;
        });
        return path;
    };

    const intraPath = generatePath('intra');
    const interPath = generatePath('inter');

    return (
        <div style={{
            background: 'linear-gradient(to bottom right, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.95) 100%)',
            border: '1px solid rgba(56, 189, 248, 0.3)',
            borderRadius: '16px',
            padding: '30px',
            marginTop: '24px',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.5)'
        }}>
            {/* 顶栏标题 */}
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px', marginBottom: '32px' }}>
                <div style={{
                    background: 'rgba(56, 189, 248, 0.15)',
                    padding: '14px',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 0 20px rgba(56, 189, 248, 0.3)'
                }}>
                    <Combine size={30} color="#38bdf8" />
                </div>
                <div>
                    <h3 style={{ margin: 0, fontSize: '22px', color: '#f0f9ff', fontWeight: '800', letterSpacing: '0.5px' }}>
                        深空结构同化与异化排斥率 (Jaccard Overlap Evolution)
                    </h3>
                    <div style={{ fontSize: '14px', color: '#bae6fd', marginTop: '8px', opacity: 0.9, lineHeight: '1.5' }}>
                        对百项概念实体在全隐层游走期间的“重叠碰撞率”做精准记录：类内相似度（特征抱团） vs 类间异构率（特征正交极化）。
                    </div>
                </div>
            </div>

            {/* 图表主区域 */}
            <div style={{ position: 'relative', width: '100%', height: '320px', background: 'rgba(0,0,0,0.3)', borderRadius: '12px', padding: '10px 0' }}>
                <svg width="100%" height="100%" viewBox="0 0 850 300" style={{ overflow: 'visible' }}>

                    {/* 背景网络辅助线 */}
                    {[50, 60, 70, 80, 90, 100].map(val => (
                        <g key={val}>
                            <line
                                x1={paddingX} y1={scaleY(val)}
                                x2={850 - paddingX} y2={scaleY(val)}
                                stroke="rgba(255,255,255,0.05)" strokeWidth="1" strokeDasharray="4,4"
                            />
                            <text x={paddingX - 10} y={scaleY(val) + 4} fill="#64748b" fontSize="12" textAnchor="end">{val}%</text>
                        </g>
                    ))}

                    {/* 收敛/排斥极值标注带 */}
                    <rect x={scaleX(6) - 15} y={paddingY} width={130} height={chartHeight} fill="rgba(16, 185, 129, 0.05)" />
                    <rect x={scaleX(9) - 15} y={paddingY} width={40} height={chartHeight} fill="rgba(239, 68, 68, 0.1)" />

                    {/* 类间异构曲线 (Heterogeneity - 紫红) */}
                    <motion.path
                        d={interPath}
                        fill="none"
                        stroke="#f43f5e"
                        strokeWidth="3"
                        initial={{ pathLength: 0, opacity: 0 }}
                        animate={{ pathLength: 1, opacity: 1 }}
                        transition={{ duration: 2, ease: "easeInOut" }}
                    />

                    {/* 类内同质曲线 (Homogeneity - 青蓝) */}
                    <motion.path
                        d={intraPath}
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth="3"
                        initial={{ pathLength: 0, opacity: 0 }}
                        animate={{ pathLength: 1, opacity: 1 }}
                        transition={{ duration: 2, delay: 0.5, ease: "easeInOut" }}
                    />

                    {/* 数据节点绘制 */}
                    {evolutionData.map((d, i) => (
                        <g key={i}>
                            {/* X轴标签 */}
                            <text x={scaleX(i)} y={300 - 10} fill="#94a3b8" fontSize="11" textAnchor="middle" fontWeight="bold">
                                L{d.layer}
                            </text>

                            {/* 类间异构节点 */}
                            <motion.circle
                                cx={scaleX(i)} cy={scaleY(d.inter)} r="5" fill="#1e293b" stroke="#f43f5e" strokeWidth="2"
                                initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 2 + i * 0.1 }}
                            />

                            {/* 类内同化节点 */}
                            <motion.circle
                                cx={scaleX(i)} cy={scaleY(d.intra)} r="5" fill="#1e293b" stroke="#38bdf8" strokeWidth="2"
                                initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 2.5 + i * 0.1 }}
                            />

                            {/* 注解极值点：34层和35层 */}
                            {d.layer === 34 && (
                                <text x={scaleX(i)} y={scaleY(d.inter) + 20} fill="#10b981" fontSize="12" textAnchor="middle" fontWeight="bold">
                                    55.4% (最大隔离)
                                </text>
                            )}
                            {d.layer === 35 && (
                                <g>
                                    <text x={scaleX(i) - 10} y={scaleY(d.intra) - 15} fill="#38bdf8" fontSize="12" textAnchor="end" fontWeight="bold">
                                        81.4%
                                    </text>
                                    <text x={scaleX(i) - 10} y={scaleY(d.inter) + 15} fill="#f43f5e" fontSize="12" textAnchor="end" fontWeight="bold">
                                        76.6% 奇点反弹
                                    </text>
                                </g>
                            )}
                        </g>
                    ))}
                </svg>
            </div>

            {/* 图例及解读 */}
            <div style={{ display: 'flex', gap: '20px', marginTop: '24px' }}>
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '10px', background: 'rgba(56, 189, 248, 0.1)', padding: '12px 16px', borderRadius: '8px' }}>
                    <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#38bdf8' }} />
                    <span style={{ fontSize: '13px', color: '#e0f2fe' }}><strong>类内同质 (Intra-Similarity):</strong> 越高代表该类别在流形宇宙中的特征越聚拢如星系核心。</span>
                </div>
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '10px', background: 'rgba(244, 63, 94, 0.1)', padding: '12px 16px', borderRadius: '8px' }}>
                    <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#f43f5e' }} />
                    <span style={{ fontSize: '13px', color: '#ffe4e6' }}><strong>跨界异构 (Inter-Similarity):</strong> 下降代表不同分类概念保持完美正交；但末端层飙升意味着万流收敛到奇点隧道。</span>
                </div>
            </div>
        </div>
    );
};

export default ConceptSimilarityGraph;
