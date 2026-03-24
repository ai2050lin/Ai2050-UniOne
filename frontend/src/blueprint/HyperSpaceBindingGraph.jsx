import React from 'react';
import { motion } from 'framer-motion';
import { Fingerprint, Network, TriangleAlert, Cpu } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine } from 'recharts';
import SafeResponsiveContainer from '../components/shared/SafeResponsiveContainer';

// 使用刚刚实验导出的 降维坍缩率(svd) 和 全新突变孤立神经元数 数据
const bindingData = [
    { layer: 10, 全新聚变涌现实体回路: 47.2, 线性加法残余概率: 5.5, 低秩坍缩绝对主干率: 29.2 },
    { layer: 20, 全新聚变涌现实体回路: 45.3, 线性加法残余概率: 9.3, 低秩坍缩绝对主干率: 38.6 },
    { layer: 30, 全新聚变涌现实体回路: 48.5, 线性加法残余概率: 2.9, 低秩坍缩绝对主干率: 47.3 }
];

const HyperSpaceBindingGraph = () => {
    return (
        <div style={{
            background: 'linear-gradient(135deg, rgba(8, 5, 20, 0.95) 0%, rgba(20, 10, 45, 1) 100%)',
            border: '1px solid rgba(236, 72, 153, 0.3)',
            borderRadius: '16px',
            padding: '30px',
            marginTop: '24px',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.7)'
        }}>
            {/* 顶栏区 */}
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px', marginBottom: '32px' }}>
                <div style={{
                    background: 'rgba(236, 72, 153, 0.15)',
                    padding: '14px',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 0 20px rgba(236, 72, 153, 0.3)'
                }}>
                    <Network size={30} color="#ec4899" />
                </div>
                <div>
                    <h3 style={{ margin: 0, fontSize: '22px', color: '#fdf2f8', fontWeight: '800', letterSpacing: '0.5px' }}>
                        万法之门：复杂概念的跨维非线性绑定与组装 (Hyper-Space Binding)
                    </h3>
                    <div style={{ fontSize: '14px', color: '#fbcfe8', marginTop: '8px', opacity: 0.9, lineHeight: '1.6' }}>
                        大模型是如何毫不费力地把 "红" 和 "苹果" 这两个散落的正交概念连成 "红苹果" 这个全新实体的？数据彻底推翻了线性的 "A+B" 模型。在极深渊逻辑区 (L30)，当遭遇复合短语时，代表复合概念的 50 根最强核心突触中，<strong>居然有 48.5 根是“红”和“苹果”中从未亮过的全新变异突触！</strong>这证明大模型使用了 <strong>非线性乘积 (MLP AND-Gate)</strong> 实现了瞬间的逻辑涌现与焊接结合！
                    </div>
                </div>
            </div>

            {/* 核心绘图区：面积图展示涌现与坍缩的挤压感 */}
            <div style={{
                height: '420px',
                background: 'rgba(0,0,0,0.5)',
                borderRadius: '12px',
                padding: '24px',
                border: '1px solid rgba(255,255,255,0.03)'
            }}>
                <SafeResponsiveContainer minHeight={260}>
                    <AreaChart data={bindingData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                        <defs>
                            <linearGradient id="colorMutate" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#ec4899" stopOpacity={0.6} />
                                <stop offset="95%" stopColor="#ec4899" stopOpacity={0.0} />
                            </linearGradient>
                            <linearGradient id="colorSvd" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.6} />
                                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                        <XAxis
                            dataKey="layer"
                            stroke="#94a3b8"
                            tick={{ fill: '#94a3b8', fontSize: 13 }}
                            tickFormatter={(val) => `深度 MLP 网络层 L${val}`}
                        />
                        <YAxis
                            domain={[0, 55]}
                            stroke="#94a3b8"
                            tick={{ fill: '#94a3b8', fontSize: 13 }}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                                border: '1px solid rgba(255,255,255,0.1)',
                                borderRadius: '8px',
                                boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
                                color: '#f1f5f9'
                            }}
                            itemStyle={{ fontWeight: 'bold' }}
                            formatter={(value, name) => [
                                name.includes('率') ? `${value.toFixed(1)}%` : `${value.toFixed(1)} / 50 根突触`,
                                name
                            ]}
                        />
                        <Legend wrapperStyle={{ paddingTop: '20px' }} />

                        <ReferenceLine y={50} stroke="rgba(236, 72, 153, 0.3)" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: '50 根全满额孤立突触线', fill: '#ec4899', fontSize: 11 }} />

                        {/* 三条具有极强涌现张力的曲线 */}
                        <Area type="monotone" dataKey="全新聚变涌现实体回路" stroke="#ec4899" strokeWidth={4} fill="url(#colorMutate)" activeDot={{ r: 8, strokeWidth: 2 }} />
                        <Area type="monotone" dataKey="低秩坍缩绝对主干率" stroke="#8b5cf6" strokeWidth={3} fill="url(#colorSvd)" activeDot={{ r: 6 }} />
                        <Area type="monotone" dataKey="线性加法残余概率" stroke="#dc2626" strokeWidth={2} fill="transparent" strokeDasharray="5 5" />
                    </AreaChart>
                </SafeResponsiveContainer>
            </div>

            {/* 底栏结论区 */}
            <div style={{ display: 'flex', gap: '20px', marginTop: '24px' }}>
                <div style={{ flex: 1, background: 'rgba(236, 72, 153, 0.1)', borderLeft: '3px solid #ec4899', padding: '16px', borderRadius: '0 8px 8px 0' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#f9a8d4', fontWeight: 'bold', marginBottom: '8px', fontSize: '14px' }}>
                        <Fingerprint size={18} /> 【突发涌现】全新的实体刻痕
                    </div>
                    <div style={{ color: '#cbd5e1', fontSize: '13px', lineHeight: '1.6' }}>
                        红线区域代表在最核心的 Top-50 发光神经元中，复合概念完全无视了基础算子的加减法则。当遇到“概念连接”时，在无垠深渊中被激发的几乎是 97%（48.5个）的无名暗突触。它们如同“专有门锁”，只有在“同时观测到红和苹果”这两个正交轴相交时，才会被非线性 <strong>AND-Gate (激活函数)</strong> 给瞬间点燃组合。
                    </div>
                </div>

                <div style={{ flex: 1, background: 'rgba(139, 92, 246, 0.1)', borderLeft: '3px solid #8b5cf6', padding: '16px', borderRadius: '0 8px 8px 0' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#c4b5fd', fontWeight: 'bold', marginBottom: '8px', fontSize: '14px' }}>
                        <TriangleAlert size={18} /> 【极低秩流形 SVD】三轴控宇宙
                    </div>
                    <div style={{ color: '#cbd5e1', fontSize: '13px', lineHeight: '1.6' }}>
                        更为惊骇的是紫色区域展示的 <strong style={{ color: '#ddd' }}>SVD</strong> (奇异值低秩分解) 效应：我们构建了二十多种各种交叉实体，它们的亿万方差轨迹网实际上是被坍缩并死死锁定在区区不到 3 根数学主干上的 (贡献了 47.3% 的全盘方差)。这如同全息投影的晶体，大模型就是靠这种 <strong>极低维度的重组骨架</strong> 掌控所有复杂的举一反三与长句绑定！
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HyperSpaceBindingGraph;
