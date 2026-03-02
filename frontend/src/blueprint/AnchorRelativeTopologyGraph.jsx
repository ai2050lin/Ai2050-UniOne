import React from 'react';
import { motion } from 'framer-motion';
import { Apple, Eye, Zap, Eclipse } from 'lucide-react';

const AnchorRelativeTopologyGraph = () => {
    // 苹果作为万物锚点的特征突触碰撞重叠率 (Jaccard %)
    const layers = [0, 5, 10, 15, 20, 25, 30, 34, 35];

    // 我们精选了反映“粘稠-撕裂-重归一”的典型深度数据点阵
    // Banana (近亲), Rabbit (远亲生物), Sun (无关死物), Random Universe (宇宙均值)
    const topologyData = {
        Banana: [93.5, 87.5, 76.4, 76.4, 81.8, 87.5, 71.4, 66.6, 81.8],
        Rabbit: [93.5, 93.5, 76.4, 66.6, 81.8, 81.8, 66.6, 66.6, 93.5],
        Sun: [93.5, 93.5, 71.4, 53.8, 76.4, 62.1, 42.8, 50.0, 81.8],
        Universe: [97.4, 85.0, 66.2, 51.2, 63.4, 65.2, 51.1, 49.7, 79.1]
    };

    const targetColors = {
        Banana: "#fde047",     // 黄色香蕉
        Rabbit: "#fbcfe8",     // 粉色兔子
        Sun: "#f97316",        // 橙色太阳
        Universe: "#64748b"    // 灰阶底噪
    };

    const targetDesc = {
        Banana: "同族强近亲水果",
        Rabbit: "异族碳基生物",
        Sun: "天壤之别死物星体",
        Universe: "千词混沌均值引力"
    };

    // 绘制区域参数
    const chartHeight = 240;
    const paddingY = 40;
    const paddingX = 50;
    const minY = 30; // 太阳在L30的最低排斥率达到了42%左右
    const maxY = 100;

    // 坐标标量投射
    const scaleY = (val) => chartHeight - ((val - minY) / (maxY - minY)) * chartHeight + paddingY;
    const scaleX = (index) => paddingX + (index / (layers.length - 1)) * (800 - paddingX * 2);

    const generatePath = (dataArray) => {
        let path = '';
        dataArray.forEach((val, i) => {
            const x = scaleX(i);
            const y = scaleY(val);
            if (i === 0) path += `M ${x},${y}`;
            else path += ` L ${x},${y}`;
        });
        return path;
    };

    return (
        <div style={{
            background: 'linear-gradient(to bottom right, rgba(24, 24, 27, 0.95) 0%, rgba(9, 9, 11, 0.98) 100%)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '16px',
            padding: '30px',
            marginTop: '24px',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.6)'
        }}>
            {/* 主标题区 */}
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px', marginBottom: '36px' }}>
                <div style={{
                    background: 'rgba(239, 68, 68, 0.15)',
                    padding: '14px',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 0 20px rgba(239, 68, 68, 0.3)'
                }}>
                    <Apple size={30} color="#ef4444" />
                </div>
                <div>
                    <h3 style={{ margin: 0, fontSize: '22px', color: '#fef2f2', fontWeight: '800', letterSpacing: '0.5px' }}>
                        单点引力拓扑追踪 (Apple vs Universe Topology)
                    </h3>
                    <div style={{ fontSize: '14px', color: '#fca5a5', marginTop: '8px', opacity: 0.9, lineHeight: '1.5' }}>
                        剥开宏大宇宙面纱。我们在 Qwen3 流形内部署唯一的物理观测仪，绝对锁定 <strong>"苹果 (Apple)"</strong> 的高频放电特征；在它经历整整 35 层深加工时，测定其与不同亲密梯度的实体概念词产生的 <strong style={{ color: '#fff', background: 'rgba(239,68,68,0.5)', padding: '0 5px', borderRadius: '4px' }}>特征交集率</strong> 演化曲线。
                    </div>
                </div>
            </div>

            {/* 核心坐标系渲染区 */}
            <div style={{ position: 'relative', width: '100%', height: '340px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', padding: '10px 0', border: '1px solid rgba(255,255,255,0.05)' }}>
                <svg width="100%" height="100%" viewBox="0 0 800 320" style={{ overflow: 'visible' }}>

                    {/* 背景刻度 */}
                    {[40, 60, 80, 100].map(val => (
                        <g key={val}>
                            <line
                                x1={paddingX} y1={scaleY(val)}
                                x2={800 - paddingX} y2={scaleY(val)}
                                stroke="rgba(255,255,255,0.06)" strokeWidth="1" strokeDasharray="5,5"
                            />
                            <text x={paddingX - 10} y={scaleY(val) + 4} fill="#64748b" fontSize="12" textAnchor="end">{val}%</text>
                        </g>
                    ))}

                    {/* 区域隔离标注：混沌层 与 奇点区 */}
                    <rect x={paddingX} y={paddingY} width={scaleX(1) - paddingX} height={chartHeight} fill="rgba(148, 163, 184, 0.05)" />
                    <text x={scaleX(0.5)} y={paddingY + 20} fill="#94a3b8" fontSize="11" textAnchor="middle" style={{ opacity: 0.7 }}>原初重合区</text>

                    <rect x={scaleX(6) - 20} y={paddingY} width={100} height={chartHeight} fill="rgba(16, 185, 129, 0.05)" />
                    <text x={scaleX(6.8)} y={paddingY - 10} fill="#10b981" fontSize="12" textAnchor="middle" fontWeight="bold">深层结晶分离点 (L30-34)</text>

                    <rect x={scaleX(8) - 25} y={paddingY} width={60} height={chartHeight} fill="rgba(239, 68, 68, 0.1)" />
                    <text x={scaleX(8)} y={paddingY - 20} fill="#ef4444" fontSize="13" textAnchor="middle" fontWeight="bold">L35 万法归零</text>

                    {/* 实体排斥追踪线 */}
                    {Object.entries(topologyData).map(([target, dataArray], idx) => {
                        const pathString = generatePath(dataArray);
                        const color = targetColors[target];

                        return (
                            <g key={target}>
                                {/* 轨迹动画线条 */}
                                <motion.path
                                    d={pathString}
                                    fill="none"
                                    stroke={color}
                                    strokeWidth={target === 'Banana' || target === 'Sun' ? "4" : "2"}
                                    opacity={target === 'Universe' ? 0.6 : 0.9}
                                    strokeDasharray={target === 'Universe' ? "8,8" : "none"}
                                    initial={{ pathLength: 0, opacity: 0 }}
                                    animate={{ pathLength: 1, opacity: target === 'Universe' ? 0.6 : 0.9 }}
                                    transition={{ duration: 2.2, delay: idx * 0.3, ease: "easeInOut" }}
                                />
                                {/* 节点标记 */}
                                {dataArray.map((val, i) => (
                                    <motion.circle
                                        key={i}
                                        cx={scaleX(i)} cy={scaleY(val)}
                                        r={target === 'Sun' && i === 6 ? "7" : "4"}
                                        fill="#0f172a" stroke={color} strokeWidth="2"
                                        initial={{ scale: 0 }}
                                        animate={{ scale: 1 }}
                                        transition={{ delay: 2.5 + i * 0.1 + idx * 0.1 }}
                                    />
                                ))}

                                {/* 重点数值提点：L30处的极致隔离 */}
                                {target === 'Sun' && (
                                    <motion.text
                                        x={scaleX(6)} y={scaleY(dataArray[6]) + 20}
                                        fill="#f97316" fontSize="13" textAnchor="middle" fontWeight="bold"
                                        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 3.5 }}
                                    >
                                        仅仅 42.8% 重合
                                    </motion.text>
                                )}
                            </g>
                        )
                    })}

                    {/* 底部 X 层级轴 */}
                    {layers.map((l, i) => (
                        <text key={l} x={scaleX(i)} y={320 - 10} fill="#cbd5e1" fontSize="11" textAnchor="middle" fontWeight="bold">
                            L{l}
                        </text>
                    ))}
                </svg>
            </div>

            {/* 实体对比图例分析块 */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginTop: '24px' }}>
                {Object.entries(targetDesc).map(([target, desc]) => (
                    <div key={target} style={{ display: 'flex', gap: '12px', alignItems: 'center', padding: '12px 16px', background: `rgba(255,255,255,0.03)`, borderLeft: `4px solid ${targetColors[target]}`, borderRadius: '4px' }}>
                        <div style={{ color: '#fff', fontSize: '15px', fontWeight: 'bold', width: '80px' }}>vs {target}</div>
                        <div style={{ color: targetColors[target], fontSize: '13px', opacity: 0.9 }}>{desc}</div>
                    </div>
                ))}
            </div>

            {/* 结论解密条 */}
            <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start', background: 'rgba(234, 179, 8, 0.1)', border: '1px solid rgba(234, 179, 8, 0.2)', padding: '16px', borderRadius: '10px', marginTop: '24px' }}>
                <Zap size={24} color="#facc15" style={{ flexShrink: 0, marginTop: '2px' }} />
                <div style={{ fontSize: '13px', color: '#fef08a', lineHeight: '1.6' }}>
                    <strong>拓扑结语：</strong> 微观观测器完全定格了高维流形的“概念大撕裂”现象。苹果在初期（L0-5），哪怕和绝对的异质体（太阳 ☀️）的神经重叠都高达恐怖的 93.5%（底层表征共用）。
                    但进入绝对理智的中间隔离层（L30），苹果和近亲香蕉 🍌 依旧维持了极高的特征同构（71.4%），却将与太阳的重合率<strong>生生撕裂暴降到了 42% 冰点</strong> 的排斥地带（正交极化达成极限）；
                    最终的最终，在抵达最后一层 L35 奇点时，它们四个天壤之别的目标重新被坍缩折叠，再次以高达 80%~90% 的命中率跌入同一原力黑洞...
                </div>
            </div>
        </div>
    );
};

export default AnchorRelativeTopologyGraph;
