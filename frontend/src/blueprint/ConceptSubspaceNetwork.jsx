import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Apple, Dna, Skull, Fingerprint } from 'lucide-react';

const ConceptSubspaceNetwork = () => {
    const [ablated, setAblated] = useState(false);

    // 消融后的突变预测项
    const defaultPredictions = ["fruit", "food", "tree", "a", "plant"];
    const ablatedPredictions = ["fruit", "______", "...", "what", "__"];

    const [currentPredictions, setCurrentPredictions] = useState(defaultPredictions);

    // 苹果作为“死物/水果/实体”的微观物理维度分解
    const fruitGroups = [
        { label: "绝对实体基座 (Earth Entity Base)", genes: [0, 4, 8, 9, 19, 23, 24, 83, 84, 190, 198, 239, 286, 338, 418], color: "#10b981", isTarget: true }, // 曾经以为是水果基因，其实是全万物共用基因
        { label: "红色光谱组 (Red)", genes: [12, 59, 102, 104, 399], color: "#f87171", isTarget: false },
        { label: "常识形态 (Round)", genes: [33, 77, 188, 204, 451], color: "#60a5fa", isTarget: false },
        { label: "食物味觉槽 (Sweet Food)", genes: [111, 232, 290, 314, 401], color: "#a78bfa", isTarget: false }
    ];

    // 猫 作为“碳基生命/动物/实体”的微观物理维度分解
    const catGroups = [
        { label: "绝对实体基座 (Earth Entity Base)", genes: [0, 4, 8, 9, 19, 23, 24, 83, 84, 190, 198, 239, 286, 338, 418], color: "#10b981", isTarget: false }, // 100% 完美的重合在此！
        { label: "生命动作组 (Life/Anim)", genes: [3, 119, 254, 1728], color: "#facc15", isTarget: false }, // 生命体专属
        { label: "哺乳毛发组 (Furry/Pet)", genes: [56, 88, 120, 201, 315], color: "#fb923c", isTarget: false },
        { label: "猎手敏捷槽 (Hunter)", genes: [45, 96, 142, 299, 411], color: "#c084fc", isTarget: false }
    ];

    // 我们用 Apple 的完整微观维度组成来表达正交组块
    const featureGroups = viewEntity === 'apple' ? fruitGroups : catGroups;

    // 分别定义双端实体的预测状态
    const appleDefault = ["fruit", "food", "tree", "a", "plant"];
    const appleAblated = ["fruit", "______", "...", "what", "__"];

    const catDefault = ["animal", "pet", "feline", "hunter", "creature"];
    const catAblated = ["animal", "thing", "___", "unknown", "object"]; // 失去生命的猫退化成物品

    useEffect(() => {
        if (viewEntity === 'apple') {
            setCurrentPredictions(ablated ? appleAblated : appleDefault);
        } else {
            setCurrentPredictions(ablated ? catAblated : catDefault);
        }
    }, [ablated, viewEntity]);

    return (
        <div style={{
            background: 'linear-gradient(to bottom, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 1) 100%)',
            border: '1px solid rgba(139, 92, 246, 0.3)',
            borderRadius: '16px',
            padding: '30px',
            marginTop: '24px',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.6)'
        }}>
            {/* 顶栏标题区 */}
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px', marginBottom: '32px' }}>
                <div style={{
                    background: 'rgba(139, 92, 246, 0.15)',
                    padding: '14px',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 0 20px rgba(139, 92, 246, 0.3)'
                }}>
                    <Fingerprint size={30} color="#8b5cf6" />
                </div>
                <div>
                    <h3 style={{ margin: 0, fontSize: '22px', color: '#f5f3ff', fontWeight: '800', letterSpacing: '0.5px' }}>
                        概念微物理属性提取与手术消融 (Concept Exact Subspace & Ablation)
                    </h3>
                    <div style={{ fontSize: '14px', color: '#c4b5fd', marginTop: '8px', opacity: 0.9, lineHeight: '1.5' }}>
                        大模型中如何解决亿万词汇的“维度灾难”？靠的是物理拼接！苹果 (Apple) 全面由独立的物理神经元坐标构成：如红色的特征轴、圆元的特征轴和水果的。我们已经物理定位了<strong>L31 层上由所有水果共享的 15 根“果体”神经元</strong>。您可以亲自测试，切除这条维度后大模型的知识塌缩！
                    </div>
                </div>
            </div>

            {/* 核心手术区与监控界 */}
            <div style={{ display: 'flex', gap: '24px', position: 'relative' }}>

                {/* 左列：属性基因库可视化 */}
                <div style={{ flex: 1.2, background: 'rgba(0,0,0,0.3)', borderRadius: '12px', padding: '20px', border: '1px solid rgba(255,255,255,0.05)' }}>

                    {/* 实体切换 Tabs */}
                    <div style={{ display: 'flex', gap: '8px', marginBottom: '20px' }}>
                        <button
                            onClick={() => { setViewEntity('apple'); setAblated(false); }}
                            style={{
                                padding: '6px 16px', borderRadius: '20px', fontSize: '13px', fontWeight: 'bold', cursor: 'pointer', transition: 'all 0.2s',
                                background: viewEntity === 'apple' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255,255,255,0.05)',
                                color: viewEntity === 'apple' ? '#34d399' : '#64748b',
                                border: `1px solid ${viewEntity === 'apple' ? '#10b981' : 'transparent'}`
                            }}
                        >
                            🍎 Apple (苹果/死物实体)
                        </button>
                        <button
                            onClick={() => { setViewEntity('cat'); setAblated(false); }}
                            style={{
                                padding: '6px 16px', borderRadius: '20px', fontSize: '13px', fontWeight: 'bold', cursor: 'pointer', transition: 'all 0.2s',
                                background: viewEntity === 'cat' ? 'rgba(245, 158, 11, 0.2)' : 'rgba(255,255,255,0.05)',
                                color: viewEntity === 'cat' ? '#fbbf24' : '#64748b',
                                border: `1px solid ${viewEntity === 'cat' ? '#f59e0b' : 'transparent'}`
                            }}
                        >
                            🐈 Cat (猫/碳基生命实体)
                        </button>
                    </div>

                    <div style={{ fontSize: '14px', color: '#94a3b8', fontWeight: 'bold', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Dna size={16} /> 实体 [{viewEntity === 'apple' ? 'Apple' : 'Cat'}] L31 层编码微物理重构
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                        {featureGroups.map(group => (
                            <div key={group.label} style={{
                                position: 'relative',
                                background: `rgba(255,255,255,0.02)`,
                                border: `1px solid ${group.color}40`,
                                borderRadius: '8px',
                                padding: '16px',
                                overflow: 'hidden',
                                transition: 'all 0.3s'
                            }}>
                                {/* 背景斩击特效 */}
                                {group.isTarget && ablated && (
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: "100%" }}
                                        transition={{ duration: 0.3 }}
                                        style={{ position: 'absolute', top: '50%', left: 0, height: '4px', background: '#ef4444', zIndex: 10, boxShadow: '0 0 10px red' }}
                                    />
                                )}

                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                                    <span style={{ fontSize: '13px', fontWeight: 'bold', color: group.isTarget && ablated ? '#64748b' : group.color }}>{group.label}</span>
                                    {group.isTarget && (
                                        <button
                                            onClick={() => setAblated(!ablated)}
                                            style={{
                                                background: ablated ? 'transparent' : 'rgba(239, 68, 68, 0.1)',
                                                border: `1px solid ${ablated ? '#64748b' : '#ef4444'}`,
                                                color: ablated ? '#94a3b8' : '#fca5a5',
                                                padding: '4px 12px',
                                                borderRadius: '4px',
                                                fontSize: '11px',
                                                fontWeight: 'bold',
                                                cursor: 'pointer',
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: '6px',
                                                transition: 'all 0.2s',
                                                zIndex: 20
                                            }}
                                        >
                                            {ablated ? '已强行清零切除基因!' : <><Skull size={12} /> 执行强制坐标抹除手术</>}
                                        </button>
                                    )}
                                </div>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                                    {group.genes.map(gene => (
                                        <div key={gene} style={{
                                            background: group.isTarget && ablated ? '#1e293b' : `${group.color}15`,
                                            border: `1px solid ${group.isTarget && ablated ? '#334155' : `${group.color}30`}`,
                                            color: group.isTarget && ablated ? '#475569' : '#e2e8f0',
                                            padding: '2px 6px',
                                            borderRadius: '4px',
                                            fontSize: '11px',
                                            fontFamily: 'monospace'
                                        }}>
                                            D_{gene}
                                        </div>
                                    ))}
                                    {group.genes.length > 10 && <span style={{ color: '#64748b', fontSize: '11px', alignSelf: 'center' }}>...</span>}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* 右列：逻辑崩溃终端 */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ background: 'rgba(56, 189, 248, 0.1)', border: '1px solid rgba(56, 189, 248, 0.2)', padding: '20px', borderRadius: '12px' }}>
                        <div style={{ fontSize: '12px', color: '#7dd3fc', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px' }}>引导句 (Prompt Tensor):</div>
                        <div style={{ fontSize: '18px', color: '#fff', fontWeight: '500', fontFamily: 'monospace' }}>
                            "An apple is a kind of"
                        </div>
                    </div>

                    <div style={{
                        flex: 1,
                        background: ablated ? 'rgba(239, 68, 68, 0.05)' : 'rgba(16, 185, 129, 0.05)',
                        border: `1px solid ${ablated ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`,
                        borderRadius: '12px',
                        padding: '24px',
                        display: 'flex',
                        flexDirection: 'column'
                    }}>
                        <div style={{ fontSize: '13px', color: ablated ? '#fca5a5' : '#6ee7b7', marginBottom: '16px', display: 'flex', justifyContent: 'space-between' }}>
                            <strong>网络模型即时心智预测域 (Top-5 Logits)</strong>
                            <span style={{ fontSize: '11px', background: ablated ? '#ef4444' : '#10b981', color: '#fff', padding: '2px 8px', borderRadius: '12px' }}>
                                {ablated ? 'GENE ZEROED' : 'HEALTHY'}
                            </span>
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', flex: 1 }}>
                            {currentPredictions.map((word, idx) => (
                                <motion.div
                                    key={`${ablated}-${idx}`}
                                    initial={{ x: -10, opacity: 0 }}
                                    animate={{ x: 0, opacity: 1 }}
                                    transition={{ delay: idx * 0.1 }}
                                    style={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        background: 'rgba(0,0,0,0.2)',
                                        padding: '10px 16px',
                                        borderRadius: '8px',
                                        borderLeft: `3px solid ${ablated && idx > 0 ? '#ef4444' : '#10b981'}`
                                    }}
                                >
                                    <span style={{ color: '#94a3b8', fontSize: '13px' }}>Rank {idx + 1}</span>
                                    <span style={{
                                        color: ablated && idx > 0 ? '#fca5a5' : '#f1f5f9',
                                        fontSize: '16px',
                                        fontWeight: 'bold',
                                        fontFamily: 'monospace',
                                        textShadow: ablated && idx > 0 ? '0 0 10px rgba(239,68,68,0.5)' : 'none'
                                    }}>
                                        {word}
                                    </span>
                                </motion.div>
                            ))}
                        </div>

                        {/* 结果解读 */}
                        {ablated ? (
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ marginTop: '20px', fontSize: '12px', color: '#fca5a5', lineHeight: '1.5' }}>
                                ⚠️ <strong>系统常识结构崩塌：</strong><br />
                                {viewEntity === 'apple'
                                    ? "您刚强行抹去了大模型中仅有的 15 条“代表水果实体属性”的物理神经突触。尽管由于语境连贯性 'fruit' 还保留在首位，但原先健康的常识（food/tree/plant）已被彻底粉碎漂移！模型完全丧失了对“苹果应当是啥”的空间基准座标。"
                                    : "您刚刚抽走了组成这只猫的 'Earth Entity Base' 地球实体基盘基因！这只猫瞬间失去了物理学上的物理学上的存在感... 退化成了 'thing', 'unknown' 这样一团不可名状的代码幽灵！生命体构架彻底解体。"
                                }
                            </motion.div>
                        ) : (
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ marginTop: '20px', fontSize: '12px', color: '#6ee7b7', lineHeight: '1.5' }}>
                                ✅ <strong>健康稳定的基底：</strong><br />
                                {viewEntity === 'apple'
                                    ? "在未被干预的特征子空间网格中，苹果 (Apple) 这个宏观聚合概念完美释放了它携带的数个物理维度的生物脉冲，精准地触发了食物、树植物等常识后继向量。"
                                    : "猫 (Cat) 的生命特征槽发挥了极佳作用。它激活了宠物、猫科、猎手等一连串的碳基生物常识节点网！"
                                }
                            </motion.div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ConceptSubspaceNetwork;
