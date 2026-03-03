import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Line } from '@react-three/drei';
import * as THREE from 'three';

// 神经元组件
const Neuron = ({ position, layerIdx, isActive, isConcept, label, color }) => {
    const meshRef = useRef();
    const materialRef = useRef();
    const labelRef = useRef();

    useFrame((state, delta) => {
        const time = state.clock.elapsedTime;
        const cycleLength = 6;
        const currentPhase = time % cycleLength;

        // 时序逐层点亮控制 (保持节奏, 但放大特效)
        const layerActivationTime = layerIdx * 1.0;
        let layerActiveOpacity = 0;

        if (currentPhase >= layerActivationTime) {
            if (currentPhase < 4.5) {
                // 亮起阶段变快
                layerActiveOpacity = Math.min(1.0, (currentPhase - layerActivationTime) * 3.5);
            } else {
                layerActiveOpacity = Math.max(0, 1.0 - (currentPhase - 4.5) * 2.0);
            }
        }

        // 基础动画: 旋转速度加快
        if (isActive && layerActiveOpacity > 0.05 && meshRef.current) {
            meshRef.current.rotation.x += delta * (isConcept ? 3 : 1.5);
            meshRef.current.rotation.y += delta * (isConcept ? 3 : 1.5);
        }

        // 材质更新
        if (materialRef.current && meshRef.current) {
            // 休眠时刻变得更暗更萎缩，以此衬托点亮时的爆发感
            let currentScale = 0.5;
            let currentOpacity = 0.1;
            let currentEmissive = 0;

            if (isActive) {
                // 点亮瞬间给个"脉冲式爆发"的缩放弹射感 (Bouncing scale)
                const pulse = Math.max(0, Math.sin(layerActiveOpacity * Math.PI));

                // 基础尺寸变大
                currentScale = 0.6 + ((isConcept ? 1.8 : 0.8) * layerActiveOpacity) + (pulse * 0.4);
                currentOpacity = 0.1 + (0.9 * layerActiveOpacity);

                // 基础发光极大增强
                let baseEmissive = isConcept ? 4.5 : 2.5;

                // 核心概念节点的呼吸灯效果: 加快频率并极大地增强光度
                if (isConcept && layerActiveOpacity > 0.8) {
                    const breathe = (Math.sin(time * 8.0) + 1) / 2;
                    baseEmissive += breathe * 6.5; // 之前是3.5
                    currentScale += breathe * 0.3;
                }

                currentEmissive = baseEmissive * layerActiveOpacity;
            }

            meshRef.current.scale.set(currentScale, currentScale, currentScale);
            materialRef.current.opacity = currentOpacity;
            materialRef.current.emissiveIntensity = currentEmissive;
        }

        // 标签透明度渐变 (也做一个提前显示)
        if (labelRef.current && isActive) {
            labelRef.current.style.opacity = Math.min(1.0, layerActiveOpacity * 1.5);
            // 让字体背景跟着活跃度变化，使其更明显
            labelRef.current.style.transform = `translate3d(-50%, -150%, 0) scale(${0.8 + layerActiveOpacity * 0.4})`;
        }
    });

    return (
        <group position={position}>
            <mesh ref={meshRef}>
                <sphereGeometry args={[0.3, 32, 32]} />
                <meshStandardMaterial
                    ref={materialRef}
                    color={color}
                    emissive={color}
                    transparent
                    roughness={0.1} // 让表面更光滑反光
                    metalness={0.9} // 更金属感
                />
            </mesh>
            {isActive && label && (
                <Html distanceFactor={15} zIndexRange={[100, 0]}>
                    <div ref={labelRef} style={{
                        color: 'white',
                        background: isConcept ? 'rgba(255, 40, 40, 0.9)' : 'rgba(20, 150, 255, 0.8)', // 增强背景不透明度
                        padding: '6px 10px',
                        borderRadius: '6px',
                        fontSize: isConcept ? '16px' : '14px', // 放大字号
                        border: isConcept ? '1px solid #ffcc00' : '1px solid rgba(255,255,255,0.2)', // 概念节点加个边框
                        fontWeight: 900,
                        whiteSpace: 'nowrap',
                        transformOrigin: 'bottom center',
                        pointerEvents: 'none',
                        opacity: 0,
                        boxShadow: `0 0 10px ${color}`, // 加个外发光
                        transition: 'opacity 0.1s, transform 0.1s'
                    }}>
                        {label}
                    </div>
                </Html>
            )}
        </group>
    );
};

// 神经连接连线组件
const NeuralConnection = ({ start, end, isActive, startLayerIdx }) => {
    const materialRef = useRef();

    useFrame((state) => {
        if (!materialRef.current) return;

        const time = state.clock.elapsedTime;
        const cycleLength = 6;
        const currentPhase = time % cycleLength;

        // 连线比源节点晚0.3秒亮起，模拟流动感
        const activationStart = startLayerIdx * 1.0 + 0.3;

        // 非激活连线降为极底透明度，激活连线则极高
        let currentOpacity = isActive ? 0.05 : 0.02;

        if (isActive && currentPhase >= activationStart) {
            if (currentPhase < 4.5) {
                // 连线爆发
                currentOpacity = Math.min(1.0, 0.05 + (currentPhase - activationStart) * 3.0);
            } else {
                currentOpacity = Math.max(0.05, 1.0 - (currentPhase - 4.5) * 2.0);
            }
        }

        materialRef.current.opacity = currentOpacity;
        materialRef.current.linewidth = currentOpacity > 0.5 ? 4 : 1; // 线宽加大
    });

    const geometry = useMemo(() => {
        const geo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(...start),
            new THREE.Vector3(...end)
        ]);
        return geo;
    }, [start, end]);

    return (
        <line geometry={geometry}>
            <lineBasicMaterial ref={materialRef} color={isActive ? 0xffea00 : 0x444444} transparent={true} opacity={isActive ? 0.8 : 0.1} linewidth={isActive ? 2 : 1} />
        </line>
    );
};

// 生成网络结构和“苹果”相关数据
const generateAppleNetwork = () => {
    const layers = [
        { name: 'Input', count: 25, z: 0, spread: 8 },
        { name: 'Visual Features', count: 18, z: 4, spread: 6 },
        { name: 'Semantic Components', count: 12, z: 8, spread: 5 },
        { name: 'Concept Abstraction', count: 5, z: 12, spread: 3 }
    ];

    const neurons = [];
    const connections = [];

    // "苹果" 概念在各层的激活模式
    const targetIndices = {
        0: [2, 5, 8, 12, 17, 21], // 对应底层像素刺激
        1: [1, 4, 9, 14], // 对应: 红色，圆形，反光，曲面边缘
        2: [2, 7], // 对应: 水果语义，可食用甜味物理属性
        3: [2] // 终极 "苹果" 概念节点
    };

    const labels = {
        "1-1": "RGB:红色特征",
        "1-4": "几何:圆形",
        "1-9": "质感:反光/光滑",
        "2-2": "属性:水果类别",
        "2-7": "多模态:甜味/脆感",
        "3-2": "Concept: 苹果 (Apple)"
    };

    const layerColors = ["#888888", "#4facfe", "#43e97b", "#fa709a"];
    const activeLayerColors = ["#ffcc00", "#00f2fe", "#38f9d7", "#ff0844"];

    // 1. 生成神经元
    let idCounter = 0;
    layers.forEach((layer, lIdx) => {
        const layerNeurons = [];
        for (let i = 0; i < layer.count; i++) {
            const isTarget = targetIndices[lIdx].includes(i);
            // 按照圆形或网格分布
            const angle = (i / layer.count) * Math.PI * 2;
            const radius = isTarget ? layer.spread * 0.5 : layer.spread * (0.8 + Math.random() * 0.4);
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;

            const neuron = {
                id: idCounter++,
                layerIdx: lIdx,
                index: i,
                pos: [x, y, layer.z],
                isActive: isTarget,
                isConcept: lIdx === 3 && isTarget,
                label: isTarget ? labels[`${lIdx}-${i}`] : null,
                color: isTarget ? activeLayerColors[lIdx] : layerColors[lIdx]
            };
            layerNeurons.push(neuron);
            neurons.push(neuron);
        }
    });

    // 2. 生成连线
    for (let l = 0; l < layers.length - 1; l++) {
        const currentLayer = neurons.filter(n => n.layerIdx === l);
        const nextLayer = neurons.filter(n => n.layerIdx === l + 1);

        currentLayer.forEach(src => {
            // 随机连接到下一层的一些节点，但如果当前节点和目标节点都是激活的，强制连接
            nextLayer.forEach(tgt => {
                const isBothActive = src.isActive && tgt.isActive;
                const connectProb = isBothActive ? 1.0 : 0.15; // 提高稀疏性，除非都是激活路线

                if (Math.random() < connectProb) {
                    connections.push({
                        start: src.pos,
                        end: tgt.pos,
                        isActive: isBothActive,
                        startLayerIdx: src.layerIdx
                    });
                }
            });
        });
    }

    return { neurons, connections };
};

const AppleNeuronVis3D = () => {
    const { neurons, connections } = useMemo(() => generateAppleNetwork(), []);

    return (
        <div style={{ width: '100%', height: '500px', background: '#0a0a1a', borderRadius: '12px', overflow: 'hidden' }}>
            <Canvas camera={{ position: [15, 10, 30], fov: 50 }}>
                <color attach="background" args={['#050510']} />

                <ambientLight intensity={0.8} />
                <pointLight position={[10, 10, 10]} intensity={2.0} />

                <group position={[0, -2, -6]}>
                    {neurons.map(n => (
                        <Neuron
                            key={`neuron-${n.id}`}
                            layerIdx={n.layerIdx}
                            position={n.pos}
                            isActive={n.isActive}
                            isConcept={n.isConcept}
                            label={n.label}
                            color={n.color}
                        />
                    ))}

                    {connections.map((c, i) => (
                        <NeuralConnection
                            key={`conn-${i}`}
                            start={c.start}
                            end={c.end}
                            isActive={c.isActive}
                            startLayerIdx={c.startLayerIdx}
                        />
                    ))}
                </group>

                <OrbitControls
                    enablePan={true}
                    enableZoom={true}
                    enableRotate={true}
                    autoRotate={true}
                    autoRotateSpeed={0.5}
                />
            </Canvas>
        </div>
    );
};

export default AppleNeuronVis3D;
