import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Line } from '@react-three/drei';
import * as THREE from 'three';

// 神经元组件
const Neuron = ({ position, layerTarget, isActive, isConcept, label, color }) => {
    const meshRef = useRef();

    useFrame((state, delta) => {
        if (isActive) {
            meshRef.current.rotation.x += delta;
            meshRef.current.rotation.y += delta;
        }
    });

    const scale = isActive ? (isConcept ? 2.0 : 1.3) : 0.8;
    const opacity = isActive ? 1.0 : 0.2;
    const emissiveInt = isActive ? (isConcept ? 3.0 : 1.5) : 0;

    return (
        <group position={position}>
            <mesh ref={meshRef} scale={[scale, scale, scale]}>
                <sphereGeometry args={[0.3, 32, 32]} />
                <meshStandardMaterial
                    color={color}
                    emissive={color}
                    emissiveIntensity={emissiveInt}
                    transparent
                    opacity={opacity}
                    roughness={0.2}
                    metalness={0.8}
                />
            </mesh>
            {isActive && label && (
                <Html distanceFactor={15}>
                    <div style={{
                        color: 'white',
                        background: isConcept ? 'rgba(255, 50, 50, 0.8)' : 'rgba(0, 150, 255, 0.6)',
                        padding: '4px 8px',
                        borderRadius: '4px',
                        fontSize: isConcept ? '12px' : '10px',
                        fontWeight: 'bold',
                        whiteSpace: 'nowrap',
                        transform: 'translate3d(-50%, -150%, 0)',
                        pointerEvents: 'none'
                    }}>
                        {label}
                    </div>
                </Html>
            )}
        </group>
    );
};

// 神经连接连线组件
const NeuralConnection = ({ start, end, isActive }) => {
    const lineMaterial = useMemo(() => {
        return new THREE.LineBasicMaterial({
            color: isActive ? 0xffea00 : 0x444444,
            transparent: true,
            opacity: isActive ? 0.8 : 0.1,
            linewidth: isActive ? 2 : 1,
        });
    }, [isActive]);

    const geometry = useMemo(() => {
        const geo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(...start),
            new THREE.Vector3(...end)
        ]);
        return geo;
    }, [start, end]);

    return (
        <line geometry={geometry} material={lineMaterial} />
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
                        isActive: isBothActive
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
