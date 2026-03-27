import { Html, Line, Text } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { agiLayerRawScene } from './blueprint/data/agi_layer_raw_scene_v1';

const LAYER_COUNT = agiLayerRawScene.layerCount || 28;
const LAYER_STEP = 1.05;
const Y_OFFSET = -14;

const CATEGORY_COLORS = {
  共享承载: '#60a5fa',
  偏置偏转: '#f97316',
  多空间角色: '#a78bfa',
  逐层放大: '#22c55e',
  其他: '#94a3b8',
};

const RESEARCH_LAYER_OVERLAY = {
  static_encoding: ['共享承载'],
  dynamic_route: ['偏置偏转'],
  result_recovery: ['逐层放大'],
  propagation_encoding: ['逐层放大'],
  semantic_roles: ['多空间角色'],
};

const laneX = {
  对象: -14,
  属性: -8,
  位置: -2,
  操作: 4,
  共享承载: -10,
  偏置偏转: 10,
};

function layerY(layerIndex) {
  return Y_OFFSET + (layerIndex - 1) * LAYER_STEP;
}

function usePulse(base = 1, amplitude = 0.08, speed = 1.6) {
  const ref = useRef();
  useFrame((state) => {
    if (!ref.current) return;
    const pulse = base + Math.sin(state.clock.elapsedTime * speed) * amplitude;
    ref.current.scale.setScalar(pulse);
  });
  return ref;
}

function LayerBand({ index, active }) {
  const y = layerY(index);
  return (
    <group position={[0, y, 0]}>
      <mesh>
        <boxGeometry args={[30, 0.04, 8]} />
        <meshBasicMaterial
          color={active ? '#5eead4' : '#203246'}
          transparent
          opacity={active ? 0.38 : 0.16}
        />
      </mesh>
      <Line
        points={[
          [-15, 0.02, -4.4],
          [15, 0.02, -4.4],
        ]}
        color={active ? '#93c5fd' : '#35506b'}
        transparent
        opacity={0.85}
        lineWidth={1.2}
      />
      <Text
        position={[-17.2, 0.18, 0]}
        fontSize={0.28}
        color={active ? '#dbeafe' : '#7f94ad'}
        anchorX="right"
        anchorY="middle"
      >
        {`L${index}`}
      </Text>
    </group>
  );
}

function LayerSpine() {
  const points = useMemo(
    () => Array.from({ length: LAYER_COUNT }, (_, i) => new THREE.Vector3(0, layerY(i + 1), 0)),
    [],
  );

  return (
    <group>
      <Line points={points} color="#38bdf8" transparent opacity={0.55} lineWidth={1.5} />
      {Array.from({ length: LAYER_COUNT }, (_, i) => (
        <LayerBand key={i + 1} index={i + 1} active={i + 1 === 5 || i + 1 === 14 || i + 1 === 24 || i + 1 === 27} />
      ))}
    </group>
  );
}

function FloatingNode({ position, color, label, size = 0.24, onHover, onSelect, payload, shape = 'sphere' }) {
  const pulseRef = usePulse(1, 0.06, 1.4);

  return (
    <group position={position}>
      <mesh
        ref={pulseRef}
        onPointerOver={(event) => {
          event.stopPropagation();
          onHover?.(payload);
        }}
        onPointerOut={(event) => {
          event.stopPropagation();
          onHover?.(null);
        }}
        onClick={(event) => {
          event.stopPropagation();
          onSelect?.(payload);
        }}
      >
        {shape === 'box' ? (
          <boxGeometry args={[size, size, size]} />
        ) : (
          <sphereGeometry args={[size, 18, 18]} />
        )}
        <meshBasicMaterial color={color} transparent opacity={0.95} />
      </mesh>
      <mesh scale={[1.65, 1.65, 1.65]}>
        {shape === 'box' ? (
          <boxGeometry args={[size, size, size]} />
        ) : (
          <sphereGeometry args={[size, 18, 18]} />
        )}
        <meshBasicMaterial color={color} transparent opacity={0.16} />
      </mesh>
      <Text position={[0, size * 2.4, 0]} fontSize={0.18} color="#dbeafe" anchorX="center" anchorY="bottom">
        {label}
      </Text>
    </group>
  );
}

function LayerAnchors({ nodes, onHover, onSelect }) {
  return (
    <group>
      {nodes.map((node, index) => {
        const x = -3 + ((index % 4) - 1.5) * 2.1;
        const z = 4.8 + Math.floor(index / 4) * 1.1;
        const position = [x, layerY(node.layerIndex), z];
        const payload = {
          type: 'layerfirst_runtime',
          label: node.label,
          nodeKind: '层级运行锚点',
          category: node.category,
          layerIndex: node.layerIndex,
          layerLabel: `L${node.layerIndex}`,
          position,
          sourceStage: node.sourceStage,
          sourceOutput: node.sourceOutput,
          detailText: `${node.category} · ${node.layerKind}`,
        };
        return (
          <FloatingNode
            key={node.id}
            position={position}
            color={CATEGORY_COLORS[node.category] || CATEGORY_COLORS.其他}
            label={node.label}
            size={0.26}
            onHover={onHover}
            onSelect={onSelect}
            payload={payload}
          />
        );
      })}
    </group>
  );
}

function ParameterRack({ nodes, activeCategories, onHover, onSelect }) {
  return (
    <group>
      {nodes.map((node, index) => {
        const x = node.category === '共享承载' ? -11.2 : 11.2;
        const z = -1.8 + (index % 5) * 0.95;
        const position = [x, layerY(node.suggestedLayer), z];
        const active = activeCategories.includes(node.category);
        const payload = {
          type: 'layerfirst_parameter',
          label: node.label,
          nodeKind: '参数位节点',
          role: node.role,
          category: node.category,
          layerIndex: node.suggestedLayer,
          layerLabel: `L${node.suggestedLayer}`,
          dimIndex: node.dimIndex,
          members: [node.dimIndex],
          position,
          sourceStage: node.sourceStage,
          sourceOutput: node.sourceOutput,
          detailText: `${node.category} · ${node.role}`,
        };
        return (
          <group key={node.id}>
            <Line
              points={[
                [x > 0 ? 6.2 : -6.2, layerY(node.suggestedLayer), 0],
                position,
              ]}
              color={active ? (CATEGORY_COLORS[node.category] || '#94a3b8') : '#34495e'}
              transparent
              opacity={active ? 0.6 : 0.18}
              lineWidth={active ? 1.4 : 1}
            />
            <FloatingNode
              position={position}
              color={active ? (CATEGORY_COLORS[node.category] || '#94a3b8') : '#5b6c7d'}
              label={node.label}
              size={active ? 0.28 : 0.22}
              shape="box"
              onHover={onHover}
              onSelect={onSelect}
              payload={payload}
            />
          </group>
        );
      })}
    </group>
  );
}

function RoleLanes({ nodes, activeCategories, onHover, onSelect }) {
  return (
    <group>
      {nodes.map((node, index) => {
        const x = laneX[node.lane] ?? 0;
        const y = layerY(4 + (index % 12));
        const z = -6 + Math.floor(index / 4) * 1.1;
        const active = activeCategories.includes(node.category);
        const payload = {
          type: 'layerfirst_role',
          label: node.label,
          nodeKind: '角色节点',
          role: node.positionKind,
          category: node.category,
          layerLabel: `L${4 + (index % 12)}`,
          position: [x, y, z],
          sourceStage: node.sourceStage,
          sourceOutput: node.sourceOutput,
          detailText: `${node.category} · ${node.positionKind}`,
        };
        return (
          <FloatingNode
            key={node.id}
            position={[x, y, z]}
            color={active ? (CATEGORY_COLORS[node.category] || '#94a3b8') : '#516276'}
            label={node.label}
            size={active ? 0.23 : 0.18}
            onHover={onHover}
            onSelect={onSelect}
            payload={payload}
          />
        );
      })}
    </group>
  );
}

function AmplificationRails({ nodes, visible, onHover, onSelect }) {
  if (!visible) return null;

  const sorted = [...nodes]
    .filter((node) => node.category === '逐层放大')
    .sort((a, b) => a.layerIndex - b.layerIndex);

  const points = sorted.map((node, index) => new THREE.Vector3(-1.2 + index * 1.2, layerY(node.layerIndex), 7.4));

  return (
    <group>
      {points.length > 1 && <Line points={points} color="#22c55e" transparent opacity={0.8} lineWidth={2.2} />}
      {sorted.map((node, index) => {
        const position = [points[index].x, points[index].y, points[index].z];
        const payload = {
          type: 'layerfirst_amplification',
          label: node.label,
          nodeKind: '逐层放大节点',
          category: node.category,
          layerIndex: node.layerIndex,
          layerLabel: `L${node.layerIndex}`,
          position,
          sourceStage: node.sourceStage,
          sourceOutput: node.sourceOutput,
          detailText: `${node.category} · ${node.layerKind}`,
        };
        return (
          <FloatingNode
            key={node.id}
            position={position}
            color="#22c55e"
            label={node.label}
            size={0.24}
            onHover={onHover}
            onSelect={onSelect}
            payload={payload}
          />
        );
      })}
    </group>
  );
}

function SceneLegend({ researchLayer }) {
  return (
    <Html position={[-14.5, layerY(28) + 1.4, -7.5]} transform={false}>
      <div
        style={{
          width: 250,
          padding: '10px 12px',
          borderRadius: 10,
          border: '1px solid rgba(255,255,255,0.12)',
          background: 'rgba(8, 14, 26, 0.82)',
          color: '#dbeafe',
          fontSize: 12,
          lineHeight: 1.5,
          backdropFilter: 'blur(10px)',
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: 6 }}>28层优先视图</div>
        <div>当前研究层：{researchLayer}</div>
        <div>主骨架：28个 layer（层）固定显示</div>
        <div>基础节点：层级锚点、参数位、角色点</div>
        <div>高级层：以叠加方式显示，不替代 layer（层）主视图</div>
      </div>
    </Html>
  );
}

export default function LayerFirstNeuronScene({
  languageFocus,
  onHover,
  onSelect,
}) {
  const researchLayer = languageFocus?.researchLayer || 'static_encoding';
  const activeCategories = RESEARCH_LAYER_OVERLAY[researchLayer] || [];

  const layerNodes = useMemo(() => agiLayerRawScene.layerNodes, []);
  const parameterNodes = useMemo(() => agiLayerRawScene.parameterNodes, []);
  const positionNodes = useMemo(() => agiLayerRawScene.positionNodes, []);

  return (
    <group position={[0, 0, 0]}>
      <LayerSpine />
      <LayerAnchors nodes={layerNodes} onHover={onHover} onSelect={onSelect} />
      <ParameterRack nodes={parameterNodes} activeCategories={activeCategories} onHover={onHover} onSelect={onSelect} />
      <RoleLanes nodes={positionNodes} activeCategories={activeCategories} onHover={onHover} onSelect={onSelect} />
      <AmplificationRails nodes={layerNodes} visible={researchLayer === 'result_recovery' || researchLayer === 'propagation_encoding'} onHover={onHover} onSelect={onSelect} />
      <SceneLegend researchLayer={researchLayer} />
    </group>
  );
}

