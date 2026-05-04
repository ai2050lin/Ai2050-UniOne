/**
 * NeuralNetworkRenderer — DNN层结构3D可视化渲染器 (v3.0)
 * 
 * 在3D空间中展示完整的Transformer层结构:
 * - 每层是一个半透明圆盘, 颜色表示功能区域
 * - 层内展示Attention/FFN/LayerNorm/Residual组件
 * - 神经元用发光球体表示, 大小=激活强度, 颜色=子空间归属
 * - 层间连接线表示信号传播
 */
import React, { useRef, useMemo } from 'react';
import * as THREE from 'three';
import { Text, Line } from '@react-three/drei';
import {
  LAYER_GAP, PLANE_SIZE, SUBSPACE_COLORS,
  LAYER_FUNCTIONS, COMPONENT_TYPES,
  layerToFuncColor, layerToFuncLabel,
} from '../utils/constants';

// ==================== 层圆盘 ====================
function LayerDisk({ layer, nLayers, y, active, highlighted }) {
  const color = layerToFuncColor(layer, nLayers);
  const label = layerToFuncLabel(layer, nLayers);
  const opacity = highlighted ? 0.35 : active ? 0.18 : 0.08;
  const scale = highlighted ? 1.05 : 1.0;

  return (
    <group position={[0, y, 0]} scale={[scale, 1, scale]}>
      {/* 圆盘主体 */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[PLANE_SIZE / 2, 64]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={opacity}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      {/* 边缘发光环 */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[PLANE_SIZE / 2 - 0.15, PLANE_SIZE / 2, 64]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={highlighted ? 0.8 : active ? 0.4 : 0.15}
          side={THREE.DoubleSide}
        />
      </mesh>
      {/* 层号标签 */}
      <Text
        position={[-PLANE_SIZE / 2 - 1.5, 0, 0]}
        fontSize={0.7}
        color={highlighted ? '#fff' : '#64748b'}
        anchorX="right"
        anchorY="middle"
      >
        L{layer}
      </Text>
      {/* 功能标签 (仅高亮层显示) */}
      {highlighted && (
        <Text
          position={[PLANE_SIZE / 2 + 1.5, 0, 0]}
          fontSize={0.5}
          color={color}
          anchorX="left"
          anchorY="middle"
        >
          {label}
        </Text>
      )}
    </group>
  );
}

// ==================== 层内组件可视化 ====================
function LayerComponents({ layer, nLayers, y, animProgress, visibleComponents }) {
  const radius = PLANE_SIZE / 2 - 2;
  // 组件在层圆盘上的角度分布
  const componentAngles = {
    residual: 0,
    attention: Math.PI / 2,
    ffn: Math.PI,
    layer_norm: -Math.PI / 2,
  };

  return (
    <group position={[0, y, 0]}>
      {Object.entries(COMPONENT_TYPES).map(([key, comp]) => {
        if (visibleComponents && !visibleComponents.includes(key)) return null;
        const angle = componentAngles[key];
        const cx = Math.cos(angle) * radius * 0.5;
        const cz = Math.sin(angle) * radius * 0.5;
        const compOpacity = Math.min(1, animProgress * 2) * comp.opacity;

        return (
          <group key={key} position={[cx, 0.1, cz]}>
            {/* 组件节点 */}
            <mesh>
              <sphereGeometry args={[0.6, 16, 16]} />
              <meshStandardMaterial
                color={comp.color}
                emissive={comp.color}
                emissiveIntensity={0.3 * compOpacity}
                transparent
                opacity={compOpacity}
              />
            </mesh>
            {/* 组件标签 */}
            <Text
              position={[0, 1.0, 0]}
              fontSize={0.35}
              color={comp.color}
              anchorX="center"
              anchorY="bottom"
            >
              {comp.label}
            </Text>
          </group>
        );
      })}
    </group>
  );
}

// ==================== 神经元球体 ====================
function NeuronSpheres({ neurons, y, animProgress }) {
  if (!neurons || neurons.length === 0) return null;

  return (
    <group position={[0, y, 0]}>
      {neurons.map((n, i) => {
        const color = n.subspace === 'w_u' ? SUBSPACE_COLORS.semantic
          : n.subspace === 'w_u_perp' ? SUBSPACE_COLORS.grammar
          : n.subspace === 'logic' ? SUBSPACE_COLORS.logic
          : n.subspace === 'dark_matter' ? SUBSPACE_COLORS.dark_matter
          : '#64748b';
        const size = 0.1 + (n.activation || 0.5) * 0.25;
        const progressScale = Math.min(1, animProgress * 3);

        return (
          <mesh key={i} position={[n.x, 0.2, n.z]} scale={[progressScale, progressScale, progressScale]}>
            <sphereGeometry args={[size, 8, 8]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={0.4 * (n.activation || 0.3)}
              transparent
              opacity={0.7 * progressScale}
            />
          </mesh>
        );
      })}
    </group>
  );
}

// ==================== 层间连接线 ====================
function InterLayerConnections({ nLayers, activeRange, animProgress }) {
  const lines = useMemo(() => {
    const result = [];
    const step = Math.max(1, Math.floor(nLayers / 8)); // 每8层一条主连接线
    for (let l = 0; l < nLayers - step; l += step) {
      const y1 = l * LAYER_GAP;
      const y2 = (l + step) * LAYER_GAP;
      const inRange = (!activeRange) || (l >= activeRange[0] && l <= activeRange[1]);
      result.push({
        points: [[0, y1, 0], [0, y2, 0]],
        opacity: inRange ? 0.3 : 0.05,
        layer: l,
      });
    }
    return result;
  }, [nLayers, activeRange]);

  return (
    <>
      {lines.map((line, i) => (
        <Line
          key={i}
          points={line.points}
          color="#60a5fa"
          lineWidth={1}
          transparent
          opacity={line.opacity * Math.min(1, animProgress * 2)}
          dashed
          dashSize={0.5}
          gapSize={0.3}
        />
      ))}
    </>
  );
}

// ==================== 信号传播粒子 ====================
function SignalParticles({ nLayers, animProgress, activeScenario }) {
  const particleCount = 12;
  const particles = useMemo(() => {
    const result = [];
    for (let i = 0; i < particleCount; i++) {
      const phase = i / particleCount;
      const progress = (animProgress + phase) % 1;
      const layer = Math.floor(progress * (nLayers - 1));
      const y = layer * LAYER_GAP;
      const angle = (i / particleCount) * Math.PI * 2;
      const r = 2 + Math.sin(i * 1.7) * 3;
      result.push({
        position: [Math.cos(angle) * r, y, Math.sin(angle) * r],
        key: i,
        color: layerToFuncColor(layer, nLayers),
      });
    }
    return result;
  }, [animProgress, nLayers]);

  if (!activeScenario) return null;

  return (
    <>
      {particles.map(p => (
        <mesh key={p.key} position={p.position}>
          <sphereGeometry args={[0.15, 8, 8]} />
          <meshBasicMaterial color={p.color} transparent opacity={0.8} />
        </mesh>
      ))}
    </>
  );
}

// ==================== 功能区域标注 ====================
function FunctionRegionLabels({ nLayers }) {
  const regions = useMemo(() => {
    const result = [];
    const funcEntries = Object.entries(LAYER_FUNCTIONS);
    for (const [key, func] of funcEntries) {
      if (key === 'embedding') continue;
      const midLayer = Math.round((func.range[0] + func.range[1]) / 2);
      if (midLayer >= nLayers) continue;
      result.push({
        position: [PLANE_SIZE / 2 + 3, midLayer * LAYER_GAP, 0],
        label: func.label,
        color: func.color,
        key,
      });
    }
    return result;
  }, [nLayers]);

  return (
    <>
      {regions.map(r => (
        <group key={r.key} position={r.position}>
          <Text
            fontSize={0.6}
            color={r.color}
            anchorX="left"
            anchorY="middle"
            maxWidth={5}
          >
            {r.label}
          </Text>
          {/* 连接线到层区域 */}
          <Line
            points={[[-2, 0, 0], [0, 0, 0]]}
            color={r.color}
            lineWidth={1}
            transparent
            opacity={0.3}
          />
        </group>
      ))}
    </>
  );
}

// ==================== 主渲染器 ====================
export default function NeuralNetworkRenderer({
  nLayers = 36,
  dModel = 2560,
  activeLayerRange = null,
  highlightedLayer = null,
  visibleComponents = null,
  neurons = null,
  animProgress = 1,
  activeScenario = null,
  onHoverLayer = null,
}) {
  const groupRef = useRef();

  const layers = useMemo(() => {
    return Array.from({ length: nLayers }, (_, i) => i);
  }, [nLayers]);

  const isActive = (layer) => {
    if (!activeLayerRange) return true;
    return layer >= activeLayerRange[0] && layer <= activeLayerRange[1];
  };

  return (
    <group ref={groupRef}>
      {/* 层圆盘 */}
      {layers.map(l => (
        <LayerDisk
          key={l}
          layer={l}
          nLayers={nLayers}
          y={l * LAYER_GAP}
          active={isActive(l)}
          highlighted={highlightedLayer === l}
        />
      ))}

      {/* 层内组件 */}
      {layers.filter(l => isActive(l)).map(l => (
        <LayerComponents
          key={`comp-${l}`}
          layer={l}
          nLayers={nLayers}
          y={l * LAYER_GAP}
          animProgress={animProgress}
          visibleComponents={visibleComponents}
        />
      ))}

      {/* 神经元球体 */}
      {layers.filter(l => isActive(l) && neurons?.[l]).map(l => (
        <NeuronSpheres
          key={`neuron-${l}`}
          neurons={neurons[l]}
          y={l * LAYER_GAP}
          animProgress={animProgress}
        />
      ))}

      {/* 层间连接 */}
      <InterLayerConnections
        nLayers={nLayers}
        activeRange={activeLayerRange}
        animProgress={animProgress}
      />

      {/* 信号传播粒子 */}
      <SignalParticles
        nLayers={nLayers}
        animProgress={animProgress}
        activeScenario={activeScenario}
      />

      {/* 功能区域标注 */}
      <FunctionRegionLabels nLayers={nLayers} />
    </group>
  );
}
