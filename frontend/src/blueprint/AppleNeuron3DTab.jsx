import { Html, Line, OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import { Canvas, useFrame } from '@react-three/fiber';
import { useEffect, useMemo, useRef, useState } from 'react';

const LAYER_COUNT = 28;
const DFF = 18944;
const QUERY_NODE_COUNT = 12;

const APPLE_CORE_NEURONS = [
  {
    id: 'apple-micro-l8n7574',
    label: 'Apple Micro A',
    role: 'micro',
    layer: 8,
    neuron: 7574,
    strength: 0.0008401030507210194,
    metric: 'drop_target',
    value: 0.0008401358791161329,
    source: 'triscale_20260302_115048',
  },
  {
    id: 'apple-micro-l9n14608',
    label: 'Apple Micro B',
    role: 'micro',
    layer: 9,
    neuron: 14608,
    strength: 0.00014841170145061256,
    metric: 'drop_target',
    value: 0.00015364339924417436,
    source: 'triscale_20260302_115048',
  },
  {
    id: 'apple-macro-l23n16819',
    label: 'Apple Macro',
    role: 'macro',
    layer: 23,
    neuron: 16819,
    strength: 0.00003632260127393039,
    metric: 'drop_target',
    value: 0.00006799021775805159,
    source: 'triscale_20260302_115048',
  },
  {
    id: 'route-shared-l24n8124',
    label: 'Route Shared A',
    role: 'route',
    layer: 24,
    neuron: 8124,
    strength: 0.0006424156169391173,
    metric: 'drop_h3',
    value: 0.0006720473178789058,
    source: 'multihop_large_20260302_145153',
  },
  {
    id: 'route-shared-l27n16649',
    label: 'Route Shared B',
    role: 'route',
    layer: 27,
    neuron: 16649,
    strength: 0.0006115013281636399,
    metric: 'drop_h3',
    value: 0.0006115264830245798,
    source: 'multihop_large_20260302_145153',
  },
  {
    id: 'route-shared-l27n16936',
    label: 'Route Shared C',
    role: 'route',
    layer: 27,
    neuron: 16936,
    strength: 0.0011910316618790559,
    metric: 'drop_h3',
    value: 0.0011917482582827904,
    source: 'multihop_large_20260302_145153',
  },
];

const FRUIT_GENERAL_NEURONS = [
  { layer: 3, neuron: 11990, score: 2.9764 },
  { layer: 3, neuron: 11542, score: 2.7930 },
  { layer: 5, neuron: 0, score: 2.7226 },
  { layer: 3, neuron: 7001, score: 2.6621 },
  { layer: 6, neuron: 13742, score: 2.6597 },
  { layer: 4, neuron: 11956, score: 2.6546 },
  { layer: 4, neuron: 2250, score: 2.6116 },
  { layer: 1, neuron: 18127, score: 2.5913 },
];

const FRUIT_SPECIFIC_NEURONS = {
  apple: [
    { layer: 3, neuron: 13834, score: 1.86 },
    { layer: 0, neuron: 3167, score: 1.84 },
    { layer: 0, neuron: 6511, score: 1.78 },
    { layer: 3, neuron: 10117, score: 1.76 },
    { layer: 3, neuron: 18787, score: 1.75 },
  ],
  banana: [
    { layer: 0, neuron: 17808, score: 1.7 },
    { layer: 0, neuron: 7250, score: 1.68 },
    { layer: 0, neuron: 9767, score: 1.63 },
    { layer: 3, neuron: 11960, score: 1.61 },
    { layer: 0, neuron: 15284, score: 1.6 },
  ],
  orange: [
    { layer: 3, neuron: 7491, score: 3.19 },
    { layer: 0, neuron: 3600, score: 2.93 },
    { layer: 13, neuron: 8936, score: 2.78 },
    { layer: 0, neuron: 12643, score: 2.74 },
    { layer: 0, neuron: 17566, score: 2.74 },
  ],
  grape: [
    { layer: 0, neuron: 15727, score: 1.87 },
    { layer: 0, neuron: 16560, score: 1.73 },
    { layer: 0, neuron: 4714, score: 1.65 },
    { layer: 5, neuron: 10366, score: 1.65 },
    { layer: 0, neuron: 17870, score: 1.64 },
  ],
};

const FRUIT_COLORS = {
  apple: '#ff6b6b',
  banana: '#ffd166',
  orange: '#ff9f40',
  grape: '#b286ff',
};

const ROLE_COLORS = {
  micro: '#ff8d3b',
  macro: '#f6d365',
  route: '#39d0ff',
  fruitGeneral: '#6cf7d4',
  background: '#ffffff',
};

const DEFAULT_PREDICT_PROMPT = '苹果 是 一种';
const PREDICT_CHAIN_LENGTH = 10;

const TOKEN_TRANSITIONS = {
  苹果: ['是', '通常', '属于', '味道', '颜色'],
  apple: ['is', 'a', 'fruit', 'usually', 'sweet'],
  香蕉: ['是', '一种', '水果', '偏', '软'],
  banana: ['is', 'a', 'fruit', 'that', 'is'],
  水果: ['通常', '富含', '维生素', '可以', '食用'],
  fruit: ['is', 'rich', 'in', 'vitamins', 'and'],
  猫: ['是', '一种', '动物', '常见', '宠物'],
  dog: ['is', 'a', 'pet', 'animal', 'with'],
  是: ['一种', '一个', '在', '并且', '可'],
  is: ['a', 'an', 'often', 'related', 'to'],
  一种: ['水果', '动物', '概念', '实体', '结构'],
  a: ['fruit', 'concept', 'model', 'pet', 'node'],
};

const TOPIC_FALLBACKS = [
  {
    keywords: ['苹果', 'apple'],
    tokens: ['是', '一种', '水果', '通常', '偏甜', '富含', '纤维'],
  },
  {
    keywords: ['香蕉', 'banana'],
    tokens: ['是', '一种', '水果', '口感', '较软', '富含', '钾'],
  },
  {
    keywords: ['猫', 'cat'],
    tokens: ['是', '一种', '动物', '常见', '宠物', '动作', '灵活'],
  },
];

const DEFAULT_CHAIN_TOKENS = ['is', 'a', 'concept', 'mapped', 'through', 'layers', 'into', 'next', 'token'];

const ANALYSIS_MODE_OPTIONS = [
  { id: 'static', label: '静态分析', desc: '结构分布观察' },
  { id: 'dynamic_prediction', label: '动态预测', desc: 'next-token 动画' },
  { id: 'causal_intervention', label: '因果干预', desc: '必要/充分性打靶' },
  { id: 'subspace_geometry', label: '子空间编码', desc: '方向与子空间表示' },
  { id: 'feature_decomposition', label: '特征分解', desc: '特征簇与可解释轴' },
  { id: 'cross_layer_transport', label: '跨层传输', desc: '层间编码迁移' },
  { id: 'compositionality', label: '组合性测试', desc: '属性组合编码' },
  { id: 'counterfactual', label: '反事实编码', desc: '最小语义改动差分' },
  { id: 'robustness', label: '鲁棒不变性', desc: '扰动下稳定编码' },
  { id: 'minimal_circuit', label: '最小子回路', desc: '最小因果子集' },
];

const FEATURE_AXES = ['color', 'taste', 'shape', 'category'];

const MODE_VISUALS = {
  static: { accent: '#e5e7eb', nodePulse: 0.7, nodeSpeed: 0.85, linkOpacityBoost: 0.02, linkWidthBoost: 0, carrier: 'none' },
  dynamic_prediction: { accent: '#7ee0ff', nodePulse: 1.0, nodeSpeed: 1.0, linkOpacityBoost: 0.18, linkWidthBoost: 0.2, carrier: 'torus' },
  causal_intervention: { accent: '#ff6b6b', nodePulse: 1.3, nodeSpeed: 1.3, linkOpacityBoost: 0.32, linkWidthBoost: 0.45, carrier: 'octa' },
  subspace_geometry: { accent: '#c084fc', nodePulse: 0.95, nodeSpeed: 0.9, linkOpacityBoost: 0.22, linkWidthBoost: 0.25, carrier: 'plane' },
  feature_decomposition: { accent: '#f59e0b', nodePulse: 1.12, nodeSpeed: 1.05, linkOpacityBoost: 0.26, linkWidthBoost: 0.3, carrier: 'tetra' },
  cross_layer_transport: { accent: '#22d3ee', nodePulse: 1.08, nodeSpeed: 1.15, linkOpacityBoost: 0.28, linkWidthBoost: 0.28, carrier: 'cylinder' },
  compositionality: { accent: '#34d399', nodePulse: 1.2, nodeSpeed: 1.1, linkOpacityBoost: 0.26, linkWidthBoost: 0.35, carrier: 'tri_ring' },
  counterfactual: { accent: '#fb7185', nodePulse: 1.22, nodeSpeed: 1.28, linkOpacityBoost: 0.3, linkWidthBoost: 0.35, carrier: 'dual_ring' },
  robustness: { accent: '#a3e635', nodePulse: 0.88, nodeSpeed: 0.82, linkOpacityBoost: 0.14, linkWidthBoost: 0.18, carrier: 'shield' },
  minimal_circuit: { accent: '#f97316', nodePulse: 1.35, nodeSpeed: 1.38, linkOpacityBoost: 0.34, linkWidthBoost: 0.5, carrier: 'hex' },
};

function pseudoRandom(seed) {
  const v = Math.sin(seed * 12.9898) * 43758.5453;
  return v - Math.floor(v);
}

function hashString(value) {
  let h = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function extractPromptTokens(prompt) {
  return (prompt || '')
    .toLowerCase()
    .replace(/[，。！？,.!?;:]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

function getFallbackTokens(prompt) {
  const normalized = (prompt || '').toLowerCase();
  const topic = TOPIC_FALLBACKS.find((item) => item.keywords.some((k) => normalized.includes(k.toLowerCase())));
  return topic?.tokens || DEFAULT_CHAIN_TOKENS;
}

function generatePredictChain(prompt) {
  const tokens = extractPromptTokens(prompt);
  const fallback = getFallbackTokens(prompt);
  let context = tokens[tokens.length - 1] || fallback[0];
  const chain = [];
  for (let i = 0; i < PREDICT_CHAIN_LENGTH; i += 1) {
    const candidates = TOKEN_TRANSITIONS[context] || fallback;
    const pickSeed = hashString(`${prompt}|${context}|${i}`);
    const idx = Math.floor(pseudoRandom(pickSeed + i * 17) * candidates.length);
    const token = candidates[idx] || fallback[i % fallback.length];
    const base = Math.exp(-i * 0.18);
    const jitter = 0.04 * pseudoRandom(pickSeed + 101);
    const prob = Math.max(0.06, Math.min(0.96, 0.68 * base + 0.18 + jitter));
    chain.push({ token, prob });
    context = token;
  }
  return chain;
}

function buildConceptNeuronSet(name, idx = 0) {
  const normalized = name.trim().toLowerCase();
  const baseHash = hashString(`${normalized}-${idx}`);
  const setId = `query-${normalized.replace(/[^a-z0-9\u4e00-\u9fa5]+/gi, '-')}-${baseHash}`;
  const color = `hsl(${baseHash % 360}, 82%, 62%)`;

  const nodes = Array.from({ length: QUERY_NODE_COUNT }, (_, i) => {
    const seed = baseHash + i * 10007;
    const baseLayer = Math.floor((i / QUERY_NODE_COUNT) * LAYER_COUNT);
    const layer = (baseLayer + Math.floor(pseudoRandom(seed + 3) * 5)) % LAYER_COUNT;
    const neuron = Math.floor(pseudoRandom(seed + 17) * DFF);
    const score = 0.35 + pseudoRandom(seed + 29) * 0.65;

    return {
      id: `${setId}-${i}`,
      label: `${name} Query ${i + 1}`,
      role: 'query',
      concept: name,
      layer,
      neuron,
      metric: 'query_score',
      value: score,
      strength: score,
      source: 'textbox-query-generator',
      color,
      position: neuronToPosition(layer, neuron, 0.18 + i * 0.025),
      size: 0.13 + score * 0.12,
      phase: i * 0.31,
    };
  });

  return {
    id: setId,
    name,
    normalized,
    color,
    nodes,
  };
}

function neuronToPosition(layer, neuron, radialJitter = 0) {
  const angle = ((neuron % 4096) / 4096) * Math.PI * 2;
  const radius = 2.7 + ((neuron % 2048) / 2048) * 3.3 + radialJitter;
  const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
  const x = Math.cos(angle) * radius;
  const y = Math.sin(angle) * radius;
  return [x, y, z];
}

function PulsingNeuron({ node, selected, onSelect, predictionStrength = 0, mode = 'static' }) {
  const ref = useRef(null);
  const modeStyle = MODE_VISUALS[mode] || MODE_VISUALS.static;

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    const pulse = (node.role === 'background' ? 0.04 : 0.14) * modeStyle.nodePulse;
    const speed = (node.role === 'background' ? 1.2 : 2.1) * modeStyle.nodeSpeed;
    const base = node.size;
    const predictionBoost = predictionStrength * (node.role === 'background' ? 0.18 : 0.5);
    const modeWave = mode === 'counterfactual' ? Math.sin(state.clock.elapsedTime * speed * 0.7 + node.phase * 1.3) * 0.06 : 0;
    const scale = base * (1 + Math.sin(state.clock.elapsedTime * speed + node.phase) * pulse + predictionBoost + modeWave);
    ref.current.scale.set(scale, scale, scale);
  });

  return (
    <mesh
      ref={ref}
      position={node.position}
      onClick={(e) => {
        e.stopPropagation();
        onSelect(node);
      }}
    >
      <sphereGeometry args={[1, 20, 20]} />
      <meshStandardMaterial
        color={predictionStrength > 0.66 && mode !== 'static' ? modeStyle.accent : node.color}
        emissive={predictionStrength > 0.5 && mode !== 'static' ? modeStyle.accent : node.color}
        emissiveIntensity={
          (selected ? 1.8 : node.role === 'background' ? 0.08 : 0.55)
          + predictionStrength * (node.role === 'background' ? 0.2 : 1.6)
          + (mode !== 'static' ? 0.12 : 0)
        }
        roughness={0.2}
        metalness={0.15}
        transparent
        opacity={node.role === 'background' ? 0.24 + predictionStrength * 0.08 : 0.92}
      />
    </mesh>
  );
}

function LayerGuides({ activeLayer = null }) {
  const layers = useMemo(() => Array.from({ length: LAYER_COUNT }, (_, i) => i), []);
  const hasActiveLayer = Number.isFinite(activeLayer);
  const activeLayerIndex = hasActiveLayer
    ? Math.max(0, Math.min(LAYER_COUNT - 1, Math.round(activeLayer)))
    : null;
  return (
    <group>
      {layers.map((layer) => {
        const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
        const isMajor = layer % 4 === 0 || layer === LAYER_COUNT - 1;
        const isActive = activeLayerIndex === layer;
        const lineColor = isActive ? '#ffffff' : isMajor ? '#dbeafe' : '#8ea4c7';
        const lineOpacity = isActive ? 0.8 : isMajor ? 0.2 : 0.1;
        const labelColor = isActive ? '#ffffff' : isMajor ? '#d8ecff' : '#9cb6dc';
        const labelSize = isActive ? 0.38 : isMajor ? 0.3 : 0.22;
        return (
          <group key={`layer-${layer}`}>
            <Line
              points={[
                [-7.5, -7.5, z],
                [7.5, -7.5, z],
                [7.5, 7.5, z],
                [-7.5, 7.5, z],
                [-7.5, -7.5, z],
              ]}
              color={lineColor}
              transparent
              opacity={lineOpacity}
              lineWidth={1}
            />
            <Text
              position={[-8.55, 0, z]}
              color={labelColor}
              fontSize={labelSize}
              anchorX="left"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="#0a1022"
            >
              {`L${layer}`}
            </Text>
            <Text
              position={[8.55, 0, z]}
              color={labelColor}
              fontSize={labelSize}
              anchorX="right"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="#0a1022"
            >
              {`L${layer}`}
            </Text>
            {isActive && (
              <Line
                points={[
                  [-6.2, -6.2, z],
                  [6.2, -6.2, z],
                  [6.2, 6.2, z],
                  [-6.2, 6.2, z],
                  [-6.2, -6.2, z],
                ]}
                color="#ffffff"
                transparent
                opacity={0.58}
                lineWidth={1.6}
              />
            )}
          </group>
        );
      })}
      <Line points={[[0, 0, -13.2], [0, 0, 13.2]]} color="#ffffff" transparent opacity={0.7} lineWidth={1.2} />
      <Text position={[0, 0.95, -13.2]} color="#cde4ff" fontSize={0.28} anchorX="center" anchorY="middle" outlineWidth={0.015} outlineColor="#0a1022">
        Layer 0
      </Text>
      <Text position={[0, 0.95, 13.2]} color="#cde4ff" fontSize={0.28} anchorX="center" anchorY="middle" outlineWidth={0.015} outlineColor="#0a1022">
        Layer 27
      </Text>
    </group>
  );
}

function TokenPredictionCarrier({ prediction, mode = 'static' }) {
  const ref = useRef(null);
  const modeStyle = MODE_VISUALS[mode] || MODE_VISUALS.static;
  const movingColor = '#ffffff';

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    ref.current.rotation.y = state.clock.elapsedTime * (1.1 + modeStyle.nodeSpeed * 0.7);
  });

  if (!prediction?.currentToken || modeStyle.carrier === 'none') {
    return null;
  }

  const z = (prediction.layerProgress - 0.5) * (LAYER_COUNT - 1) * 0.92;
  const radius = 0.5 + prediction.currentToken.prob * 0.75;
  return (
    <group position={[0, 0, z]}>
      {modeStyle.carrier === 'torus' && (
        <mesh ref={ref}>
          <torusGeometry args={[radius, 0.08, 14, 42]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.4} transparent opacity={0.75} />
        </mesh>
      )}
      {modeStyle.carrier === 'octa' && (
        <mesh ref={ref}>
          <octahedronGeometry args={[radius * 0.92]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.72} wireframe />
        </mesh>
      )}
      {modeStyle.carrier === 'plane' && (
        <mesh ref={ref} rotation={[0.55, 0.25, 0.15]}>
          <boxGeometry args={[radius * 1.95, 0.08, radius * 1.1]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.0} transparent opacity={0.55} />
        </mesh>
      )}
      {modeStyle.carrier === 'tetra' && (
        <mesh ref={ref}>
          <tetrahedronGeometry args={[radius * 0.95]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.72} />
        </mesh>
      )}
      {modeStyle.carrier === 'cylinder' && (
        <mesh ref={ref}>
          <cylinderGeometry args={[radius * 0.22, radius * 0.22, radius * 2.0, 16]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.15} transparent opacity={0.72} />
        </mesh>
      )}
      {modeStyle.carrier === 'tri_ring' && (
        <group ref={ref}>
          <mesh rotation={[0, 0, 0]}>
            <torusGeometry args={[radius * 0.9, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.72} />
          </mesh>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[radius * 0.7, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.54} />
          </mesh>
          <mesh rotation={[0, Math.PI / 2, 0]}>
            <torusGeometry args={[radius * 0.52, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.4} />
          </mesh>
        </group>
      )}
      {modeStyle.carrier === 'dual_ring' && (
        <group ref={ref}>
          <mesh position={[-0.36, 0, 0]}>
            <torusGeometry args={[radius * 0.6, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.1} transparent opacity={0.68} />
          </mesh>
          <mesh position={[0.36, 0, 0]}>
            <torusGeometry args={[radius * 0.6, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.78} />
          </mesh>
        </group>
      )}
      {modeStyle.carrier === 'shield' && (
        <mesh ref={ref}>
          <sphereGeometry args={[radius * 0.9, 20, 20]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={0.95} transparent opacity={0.2} wireframe />
        </mesh>
      )}
      {modeStyle.carrier === 'hex' && (
        <mesh ref={ref}>
          <cylinderGeometry args={[radius * 0.8, radius * 0.8, radius * 0.95, 6]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.72} wireframe />
        </mesh>
      )}
      <Text position={[0, 0.9, 0]} color="#dff6ff" fontSize={0.34} anchorX="center" anchorY="middle">
        {`${prediction.currentToken.token} (${(prediction.currentToken.prob * 100).toFixed(1)}%)`}
      </Text>
    </group>
  );
}

function ModeVisualOverlay({ mode = 'static', prediction = null }) {
  const ref = useRef(null);
  const modeStyle = MODE_VISUALS[mode] || MODE_VISUALS.static;

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    ref.current.rotation.y = state.clock.elapsedTime * (0.25 + modeStyle.nodeSpeed * 0.2);
  });

  if (mode === 'static') {
    return null;
  }

  const z = ((prediction?.layerProgress ?? 0.5) - 0.5) * (LAYER_COUNT - 1) * 0.92;
  return (
    <group ref={ref} position={[0, 0, z]}>
      {mode === 'causal_intervention' && (
        <mesh>
          <torusKnotGeometry args={[1.2, 0.08, 120, 16]} />
          <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={0.95} transparent opacity={0.45} wireframe />
        </mesh>
      )}
      {mode === 'subspace_geometry' && (
        <mesh rotation={[0.62, 0.15, 0.42]}>
          <boxGeometry args={[3.6, 0.05, 1.6]} />
          <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={0.8} transparent opacity={0.28} />
        </mesh>
      )}
      {mode === 'feature_decomposition' && (
        <>
          <Line points={[[-1.9, 0, 0], [1.9, 0, 0]]} color="#f59e0b" transparent opacity={0.8} lineWidth={2} />
          <Line points={[[0, -1.9, 0], [0, 1.9, 0]]} color="#38bdf8" transparent opacity={0.8} lineWidth={2} />
          <Line points={[[0, 0, -1.9], [0, 0, 1.9]]} color="#a78bfa" transparent opacity={0.8} lineWidth={2} />
        </>
      )}
      {mode === 'cross_layer_transport' && (
        <>
          <Line points={[[0, 0, -2.8], [0, 0, 2.8]]} color={modeStyle.accent} transparent opacity={0.85} lineWidth={2} />
          <mesh position={[0, 0.2, Math.sin((prediction?.layerProgress || 0) * Math.PI * 2) * 2.2]}>
            <sphereGeometry args={[0.16, 12, 12]} />
            <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={1.35} />
          </mesh>
        </>
      )}
      {mode === 'compositionality' && (
        <>
          <mesh rotation={[0, 0, 0]}>
            <torusGeometry args={[1.2, 0.05, 12, 42]} />
            <meshStandardMaterial color="#34d399" emissive="#34d399" emissiveIntensity={1.0} transparent opacity={0.62} />
          </mesh>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[1.0, 0.05, 12, 42]} />
            <meshStandardMaterial color="#f59e0b" emissive="#f59e0b" emissiveIntensity={1.0} transparent opacity={0.62} />
          </mesh>
          <mesh rotation={[0, Math.PI / 2, 0]}>
            <torusGeometry args={[0.8, 0.05, 12, 42]} />
            <meshStandardMaterial color="#60a5fa" emissive="#60a5fa" emissiveIntensity={1.0} transparent opacity={0.62} />
          </mesh>
        </>
      )}
      {mode === 'counterfactual' && (
        <>
          <mesh position={[-0.8, 0, 0]}>
            <sphereGeometry args={[0.42, 16, 16]} />
            <meshStandardMaterial color="#fda4af" emissive="#fda4af" emissiveIntensity={1.05} transparent opacity={0.58} />
          </mesh>
          <mesh position={[0.8, 0, 0]}>
            <sphereGeometry args={[0.42, 16, 16]} />
            <meshStandardMaterial color="#fb7185" emissive="#fb7185" emissiveIntensity={1.2} transparent opacity={0.58} />
          </mesh>
          <Line points={[[-0.4, 0, 0], [0.4, 0, 0]]} color="#fda4af" transparent opacity={0.85} lineWidth={2} />
        </>
      )}
      {mode === 'robustness' && (
        <mesh>
          <sphereGeometry args={[1.45, 24, 24]} />
          <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={0.72} transparent opacity={0.16} wireframe />
        </mesh>
      )}
      {mode === 'minimal_circuit' && (
        <>
          <mesh>
            <cylinderGeometry args={[1.2, 1.2, 1.6, 6]} />
            <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={0.9} transparent opacity={0.26} wireframe />
          </mesh>
          <Line points={[[0, 0.8, 0], [0, -0.8, 0]]} color={modeStyle.accent} transparent opacity={0.9} lineWidth={2} />
        </>
      )}
    </group>
  );
}

export function AppleNeuronSceneContent({ nodes, links, selected, onSelect, prediction = null, mode = 'static' }) {
  const activationMap = prediction?.activationMap || {};
  const modeStyle = MODE_VISUALS[mode] || MODE_VISUALS.static;
  const activeLayer = Number.isFinite(prediction?.layerProgress)
    ? prediction.layerProgress * (LAYER_COUNT - 1)
    : null;

  return (
    <>
      <LayerGuides activeLayer={activeLayer} />

      {links.map((link) => (
        <Line
          key={link.id}
          points={link.points}
          color={mode === 'dynamic_prediction' || mode === 'static' ? link.color : modeStyle.accent}
          transparent
          opacity={0.42 + (prediction?.isRunning ? 0.18 : 0) + modeStyle.linkOpacityBoost}
          lineWidth={1.6 + modeStyle.linkWidthBoost}
        />
      ))}

      {nodes.map((node) => (
        <PulsingNeuron
          key={node.id}
          node={node}
          selected={selected?.id === node.id}
          onSelect={onSelect}
          predictionStrength={activationMap[node.id] || 0}
          mode={mode}
        />
      ))}

      <ModeVisualOverlay mode={mode} prediction={prediction} />
      <TokenPredictionCarrier prediction={prediction} mode={mode} />

      {selected && selected.role !== 'background' && (
        <Html position={[selected.position[0], selected.position[1] + 1.25, selected.position[2]]} center>
          <div
            style={{
              padding: '8px 10px',
              borderRadius: 8,
              background: 'rgba(255,255,255,0.95)',
              border: '1px solid rgba(180, 198, 228, 0.85)',
              color: '#1f2937',
              fontSize: 11,
              whiteSpace: 'nowrap',
              pointerEvents: 'none',
            }}
          >
            {`${selected.label} | L${selected.layer}N${selected.neuron}`}
          </div>
        </Html>
      )}
    </>
  );
}

function AppleNeuronScene({ nodes, links, selected, onSelect, prediction, mode = 'static' }) {
  return (
    <Canvas shadows dpr={[1, 1.5]}>
      <color attach="background" args={['#090b15']} />
      <fog attach="fog" args={['#090b15', 14, 42]} />

      <ambientLight intensity={0.5} />
      <pointLight position={[12, 12, 16]} intensity={70} color="#8fc4ff" />
      <pointLight position={[-14, -8, -15]} intensity={30} color="#ff9e6b" />

      <PerspectiveCamera makeDefault position={[16, 12, 26]} fov={42} />
      <OrbitControls enablePan enableZoom minDistance={10} maxDistance={44} />

      <AppleNeuronSceneContent nodes={nodes} links={links} selected={selected} onSelect={onSelect} prediction={prediction} mode={mode} />
    </Canvas>
  );
}

function buildFruitSpecificNodes() {
  const nodes = [];
  Object.entries(FRUIT_SPECIFIC_NEURONS).forEach(([fruit, items], fruitIdx) => {
    items.forEach((item, idx) => {
      nodes.push({
        id: `fruit-${fruit}-l${item.layer}-n${item.neuron}`,
        label: `${fruit} specific ${idx + 1}`,
        role: 'fruitSpecific',
        fruit,
        layer: item.layer,
        neuron: item.neuron,
        metric: 'fruit_specific_score',
        value: item.score,
        strength: item.score / 3.2,
        source: 'multi_fruit_20260301_194541',
        color: FRUIT_COLORS[fruit],
        position: neuronToPosition(item.layer, item.neuron, 0.22 + idx * 0.04 + fruitIdx * 0.03),
        size: 0.12 + (item.score / 3.2) * 0.2,
        phase: fruitIdx * 0.6 + idx * 0.35,
      });
    });
  });
  return nodes;
}

function buildFruitGeneralNodes() {
  return FRUIT_GENERAL_NEURONS.map((item, idx) => ({
    id: `fruit-general-l${item.layer}-n${item.neuron}`,
    label: `Fruit General ${idx + 1}`,
    role: 'fruitGeneral',
    layer: item.layer,
    neuron: item.neuron,
    metric: 'fruit_general_score',
    value: item.score,
    strength: item.score / 3.1,
    source: 'multi_fruit_20260301_194541',
    color: ROLE_COLORS.fruitGeneral,
    position: neuronToPosition(item.layer, item.neuron, 0.12 + idx * 0.03),
    size: 0.12 + (item.score / 3.1) * 0.18,
    phase: idx * 0.42,
  }));
}

function buildAppleCoreNodes() {
  return APPLE_CORE_NEURONS.map((n, idx) => {
    const size = 0.16 + Math.sqrt(n.strength / 0.0012) * 0.22;
    return {
      ...n,
      color: ROLE_COLORS[n.role],
      position: neuronToPosition(n.layer, n.neuron, 0.15 + idx * 0.02),
      size,
      phase: idx * 0.9,
    };
  });
}

function buildBackgroundNodes() {
  const background = [];
  for (let layer = 0; layer < LAYER_COUNT; layer += 1) {
    for (let i = 0; i < 11; i += 1) {
      const seed = layer * 97 + i * 29 + 13;
      const neuron = Math.floor(pseudoRandom(seed) * DFF);
      background.push({
        id: `bg-${layer}-${i}`,
        label: `Background L${layer}`,
        role: 'background',
        layer,
        neuron,
        metric: 'activation',
        value: pseudoRandom(seed + 7) * 0.1,
        strength: pseudoRandom(seed + 11) * 0.1,
        source: 'synthetic-grid',
        color: ROLE_COLORS.background,
        position: neuronToPosition(layer, neuron, pseudoRandom(seed + 5) * 0.6),
        size: 0.07 + pseudoRandom(seed + 19) * 0.08,
        phase: pseudoRandom(seed + 23) * Math.PI * 2,
      });
    }
  }
  return background;
}

export function useAppleNeuronWorkspace() {
  const [analysisMode, setAnalysisMode] = useState('dynamic_prediction');
  const [showFruitGeneral, setShowFruitGeneral] = useState(true);
  const [showFruit, setShowFruit] = useState({
    apple: true,
    banana: true,
    orange: true,
    grape: true,
  });
  const [queryInput, setQueryInput] = useState('');
  const [querySets, setQuerySets] = useState([]);
  const [predictPrompt, setPredictPrompt] = useState(DEFAULT_PREDICT_PROMPT);
  const [predictStep, setPredictStep] = useState(0);
  const [predictLayerProgress, setPredictLayerProgress] = useState(0);
  const [predictPlaying, setPredictPlaying] = useState(false);
  const [predictSpeed, setPredictSpeed] = useState(1);
  const [mechanismPlaying, setMechanismPlaying] = useState(false);
  const [mechanismSpeed, setMechanismSpeed] = useState(1);
  const [mechanismTick, setMechanismTick] = useState(0);
  const [interventionSparsity, setInterventionSparsity] = useState(0.45);
  const [featureAxis, setFeatureAxis] = useState(0);
  const [compositionWeights, setCompositionWeights] = useState({
    size: 0.34,
    sweetness: 0.33,
    color: 0.33,
  });
  const [counterfactualPrompt, setCounterfactualPrompt] = useState('苹果 不是 一种 水果');
  const [robustnessTrials, setRobustnessTrials] = useState(6);
  const [minimalSubsetSize, setMinimalSubsetSize] = useState(12);

  const backgroundNodes = useMemo(() => buildBackgroundNodes(), []);
  const appleCoreNodes = useMemo(() => buildAppleCoreNodes(), []);
  const fruitGeneralNodes = useMemo(() => buildFruitGeneralNodes(), []);
  const fruitSpecificNodes = useMemo(() => buildFruitSpecificNodes(), []);
  const queryNodes = useMemo(() => querySets.flatMap((set) => set.nodes), [querySets]);
  const predictChain = useMemo(() => generatePredictChain(predictPrompt), [predictPrompt]);
  const dynamicEnabled = analysisMode === 'dynamic_prediction';
  const mechanismEnabled = !['static', 'dynamic_prediction'].includes(analysisMode);

  const nodes = useMemo(() => {
    const visibleFruitSpecific = fruitSpecificNodes.filter((n) => showFruit[n.fruit]);
    const visibleFruitGeneral = showFruitGeneral ? fruitGeneralNodes : [];
    return [...backgroundNodes, ...appleCoreNodes, ...visibleFruitGeneral, ...visibleFruitSpecific, ...queryNodes];
  }, [appleCoreNodes, backgroundNodes, fruitGeneralNodes, fruitSpecificNodes, queryNodes, showFruit, showFruitGeneral]);

  const keyNodes = useMemo(() => nodes.filter((n) => n.role !== 'background'), [nodes]);
  const [selected, setSelected] = useState(appleCoreNodes[0] || null);

  useEffect(() => {
    if (analysisMode !== 'dynamic_prediction') {
      setPredictPlaying(false);
    }
    if (!mechanismEnabled) {
      setMechanismPlaying(false);
    }
  }, [analysisMode, mechanismEnabled]);

  useEffect(() => {
    if (!predictChain.length) {
      setPredictPlaying(false);
      return;
    }
    setPredictStep(0);
    setPredictLayerProgress(0);
  }, [predictChain]);

  useEffect(() => {
    if (!predictPlaying || !predictChain.length) {
      return undefined;
    }
    const interval = setInterval(() => {
      setPredictLayerProgress((prev) => {
        const next = prev + 0.038 * predictSpeed;
        if (next >= 1) {
          setPredictStep((s) => (s + 1) % predictChain.length);
          return 0;
        }
        return next;
      });
    }, 40);
    return () => clearInterval(interval);
  }, [predictPlaying, predictChain, predictSpeed]);

  useEffect(() => {
    if (!mechanismPlaying || !mechanismEnabled) {
      return undefined;
    }
    const interval = setInterval(() => {
      setMechanismTick((tick) => tick + 1);
    }, Math.max(30, 80 - mechanismSpeed * 18));
    return () => clearInterval(interval);
  }, [mechanismEnabled, mechanismPlaying, mechanismSpeed]);

  const handleGenerateQuery = () => {
    const concept = queryInput.trim();
    if (!concept) {
      return;
    }
    setQuerySets((prev) => {
      if (prev.some((set) => set.normalized === concept.toLowerCase())) {
        return prev;
      }
      const nextSet = buildConceptNeuronSet(concept, prev.length);
      if (nextSet.nodes[0]) {
        setSelected(nextSet.nodes[0]);
      }
      return [...prev, nextSet];
    });
    setQueryInput('');
  };

  const removeQuerySet = (setId) => {
    setQuerySets((prev) => prev.filter((set) => set.id !== setId));
  };

  const links = useMemo(() => {
    const byId = Object.fromEntries(keyNodes.map((n) => [n.id, n]));
    const linkSpecs = [
      ['apple-micro-l8n7574', 'apple-micro-l9n14608', '#ffad66'],
      ['apple-micro-l9n14608', 'apple-macro-l23n16819', '#ffd48a'],
      ['apple-macro-l23n16819', 'route-shared-l24n8124', '#a3ddff'],
      ['route-shared-l24n8124', 'route-shared-l27n16649', '#7fd9ff'],
      ['route-shared-l24n8124', 'route-shared-l27n16936', '#7fd9ff'],
    ];

    const fruitLinks = Object.keys(FRUIT_COLORS)
      .flatMap((fruit) => {
        if (!showFruit[fruit]) {
          return [];
        }
        const items = keyNodes.filter((n) => n.role === 'fruitSpecific' && n.fruit === fruit);
        if (items.length < 2) {
          return [];
        }
        return items.slice(1).map((node) => [items[0].id, node.id, FRUIT_COLORS[fruit]]);
      });

    const queryLinks = querySets.flatMap((set) => {
      if (set.nodes.length < 2) {
        return [];
      }
      return set.nodes.slice(1).map((node) => [set.nodes[0].id, node.id, set.color]);
    });

    return [...linkSpecs, ...fruitLinks, ...queryLinks]
      .filter(([from, to]) => byId[from] && byId[to])
      .map(([from, to, color]) => ({
        id: `${from}->${to}`,
        color,
        points: [byId[from].position, byId[to].position],
      }));
  }, [keyNodes, querySets, showFruit]);

  const currentPredictToken = dynamicEnabled && predictChain.length ? predictChain[predictStep % predictChain.length] : null;
  const predictLayer = predictLayerProgress * (LAYER_COUNT - 1);
  const mechanismPhase = (mechanismTick % 240) / 240;

  const dynamicActivationMap = useMemo(() => {
    if (!currentPredictToken) {
      return {};
    }
    const map = {};
    keyNodes.forEach((node) => {
      const seed = hashString(`${currentPredictToken.token}|${predictStep}|${node.id}`);
      const lexical = 0.25 + pseudoRandom(seed) * 0.75;
      const layerGate = Math.max(0, 1 - Math.abs(node.layer - predictLayer) / 8.2);
      const roleBoost = node.role === 'micro' ? 1.2 : node.role === 'macro' ? 1.08 : node.role === 'route' ? 1.15 : 1;
      map[node.id] = Math.min(1, lexical * layerGate * roleBoost * (0.65 + currentPredictToken.prob));
    });
    return map;
  }, [currentPredictToken, keyNodes, predictLayer, predictStep]);

  const modeOverlay = useMemo(() => {
    const overlay = {
      activationMap: {},
      currentToken: { token: '静态分析', prob: 0 },
      layerProgress: 0,
      focusNodeIds: [],
      metrics: [],
      statusText: '',
    };

    if (!keyNodes.length) {
      return overlay;
    }

    if (analysisMode === 'static') {
      keyNodes.forEach((node) => {
        overlay.activationMap[node.id] = Math.min(0.25, 0.06 + Math.sqrt(Math.max(node.strength, 1e-6)) * 0.5);
      });
      overlay.statusText = '结构分布快照';
      overlay.metrics = [{ label: 'Mode', value: 'Static' }];
      return overlay;
    }

    if (analysisMode === 'dynamic_prediction') {
      overlay.activationMap = dynamicActivationMap;
      overlay.currentToken = currentPredictToken || { token: '-', prob: 0 };
      overlay.layerProgress = predictLayerProgress;
      overlay.statusText = 'Autoregressive decoding';
      overlay.metrics = [
        { label: 'Step', value: `${predictStep + 1}/${predictChain.length || 0}` },
        { label: 'Layer', value: `L${predictLayer.toFixed(1)}` },
      ];
      return overlay;
    }

    if (analysisMode === 'causal_intervention') {
      const scores = keyNodes.map((node) => {
        const roleBoost = node.role === 'route' ? 1.25 : node.role === 'macro' ? 1.15 : 1;
        const score = pseudoRandom(hashString(`causal|${predictPrompt}|${node.id}`)) * roleBoost;
        return { id: node.id, score };
      });
      scores.sort((a, b) => b.score - a.score);
      const topCount = Math.max(4, Math.floor(4 + interventionSparsity * 20));
      const focus = scores.slice(0, topCount);
      const focusIds = new Set(focus.map((v) => v.id));
      keyNodes.forEach((node) => {
        const item = scores.find((s) => s.id === node.id);
        overlay.activationMap[node.id] = focusIds.has(node.id) ? 0.55 + item.score * 0.45 : 0.02;
      });
      overlay.focusNodeIds = [...focusIds];
      overlay.currentToken = { token: 'do(intervene)', prob: Math.min(0.99, focus.reduce((a, b) => a + b.score, 0) / topCount) };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'Ablation + patching target set';
      overlay.metrics = [
        { label: 'Top Nodes', value: `${topCount}` },
        { label: 'Sparsity', value: interventionSparsity.toFixed(2) },
      ];
      return overlay;
    }

    if (analysisMode === 'subspace_geometry') {
      const a = pseudoRandom(hashString(`${predictPrompt}|subspace|a`)) * 2 - 1;
      const b = pseudoRandom(hashString(`${predictPrompt}|subspace|b`)) * 2 - 1;
      const c = pseudoRandom(hashString(`${predictPrompt}|subspace|c`)) * 2 - 1;
      keyNodes.forEach((node) => {
        const x = node.layer / (LAYER_COUNT - 1) - 0.5;
        const y = (node.neuron / DFF) * 2 - 1;
        const z = Math.sin((node.layer + 1) * 0.35 + (node.neuron % 97) * 0.02);
        const projection = Math.abs(a * x + b * y + c * z);
        overlay.activationMap[node.id] = Math.min(1, 0.15 + projection * 0.95);
      });
      overlay.currentToken = { token: 'subspace', prob: 0.72 };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'Direction / subspace encoding';
      overlay.metrics = [
        { label: 'Basis', value: `[${a.toFixed(2)}, ${b.toFixed(2)}, ${c.toFixed(2)}]` },
      ];
      return overlay;
    }

    if (analysisMode === 'feature_decomposition') {
      const axisName = FEATURE_AXES[featureAxis] || FEATURE_AXES[0];
      keyNodes.forEach((node) => {
        const axis = hashString(`feature-axis|${node.id}`) % FEATURE_AXES.length;
        const local = pseudoRandom(hashString(`feature-val|${axisName}|${node.id}`));
        overlay.activationMap[node.id] = axis === featureAxis ? 0.58 + local * 0.4 : 0.08 + local * 0.2;
      });
      overlay.currentToken = { token: `axis:${axisName}`, prob: 0.78 };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'SAE-like feature slots';
      overlay.metrics = [
        { label: 'Axis', value: axisName },
        { label: 'Slots', value: `${FEATURE_AXES.length}` },
      ];
      return overlay;
    }

    if (analysisMode === 'cross_layer_transport') {
      const currentLayer = mechanismPhase * (LAYER_COUNT - 1);
      keyNodes.forEach((node) => {
        const layerGate = Math.exp(-Math.abs(node.layer - currentLayer) / 3.4);
        const routeBoost = node.role === 'route' ? 1.2 : 1;
        const lexical = 0.45 + pseudoRandom(hashString(`transport|${node.id}|${Math.floor(currentLayer)}`)) * 0.55;
        overlay.activationMap[node.id] = Math.min(1, layerGate * lexical * routeBoost);
      });
      overlay.currentToken = { token: `transport@L${currentLayer.toFixed(1)}`, prob: 0.75 };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'Layer-wise representational flow';
      overlay.metrics = [{ label: 'Current Layer', value: currentLayer.toFixed(1) }];
      return overlay;
    }

    if (analysisMode === 'compositionality') {
      const total = compositionWeights.size + compositionWeights.sweetness + compositionWeights.color;
      const ws = {
        size: compositionWeights.size / total,
        sweetness: compositionWeights.sweetness / total,
        color: compositionWeights.color / total,
      };
      keyNodes.forEach((node) => {
        const sizeSig = pseudoRandom(hashString(`comp-size|${node.id}`));
        const sweetSig = pseudoRandom(hashString(`comp-sweet|${node.id}`));
        const colorSig = pseudoRandom(hashString(`comp-color|${node.id}`));
        overlay.activationMap[node.id] = Math.min(1, 0.08 + ws.size * sizeSig + ws.sweetness * sweetSig + ws.color * colorSig);
      });
      overlay.currentToken = { token: 'compose(size,sweet,color)', prob: 0.8 };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'Attribute composition';
      overlay.metrics = [
        { label: 'w(size)', value: ws.size.toFixed(2) },
        { label: 'w(sweet)', value: ws.sweetness.toFixed(2) },
        { label: 'w(color)', value: ws.color.toFixed(2) },
      ];
      return overlay;
    }

    if (analysisMode === 'counterfactual') {
      keyNodes.forEach((node) => {
        const base = pseudoRandom(hashString(`base|${predictPrompt}|${node.id}`));
        const cf = pseudoRandom(hashString(`cf|${counterfactualPrompt}|${node.id}`));
        overlay.activationMap[node.id] = Math.abs(base - cf);
      });
      overlay.currentToken = { token: 'counterfactual Δ', prob: 0.7 };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'Minimal semantic edit response';
      overlay.metrics = [
        { label: 'Base', value: predictPrompt.slice(0, 16) || '-' },
        { label: 'CF', value: counterfactualPrompt.slice(0, 16) || '-' },
      ];
      return overlay;
    }

    if (analysisMode === 'robustness') {
      const trials = Math.max(2, robustnessTrials);
      keyNodes.forEach((node) => {
        const values = [];
        for (let t = 0; t < trials; t += 1) {
          values.push(pseudoRandom(hashString(`robust|${t}|${node.id}`)));
        }
        const mean = values.reduce((a, b) => a + b, 0) / trials;
        const variance = values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / trials;
        const std = Math.sqrt(variance);
        const stability = Math.max(0, 1 - std * 3.6);
        overlay.activationMap[node.id] = 0.08 + stability * 0.92;
      });
      overlay.currentToken = { token: `robust@${trials}`, prob: 0.76 };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'Noise / paraphrase invariance';
      overlay.metrics = [{ label: 'Trials', value: `${trials}` }];
      return overlay;
    }

    if (analysisMode === 'minimal_circuit') {
      const k = Math.max(3, Math.min(minimalSubsetSize, keyNodes.length));
      const scores = keyNodes
        .map((node) => ({ id: node.id, score: pseudoRandom(hashString(`mcs|${predictPrompt}|${node.id}`)) }))
        .sort((a, b) => b.score - a.score);
      const focusIds = new Set(scores.slice(0, k).map((v) => v.id));
      keyNodes.forEach((node) => {
        const s = scores.find((x) => x.id === node.id)?.score || 0;
        overlay.activationMap[node.id] = focusIds.has(node.id) ? 0.6 + s * 0.4 : 0.015;
      });
      overlay.focusNodeIds = [...focusIds];
      overlay.currentToken = { token: `MCS(k=${k})`, prob: Math.min(0.99, scores.slice(0, k).reduce((a, b) => a + b.score, 0) / k) };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'Minimal causal subset';
      overlay.metrics = [{ label: 'Subset Size', value: `${k}` }];
      return overlay;
    }

    return overlay;
  }, [
    analysisMode,
    compositionWeights.color,
    compositionWeights.size,
    compositionWeights.sweetness,
    counterfactualPrompt,
    currentPredictToken,
    dynamicActivationMap,
    featureAxis,
    interventionSparsity,
    keyNodes,
    mechanismPhase,
    minimalSubsetSize,
    predictChain.length,
    predictLayer,
    predictLayerProgress,
    predictPrompt,
    predictStep,
    robustnessTrials,
  ]);

  useEffect(() => {
    const map = modeOverlay.activationMap || {};
    let bestNode = null;
    let bestScore = -1;
    keyNodes.forEach((node) => {
      const score = map[node.id] || 0;
      if (score > bestScore) {
        bestScore = score;
        bestNode = node;
      }
    });
    if (bestNode) {
      setSelected(bestNode);
    }
  }, [keyNodes, modeOverlay.activationMap]);

  const handlePredictReset = () => {
    setPredictPlaying(false);
    setPredictStep(0);
    setPredictLayerProgress(0);
  };

  const handlePredictStepForward = () => {
    if (!predictChain.length) {
      return;
    }
    setPredictPlaying(false);
    setPredictLayerProgress(0);
    setPredictStep((s) => (s + 1) % predictChain.length);
  };

  const handleMechanismReset = () => {
    setMechanismPlaying(false);
    setMechanismTick(0);
  };

  const handleMechanismStepForward = () => {
    setMechanismPlaying(false);
    setMechanismTick((t) => t + 18);
  };

  const summary = useMemo(() => {
    const fruitSpecific = keyNodes.filter((n) => n.role === 'fruitSpecific');
    const perFruit = Object.keys(FRUIT_COLORS).reduce((acc, fruit) => {
      acc[fruit] = fruitSpecific.filter((n) => n.fruit === fruit).length;
      return acc;
    }, {});

    return {
      micro: keyNodes.filter((n) => n.role === 'micro').length,
      macro: keyNodes.filter((n) => n.role === 'macro').length,
      route: keyNodes.filter((n) => n.role === 'route').length,
      fruitGeneral: keyNodes.filter((n) => n.role === 'fruitGeneral').length,
      fruitSpecific: fruitSpecific.length,
      query: keyNodes.filter((n) => n.role === 'query').length,
      total: keyNodes.length,
      perFruit,
      currentToken: modeOverlay.currentToken?.token || '-',
      currentTokenProb: modeOverlay.currentToken?.prob || 0,
      analysisMode,
      statusText: modeOverlay.statusText || '',
    };
  }, [analysisMode, keyNodes, modeOverlay.currentToken, modeOverlay.statusText]);

  return {
    analysisMode,
    setAnalysisMode,
    analysisModes: ANALYSIS_MODE_OPTIONS,
    showFruitGeneral,
    setShowFruitGeneral,
    showFruit,
    setShowFruit,
    queryInput,
    setQueryInput,
    querySets,
    handleGenerateQuery,
    removeQuerySet,
    nodes,
    links,
    selected,
    setSelected,
    summary,
    predictPrompt,
    setPredictPrompt,
    predictChain,
    predictStep,
    predictLayerProgress,
    predictPlaying,
    setPredictPlaying,
    predictSpeed,
    setPredictSpeed,
    handlePredictReset,
    handlePredictStepForward,
    mechanismPlaying,
    setMechanismPlaying,
    mechanismSpeed,
    setMechanismSpeed,
    mechanismTick,
    handleMechanismReset,
    handleMechanismStepForward,
    interventionSparsity,
    setInterventionSparsity,
    featureAxis,
    setFeatureAxis,
    compositionWeights,
    setCompositionWeights,
    counterfactualPrompt,
    setCounterfactualPrompt,
    robustnessTrials,
    setRobustnessTrials,
    minimalSubsetSize,
    setMinimalSubsetSize,
    modeMetrics: modeOverlay.metrics,
    prediction: analysisMode === 'static'
      ? null
      : {
          isRunning: dynamicEnabled ? predictPlaying : mechanismPlaying,
          currentToken: modeOverlay.currentToken,
          step: dynamicEnabled ? predictStep : mechanismTick,
          layerProgress: modeOverlay.layerProgress,
          activationMap: modeOverlay.activationMap,
          chain: dynamicEnabled ? predictChain : [],
          mode: analysisMode,
          metrics: modeOverlay.metrics,
          statusText: modeOverlay.statusText,
        },
  };
}

export function AppleNeuronMainScene({ workspace, sceneHeight = '74vh' }) {
  return (
    <div
      style={{
        height: sceneHeight,
        borderRadius: 18,
        border: '1px solid rgba(122, 162, 255, 0.28)',
        overflow: 'hidden',
        background: 'radial-gradient(circle at 20% 0%, rgba(43, 84, 165, 0.2), rgba(8, 10, 18, 0.95) 55%)',
        boxShadow: '0 18px 44px rgba(0,0,0,0.45)',
      }}
    >
      <AppleNeuronScene
        nodes={workspace.nodes}
        links={workspace.links}
        selected={workspace.selected}
        onSelect={workspace.setSelected}
        prediction={workspace.prediction}
        mode={workspace.analysisMode}
      />
    </div>
  );
}

const smallActionButtonStyle = {
  borderRadius: 8,
  border: '1px solid rgba(122, 162, 255, 0.5)',
  background: 'rgba(28, 53, 102, 0.75)',
  color: '#dbe9ff',
  fontSize: 12,
  padding: '7px 10px',
  cursor: 'pointer',
};

const panelCardStyle = {
  borderRadius: 14,
  padding: 14,
  border: '1px solid rgba(118, 170, 255, 0.25)',
  background: 'linear-gradient(170deg, rgba(15,24,42,0.94), rgba(7,12,25,0.95))',
};

const inputStyle = {
  width: '100%',
  borderRadius: 8,
  border: '1px solid rgba(122, 162, 255, 0.3)',
  background: 'rgba(7, 12, 25, 0.8)',
  color: '#dbe9ff',
  padding: '8px 10px',
  fontSize: 12,
};

const textAreaStyle = {
  width: '100%',
  borderRadius: 8,
  border: '1px solid rgba(122, 162, 255, 0.3)',
  background: 'rgba(7, 12, 25, 0.8)',
  color: '#dbe9ff',
  padding: '8px 10px',
  fontSize: 12,
  resize: 'vertical',
};

export function AppleNeuronControlPanels({ workspace }) {
  const {
    analysisMode,
    setAnalysisMode,
    analysisModes,
    summary,
    queryInput,
    setQueryInput,
    handleGenerateQuery,
    querySets,
    removeQuerySet,
    showFruitGeneral,
    setShowFruitGeneral,
    showFruit,
    setShowFruit,
    selected,
    predictPrompt,
    setPredictPrompt,
    predictChain,
    predictStep,
    predictLayerProgress,
    predictPlaying,
    setPredictPlaying,
    predictSpeed,
    setPredictSpeed,
    handlePredictReset,
    handlePredictStepForward,
    mechanismPlaying,
    setMechanismPlaying,
    mechanismSpeed,
    setMechanismSpeed,
    mechanismTick,
    handleMechanismReset,
    handleMechanismStepForward,
    interventionSparsity,
    setInterventionSparsity,
    featureAxis,
    setFeatureAxis,
    compositionWeights,
    setCompositionWeights,
    counterfactualPrompt,
    setCounterfactualPrompt,
    robustnessTrials,
    setRobustnessTrials,
    minimalSubsetSize,
    setMinimalSubsetSize,
    modeMetrics,
  } = workspace;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>分析类型</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          {analysisModes.map((mode) => {
            const active = analysisMode === mode.id;
            return (
              <button
                key={mode.id}
                type="button"
                onClick={() => setAnalysisMode(mode.id)}
                style={{
                  borderRadius: 8,
                  border: `1px solid ${active ? 'rgba(126, 224, 255, 0.72)' : 'rgba(122, 162, 255, 0.35)'}`,
                  background: active ? 'rgba(24, 101, 134, 0.42)' : 'rgba(11, 18, 35, 0.86)',
                  color: active ? '#dff6ff' : '#bcd1f5',
                  fontSize: 12,
                  padding: '8px 10px',
                  cursor: 'pointer',
                }}
                title={mode.desc}
              >
                {mode.label}
              </button>
            );
          })}
        </div>
        <div style={{ fontSize: 11, color: '#7f95bb', marginTop: 8, lineHeight: 1.6 }}>
          {analysisModes.find((m) => m.id === analysisMode)?.desc || ''}
        </div>
        {summary.statusText ? (
          <div style={{ fontSize: 11, color: '#9bb3de', marginTop: 6 }}>{summary.statusText}</div>
        ) : null}
        {modeMetrics?.length > 0 && (
          <div style={{ marginTop: 8, display: 'grid', gap: 4 }}>
            {modeMetrics.map((metric, idx) => (
              <div key={`${metric.label}-${idx}`} style={{ fontSize: 11, color: '#9bb3de' }}>
                {`${metric.label}: ${metric.value}`}
              </div>
            ))}
          </div>
        )}
      </div>

      {analysisMode === 'dynamic_prediction' && (
        <div style={panelCardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>Next-Token Prediction Animation</div>
          <textarea
            value={predictPrompt}
            onChange={(e) => setPredictPrompt(e.target.value)}
            rows={2}
            placeholder="输入上下文，例如：苹果 是 一种"
            style={textAreaStyle}
          />
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button type="button" onClick={() => setPredictPlaying((v) => !v)} style={smallActionButtonStyle}>
              {predictPlaying ? 'Pause' : 'Play'}
            </button>
            <button type="button" onClick={handlePredictStepForward} style={smallActionButtonStyle}>Step</button>
            <button type="button" onClick={handlePredictReset} style={smallActionButtonStyle}>Reset</button>
          </div>
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 11, color: '#9eb4dd', marginBottom: 4 }}>
              {`Speed ${predictSpeed.toFixed(2)}x | Step ${predictStep + 1}/${predictChain.length || 0} | Layer ${(predictLayerProgress * 27).toFixed(1)}`}
            </div>
            <input type="range" min={0.4} max={2.4} step={0.1} value={predictSpeed} onChange={(e) => setPredictSpeed(Number(e.target.value))} style={{ width: '100%' }} />
          </div>
        </div>
      )}

      {analysisMode !== 'dynamic_prediction' && analysisMode !== 'static' && (
        <div style={panelCardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>机制控制</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button type="button" onClick={() => setMechanismPlaying((v) => !v)} style={smallActionButtonStyle}>{mechanismPlaying ? 'Pause' : 'Play'}</button>
            <button type="button" onClick={handleMechanismStepForward} style={smallActionButtonStyle}>Step</button>
            <button type="button" onClick={handleMechanismReset} style={smallActionButtonStyle}>Reset</button>
          </div>
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 11, color: '#9eb4dd', marginBottom: 4 }}>{`Mechanism Speed ${mechanismSpeed.toFixed(2)}x | Tick ${mechanismTick}`}</div>
            <input type="range" min={0.4} max={2.4} step={0.1} value={mechanismSpeed} onChange={(e) => setMechanismSpeed(Number(e.target.value))} style={{ width: '100%' }} />
          </div>

          {analysisMode === 'causal_intervention' && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 11, color: '#9eb4dd', marginBottom: 4 }}>{`Intervention Sparsity ${interventionSparsity.toFixed(2)}`}</div>
              <input type="range" min={0.1} max={1} step={0.05} value={interventionSparsity} onChange={(e) => setInterventionSparsity(Number(e.target.value))} style={{ width: '100%' }} />
            </div>
          )}

          {analysisMode === 'feature_decomposition' && (
            <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
              {FEATURE_AXES.map((axis, idx) => (
                <button
                  key={axis}
                  type="button"
                  onClick={() => setFeatureAxis(idx)}
                  style={{
                    borderRadius: 8,
                    border: `1px solid ${featureAxis === idx ? 'rgba(126, 224, 255, 0.75)' : 'rgba(122, 162, 255, 0.35)'}`,
                    background: featureAxis === idx ? 'rgba(24, 101, 134, 0.38)' : 'rgba(7, 12, 25, 0.82)',
                    color: '#dbe9ff',
                    fontSize: 11,
                    padding: '6px 8px',
                    cursor: 'pointer',
                  }}
                >
                  {axis}
                </button>
              ))}
            </div>
          )}

          {analysisMode === 'compositionality' && (
            <div style={{ marginTop: 8, display: 'grid', gap: 6 }}>
              {['size', 'sweetness', 'color'].map((k) => (
                <div key={k}>
                  <div style={{ fontSize: 11, color: '#9eb4dd', marginBottom: 2 }}>{`${k}: ${compositionWeights[k].toFixed(2)}`}</div>
                  <input
                    type="range"
                    min={0.05}
                    max={1}
                    step={0.01}
                    value={compositionWeights[k]}
                    onChange={(e) => setCompositionWeights((prev) => ({ ...prev, [k]: Number(e.target.value) }))}
                    style={{ width: '100%' }}
                  />
                </div>
              ))}
            </div>
          )}

          {analysisMode === 'counterfactual' && (
            <textarea
              value={counterfactualPrompt}
              onChange={(e) => setCounterfactualPrompt(e.target.value)}
              rows={2}
              placeholder="输入反事实提示词"
              style={{ ...textAreaStyle, marginTop: 8 }}
            />
          )}

          {analysisMode === 'robustness' && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 11, color: '#9eb4dd', marginBottom: 4 }}>{`Perturb Trials ${robustnessTrials}`}</div>
              <input type="range" min={2} max={12} step={1} value={robustnessTrials} onChange={(e) => setRobustnessTrials(Number(e.target.value))} style={{ width: '100%' }} />
            </div>
          )}

          {analysisMode === 'minimal_circuit' && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 11, color: '#9eb4dd', marginBottom: 4 }}>{`Subset Size ${minimalSubsetSize}`}</div>
              <input type="range" min={3} max={32} step={1} value={minimalSubsetSize} onChange={(e) => setMinimalSubsetSize(Number(e.target.value))} style={{ width: '100%' }} />
            </div>
          )}
        </div>
      )}

      <div style={panelCardStyle}>
        <div style={{ fontSize: 15, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>Apple + Fruit Neuron Compare</div>
        <div style={{ fontSize: 12, color: '#92a6cc', lineHeight: 1.6 }}>
          {`Layers: L0-L27 (all visible) | Total ${summary.total} | Fruit-General ${summary.fruitGeneral} | Fruit-Specific ${summary.fruitSpecific} | Query ${summary.query}`}
        </div>
        <div style={{ fontSize: 11, color: '#7ea2c9', marginTop: 4 }}>
          {`Current token: ${summary.currentToken} (${(summary.currentTokenProb * 100).toFixed(1)}%)`}
        </div>
      </div>

      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>Quick Concept Generator</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 8 }}>
          <input
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleGenerateQuery();
              }
            }}
            placeholder="输入名称，例如：猫 / 太阳 / 量子"
            style={inputStyle}
          />
          <button type="button" onClick={handleGenerateQuery} style={smallActionButtonStyle}>
            Generate
          </button>
        </div>
        <div style={{ marginTop: 10, display: 'grid', gap: 6 }}>
          {querySets.length === 0 ? (
            <div style={{ fontSize: 11, color: '#6f84ad' }}>尚未生成概念神经元。</div>
          ) : (
            querySets.map((set) => (
              <div key={set.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 12, color: '#9eb4dd' }}>
                <span><span style={{ color: set.color }}>●</span>{` ${set.name} (${set.nodes.length})`}</span>
                <button type="button" onClick={() => removeQuerySet(set.id)} style={{ ...smallActionButtonStyle, padding: '2px 8px', fontSize: 11 }}>Remove</button>
              </div>
            ))
          )}
        </div>
      </div>

      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>Compare Filter</div>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#9eb4dd', marginBottom: 10 }}>
          <input type="checkbox" checked={showFruitGeneral} onChange={(e) => setShowFruitGeneral(e.target.checked)} />
          Show fruit-general neurons ({summary.fruitGeneral})
        </label>
        {Object.keys(FRUIT_COLORS).map((fruit) => (
          <label key={fruit} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#9eb4dd', marginBottom: 8 }}>
            <input type="checkbox" checked={showFruit[fruit]} onChange={(e) => setShowFruit((prev) => ({ ...prev, [fruit]: e.target.checked }))} />
            <span style={{ color: FRUIT_COLORS[fruit] }}>●</span>
            {`${fruit} specific (${summary.perFruit[fruit] || 0})`}
          </label>
        ))}
      </div>

      <div style={{ ...panelCardStyle, minHeight: 160 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>Selected Neuron</div>
        {selected ? (
          <div style={{ fontSize: 12, color: '#9eb4dd', display: 'grid', gap: 6 }}>
            <div style={{ color: '#e5eeff', fontWeight: 700 }}>{selected.label}</div>
            <div>{`Role: ${selected.role}`}</div>
            {'fruit' in selected ? <div>{`Fruit: ${selected.fruit}`}</div> : null}
            {'concept' in selected ? <div>{`Concept: ${selected.concept}`}</div> : null}
            <div>{`Layer / Neuron: L${selected.layer} / N${selected.neuron}`}</div>
            <div>{`Strength: ${selected.strength.toExponential(3)}`}</div>
            <div>{`${selected.metric}: ${selected.value.toExponential(3)}`}</div>
            <div style={{ color: '#6f84ad' }}>{`Source: ${selected.source}`}</div>
          </div>
        ) : (
          <div style={{ fontSize: 12, color: '#7d93bd' }}>Click a highlighted neuron in the 3D scene.</div>
        )}
      </div>

      <div style={{ ...panelCardStyle, fontSize: 12, color: '#9eb4dd', lineHeight: 1.7 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>Legend</div>
        <div><span style={{ color: ROLE_COLORS.micro }}>●</span> Apple micro</div>
        <div><span style={{ color: ROLE_COLORS.macro }}>●</span> Apple macro</div>
        <div><span style={{ color: ROLE_COLORS.route }}>●</span> Route shared</div>
        <div><span style={{ color: ROLE_COLORS.fruitGeneral }}>●</span> Fruit-general</div>
        <div><span style={{ color: FRUIT_COLORS.apple }}>●</span> Apple specific</div>
        <div><span style={{ color: FRUIT_COLORS.banana }}>●</span> Banana specific</div>
        <div><span style={{ color: FRUIT_COLORS.orange }}>●</span> Orange specific</div>
        <div><span style={{ color: FRUIT_COLORS.grape }}>●</span> Grape specific</div>
        <div><span style={{ color: '#84f1ff' }}>●</span> Query concept neurons</div>
        <div><span style={{ color: ROLE_COLORS.background }}>●</span> Background network sample</div>
        <div style={{ color: '#6f84ad', marginTop: 8 }}>{`Apple core: ${summary.micro + summary.macro + summary.route}`}</div>
      </div>
    </div>
  );
}

export function AppleNeuron3DTab({ panelPosition = 'right', sceneHeight = '74vh', workspace: externalWorkspace } = {}) {
  const internalWorkspace = useAppleNeuronWorkspace();
  const workspace = externalWorkspace || internalWorkspace;
  const isPanelLeft = panelPosition === 'left';

  return (
    <div style={{ animation: 'roadmapFade 0.6s ease-out', display: 'grid', gridTemplateColumns: isPanelLeft ? '340px 1fr' : '1fr 340px', gap: 20 }}>
      {isPanelLeft ? (
        <>
          <AppleNeuronControlPanels workspace={workspace} />
          <AppleNeuronMainScene workspace={workspace} sceneHeight={sceneHeight} />
        </>
      ) : (
        <>
          <AppleNeuronMainScene workspace={workspace} sceneHeight={sceneHeight} />
          <AppleNeuronControlPanels workspace={workspace} />
        </>
      )}
    </div>
  );
}
