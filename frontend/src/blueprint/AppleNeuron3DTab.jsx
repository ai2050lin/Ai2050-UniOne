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
  background: '#2e3554',
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

function PulsingNeuron({ node, selected, onSelect, predictionStrength = 0 }) {
  const ref = useRef(null);

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    const pulse = node.role === 'background' ? 0.04 : 0.14;
    const speed = node.role === 'background' ? 1.2 : 2.1;
    const base = node.size;
    const predictionBoost = predictionStrength * (node.role === 'background' ? 0.18 : 0.5);
    const scale = base * (1 + Math.sin(state.clock.elapsedTime * speed + node.phase) * pulse + predictionBoost);
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
        color={node.color}
        emissive={node.color}
        emissiveIntensity={
          (selected ? 1.8 : node.role === 'background' ? 0.08 : 0.55)
          + predictionStrength * (node.role === 'background' ? 0.2 : 1.6)
        }
        roughness={0.2}
        metalness={0.15}
        transparent
        opacity={node.role === 'background' ? 0.24 + predictionStrength * 0.08 : 0.92}
      />
    </mesh>
  );
}

function LayerGuides() {
  const layers = useMemo(() => Array.from({ length: LAYER_COUNT }, (_, i) => i), []);
  return (
    <group>
      {layers.map((layer) => {
        const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
        const isMajor = layer % 4 === 0 || layer === LAYER_COUNT - 1;
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
              color={isMajor ? '#5f7fb1' : '#33405f'}
              transparent
              opacity={isMajor ? 0.35 : 0.12}
              lineWidth={1}
            />
            <Text
              position={[-8.55, 0, z]}
              color={isMajor ? '#d8ecff' : '#9cb6dc'}
              fontSize={isMajor ? 0.3 : 0.22}
              anchorX="left"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="#0a1022"
            >
              {`L${layer}`}
            </Text>
            <Text
              position={[8.55, 0, z]}
              color={isMajor ? '#d8ecff' : '#9cb6dc'}
              fontSize={isMajor ? 0.3 : 0.22}
              anchorX="right"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="#0a1022"
            >
              {`L${layer}`}
            </Text>
          </group>
        );
      })}
      <Line points={[[0, 0, -13.2], [0, 0, 13.2]]} color="#90a4d4" transparent opacity={0.6} lineWidth={1.2} />
      <Text position={[0, 0.95, -13.2]} color="#cde4ff" fontSize={0.28} anchorX="center" anchorY="middle" outlineWidth={0.015} outlineColor="#0a1022">
        Layer 0
      </Text>
      <Text position={[0, 0.95, 13.2]} color="#cde4ff" fontSize={0.28} anchorX="center" anchorY="middle" outlineWidth={0.015} outlineColor="#0a1022">
        Layer 27
      </Text>
    </group>
  );
}

function TokenPredictionCarrier({ prediction }) {
  const ref = useRef(null);

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    ref.current.rotation.y = state.clock.elapsedTime * 1.6;
  });

  if (!prediction?.currentToken) {
    return null;
  }

  const z = (prediction.layerProgress - 0.5) * (LAYER_COUNT - 1) * 0.92;
  const radius = 0.5 + prediction.currentToken.prob * 0.75;
  return (
    <group position={[0, 0, z]}>
      <mesh ref={ref}>
        <torusGeometry args={[radius, 0.08, 14, 42]} />
        <meshStandardMaterial
          color="#7ee0ff"
          emissive="#7ee0ff"
          emissiveIntensity={1.4}
          transparent
          opacity={0.75}
        />
      </mesh>
      <Text position={[0, 0.9, 0]} color="#dff6ff" fontSize={0.34} anchorX="center" anchorY="middle">
        {`${prediction.currentToken.token} (${(prediction.currentToken.prob * 100).toFixed(1)}%)`}
      </Text>
    </group>
  );
}

export function AppleNeuronSceneContent({ nodes, links, selected, onSelect, prediction = null }) {
  const activationMap = prediction?.activationMap || {};

  return (
    <>
      <LayerGuides />

      {links.map((link) => (
        <Line
          key={link.id}
          points={link.points}
          color={link.color}
          transparent
          opacity={0.54 + (prediction?.isRunning ? 0.18 : 0)}
          lineWidth={1.6}
        />
      ))}

      {nodes.map((node) => (
        <PulsingNeuron
          key={node.id}
          node={node}
          selected={selected?.id === node.id}
          onSelect={onSelect}
          predictionStrength={activationMap[node.id] || 0}
        />
      ))}

      <TokenPredictionCarrier prediction={prediction} />

      {selected && selected.role !== 'background' && (
        <Html position={[selected.position[0], selected.position[1] + 1.25, selected.position[2]]} center>
          <div
            style={{
              padding: '8px 10px',
              borderRadius: 8,
              background: 'rgba(9,12,22,0.9)',
              border: '1px solid rgba(122, 175, 255, 0.45)',
              color: '#dbe9ff',
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

function AppleNeuronScene({ nodes, links, selected, onSelect, prediction }) {
  return (
    <Canvas shadows dpr={[1, 1.5]}>
      <color attach="background" args={['#090b15']} />
      <fog attach="fog" args={['#090b15', 14, 42]} />

      <ambientLight intensity={0.5} />
      <pointLight position={[12, 12, 16]} intensity={70} color="#8fc4ff" />
      <pointLight position={[-14, -8, -15]} intensity={30} color="#ff9e6b" />

      <PerspectiveCamera makeDefault position={[16, 12, 26]} fov={42} />
      <OrbitControls enablePan enableZoom minDistance={10} maxDistance={44} />

      <AppleNeuronSceneContent nodes={nodes} links={links} selected={selected} onSelect={onSelect} prediction={prediction} />
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
  const [analysisType, setAnalysisType] = useState('dynamic');
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

  const backgroundNodes = useMemo(() => buildBackgroundNodes(), []);
  const appleCoreNodes = useMemo(() => buildAppleCoreNodes(), []);
  const fruitGeneralNodes = useMemo(() => buildFruitGeneralNodes(), []);
  const fruitSpecificNodes = useMemo(() => buildFruitSpecificNodes(), []);
  const queryNodes = useMemo(() => querySets.flatMap((set) => set.nodes), [querySets]);
  const predictChain = useMemo(() => generatePredictChain(predictPrompt), [predictPrompt]);

  const nodes = useMemo(() => {
    const visibleFruitSpecific = fruitSpecificNodes.filter((n) => showFruit[n.fruit]);
    const visibleFruitGeneral = showFruitGeneral ? fruitGeneralNodes : [];
    return [...backgroundNodes, ...appleCoreNodes, ...visibleFruitGeneral, ...visibleFruitSpecific, ...queryNodes];
  }, [appleCoreNodes, backgroundNodes, fruitGeneralNodes, fruitSpecificNodes, queryNodes, showFruit, showFruitGeneral]);

  const keyNodes = useMemo(() => nodes.filter((n) => n.role !== 'background'), [nodes]);
  const [selected, setSelected] = useState(appleCoreNodes[0] || null);

  useEffect(() => {
    if (analysisType === 'static') {
      setPredictPlaying(false);
    }
  }, [analysisType]);

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

  const dynamicEnabled = analysisType === 'dynamic';
  const currentPredictToken = dynamicEnabled && predictChain.length ? predictChain[predictStep % predictChain.length] : null;
  const predictLayer = predictLayerProgress * (LAYER_COUNT - 1);

  const predictActivationMap = useMemo(() => {
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

  useEffect(() => {
    if (!currentPredictToken) {
      return;
    }
    let bestNode = null;
    let bestScore = -1;
    keyNodes.forEach((node) => {
      const seed = hashString(`${currentPredictToken.token}|${predictStep}|${node.id}|step`);
      const roleBoost = node.role === 'micro' ? 1.2 : node.role === 'macro' ? 1.08 : node.role === 'route' ? 1.15 : 1;
      const score = (0.2 + pseudoRandom(seed) * 0.8) * roleBoost;
      if (score > bestScore) {
        bestScore = score;
        bestNode = node;
      }
    });
    if (bestNode) {
      setSelected(bestNode);
    }
  }, [currentPredictToken, keyNodes, predictStep]);

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
      currentToken: dynamicEnabled ? (currentPredictToken?.token || '-') : '静态分析',
      currentTokenProb: dynamicEnabled ? (currentPredictToken?.prob || 0) : 0,
      analysisType,
    };
  }, [analysisType, currentPredictToken, dynamicEnabled, keyNodes]);

  return {
    analysisType,
    setAnalysisType,
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
    prediction: dynamicEnabled
      ? {
          isRunning: predictPlaying,
          currentToken: currentPredictToken,
          step: predictStep,
          layerProgress: predictLayerProgress,
          activationMap: predictActivationMap,
          chain: predictChain,
        }
      : null,
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
      />
    </div>
  );
}

export function AppleNeuronControlPanels({ workspace }) {
  const {
    analysisType,
    setAnalysisType,
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
  } = workspace;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div
        style={{
          borderRadius: 14,
          padding: 14,
          border: '1px solid rgba(118, 170, 255, 0.25)',
          background: 'linear-gradient(170deg, rgba(15,24,42,0.94), rgba(7,12,25,0.95))',
        }}
      >
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>分析类型</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <button
            type="button"
            onClick={() => setAnalysisType('static')}
            style={{
              borderRadius: 8,
              border: `1px solid ${analysisType === 'static' ? 'rgba(122, 220, 160, 0.65)' : 'rgba(122, 162, 255, 0.35)'}`,
              background: analysisType === 'static' ? 'rgba(28, 116, 80, 0.35)' : 'rgba(11, 18, 35, 0.86)',
              color: analysisType === 'static' ? '#d8ffe9' : '#bcd1f5',
              fontSize: 12,
              padding: '8px 10px',
              cursor: 'pointer',
            }}
          >
            静态分析
          </button>
          <button
            type="button"
            onClick={() => setAnalysisType('dynamic')}
            style={{
              borderRadius: 8,
              border: `1px solid ${analysisType === 'dynamic' ? 'rgba(126, 224, 255, 0.7)' : 'rgba(122, 162, 255, 0.35)'}`,
              background: analysisType === 'dynamic' ? 'rgba(24, 101, 134, 0.4)' : 'rgba(11, 18, 35, 0.86)',
              color: analysisType === 'dynamic' ? '#dff6ff' : '#bcd1f5',
              fontSize: 12,
              padding: '8px 10px',
              cursor: 'pointer',
            }}
          >
            动态预测
          </button>
        </div>
        <div style={{ fontSize: 11, color: '#7f95bb', marginTop: 8 }}>
          {analysisType === 'dynamic'
            ? '动态预测: 按下一词预测链驱动层间激活动画。'
            : '静态分析: 关闭预测动画，专注结构与神经元分布观察。'}
        </div>
      </div>

      <div
        style={{
          borderRadius: 14,
          padding: 14,
          border: '1px solid rgba(118, 170, 255, 0.25)',
          background: 'linear-gradient(170deg, rgba(15,24,42,0.94), rgba(7,12,25,0.95))',
        }}
      >
        <div style={{ fontSize: 15, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>Apple + Fruit Neuron Compare</div>
          <div style={{ fontSize: 12, color: '#92a6cc', lineHeight: 1.6 }}>
            {`Layers: L0-L27 (all visible) | Total ${summary.total} | Fruit-General ${summary.fruitGeneral} | Fruit-Specific ${summary.fruitSpecific} | Query ${summary.query}`}
          </div>
        <div style={{ fontSize: 11, color: '#7ea2c9', marginTop: 4 }}>
          {`Current token: ${summary.currentToken} (${(summary.currentTokenProb * 100).toFixed(1)}%)`}
        </div>
        <div style={{ fontSize: 11, color: '#6f84ad', marginTop: 8 }}>
          Z-axis is layer depth. Use toggles to compare apple core neurons with banana/orange/grape specific neurons.
        </div>
      </div>

      {analysisType === 'dynamic' && (
        <div
          style={{
            borderRadius: 14,
            padding: 14,
            border: '1px solid rgba(118, 170, 255, 0.25)',
            background: 'linear-gradient(170deg, rgba(15,24,42,0.94), rgba(7,12,25,0.95))',
          }}
        >
          <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>Next-Token Prediction Animation</div>
          <div style={{ fontSize: 11, color: '#7f95bb', lineHeight: 1.6, marginBottom: 8 }}>
            Simulate autoregressive decoding: each token advances through L0-L27 and drives neuron activation.
          </div>
          <textarea
            value={predictPrompt}
            onChange={(e) => setPredictPrompt(e.target.value)}
            rows={2}
            placeholder="输入上下文，例如：苹果 是 一种"
            style={{
              width: '100%',
              borderRadius: 8,
              border: '1px solid rgba(122, 162, 255, 0.3)',
              background: 'rgba(7, 12, 25, 0.8)',
              color: '#dbe9ff',
              padding: '8px 10px',
              fontSize: 12,
              resize: 'vertical',
            }}
          />
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button
              type="button"
              onClick={() => setPredictPlaying((v) => !v)}
              style={{
                borderRadius: 8,
                border: '1px solid rgba(122, 162, 255, 0.5)',
                background: predictPlaying ? 'rgba(120, 162, 255, 0.28)' : 'rgba(28, 53, 102, 0.75)',
                color: '#dbe9ff',
                fontSize: 12,
                padding: '7px 10px',
                cursor: 'pointer',
              }}
            >
              {predictPlaying ? 'Pause' : 'Play'}
            </button>
            <button
              type="button"
              onClick={handlePredictStepForward}
              style={{
                borderRadius: 8,
                border: '1px solid rgba(122, 162, 255, 0.5)',
                background: 'rgba(28, 53, 102, 0.75)',
                color: '#dbe9ff',
                fontSize: 12,
                padding: '7px 10px',
                cursor: 'pointer',
              }}
            >
              Step
            </button>
            <button
              type="button"
              onClick={handlePredictReset}
              style={{
                borderRadius: 8,
                border: '1px solid rgba(122, 162, 255, 0.38)',
                background: 'rgba(11, 18, 35, 0.9)',
                color: '#bcd1f5',
                fontSize: 12,
                padding: '7px 10px',
                cursor: 'pointer',
              }}
            >
              Reset
            </button>
          </div>
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 11, color: '#9eb4dd', marginBottom: 4 }}>
              {`Speed ${predictSpeed.toFixed(2)}x | Step ${predictStep + 1}/${predictChain.length || 0} | Layer ${(predictLayerProgress * 27).toFixed(1)}`}
            </div>
            <input
              type="range"
              min={0.4}
              max={2.4}
              step={0.1}
              value={predictSpeed}
              onChange={(e) => setPredictSpeed(Number(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
            {predictChain.map((item, idx) => {
              const active = idx === predictStep;
              return (
                <div
                  key={`${item.token}-${idx}`}
                  style={{
                    borderRadius: 999,
                    border: `1px solid ${active ? 'rgba(126, 224, 255, 0.7)' : 'rgba(122, 162, 255, 0.25)'}`,
                    background: active ? 'rgba(24, 101, 134, 0.38)' : 'rgba(7, 12, 25, 0.72)',
                    color: active ? '#dff6ff' : '#9eb4dd',
                    fontSize: 11,
                    padding: '3px 8px',
                  }}
                >
                  {`${item.token} ${(item.prob * 100).toFixed(0)}%`}
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div
        style={{
          borderRadius: 14,
          padding: 14,
          border: '1px solid rgba(118, 170, 255, 0.25)',
          background: 'linear-gradient(170deg, rgba(15,24,42,0.94), rgba(7,12,25,0.95))',
        }}
      >
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
            style={{
              width: '100%',
              borderRadius: 8,
              border: '1px solid rgba(122, 162, 255, 0.3)',
              background: 'rgba(7, 12, 25, 0.8)',
              color: '#dbe9ff',
              padding: '8px 10px',
              fontSize: 12,
            }}
          />
          <button
            type="button"
            onClick={handleGenerateQuery}
            style={{
              borderRadius: 8,
              border: '1px solid rgba(122, 162, 255, 0.5)',
              background: 'rgba(28, 53, 102, 0.75)',
              color: '#dbe9ff',
              fontSize: 12,
              padding: '8px 10px',
              cursor: 'pointer',
            }}
          >
            Generate
          </button>
        </div>
        <div style={{ marginTop: 10, display: 'grid', gap: 6 }}>
          {querySets.length === 0 ? (
            <div style={{ fontSize: 11, color: '#6f84ad' }}>尚未生成概念神经元。</div>
          ) : (
            querySets.map((set) => (
              <div key={set.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 12, color: '#9eb4dd' }}>
                <span>
                  <span style={{ color: set.color }}>●</span>
                  {` ${set.name} (${set.nodes.length})`}
                </span>
                <button
                  type="button"
                  onClick={() => removeQuerySet(set.id)}
                  style={{
                    borderRadius: 6,
                    border: '1px solid rgba(122, 162, 255, 0.35)',
                    background: 'rgba(14, 23, 43, 0.8)',
                    color: '#9eb4dd',
                    fontSize: 11,
                    padding: '2px 8px',
                    cursor: 'pointer',
                  }}
                >
                  Remove
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div
        style={{
          borderRadius: 14,
          padding: 14,
          border: '1px solid rgba(118, 170, 255, 0.25)',
          background: 'linear-gradient(170deg, rgba(15,24,42,0.94), rgba(7,12,25,0.95))',
        }}
      >
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>Compare Filter</div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#9eb4dd', marginBottom: 10 }}>
            <input type="checkbox" checked={showFruitGeneral} onChange={(e) => setShowFruitGeneral(e.target.checked)} />
            Show fruit-general neurons ({summary.fruitGeneral})
        </label>
        {Object.keys(FRUIT_COLORS).map((fruit) => (
          <label key={fruit} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#9eb4dd', marginBottom: 8 }}>
            <input
              type="checkbox"
              checked={showFruit[fruit]}
              onChange={(e) => setShowFruit((prev) => ({ ...prev, [fruit]: e.target.checked }))}
            />
            <span style={{ color: FRUIT_COLORS[fruit] }}>●</span>
            {`${fruit} specific (${summary.perFruit[fruit] || 0})`}
          </label>
        ))}
      </div>

      <div
        style={{
          borderRadius: 14,
          padding: 14,
          border: '1px solid rgba(118, 170, 255, 0.25)',
          background: 'linear-gradient(170deg, rgba(15,24,42,0.94), rgba(7,12,25,0.95))',
          minHeight: 160,
        }}
      >
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

      <div
        style={{
          borderRadius: 14,
          padding: 14,
          border: '1px solid rgba(118, 170, 255, 0.25)',
          background: 'linear-gradient(170deg, rgba(15,24,42,0.94), rgba(7,12,25,0.95))',
          fontSize: 12,
          color: '#9eb4dd',
          lineHeight: 1.7,
        }}
      >
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>Legend</div>
        <div>
          <span style={{ color: ROLE_COLORS.micro }}>●</span> Apple micro
        </div>
        <div>
          <span style={{ color: ROLE_COLORS.macro }}>●</span> Apple macro
        </div>
        <div>
          <span style={{ color: ROLE_COLORS.route }}>●</span> Route shared
        </div>
        <div>
          <span style={{ color: ROLE_COLORS.fruitGeneral }}>●</span> Fruit-general
        </div>
        <div>
          <span style={{ color: FRUIT_COLORS.apple }}>●</span> Apple specific
        </div>
        <div>
          <span style={{ color: FRUIT_COLORS.banana }}>●</span> Banana specific
        </div>
        <div>
          <span style={{ color: FRUIT_COLORS.orange }}>●</span> Orange specific
        </div>
        <div>
          <span style={{ color: FRUIT_COLORS.grape }}>●</span> Grape specific
        </div>
        <div>
          <span style={{ color: '#84f1ff' }}>●</span> Query concept neurons
        </div>
        <div>
          <span style={{ color: ROLE_COLORS.background }}>●</span> Background network sample
        </div>
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
