import { Html, Line, OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import { Canvas, useFrame } from '@react-three/fiber';
import { Activity, ArrowRightLeft, BarChart2, CheckCircle, GitBranch, Network, Scale, Search, Sparkles, Target } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

const LAYER_COUNT = 28;
const DFF = 18944;
const QUERY_NODE_COUNT = 12;
const IMPORTED_QUERY_NODE_MAX = 18;
const MAIN_API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

const APPLE_CORE_NEURONS = [];

const FRUIT_GENERAL_NEURONS = [];

const FRUIT_SPECIFIC_NEURONS = {};

const FRUIT_COLORS = {};

const ROLE_COLORS = {
  micro: '#ff8d3b',
  macro: '#f6d365',
  route: '#39d0ff',
  fruitGeneral: '#6cf7d4',
  style: '#7dd3fc',
  logic: '#fca5a5',
  syntax: '#a7f3d0',
  hardBinding: '#fb7185',
  hardLong: '#38bdf8',
  hardLocal: '#f59e0b',
  hardTriplet: '#a78bfa',
  unifiedDecode: '#22c55e',
  background: '#ffffff',
};

const DIMENSION_LABELS = {
  style: '风格维度',
  logic: '逻辑维度',
  syntax: '句法维度',
};

const DEFAULT_PREDICT_PROMPT = '';
const PREDICT_CHAIN_LENGTH = 10;

const TOKEN_TRANSITIONS = {
  概念: ['是', '一种', '结构', '系统', '表达'],
  模型: ['通过', '层级', '编码', '形成', '预测'],
  concept: ['is', 'a', 'structured', 'representation', 'in'],
  model: ['builds', 'multi-layer', 'features', 'for', 'prediction'],
  is: ['a', 'structured', 'mapping', 'inside', 'the'],
  a: ['concept', 'model', 'token', 'signal', 'pattern'],
};

const TOPIC_FALLBACKS = [
  {
    keywords: ['概念', 'concept'],
    tokens: ['是', '一种', '结构', '可以', '在', '层级', '传播'],
  },
  {
    keywords: ['模型', 'model'],
    tokens: ['通过', '多层', '机制', '进行', '编码', '并', '预测'],
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


const ANALYSIS_MODE_ICONS = {
  static: Search,
  dynamic_prediction: Sparkles,
  causal_intervention: Target,
  subspace_geometry: Scale,
  feature_decomposition: BarChart2,
  cross_layer_transport: ArrowRightLeft,
  compositionality: Activity,
  counterfactual: GitBranch,
  robustness: CheckCircle,
  minimal_circuit: Network,
};

const ANALYSIS_MODE_STAGE_GROUPS = [
  {
    id: 'observation',
    label: '观测',
    icon: Search,
    items: ['static', 'dynamic_prediction', 'cross_layer_transport'],
  },
  {
    id: 'extraction',
    label: '提取',
    icon: Sparkles,
    items: ['subspace_geometry', 'feature_decomposition', 'compositionality'],
  },
  {
    id: 'validation',
    label: '验证',
    icon: CheckCircle,
    items: ['causal_intervention', 'counterfactual', 'robustness'],
  },
  {
    id: 'system',
    label: '系统',
    icon: Network,
    items: ['minimal_circuit'],
  },
];
const FEATURE_AXES = ['color', 'taste', 'shape', 'category'];

const HARD_PROBLEM_EXPERIMENT_LABELS = {
  hard_problem_dynamic_binding_v1: '动态绑定',
  hard_problem_long_horizon_trace_v1: '长程因果链路',
  hard_problem_local_credit_assignment_v1: '局部信用分配',
  triplet_targeted_causal_scan_v1: '三元组定向因果',
  triplet_targeted_multiseed_stability_v1: '三元组多seed稳定性',
  hard_problem_variable_binding_verification_v1: '变量绑定硬验证',
  minimal_causal_circuit_search_v1: '最小因果回路搜索',
  unified_coordinate_system_test_v1: '统一坐标系',
  concept_family_parallel_scale_v1: '规模化概念族',
};

function isHardProblemResultPayload(data) {
  return Boolean(
    data
    && data.schema_version === 'agi_research_result.v1'
    && typeof data.experiment_id === 'string'
    && data.experiment_id in HARD_PROBLEM_EXPERIMENT_LABELS
    && data.metrics
  );
}

function isUnifiedDecodePayload(data) {
  return Boolean(
    data
    && data.axis_stability
    && data.causal_separation
    && data.concept_hierarchy
  );
}

function isBundleManifestPayload(data) {
  return Boolean(data && data.bundle_id === 'agi_research_stage_bundle_v1');
}

function isFourTasksManifestPayload(data) {
  return Boolean(data && data.suite_id === 'agi_four_tasks_suite_v1');
}

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

function buildConceptNeuronSet(name, category = '未分类', idx = 0) {
  const normalized = name.trim().toLowerCase();
  const normalizedCategory = category.trim().toLowerCase() || '未分类';
  const baseHash = hashString(`${normalized}-${normalizedCategory}-${idx}`);
  const setId = `query-${normalized.replace(/[^a-z0-9\u4e00-\u9fa5]+/gi, '-')}-${normalizedCategory.replace(/[^a-z0-9\u4e00-\u9fa5]+/gi, '-')}-${baseHash}`;
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
      category,
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
    category,
    normalized,
    normalizedCategory,
    color,
    nodes,
  };
}

function toSafeNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeConceptKey(value) {
  return String(value || '').trim().toLowerCase();
}

function buildConceptNeuronSetFromSignature(name, category = '未分类', signatureIndices = [], idx = 0, dff = DFF, maxNodes = IMPORTED_QUERY_NODE_MAX) {
  const normalized = name.trim().toLowerCase();
  const normalizedCategory = category.trim().toLowerCase() || '未分类';
  const baseHash = hashString(`import-${normalized}-${normalizedCategory}-${idx}`);
  const setId = `import-${normalized.replace(/[^a-z0-9\u4e00-\u9fa5]+/gi, '-')}-${normalizedCategory.replace(/[^a-z0-9\u4e00-\u9fa5]+/gi, '-')}-${baseHash}`;
  const color = `hsl(${baseHash % 360}, 84%, 66%)`;

  const indices = signatureIndices
    .map((v) => toSafeNumber(v, -1))
    .filter((v) => Number.isFinite(v) && v >= 0)
    .slice(0, maxNodes);

  const nodes = indices.map((flatIdx, i) => {
    const layer = Math.floor(flatIdx / dff);
    const neuron = flatIdx % dff;
    const layerClamped = Math.max(0, Math.min(LAYER_COUNT - 1, layer));
    const neuronClamped = Math.max(0, neuron);
    const rank = i + 1;
    const score = Math.max(0.08, 1 - i / Math.max(4, indices.length));
    return {
      id: `${setId}-${i}`,
      label: `${name} Sig ${rank}`,
      role: 'query',
      concept: name,
      category,
      layer: layerClamped,
      neuron: neuronClamped,
      metric: 'signature_rank_score',
      value: score,
      strength: score,
      source: 'mass_noun_encoding_scan_import',
      color,
      position: neuronToPosition(layerClamped, neuronClamped, 0.2 + i * 0.024),
      size: 0.12 + score * 0.16,
      phase: i * 0.28,
    };
  });

  return {
    id: setId,
    name,
    category,
    normalized,
    normalizedCategory,
    color,
    nodes,
  };
}

function buildSharedReuseSet(reusedRecords = [], dff = DFF, maxNodes = IMPORTED_QUERY_NODE_MAX, idx = 0) {
  const list = reusedRecords.slice(0, maxNodes);
  const baseHash = hashString(`shared-reuse-${idx}`);
  const setId = `import-shared-reused-${baseHash}`;
  const color = '#ffd166';
  const nodes = list.map((rec, i) => {
    const layer = Number.isFinite(rec?.layer) ? rec.layer : Math.floor(toSafeNumber(rec?.flat_index, 0) / dff);
    const neuron = Number.isFinite(rec?.neuron) ? rec.neuron : toSafeNumber(rec?.flat_index, 0) % dff;
    const layerClamped = Math.max(0, Math.min(LAYER_COUNT - 1, layer));
    const neuronClamped = Math.max(0, neuron);
    const count = toSafeNumber(rec?.count, 1);
    const score = Math.max(0.1, Math.min(1, count / 12));
    return {
      id: `${setId}-${i}`,
      label: `Shared Reuse ${i + 1}`,
      role: 'query',
      concept: '共享复用神经元',
      category: '共享',
      layer: layerClamped,
      neuron: neuronClamped,
      metric: 'reuse_count_score',
      value: count,
      strength: score,
      source: 'mass_noun_encoding_scan_import',
      color,
      position: neuronToPosition(layerClamped, neuronClamped, 0.24 + i * 0.02),
      size: 0.12 + score * 0.18,
      phase: 0.18 * i,
    };
  });
  return {
    id: setId,
    name: '共享复用神经元',
    category: '共享',
    normalized: '共享复用神经元',
    normalizedCategory: '共享',
    color,
    nodes,
  };
}

function buildMultidimNodesFromProbe(probeData, visibleDims = { style: true, logic: true, syntax: true }, topN = 64) {
  if (!probeData || !probeData.dimensions) {
    return [];
  }
  const dims = ['style', 'logic', 'syntax'];
  const maxNodes = Math.max(8, Math.min(256, toSafeNumber(topN, 64)));
  const nodes = [];
  dims.forEach((dim, dimIdx) => {
    if (visibleDims[dim] === false) {
      return;
    }
    const rows = probeData?.dimensions?.[dim]?.specific_top_neurons || probeData?.dimensions?.[dim]?.top_neurons || [];
    const color = ROLE_COLORS[dim] || '#84f1ff';
    rows.slice(0, maxNodes).forEach((row, i) => {
      const layer = Math.max(0, Math.min(LAYER_COUNT - 1, toSafeNumber(row?.layer, 0)));
      const neuron = Math.max(0, toSafeNumber(row?.neuron, 0));
      const score = Math.max(0.05, toSafeNumber(row?.specific_score, toSafeNumber(row?.mean_abs_delta, 0.1)));
      nodes.push({
        id: `multidim-${dim}-${i}-l${layer}-n${neuron}`,
        label: `${DIMENSION_LABELS[dim] || dim} ${i + 1}`,
        role: dim,
        dimension: dim,
        concept: DIMENSION_LABELS[dim] || dim,
        category: '多维编码',
        layer,
        neuron,
        metric: 'dimension_specific_score',
        value: score,
        strength: score,
        source: 'multidim_encoding_probe',
        color,
        position: neuronToPosition(layer, neuron, 0.18 + i * 0.018 + dimIdx * 0.06),
        size: 0.11 + Math.min(0.28, Math.abs(score) * 0.08),
        phase: dimIdx * 0.7 + i * 0.22,
      });
    });
  });
  return nodes;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number.isFinite(Number(value)) ? Number(value) : 0));
}

function metricNodeStrength(metricKey, value) {
  const v = Number(value);
  if (!Number.isFinite(v)) {
    return 0.2;
  }
  if (metricKey.includes('error') || metricKey.includes('collision') || metricKey.includes('decay')) {
    return clamp01(1 - v);
  }
  return clamp01(v);
}

function extractMetricScalar(value) {
  if (typeof value === 'number') {
    return value;
  }
  if (value && typeof value === 'object' && typeof value.mean === 'number') {
    return Number(value.mean);
  }
  return NaN;
}

function getMetricByPath(metrics, path) {
  if (!metrics || !path) {
    return undefined;
  }
  if (!String(path).includes('.')) {
    return metrics[path];
  }
  return String(path)
    .split('.')
    .reduce((acc, key) => (acc && typeof acc === 'object' ? acc[key] : undefined), metrics);
}

function buildHardProblemNodes(hardProblemResults = {}) {
  const expEntries = Object.entries(hardProblemResults || {});
  if (expEntries.length === 0) {
    return [];
  }
  const roleByExp = {
    hard_problem_dynamic_binding_v1: 'hardBinding',
    hard_problem_long_horizon_trace_v1: 'hardLong',
    hard_problem_local_credit_assignment_v1: 'hardLocal',
    triplet_targeted_causal_scan_v1: 'hardTriplet',
    triplet_targeted_multiseed_stability_v1: 'hardTriplet',
    hard_problem_variable_binding_verification_v1: 'hardBinding',
    minimal_causal_circuit_search_v1: 'hardLocal',
    unified_coordinate_system_test_v1: 'unifiedDecode',
    concept_family_parallel_scale_v1: 'hardTriplet',
  };
  const metricPriority = {
    hard_problem_dynamic_binding_v1: ['binding_stability_index', 'role_swap_error_rate', 'collision_rate_top1', 'subject_decode_accuracy'],
    hard_problem_long_horizon_trace_v1: ['layer_transport_stability_mean', 'long_horizon_decay', 'hop_recovery_mean'],
    hard_problem_local_credit_assignment_v1: ['local_global_consistency_mean', 'local_sufficiency_mean', 'local_selectivity_mean'],
    hard_problem_variable_binding_verification_v1: [
      'mean_delta',
      'improved_dimension_count',
      'enhanced.rewrite_accuracy',
      'enhanced.role_swap_accuracy',
      'enhanced.cross_sentence_chain_accuracy',
    ],
    minimal_causal_circuit_search_v1: [
      'global.intervention_drop_mean',
      'global.reproducibility_jaccard_mean',
      'global.fidelity_mean',
      'global.min_subset_size_mean',
    ],
    unified_coordinate_system_test_v1: [
      'unified_coordinate_score',
      'probe_orthogonality.orthogonality_index',
      'ablation_coupling.decoupling_score',
      'concept_dim_alignment.concept_dim_coupling_abs_mean',
    ],
    concept_family_parallel_scale_v1: [
      'apple_chain_summary.shared_base_ratio_vs_micro_union.mean',
      'cat_chain_summary.shared_base_ratio_vs_micro_union.mean',
      'apple_vs_cat_shared_base_gap_mean',
    ],
    triplet_targeted_causal_scan_v1: [
      'triplet_minimal_records',
      'triplet_counterfactual_records',
      'axis_specificity_index',
      'triplet_separability_index',
      'global_mean_causal_margin_seq_logprob',
    ],
    triplet_targeted_multiseed_stability_v1: [
      'triplet_counterfactual_records',
      'global_mean_causal_margin_seq_logprob',
      'global_positive_causal_margin_ratio',
      'queen_recovery_ratio_mean',
      'king_recovery_ratio_mean',
    ],
  };

  const nodes = [];
  expEntries.forEach(([expId, payload], expIdx) => {
    const role = roleByExp[expId] || 'hardBinding';
    const color = ROLE_COLORS[role] || '#f97316';
    const title = HARD_PROBLEM_EXPERIMENT_LABELS[expId] || payload?.title || expId;
    const metrics = payload?.metrics || {};
    const preferredKeys = metricPriority[expId] || Object.keys(metrics);
    const keys = preferredKeys.filter((k) => k in metrics).slice(0, 6);
    const resolvedKeys = (keys.length > 0 ? keys : preferredKeys).slice(0, 6);
    resolvedKeys.forEach((k, i) => {
      const rawMetric = getMetricByPath(metrics, k);
      const val = extractMetricScalar(rawMetric);
      const strength = metricNodeStrength(k, val);
      const seed = hashString(`hard|${expId}|${k}|${i}|${expIdx}`);
      const layer = Math.max(0, Math.min(LAYER_COUNT - 1, Math.floor(pseudoRandom(seed + 7) * LAYER_COUNT)));
      const neuron = Math.max(0, Math.floor(pseudoRandom(seed + 13) * DFF));
      nodes.push({
        id: `hard-${expId}-${k}-${i}`,
        label: `${title} ${k}`,
        role,
        concept: title,
        category: '硬伤实验',
        layer,
        neuron,
        metric: k,
        value: Number.isFinite(val) ? val : 0,
        strength: Math.max(0.12, 0.2 + strength * 0.8),
        source: 'agi_research_result_v1',
        color,
        position: neuronToPosition(layer, neuron, 0.28 + i * 0.03 + expIdx * 0.05),
        size: 0.12 + Math.max(0.05, strength) * 0.18,
        phase: expIdx * 0.5 + i * 0.22,
      });
    });
  });
  return nodes;
}

function parseDominantLayers(voteObj) {
  if (!voteObj || typeof voteObj !== 'object') {
    return [];
  }
  const topPattern = Object.entries(voteObj).sort((a, b) => Number(b?.[1] || 0) - Number(a?.[1] || 0))[0]?.[0];
  if (!topPattern || typeof topPattern !== 'string') {
    return [];
  }
  return topPattern
    .split(',')
    .map((x) => Number(x))
    .filter((n) => Number.isFinite(n) && n >= 0 && n < LAYER_COUNT);
}

function buildUnifiedDecodeNodes(unifiedDecodeResult) {
  if (!unifiedDecodeResult) {
    return [];
  }
  const dims = ['style', 'logic', 'syntax'];
  const nodes = [];
  dims.forEach((dim, dimIdx) => {
    const axis = unifiedDecodeResult?.axis_stability?.dimensions?.[dim] || {};
    const causal = unifiedDecodeResult?.causal_separation?.diagonal_advantage?.[dim] || {};
    const profileCos = Number(axis?.profile_cosine_mean);
    const diagAdv = Number(causal?.mean);
    const strength = clamp01((Number.isFinite(profileCos) ? profileCos : 0) * 0.8 + (Number.isFinite(diagAdv) ? Math.max(0, diagAdv) : 0) * 2.0);
    const layers = parseDominantLayers(axis?.dominant_layer_pattern_votes);
    const fallbackLayer = Math.floor((dimIdx / Math.max(1, dims.length - 1)) * (LAYER_COUNT - 1));
    const layerList = layers.length > 0 ? layers.slice(0, 4) : [fallbackLayer];
    layerList.forEach((layer, li) => {
      const seed = hashString(`unified|${dim}|${li}|${layer}`);
      const neuron = Math.max(0, Math.floor(pseudoRandom(seed + 37) * DFF));
      nodes.push({
        id: `unified-${dim}-${li}-l${layer}`,
        label: `统一解码 ${DIMENSION_LABELS[dim] || dim}`,
        role: 'unifiedDecode',
        concept: DIMENSION_LABELS[dim] || dim,
        category: '统一解码',
        layer,
        neuron,
        metric: 'profile_cosine_mean',
        value: Number.isFinite(profileCos) ? profileCos : 0,
        strength: 0.18 + Math.max(0.08, strength) * 0.75,
        source: 'unified_math_structure_decode',
        color: ROLE_COLORS.unifiedDecode,
        position: neuronToPosition(layer, neuron, 0.25 + li * 0.03 + dimIdx * 0.06),
        size: 0.12 + Math.max(0.06, strength) * 0.16,
        phase: dimIdx * 0.7 + li * 0.21,
      });
    });
  });
  return nodes;
}

function nodeDisplayGroup(role) {
  if (role === 'background') {
    return 'background';
  }
  if (role === 'query') {
    return 'query';
  }
  if (role === 'style' || role === 'logic' || role === 'syntax') {
    return 'multidim';
  }
  if (role === 'unifiedDecode') {
    return 'unified';
  }
  if (role === 'hardBinding' || role === 'hardLong' || role === 'hardLocal' || role === 'hardTriplet') {
    return 'hard';
  }
  return 'core';
}

function buildAutoDisplayProfile(analysisMode) {
  if (['causal_intervention', 'counterfactual', 'robustness', 'minimal_circuit'].includes(analysisMode)) {
    return { core: 0.45, query: 0.65, multidim: 0.5, hard: 1, unified: 0.45, background: 0.08 };
  }
  if (['subspace_geometry', 'feature_decomposition', 'cross_layer_transport', 'compositionality'].includes(analysisMode)) {
    return { core: 0.5, query: 0.7, multidim: 0.95, hard: 0.45, unified: 1, background: 0.08 };
  }
  if (analysisMode === 'dynamic_prediction') {
    return { core: 0.9, query: 1, multidim: 0.85, hard: 0.8, unified: 0.8, background: 0.12 };
  }
  if (analysisMode === 'static') {
    return { core: 0.85, query: 0.85, multidim: 0.85, hard: 0.85, unified: 0.85, background: 0.12 };
  }
  return { core: 0.8, query: 0.8, multidim: 0.8, hard: 0.8, unified: 0.8, background: 0.1 };
}

function neuronToPosition(layer, neuron, radialJitter = 0) {
  const angle = ((neuron % 4096) / 4096) * Math.PI * 2;
  const radius = 2.7 + ((neuron % 2048) / 2048) * 3.3 + radialJitter;
  const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
  const x = Math.cos(angle) * radius;
  const y = Math.sin(angle) * radius;
  return [x, y, z];
}

function PulsingNeuron({
  node,
  selected,
  onSelect,
  predictionStrength = 0,
  mode = 'static',
  isEffectiveNode = false,
  visibilityEmphasis = 1,
}) {
  const ref = useRef(null);
  const modeStyle = MODE_VISUALS[mode] || MODE_VISUALS.static;

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    const pulse = (node.role === 'background' ? 0.04 : 0.14) * modeStyle.nodePulse;
    const speed = (node.role === 'background' ? 1.2 : 2.1) * modeStyle.nodeSpeed;
    const base = node.size;
    const predictionBoost = predictionStrength * (node.role === 'background' ? 0.18 : 0.5) * (0.6 + 0.4 * visibilityEmphasis);
    const modeWave = mode === 'counterfactual' ? Math.sin(state.clock.elapsedTime * speed * 0.7 + node.phase * 1.3) * 0.06 : 0;
    const effectiveBoost = isEffectiveNode ? 0.22 : 0;
    const scale = base * (1 + Math.sin(state.clock.elapsedTime * speed + node.phase) * pulse + predictionBoost + modeWave + effectiveBoost);
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
        color={isEffectiveNode ? '#ffffff' : predictionStrength > 0.66 && mode !== 'static' ? modeStyle.accent : node.color}
        emissive={isEffectiveNode ? '#ffffff' : predictionStrength > 0.5 && mode !== 'static' ? modeStyle.accent : node.color}
        emissiveIntensity={
          (selected ? 1.8 : node.role === 'background' ? 0.08 : 0.55)
          + predictionStrength * (node.role === 'background' ? 0.2 : 1.6)
          + (isEffectiveNode ? 0.95 : 0)
          + (mode !== 'static' ? 0.12 : 0)
        }
        roughness={0.2}
        metalness={0.15}
        transparent
        opacity={(isEffectiveNode ? 0.98 : node.role === 'background' ? 0.24 + predictionStrength * 0.08 : 0.92) * visibilityEmphasis}
      />
    </mesh>
  );
}

function LayerEffectiveNeuronOverlay({ prediction = null, mode = 'static' }) {
  if (mode !== 'feature_decomposition') {
    return null;
  }
  const layer = Number.isFinite(prediction?.effectiveLayer)
    ? Math.max(0, Math.min(LAYER_COUNT - 1, Math.round(prediction.effectiveLayer)))
    : null;
  if (!Number.isFinite(layer)) {
    return null;
  }
  const rows = Array.isArray(prediction?.effectiveNeurons) ? prediction.effectiveNeurons.slice(0, 6) : [];
  const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
  return (
    <group position={[0, 0, z]}>
      <Line points={[[3.8, 0.2, 0], [7.05, 1.95, 0]]} color="#ffffff" transparent opacity={0.82} lineWidth={1.4} />
      <Html position={[7.25, 2.12, 0]} center={false}>
        <div
          style={{
            width: 226,
            borderRadius: 10,
            border: '1px solid rgba(255,255,255,0.58)',
            background: 'rgba(8, 12, 24, 0.86)',
            color: '#e8f2ff',
            padding: '8px 10px',
            fontSize: 11,
            lineHeight: 1.55,
            boxShadow: '0 10px 24px rgba(0,0,0,0.35)',
            pointerEvents: 'none',
          }}
        >
          <div style={{ fontWeight: 700, color: '#ffffff' }}>{`L${layer} 有效神经元 Top-${rows.length}`}</div>
          {rows.length === 0 ? (
            <div style={{ color: '#9bb3de' }}>当前层暂无可显示节点</div>
          ) : (
            rows.map((item, idx) => (
              <div key={`eff-n-${item.id}-${idx}`} style={{ color: '#d4e5ff' }}>
                {`${idx + 1}. N${item.neuron} | ${item.role} | ${(Number(item.score || 0) * 100).toFixed(1)}%`}
              </div>
            ))
          )}
        </div>
      </Html>
    </group>
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

function DimensionLayerImpactGraph({ profile = [], dimension = 'style', suppression = null }) {
  if (!Array.isArray(profile) || profile.length === 0) {
    return null;
  }
  const color = ROLE_COLORS[dimension] || '#84f1ff';
  const points = profile.map((v, layer) => {
    const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
    const x = 8.2 + Math.max(0, toSafeNumber(v, 0)) * 4.8;
    const y = -6.45;
    return [x, y, z];
  });
  const diagAdv = suppression?.diagonal_advantage?.[dimension];
  const row = suppression?.suppression_matrix_mean?.[dimension];
  return (
    <group>
      <Line points={[[8.2, -6.45, -13.1], [8.2, -6.45, 13.1]]} color="#7f95bb" transparent opacity={0.28} lineWidth={1} />
      <Line points={points} color={color} transparent opacity={0.95} lineWidth={2.6} />
      {points.map((p, idx) => (
        <mesh key={`impact-${dimension}-${idx}`} position={p}>
          <sphereGeometry args={[0.045, 12, 12]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.9} />
        </mesh>
      ))}
      <Text position={[9.7, -5.8, 13.3]} color={color} fontSize={0.23} anchorX="right" anchorY="middle" outlineWidth={0.015} outlineColor="#0a1022">
        {`${DIMENSION_LABELS[dimension] || dimension} 层影响谱`}
      </Text>
      {Number.isFinite(diagAdv) && (
        <Text position={[9.7, -6.2, 13.3]} color="#cde4ff" fontSize={0.2} anchorX="right" anchorY="middle" outlineWidth={0.012} outlineColor="#0a1022">
          {`对角优势: ${diagAdv.toFixed(4)}`}
        </Text>
      )}
      {row ? (
        <Text position={[9.7, -6.55, 13.3]} color="#9eb4dd" fontSize={0.18} anchorX="right" anchorY="middle" outlineWidth={0.01} outlineColor="#0a1022">
          {`S/L/Y: ${toSafeNumber(row.style, 0).toFixed(3)} / ${toSafeNumber(row.logic, 0).toFixed(3)} / ${toSafeNumber(row.syntax, 0).toFixed(3)}`}
        </Text>
      ) : null}
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

export function AppleNeuronSceneContent({
  nodes,
  links,
  selected,
  onSelect,
  prediction = null,
  mode = 'static',
  dimensionLayerProfile = [],
  activeDimension = 'style',
  dimensionCausal = null,
  nodeDisplayEmphasis = {},
}) {
  const activationMap = prediction?.activationMap || {};
  const focusNodeIds = prediction?.focusNodeIds || [];
  const focusNodeSet = useMemo(() => new Set(focusNodeIds), [focusNodeIds]);
  const modeStyle = MODE_VISUALS[mode] || MODE_VISUALS.static;
  const activeLayer = Number.isFinite(prediction?.layerProgress)
    ? prediction.layerProgress * (LAYER_COUNT - 1)
    : null;

  const visibleNodes = useMemo(
    () => nodes.filter((node) => toSafeNumber(nodeDisplayEmphasis?.[node.id], 1) > 0.025),
    [nodeDisplayEmphasis, nodes]
  );
  const visibleNodeIdSet = useMemo(() => new Set(visibleNodes.map((n) => n.id)), [visibleNodes]);
  const visibleLinks = useMemo(
    () => links.filter((link) => visibleNodeIdSet.has(link?.from) && visibleNodeIdSet.has(link?.to)),
    [links, visibleNodeIdSet]
  );

  return (
    <>
      <LayerGuides activeLayer={activeLayer} />

      {visibleLinks.map((link) => (
        <Line
          key={link.id}
          points={link.points}
          color={mode === 'dynamic_prediction' || mode === 'static' ? link.color : modeStyle.accent}
          transparent
          opacity={0.42 + (prediction?.isRunning ? 0.18 : 0) + modeStyle.linkOpacityBoost}
          lineWidth={1.6 + modeStyle.linkWidthBoost}
        />
      ))}

      {visibleNodes.map((node) => (
        <PulsingNeuron
          key={node.id}
          node={node}
          selected={selected?.id === node.id}
          onSelect={onSelect}
          predictionStrength={activationMap[node.id] || 0}
          mode={mode}
          isEffectiveNode={focusNodeSet.has(node.id)}
          visibilityEmphasis={toSafeNumber(nodeDisplayEmphasis?.[node.id], 1)}
        />
      ))}

      <ModeVisualOverlay mode={mode} prediction={prediction} />
      <TokenPredictionCarrier prediction={prediction} mode={mode} />
      <LayerEffectiveNeuronOverlay prediction={prediction} mode={mode} />
      <DimensionLayerImpactGraph profile={dimensionLayerProfile} dimension={activeDimension} suppression={dimensionCausal} />

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

function AppleNeuronScene({
  nodes,
  links,
  selected,
  onSelect,
  prediction,
  mode = 'static',
  dimensionLayerProfile = [],
  activeDimension = 'style',
  dimensionCausal = null,
  nodeDisplayEmphasis = {},
}) {
  return (
    <Canvas shadows dpr={[1, 1.5]}>
      <color attach="background" args={['#090b15']} />
      <fog attach="fog" args={['#090b15', 14, 42]} />

      <ambientLight intensity={0.5} />
      <pointLight position={[12, 12, 16]} intensity={70} color="#8fc4ff" />
      <pointLight position={[-14, -8, -15]} intensity={30} color="#ff9e6b" />

      <PerspectiveCamera makeDefault position={[16, 12, 26]} fov={42} />
      <OrbitControls enablePan enableZoom minDistance={10} maxDistance={44} />

      <AppleNeuronSceneContent
        nodes={nodes}
        links={links}
        selected={selected}
        onSelect={onSelect}
        prediction={prediction}
        mode={mode}
        dimensionLayerProfile={dimensionLayerProfile}
        activeDimension={activeDimension}
        dimensionCausal={dimensionCausal}
        nodeDisplayEmphasis={nodeDisplayEmphasis}
      />
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
  return [];
}

export function useAppleNeuronWorkspace() {
  const [analysisMode, setAnalysisMode] = useState('dynamic_prediction');
  const [showFruitGeneral, setShowFruitGeneral] = useState(true);
  const [showFruit, setShowFruit] = useState(() => Object.fromEntries(Object.keys(FRUIT_COLORS).map((k) => [k, true])));
  const [queryInput, setQueryInput] = useState('');
  const [queryCategoryInput, setQueryCategoryInput] = useState('');
  const [querySets, setQuerySets] = useState([]);
  const [queryVisibility, setQueryVisibility] = useState({});
  const [queryFeedback, setQueryFeedback] = useState('');
  const [scanImportLimit, setScanImportLimit] = useState(20);
  const [scanImportTopK, setScanImportTopK] = useState(IMPORTED_QUERY_NODE_MAX);
  const [scanImportSummary, setScanImportSummary] = useState(null);
  const [scanMechanismData, setScanMechanismData] = useState(null);
  const [multidimProbeData, setMultidimProbeData] = useState(null);
  const [multidimCausalData, setMultidimCausalData] = useState(null);
  const [hardProblemResults, setHardProblemResults] = useState({});
  const [unifiedDecodeResult, setUnifiedDecodeResult] = useState(null);
  const [bundleManifest, setBundleManifest] = useState(null);
  const [fourTasksManifest, setFourTasksManifest] = useState(null);
  const [multidimTopN, setMultidimTopN] = useState(96);
  const [multidimVisible, setMultidimVisible] = useState({ style: true, logic: true, syntax: true });
  const [multidimActiveDimension, setMultidimActiveDimension] = useState('style');
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
  const [counterfactualPrompt, setCounterfactualPrompt] = useState('');
  const [robustnessTrials, setRobustnessTrials] = useState(6);
  const [minimalSubsetSize, setMinimalSubsetSize] = useState(12);
  const [displayStrategy, setDisplayStrategy] = useState('auto');
  const [manualDisplayGroups, setManualDisplayGroups] = useState({
    core: true,
    query: true,
    multidim: true,
    hard: true,
    unified: true,
    background: false,
  });

  const backgroundNodes = useMemo(() => buildBackgroundNodes(), []);
  const appleCoreNodes = useMemo(() => buildAppleCoreNodes(), []);
  const fruitGeneralNodes = useMemo(() => buildFruitGeneralNodes(), []);
  const fruitSpecificNodes = useMemo(() => buildFruitSpecificNodes(), []);
  const queryNodes = useMemo(
    () => querySets
      .filter((set) => queryVisibility[set.id] !== false)
      .flatMap((set) => set.nodes),
    [querySets, queryVisibility]
  );
  const multidimNodes = useMemo(
    () => buildMultidimNodesFromProbe(multidimProbeData, multidimVisible, multidimTopN),
    [multidimProbeData, multidimVisible, multidimTopN]
  );
  const hardProblemNodes = useMemo(() => buildHardProblemNodes(hardProblemResults), [hardProblemResults]);
  const unifiedDecodeNodes = useMemo(() => buildUnifiedDecodeNodes(unifiedDecodeResult), [unifiedDecodeResult]);
  const predictChain = useMemo(() => generatePredictChain(predictPrompt), [predictPrompt]);
  const dynamicEnabled = analysisMode === 'dynamic_prediction';
  const mechanismEnabled = !['static', 'dynamic_prediction'].includes(analysisMode);

  const nodes = useMemo(() => {
    const visibleFruitSpecific = fruitSpecificNodes.filter((n) => showFruit[n.fruit]);
    const visibleFruitGeneral = showFruitGeneral ? fruitGeneralNodes : [];
    return [
      ...backgroundNodes,
      ...appleCoreNodes,
      ...visibleFruitGeneral,
      ...visibleFruitSpecific,
      ...queryNodes,
      ...multidimNodes,
      ...hardProblemNodes,
      ...unifiedDecodeNodes,
    ];
  }, [
    appleCoreNodes,
    backgroundNodes,
    fruitGeneralNodes,
    fruitSpecificNodes,
    hardProblemNodes,
    multidimNodes,
    queryNodes,
    showFruit,
    showFruitGeneral,
    unifiedDecodeNodes,
  ]);

  const keyNodes = useMemo(() => nodes.filter((n) => n.role !== 'background'), [nodes]);
  const [selected, setSelected] = useState(appleCoreNodes[0] || null);
  const nodeDisplayEmphasis = useMemo(() => {
    const map = {};
    const autoProfile = buildAutoDisplayProfile(analysisMode);
    nodes.forEach((node) => {
      const group = nodeDisplayGroup(node.role);
      let emphasis = 1;
      if (displayStrategy === 'all') {
        emphasis = 1;
      } else if (displayStrategy === 'manual') {
        emphasis = manualDisplayGroups[group] === false ? 0.03 : 1;
      } else {
        emphasis = toSafeNumber(autoProfile[group], 0.8);
      }
      if (selected?.id === node.id) {
        emphasis = Math.max(emphasis, 0.95);
      }
      map[node.id] = Math.max(0, Math.min(1, emphasis));
    });
    return map;
  }, [analysisMode, displayStrategy, manualDisplayGroups, nodes, selected?.id]);

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

  useEffect(() => {
    setQueryVisibility((prev) => {
      const next = {};
      querySets.forEach((set) => {
        next[set.id] = prev[set.id] !== false;
      });
      return next;
    });
  }, [querySets]);

  const handleGenerateQuery = () => {
    const concept = queryInput.trim();
    const category = queryCategoryInput.trim() || '未分类';
    if (!concept) {
      setQueryFeedback('请输入名称后再生成。');
      return;
    }
    setQuerySets((prev) => {
      const existing = prev.find((set) => set.normalized === concept.toLowerCase() && set.normalizedCategory === category.toLowerCase());
      if (existing) {
        setQueryVisibility((visibilityPrev) => ({ ...visibilityPrev, [existing.id]: true }));
        if (existing.nodes[0]) {
          setSelected(existing.nodes[0]);
        }
        setQueryFeedback(`已存在「${existing.name} [${existing.category}]」，已定位并显示。`);
        return prev;
      }
      const nextSet = buildConceptNeuronSet(concept, category, prev.length);
      if (nextSet.nodes[0]) {
        setSelected(nextSet.nodes[0]);
      }
      setQueryVisibility((visibilityPrev) => ({ ...visibilityPrev, [nextSet.id]: true }));
      setQueryFeedback(`已生成「${nextSet.name} [${nextSet.category}]」神经元集合。`);
      return [...prev, nextSet];
    });
    setQueryInput('');
  };

  const handleImportScanJsonText = (jsonText, sourceName = 'mass_noun_encoding_scan.json') => {
    let parsed;
    try {
      parsed = JSON.parse(jsonText);
    } catch (_err) {
      setQueryFeedback('JSON 解析失败，请确认文件格式正确。');
      return;
    }

    if (isHardProblemResultPayload(parsed)) {
      const expId = parsed.experiment_id;
      const expLabel = HARD_PROBLEM_EXPERIMENT_LABELS[expId] || expId;
      setHardProblemResults((prev) => ({ ...prev, [expId]: parsed }));
      if (expId === 'minimal_causal_circuit_search_v1') {
        const targets = parsed?.metrics?.targets || {};
        const minimalByNoun = {};
        Object.entries(targets).forEach(([noun, row]) => {
          const runs = Array.isArray(row?.runs) ? row.runs : [];
          if (runs.length === 0) {
            return;
          }
          const pick = runs
            .slice()
            .sort((a, b) => toSafeNumber(b?.fidelity_ratio, 0) - toSafeNumber(a?.fidelity_ratio, 0))[0];
          const subset = Array.isArray(pick?.minimal_subset) ? pick.minimal_subset : [];
          if (subset.length === 0) {
            return;
          }
          const key = normalizeConceptKey(noun);
          if (!key) {
            return;
          }
          minimalByNoun[key] = {
            noun: key,
            subset_flat_indices: subset,
            subset_size: toSafeNumber(pick?.minimal_size, subset.length),
            recovery_ratio: toSafeNumber(pick?.fidelity_ratio, 0),
            subset_drop_seq_logprob: toSafeNumber(pick?.intervention_drop_after_remove_subset, 0),
          };
        });
        if (Object.keys(minimalByNoun).length > 0) {
          setScanMechanismData((prev) => ({
            dff: Math.max(1, toSafeNumber(prev?.dff, DFF)),
            minimalByNoun: { ...(prev?.minimalByNoun || {}), ...minimalByNoun },
            counterfactualByNoun: { ...(prev?.counterfactualByNoun || {}) },
          }));
        }
      }
      const metricKeys = Object.keys(parsed?.metrics || {});
      setQueryFeedback(`已导入硬伤实验：${expLabel}（${sourceName}），指标数=${metricKeys.length}。`);
      return;
    }

    if (isFourTasksManifestPayload(parsed)) {
      setFourTasksManifest(parsed);
      const allSuccess = Boolean(parsed?.all_success);
      setQueryFeedback(`已导入四任务清单：${allSuccess ? '全部成功' : '存在失败'}，任务数=${Object.keys(parsed?.return_codes || {}).length}。`);
      return;
    }

    if (isUnifiedDecodePayload(parsed)) {
      setUnifiedDecodeResult(parsed);
      const passRatio = toSafeNumber(parsed?.hypothesis_test?.pass_ratio, 0);
      setQueryFeedback(`已导入统一解码结果：${sourceName}，假设通过率=${(passRatio * 100).toFixed(1)}%。`);
      return;
    }

    if (isBundleManifestPayload(parsed)) {
      setBundleManifest(parsed);
      const snap = parsed?.metrics_snapshot || {};
      const dynamic = toSafeNumber(snap?.dynamic_binding?.binding_stability_index, 0);
      const longDecay = toSafeNumber(snap?.long_horizon?.long_horizon_decay, 0);
      const localSel = toSafeNumber(snap?.local_credit?.local_selectivity_mean, 0);
      setQueryFeedback(`已导入批量实验清单：动态稳定=${dynamic.toFixed(3)}，长程衰减=${longDecay.toFixed(3)}，局部选择性=${localSel.toFixed(3)}。`);
      return;
    }

    const hasMultidimProbe = Boolean(
      parsed?.dimensions?.style
      && parsed?.dimensions?.logic
      && parsed?.dimensions?.syntax
      && parsed?.cross_dimension
    );
    const hasMultidimCausal = Boolean(parsed?.suppression_matrix_mean && parsed?.diagonal_advantage);
    const hasMultidimStability = Boolean(parsed?.aggregate?.diag_adv_style && parsed?.aggregate?.specificity_margin_style);
    if (hasMultidimProbe) {
      setMultidimProbeData(parsed);
      const probeSummary = parsed?.runtime_config || {};
      setQueryFeedback(
        `已导入三维编码探针：source=${sourceName}，每维样本=${probeSummary.max_pairs_per_dim || '-'}，top_k=${probeSummary.top_k || '-'}。`
      );
      return;
    }
    if (hasMultidimCausal) {
      setMultidimCausalData(parsed);
      setQueryFeedback(`已导入三维因果消融：source=${sourceName}，top_n=${parsed?.top_n || '-'}。`);
      return;
    }
    if (hasMultidimStability) {
      const runs = toSafeNumber(parsed?.n_runs, 0);
      const style = toSafeNumber(parsed?.aggregate?.specificity_margin_style?.mean, 0);
      const logic = toSafeNumber(parsed?.aggregate?.specificity_margin_logic?.mean, 0);
      const syntax = toSafeNumber(parsed?.aggregate?.specificity_margin_syntax?.mean, 0);
      setQueryFeedback(`已导入三维多seed稳定性汇总：runs=${runs}，specificity(style/logic/syntax)=(${style.toFixed(3)}/${logic.toFixed(3)}/${syntax.toFixed(3)})。`);
      return;
    }

    const nounRecords = Array.isArray(parsed?.noun_records) ? parsed.noun_records : [];
    if (nounRecords.length === 0) {
      setQueryFeedback('未检测到 noun_records，无法导入。');
      return;
    }

    const dff = Math.max(1, toSafeNumber(parsed?.config?.d_ff, DFF));
    const limit = Math.max(1, Math.min(120, toSafeNumber(scanImportLimit, 20)));
    const perConceptTopK = Math.max(4, Math.min(64, toSafeNumber(scanImportTopK, IMPORTED_QUERY_NODE_MAX)));

    const validRecords = nounRecords.filter((rec) => Array.isArray(rec?.signature_top_indices) && rec.signature_top_indices.length > 0);
    if (validRecords.length === 0) {
      setQueryFeedback('该扫描结果没有 signature_top_indices，请先用新版脚本重新导出。');
      return;
    }

    const picked = validRecords.slice(0, limit);
    const importedSets = picked.map((rec, idx) => buildConceptNeuronSetFromSignature(
      String(rec?.noun || `concept_${idx + 1}`),
      String(rec?.category || '未分类'),
      rec?.signature_top_indices || [],
      idx,
      dff,
      perConceptTopK
    ));

    const reused = Array.isArray(parsed?.top_reused_neurons) ? parsed.top_reused_neurons : [];
    if (reused.length > 0) {
      importedSets.push(buildSharedReuseSet(reused, dff, perConceptTopK, picked.length));
    }

    const minimalRecords = Array.isArray(parsed?.causal_ablation?.minimal_circuit?.records)
      ? parsed.causal_ablation.minimal_circuit.records
      : [];
    const counterfactualRecords = Array.isArray(parsed?.causal_ablation?.counterfactual_validation?.records)
      ? parsed.causal_ablation.counterfactual_validation.records
      : [];
    const minimalByNoun = {};
    minimalRecords.forEach((rec) => {
      const key = normalizeConceptKey(rec?.noun);
      if (!key) {
        return;
      }
      minimalByNoun[key] = rec;
    });
    const counterfactualByNoun = {};
    counterfactualRecords.forEach((rec) => {
      const key = normalizeConceptKey(rec?.noun);
      if (!key) {
        return;
      }
      if (!counterfactualByNoun[key]) {
        counterfactualByNoun[key] = [];
      }
      counterfactualByNoun[key].push(rec);
    });
    setScanMechanismData({
      dff,
      minimalByNoun,
      counterfactualByNoun,
    });
    const firstCounterfactual = counterfactualRecords[0];
    if (firstCounterfactual?.counterfactual_noun) {
      setCounterfactualPrompt(String(firstCounterfactual.counterfactual_noun));
    }

    let added = 0;
    let updated = 0;
    setQuerySets((prev) => {
      const next = [...prev];
      importedSets.forEach((set) => {
        const existingIdx = next.findIndex(
          (item) => item.normalized === set.normalized && item.normalizedCategory === set.normalizedCategory
        );
        if (existingIdx >= 0) {
          next[existingIdx] = set;
          updated += 1;
        } else {
          next.push(set);
          added += 1;
        }
      });
      return next;
    });

    setQueryVisibility((prev) => {
      const next = { ...prev };
      importedSets.forEach((set) => {
        next[set.id] = true;
      });
      return next;
    });

    if (importedSets[0]?.nodes?.[0]) {
      setSelected(importedSets[0].nodes[0]);
    }

    const importedCategoryCount = new Set(importedSets.map((set) => set.category)).size;
    setScanImportSummary({
      source: sourceName,
      importedConcepts: importedSets.length,
      importedCategories: importedCategoryCount,
      totalNouns: nounRecords.length,
      minimalCircuitNouns: minimalRecords.length,
      counterfactualPairs: counterfactualRecords.length,
    });
    setQueryFeedback(`已导入扫描结果：新增 ${added}，更新 ${updated}，来源 ${sourceName}。`);
  };

  const removeQuerySet = (setId) => {
    setQuerySets((prev) => prev.filter((set) => set.id !== setId));
    setQueryVisibility((prev) => {
      const next = { ...prev };
      delete next[setId];
      return next;
    });
    setQueryFeedback('已移除该概念集合。');
  };

  const setQuerySetVisible = (setId, visible) => {
    setQueryVisibility((prev) => ({ ...prev, [setId]: visible }));
  };

  const setAllQuerySetVisible = (visible) => {
    setQueryVisibility((prev) => {
      const next = { ...prev };
      querySets.forEach((set) => {
        next[set.id] = visible;
      });
      return next;
    });
  };

  const links = useMemo(() => {
    const byId = Object.fromEntries(keyNodes.map((n) => [n.id, n]));
    const linkSpecs = [];

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

    const multidimLinks = ['style', 'logic', 'syntax'].flatMap((dim) => {
      if (multidimVisible[dim] === false) {
        return [];
      }
      const color = ROLE_COLORS[dim] || '#84f1ff';
      const group = keyNodes
        .filter((n) => n.role === dim)
        .sort((a, b) => Number(b.value || 0) - Number(a.value || 0))
        .slice(0, 16);
      if (group.length < 2) {
        return [];
      }
      return group.slice(1).map((node) => [group[0].id, node.id, color]);
    });

    return [...linkSpecs, ...fruitLinks, ...queryLinks, ...multidimLinks]
      .filter(([from, to]) => byId[from] && byId[to])
      .map(([from, to, color]) => ({
        id: `${from}->${to}`,
        from,
        to,
        color,
        points: [byId[from].position, byId[to].position],
      }));
  }, [keyNodes, multidimVisible, querySets, showFruit]);

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
      effectiveLayer: null,
      effectiveNeurons: [],
      metrics: [],
      statusText: '',
    };

    if (!keyNodes.length) {
      overlay.metrics = [{ label: '节点', value: '0（请先生成或导入概念）' }];
    }
    const selectedConceptKey = normalizeConceptKey(selected?.concept);
    const importedDff = Math.max(1, toSafeNumber(scanMechanismData?.dff, DFF));
    const importedMinimal = selectedConceptKey ? scanMechanismData?.minimalByNoun?.[selectedConceptKey] : null;
    const importedCounterfactualList = selectedConceptKey ? (scanMechanismData?.counterfactualByNoun?.[selectedConceptKey] || []) : [];

    if (analysisMode === 'static') {
      keyNodes.forEach((node) => {
        overlay.activationMap[node.id] = Math.min(0.25, 0.06 + Math.sqrt(Math.max(node.strength, 1e-6)) * 0.5);
      });
      overlay.statusText = '结构分布快照';
      overlay.metrics = [{ label: '模式', value: '静态分析' }];
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
      const currentLayer = Math.max(0, Math.min(LAYER_COUNT - 1, Math.round(mechanismPhase * (LAYER_COUNT - 1))));
      const layerEffective = [];
      keyNodes.forEach((node) => {
        const axis = hashString(`feature-axis|${node.id}`) % FEATURE_AXES.length;
        const local = pseudoRandom(hashString(`feature-val|${axisName}|${node.id}`));
        const score = axis === featureAxis ? 0.58 + local * 0.4 : 0.08 + local * 0.2;
        overlay.activationMap[node.id] = score;
        if (node.layer === currentLayer && node.role !== 'background') {
          layerEffective.push({ ...node, score: score * (axis === featureAxis ? 1.15 : 0.72) });
        }
      });
      layerEffective.sort((a, b) => b.score - a.score);
      const topLayerNodes = layerEffective.slice(0, 8);
      overlay.focusNodeIds = topLayerNodes.map((n) => n.id);
      overlay.effectiveLayer = currentLayer;
      overlay.effectiveNeurons = topLayerNodes.map((n) => ({
        id: n.id,
        label: n.label,
        role: n.role,
        layer: n.layer,
        neuron: n.neuron,
        score: n.score,
      }));
      overlay.currentToken = { token: `axis:${axisName}`, prob: 0.78 };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = `特征分解：定位 L${currentLayer} 有效神经元`;
      overlay.metrics = [
        { label: 'Axis', value: axisName },
        { label: 'Slots', value: `${FEATURE_AXES.length}` },
        { label: '当前层', value: `L${currentLayer}` },
        { label: '有效神经元', value: `${topLayerNodes.length}` },
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
      if (importedCounterfactualList.length > 0) {
        const preferred = importedCounterfactualList.find((r) => r?.relation === 'same_category') || importedCounterfactualList[0];
        const cfConcept = normalizeConceptKey(preferred?.counterfactual_noun);
        const focus = [];
        keyNodes.forEach((node) => {
          const nk = normalizeConceptKey(node.concept);
          if (nk === selectedConceptKey) {
            overlay.activationMap[node.id] = 0.88;
            focus.push(node.id);
            return;
          }
          if (cfConcept && nk === cfConcept) {
            overlay.activationMap[node.id] = 0.5;
            focus.push(node.id);
            return;
          }
          overlay.activationMap[node.id] = 0.02 + pseudoRandom(hashString(`cf-import-bg|${node.id}`)) * 0.08;
        });
        overlay.focusNodeIds = focus;
        overlay.currentToken = { token: `CF: ${preferred?.noun || '-'} -> ${preferred?.counterfactual_noun || '-'}`, prob: 0.76 };
        overlay.layerProgress = mechanismPhase;
        overlay.statusText = '反事实特异性（导入）';
        overlay.metrics = [
          { label: '关系', value: preferred?.relation === 'same_category' ? '同类反事实' : '跨类反事实' },
          { label: '特异性边际', value: `${toSafeNumber(preferred?.specificity_margin_seq_logprob, 0).toFixed(6)}` },
          { label: '子集大小', value: `${toSafeNumber(preferred?.subset_size, 0)}` },
        ];
        return overlay;
      }
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
      if (importedMinimal && Array.isArray(importedMinimal?.subset_flat_indices)) {
        const subset = new Set(importedMinimal.subset_flat_indices.map((v) => toSafeNumber(v, -1)).filter((v) => v >= 0));
        const focus = [];
        keyNodes.forEach((node) => {
          const flat = node.layer * importedDff + node.neuron;
          if (subset.has(flat)) {
            overlay.activationMap[node.id] = 0.92;
            focus.push(node.id);
          } else {
            overlay.activationMap[node.id] = 0.015;
          }
        });
        overlay.focusNodeIds = focus;
        const subsetSize = toSafeNumber(importedMinimal?.subset_size, subset.size);
        overlay.currentToken = { token: `MCS(import k=${subsetSize})`, prob: Math.min(0.99, toSafeNumber(importedMinimal?.recovery_ratio, 0)) };
        overlay.layerProgress = mechanismPhase;
        overlay.statusText = '最小因果子回路（导入）';
        overlay.metrics = [
          { label: '子集大小', value: `${subsetSize}` },
          { label: '恢复率', value: `${toSafeNumber(importedMinimal?.recovery_ratio, 0).toFixed(3)}` },
          { label: 'Seq Drop', value: `${toSafeNumber(importedMinimal?.subset_drop_seq_logprob, 0).toFixed(6)}` },
        ];
        return overlay;
      }
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
    scanMechanismData,
    selected,
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

  const multidimLayerProfile = useMemo(() => {
    const arr = multidimProbeData?.dimensions?.[multidimActiveDimension]?.layer_profile_abs_delta_norm;
    return Array.isArray(arr) ? arr : [];
  }, [multidimActiveDimension, multidimProbeData]);

  const summary = useMemo(() => {
    const fruitSpecific = keyNodes.filter((n) => n.role === 'fruitSpecific');
    const perFruit = Object.keys(FRUIT_COLORS).reduce((acc, fruit) => {
      acc[fruit] = fruitSpecific.filter((n) => n.fruit === fruit).length;
      return acc;
    }, {});
    const categoryStats = querySets.reduce((acc, set) => {
      const key = set.category || '未分类';
      if (!acc[key]) {
        acc[key] = { concepts: 0, neurons: 0 };
      }
      acc[key].concepts += 1;
      acc[key].neurons += set.nodes.length;
      return acc;
    }, {});

    return {
      micro: keyNodes.filter((n) => n.role === 'micro').length,
      macro: keyNodes.filter((n) => n.role === 'macro').length,
      route: keyNodes.filter((n) => n.role === 'route').length,
      fruitGeneral: keyNodes.filter((n) => n.role === 'fruitGeneral').length,
      fruitSpecific: fruitSpecific.length,
      query: keyNodes.filter((n) => n.role === 'query').length,
      hardProblemNodes: keyNodes.filter(
        (n) => n.role === 'hardBinding' || n.role === 'hardLong' || n.role === 'hardLocal' || n.role === 'hardTriplet'
      ).length,
      unifiedDecodeNodes: keyNodes.filter((n) => n.role === 'unifiedDecode').length,
      total: keyNodes.length,
      perFruit,
      categoryStats,
      visibleQuerySets: querySets.filter((set) => queryVisibility[set.id] !== false).length,
      hiddenQuerySets: querySets.filter((set) => queryVisibility[set.id] === false).length,
      multidimNodes: keyNodes.filter((n) => n.role === 'style' || n.role === 'logic' || n.role === 'syntax').length,
      multidimActiveDimension,
      hardProblemCount: Object.keys(hardProblemResults || {}).length,
      unifiedDecodeLoaded: Boolean(unifiedDecodeResult),
      bundleLoaded: Boolean(bundleManifest),
      fourTasksLoaded: Boolean(fourTasksManifest),
      currentToken: modeOverlay.currentToken?.token || '-',
      currentTokenProb: modeOverlay.currentToken?.prob || 0,
      analysisMode,
      displayStrategy,
      statusText: modeOverlay.statusText || '',
    };
  }, [
    analysisMode,
    bundleManifest,
    displayStrategy,
    fourTasksManifest,
    hardProblemResults,
    keyNodes,
    modeOverlay.currentToken,
    modeOverlay.statusText,
    multidimActiveDimension,
    querySets,
    queryVisibility,
    unifiedDecodeResult,
  ]);

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
    queryCategoryInput,
    setQueryCategoryInput,
    querySets,
    queryVisibility,
    queryFeedback,
    scanImportLimit,
    setScanImportLimit,
    scanImportTopK,
    setScanImportTopK,
    scanImportSummary,
    multidimProbeData,
    multidimCausalData,
    hardProblemResults,
    unifiedDecodeResult,
    bundleManifest,
    fourTasksManifest,
    multidimTopN,
    setMultidimTopN,
    multidimVisible,
    setMultidimVisible,
    multidimActiveDimension,
    setMultidimActiveDimension,
    multidimLayerProfile,
    handleGenerateQuery,
    handleImportScanJsonText,
    removeQuerySet,
    setQuerySetVisible,
    setAllQuerySetVisible,
    nodes,
    nodeDisplayEmphasis,
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
    displayStrategy,
    setDisplayStrategy,
    manualDisplayGroups,
    setManualDisplayGroups,
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
          focusNodeIds: modeOverlay.focusNodeIds,
          effectiveLayer: modeOverlay.effectiveLayer,
          effectiveNeurons: modeOverlay.effectiveNeurons,
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
        dimensionLayerProfile={workspace.multidimLayerProfile}
        activeDimension={workspace.multidimActiveDimension}
        dimensionCausal={workspace.multidimCausalData}
        nodeDisplayEmphasis={workspace.nodeDisplayEmphasis}
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
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 6,
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

const fixedFileControlWidth = 240;
const formatScanOptionLabel = (fileMeta) => {
  const rawName = String(fileMeta?.name || 'scan.json');
  const shortName = rawName.length > 24 ? `${rawName.slice(0, 12)}...${rawName.slice(-9)}` : rawName;
  const mtime = String(fileMeta?.mtime_iso || '').slice(0, 19).replace('T', ' ');
  return mtime ? `${shortName} | ${mtime}` : shortName;
};

export function AppleNeuronEncodingInfoPanels({ workspace, compact = false }) {
  const nodes = workspace?.nodes || [];
  const summary = workspace?.summary || {};
  const metrics = workspace?.modeMetrics || [];
  const multidimProbe = workspace?.multidimProbeData || null;
  const multidimCausal = workspace?.multidimCausalData || null;
  const hardProblemResults = workspace?.hardProblemResults || {};
  const unifiedDecodeResult = workspace?.unifiedDecodeResult || null;
  const bundleManifest = workspace?.bundleManifest || null;
  const fourTasksManifest = workspace?.fourTasksManifest || null;
  const activeDim = workspace?.multidimActiveDimension || 'style';

  const layerRows = useMemo(() => {
    const map = new Map();
    nodes.forEach((node) => {
      const key = Number.isFinite(node.layer) ? node.layer : 0;
      const row = map.get(key) || { layer: key, count: 0, strength: 0 };
      row.count += 1;
      row.strength += Number(node.strength || 0);
      map.set(key, row);
    });
    return Array.from(map.values())
      .sort((a, b) => a.layer - b.layer)
      .map((row) => ({
        ...row,
        strength: row.count ? row.strength / row.count : 0,
      }));
  }, [nodes]);

  const maxCount = useMemo(() => Math.max(1, ...layerRows.map((row) => row.count)), [layerRows]);
  const hardProblemRows = useMemo(() => Object.entries(hardProblemResults), [hardProblemResults]);
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>层级编码签名</div>
        <div style={{ display: 'grid', gap: 6, maxHeight: compact ? 130 : 170, overflowY: 'auto' }}>
          {layerRows.map((row) => (
            <div key={`layer-sign-${row.layer}`} style={{ display: 'grid', gridTemplateColumns: '44px 1fr', gap: 8, alignItems: 'center' }}>
              <div style={{ fontSize: 11, color: '#9bb3de' }}>{`L${row.layer}`}</div>
              <div style={{ display: 'grid', gap: 3 }}>
                <div style={{ height: 5, background: 'rgba(255,255,255,0.08)', borderRadius: 8, overflow: 'hidden' }}>
                  <div style={{ width: `${(row.count / maxCount) * 100}%`, height: '100%', background: '#67dfff' }} />
                </div>
                <div style={{ height: 4, background: 'rgba(255,255,255,0.08)', borderRadius: 8, overflow: 'hidden' }}>
                  <div style={{ width: `${Math.min(100, row.strength * 1000000)}%`, height: '100%', background: '#f59e0b' }} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>编码机制指标</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>
          <div>{`核心神经元: ${(summary.micro || 0) + (summary.macro || 0) + (summary.route || 0)}`}</div>
          <div>{`当前词元: ${summary.currentToken || '-'} (${((summary.currentTokenProb || 0) * 100).toFixed(1)}%)`}</div>
          <div>{`显示策略: ${summary.displayStrategy === 'auto' ? '自动聚焦' : summary.displayStrategy === 'all' ? '全部显示' : '手动筛选'}`}</div>
        </div>
        {metrics.length > 0 ? (
          <div style={{ marginTop: 8, display: 'grid', gap: 4 }}>
            {metrics.map((metric, idx) => (
              <div key={`m-${metric.label}-${idx}`} style={{ fontSize: 11, color: '#9bb3de' }}>{`${metric.label}: ${metric.value}`}</div>
            ))}
          </div>
        ) : null}
        {multidimProbe ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`多维探针: 已导入`}</div>
            <div>{`当前维度: ${DIMENSION_LABELS[activeDim] || activeDim}`}</div>
            {Number.isFinite(multidimCausal?.diagonal_advantage?.[activeDim]) ? (
              <div>{`对角优势: ${toSafeNumber(multidimCausal?.diagonal_advantage?.[activeDim], 0).toFixed(4)}`}</div>
            ) : (
              <div style={{ color: '#7f95bb' }}>未导入三维因果消融结果</div>
            )}
          </div>
        ) : null}
        {hardProblemRows.length > 0 ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`硬伤实验导入: ${hardProblemRows.length}`}</div>
            {hardProblemRows.map(([expId, payload]) => {
              const title = HARD_PROBLEM_EXPERIMENT_LABELS[expId] || payload?.title || expId;
              const mm = payload?.metrics || {};
              if (expId === 'hard_problem_dynamic_binding_v1') {
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 稳定性=${toSafeNumber(mm.binding_stability_index, 0).toFixed(3)} | 交换错率=${toSafeNumber(mm.role_swap_error_rate, 0).toFixed(3)}`}
                  </div>
                );
              }
              if (expId === 'hard_problem_long_horizon_trace_v1') {
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 长程衰减=${toSafeNumber(mm.long_horizon_decay, 0).toFixed(3)} | 传输稳定=${toSafeNumber(mm.layer_transport_stability_mean, 0).toFixed(3)}`}
                  </div>
                );
              }
              if (expId === 'hard_problem_local_credit_assignment_v1') {
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 局部充分=${toSafeNumber(mm.local_sufficiency_mean, 0).toFixed(3)} | 局部选择=${toSafeNumber(mm.local_selectivity_mean, 0).toFixed(3)}`}
                  </div>
                );
              }
              if (expId === 'triplet_targeted_causal_scan_v1') {
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 三联分离=${toSafeNumber(mm.triplet_separability_index, 0).toFixed(3)} | 轴特异=${toSafeNumber(mm.axis_specificity_index, 0).toFixed(3)}`}
                  </div>
                );
              }
              if (expId === 'triplet_targeted_multiseed_stability_v1') {
                const seqMargin = toSafeNumber(mm?.global_mean_causal_margin_seq_logprob?.mean, 0);
                const posRatio = toSafeNumber(mm?.global_positive_causal_margin_ratio?.mean, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: seq边际均值=${seqMargin.toFixed(4)} | 正边际比例=${(posRatio * 100).toFixed(1)}%`}
                  </div>
                );
              }
              if (expId === 'hard_problem_variable_binding_verification_v1') {
                const meanDelta = toSafeNumber(mm?.mean_delta, 0);
                const improvedDims = toSafeNumber(mm?.improved_dimension_count, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 平均提升=${meanDelta.toFixed(4)} | 提升维度=${improvedDims}`}
                  </div>
                );
              }
              if (expId === 'minimal_causal_circuit_search_v1') {
                const drop = toSafeNumber(mm?.global?.intervention_drop_mean, 0);
                const repr = toSafeNumber(mm?.global?.reproducibility_jaccard_mean, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 干预下降=${drop.toFixed(4)} | 复现Jaccard=${repr.toFixed(4)}`}
                  </div>
                );
              }
              if (expId === 'unified_coordinate_system_test_v1') {
                const us = toSafeNumber(mm?.unified_coordinate_score, 0);
                const orth = toSafeNumber(mm?.probe_orthogonality?.orthogonality_index, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 统一分数=${us.toFixed(4)} | 正交性=${orth.toFixed(4)}`}
                  </div>
                );
              }
              if (expId === 'concept_family_parallel_scale_v1') {
                const appleShared = toSafeNumber(mm?.apple_chain_summary?.shared_base_ratio_vs_micro_union?.mean, 0);
                const catShared = toSafeNumber(mm?.cat_chain_summary?.shared_base_ratio_vs_micro_union?.mean, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 苹果共享=${appleShared.toFixed(4)} | 猫共享=${catShared.toFixed(4)}`}
                  </div>
                );
              }
              return <div key={`hp-${expId}`}>{`${title}: 已导入`}</div>;
            })}
          </div>
        ) : null}
        {unifiedDecodeResult ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`统一解码: 已导入`}</div>
            <div>{`假设通过率: ${(toSafeNumber(unifiedDecodeResult?.hypothesis_test?.pass_ratio, 0) * 100).toFixed(1)}%`}</div>
            <div>{`探针文件数: ${toSafeNumber(unifiedDecodeResult?.axis_stability?.n_probe_files, 0)}`}</div>
          </div>
        ) : null}
        {bundleManifest ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`批量清单: 已导入`}</div>
            <div>{`seed=${toSafeNumber(bundleManifest?.config?.seed, 0)} | 统一解码=${bundleManifest?.config?.run_unified_decoder ? '开启' : '关闭'}`}</div>
          </div>
        ) : null}
        {fourTasksManifest ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`四任务清单: 已导入`}</div>
            <div>{`all_success=${fourTasksManifest?.all_success ? 'true' : 'false'} | 任务数=${Object.keys(fourTasksManifest?.return_codes || {}).length}`}</div>
          </div>
        ) : null}
      </div>

      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>硬伤实验与统一编码说明</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7, display: 'grid', gap: 6, maxHeight: compact ? 220 : 300, overflowY: 'auto' }}>
          <div style={{ color: '#cfe2ff', fontWeight: 700 }}>硬伤实验节点（红/蓝/橙/紫）</div>
          <div>1. 来源：导入 `agi_research_result.v1` 的实验指标。</div>
          <div>2. 映射：每个关键指标生成一个节点，颜色代表实验类型。</div>
          <div>3. 位置：层号与神经元号由指标哈希映射到主 3D 空间（用于对比，不代表真实单一神经元定位）。</div>
          <div>4. 大小与亮度：由指标强度决定；误差类指标（error/collision/decay）按“越小越好”反向映射。</div>
          <div style={{ color: '#cfe2ff', fontWeight: 700, marginTop: 4 }}>统一编码节点（绿色）</div>
          <div>1. 来源：`unified_math_structure_decode.json` 的融合结果。</div>
          <div>2. 映射：按 style / logic / syntax 三个维度生成节点簇。</div>
          <div>3. 层位：优先使用 dominant layer pattern，把节点放到对应层；无模式时使用回退层。</div>
          <div>4. 强度：综合 `profile_cosine_mean` 与 `diagonal_advantage`，用于显示“轴稳定 + 因果可分离”程度。</div>
          <div style={{ color: '#cfe2ff', fontWeight: 700, marginTop: 4 }}>读图顺序（建议）</div>
          <div>1. 先看图例颜色区分实验类型。</div>
          <div>2. 再看模型说明中的指标数值（均值/通过率）。</div>
          <div>3. 最后点选节点查看 `metric/value/source`，判断该可视化是“证据节点”还是“结构节点”。</div>
        </div>
      </div>
    </div>
  );
}

export function AppleNeuronSelectedLegendPanels({ workspace, compact = false }) {
  const selected = workspace?.selected || null;
  const summary = workspace?.summary || {};
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <div style={{ ...cardStyle, minHeight: 160 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>选中神经元</div>
        {selected ? (
          <div style={{ fontSize: 12, color: '#9eb4dd', display: 'grid', gap: 6 }}>
            <div style={{ color: '#e5eeff', fontWeight: 700 }}>{selected.label}</div>
            <div>{`角色: ${selected.role}`}</div>
            {'fruit' in selected ? <div>{`水果: ${selected.fruit}`}</div> : null}
            {'concept' in selected ? <div>{`概念: ${selected.concept}`}</div> : null}
            {'category' in selected ? <div>{`类别: ${selected.category}`}</div> : null}
            <div>{`层 / 神经元: L${selected.layer} / N${selected.neuron}`}</div>
            <div>{`强度: ${selected.strength.toExponential(3)}`}</div>
            <div>{`${selected.metric}: ${selected.value.toExponential(3)}`}</div>
            <div style={{ color: '#6f84ad' }}>{`来源: ${selected.source}`}</div>
          </div>
        ) : (
          <div style={{ fontSize: 12, color: '#7d93bd' }}>请在 3D 场景中点击高亮神经元。</div>
        )}
      </div>

      <div style={{ ...cardStyle, fontSize: 12, color: '#9eb4dd', lineHeight: 1.7 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>图例</div>
        <div><span style={{ color: ROLE_COLORS.micro }}>●</span> 微观神经元</div>
        <div><span style={{ color: ROLE_COLORS.macro }}>●</span> 中观神经元</div>
        <div><span style={{ color: ROLE_COLORS.route }}>●</span> 共享路径神经元</div>
        <div><span style={{ color: ROLE_COLORS.fruitGeneral }}>●</span> 类别通用神经元</div>
        <div><span style={{ color: ROLE_COLORS.style }}>●</span> 风格维度神经元</div>
        <div><span style={{ color: ROLE_COLORS.logic }}>●</span> 逻辑维度神经元</div>
        <div><span style={{ color: ROLE_COLORS.syntax }}>●</span> 句法维度神经元</div>
        <div><span style={{ color: ROLE_COLORS.hardBinding }}>●</span> 硬伤实验-动态绑定</div>
        <div><span style={{ color: ROLE_COLORS.hardLong }}>●</span> 硬伤实验-长程链路</div>
        <div><span style={{ color: ROLE_COLORS.hardLocal }}>●</span> 硬伤实验-局部信用</div>
        <div><span style={{ color: ROLE_COLORS.hardTriplet }}>●</span> 硬伤实验-三元组定向因果</div>
        <div><span style={{ color: ROLE_COLORS.unifiedDecode }}>●</span> 统一解码节点</div>
        <div><span style={{ color: '#84f1ff' }}>●</span> 输入概念神经元</div>
        <div><span style={{ color: ROLE_COLORS.background }}>●</span> 背景网络采样</div>
        <div style={{ color: '#6f84ad', marginTop: 8 }}>
          {`核心集合: ${(summary.micro || 0) + (summary.macro || 0) + (summary.route || 0)} | 多维集合: ${summary.multidimNodes || 0} | 硬伤: ${summary.hardProblemNodes || 0} | 统一解码: ${summary.unifiedDecodeNodes || 0}`}
        </div>
      </div>
    </div>
  );
}

export function AppleNeuronCategoryComparePanel({ workspace, compact = false }) {
  const summary = workspace?.summary || {};
  const querySets = workspace?.querySets || [];
  const categoryStats = summary?.categoryStats || {};
  const categoryRows = Object.entries(categoryStats)
    .map(([name, stat]) => ({ name, ...stat }))
    .sort((a, b) => b.neurons - a.neurons);
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>类别神经元比较</div>
      <div style={{ fontSize: 12, color: '#92a6cc', lineHeight: 1.6 }}>
        {`总神经元 ${summary.total || 0} | 查询神经元 ${summary.query || 0} | 类别数 ${categoryRows.length}`}
      </div>
      <div style={{ fontSize: 11, color: '#7ea2c9', marginTop: 4 }}>
        {`当前词元: ${summary.currentToken || '-'} (${((summary.currentTokenProb || 0) * 100).toFixed(1)}%) | 多维节点: ${summary.multidimNodes || 0}`}
      </div>
      <div style={{ marginTop: 8, display: 'grid', gap: 6 }}>
        {categoryRows.length === 0 ? (
          <div style={{ fontSize: 11, color: '#6f84ad' }}>暂无类别数据，请在左侧输入概念和类别后生成。</div>
        ) : (
          categoryRows.map((row) => (
            <div key={`cat-${row.name}`} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#9eb4dd' }}>
              <span>{row.name}</span>
              <span>{`${row.concepts} 概念 / ${row.neurons} 神经元`}</span>
            </div>
          ))
        )}
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: '#6f84ad' }}>{`已生成概念集: ${querySets.length}`}</div>
    </div>
  );
}

export function AppleNeuronCompareFilterPanel({ workspace, compact = false }) {
  const querySets = workspace?.querySets || [];
  const queryVisibility = workspace?.queryVisibility || {};
  const setQuerySetVisible = workspace?.setQuerySetVisible;
  const setAllQuerySetVisible = workspace?.setAllQuerySetVisible;
  const summary = workspace?.summary || {};
  const visibleCount = querySets.filter((set) => queryVisibility[set.id] !== false).length;
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>Compare Filter</div>
      <div style={{ fontSize: 12, color: '#92a6cc', lineHeight: 1.6 }}>
        {`按输入名称筛选：已显示 ${visibleCount}/${querySets.length}`}
      </div>
      <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
        <button type="button" onClick={() => setAllQuerySetVisible?.(true)} style={smallActionButtonStyle}>全选</button>
        <button type="button" onClick={() => setAllQuerySetVisible?.(false)} style={smallActionButtonStyle}>全不选</button>
      </div>
      <div style={{ marginTop: 10, display: 'grid', gap: 8, maxHeight: compact ? 180 : 220, overflowY: 'auto' }}>
        {querySets.length === 0 ? (
          <div style={{ fontSize: 11, color: '#6f84ad' }}>暂无输入名称。请先在左侧生成概念神经元。</div>
        ) : (
          querySets.map((set) => (
            <label key={`qf-${set.id}`} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#9eb4dd' }}>
              <input
                type="checkbox"
                checked={queryVisibility[set.id] !== false}
                onChange={(e) => setQuerySetVisible?.(set.id, e.target.checked)}
              />
              <span style={{ color: set.color }}>●</span>
              <span>{`${set.name} [${set.category}] (${set.nodes.length})`}</span>
            </label>
          ))
        )}
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: '#6f84ad' }}>
        {`当前词元: ${summary.currentToken || '-'} (${((summary.currentTokenProb || 0) * 100).toFixed(1)}%)`}
      </div>
    </div>
  );
}

export function AppleNeuronControlPanels({ workspace }) {
  const {
    analysisMode,
    setAnalysisMode,
    analysisModes,
    summary,
    queryInput,
    setQueryInput,
    queryCategoryInput,
    setQueryCategoryInput,
    handleGenerateQuery,
    querySets,
    queryFeedback,
    scanImportLimit,
    setScanImportLimit,
    scanImportTopK,
    setScanImportTopK,
    scanImportSummary,
    multidimProbeData,
    multidimCausalData,
    multidimTopN,
    setMultidimTopN,
    multidimVisible,
    setMultidimVisible,
    multidimActiveDimension,
    setMultidimActiveDimension,
    multidimLayerProfile,
    handleImportScanJsonText,
    removeQuerySet,
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
    displayStrategy,
    setDisplayStrategy,
    manualDisplayGroups,
    setManualDisplayGroups,
    modeMetrics,
  } = workspace;
  const [scanFileOptions, setScanFileOptions] = useState([]);
  const [selectedScanPath, setSelectedScanPath] = useState('');
  const [scanFileLoading, setScanFileLoading] = useState(false);
  const [scanFileImporting, setScanFileImporting] = useState(false);
  const [scanFileError, setScanFileError] = useState('');
  const [scanFileFilter, setScanFileFilter] = useState('multidim');
  const modeMetaById = Object.fromEntries(analysisModes.map((mode) => [mode.id, mode]));
  const scanFileFilterLabelMap = {
    multidim: '多维编码',
    mass_noun: '名词扫描',
    hard_problem: '硬伤实验',
    four_tasks: '四任务套件',
    unified_decode: '统一解码',
    all: '全部',
  };

  const filteredScanFileOptions = useMemo(() => {
    const rows = Array.isArray(scanFileOptions) ? scanFileOptions : [];
    if (scanFileFilter === 'all') {
      return rows;
    }
    if (scanFileFilter === 'multidim') {
      return rows.filter((f) => {
        const p = String(f?.path || '').toLowerCase();
        return p.includes('multidim_encoding_probe')
          || p.includes('multidim_causal_ablation')
          || p.includes('multidim_multiseed_stability');
      });
    }
    if (scanFileFilter === 'mass_noun') {
      return rows.filter((f) => {
        const p = String(f?.path || '').toLowerCase();
        return p.includes('mass_noun') || p.includes('noun_scan') || p.includes('encoding_scan');
      });
    }
    if (scanFileFilter === 'hard_problem') {
      return rows.filter((f) => {
        const p = String(f?.path || '').toLowerCase();
        return p.includes('dynamic_binding_stress_test')
          || p.includes('long_horizon_causal_trace_test')
          || p.includes('local_credit_assignment_proxy_test')
          || p.includes('triplet_targeted_causal_scan')
          || p.includes('triplet_targeted_multiseed_stability')
          || p.includes('variable_binding_hard_verification')
          || p.includes('minimal_causal_circuit_search')
          || p.includes('unified_coordinate_system_test')
          || p.includes('concept_family_parallel_scale')
          || p.includes('agi_research_stage_bundle_manifest');
      });
    }
    if (scanFileFilter === 'four_tasks') {
      return rows.filter((f) => {
        const p = String(f?.path || '').toLowerCase();
        return p.includes('agi_four_tasks_suite_manifest')
          || p.includes('variable_binding_hard_verification')
          || p.includes('minimal_causal_circuit_search')
          || p.includes('unified_coordinate_system_test')
          || p.includes('concept_family_parallel_scale');
      });
    }
    if (scanFileFilter === 'unified_decode') {
      return rows.filter((f) => {
        const p = String(f?.path || '').toLowerCase();
        return p.includes('unified_math_structure_decode');
      });
    }
    return rows;
  }, [scanFileFilter, scanFileOptions]);

  const refreshScanFileOptions = async () => {
    setScanFileLoading(true);
    setScanFileError('');
    try {
      const res = await fetch(`${MAIN_API_BASE}/api/main/scan_files?limit=200`);
      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload?.detail || '读取扫描文件列表失败');
      }
      const files = Array.isArray(payload?.files) ? payload.files : [];
      setScanFileOptions(files);
      setSelectedScanPath((prev) => {
        if (prev && files.some((f) => f.path === prev)) {
          return prev;
        }
        return files[0]?.path || '';
      });
      if (files.length === 0) {
        setScanFileError('未发现可导入文件：请先生成名词扫描、多维编码、硬伤实验、四任务套件或统一解码 JSON。');
      }
    } catch (err) {
      setScanFileOptions([]);
      setSelectedScanPath('');
      setScanFileError(`扫描文件列表加载失败: ${err?.message || err}`);
    } finally {
      setScanFileLoading(false);
    }
  };

  useEffect(() => {
    refreshScanFileOptions();
  }, []);

  useEffect(() => {
    setSelectedScanPath((prev) => {
      if (prev && filteredScanFileOptions.some((f) => f.path === prev)) {
        return prev;
      }
      return filteredScanFileOptions[0]?.path || '';
    });
  }, [filteredScanFileOptions]);

  const handleImportSelectedScanFile = async () => {
    if (!selectedScanPath) {
      setScanFileError('请先在下拉框选择一个扫描 JSON 文件。');
      return;
    }
    setScanFileImporting(true);
    setScanFileError('');
    try {
      const res = await fetch(`${MAIN_API_BASE}/api/main/scan_file?path=${encodeURIComponent(selectedScanPath)}`);
      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload?.detail || '读取扫描文件失败');
      }
      if (!payload?.data) {
        throw new Error('返回数据为空');
      }
      const data = payload.data;
      const sourcePath = payload.path || selectedScanPath;

      const looksLikeBundle = Boolean(data?.bundle_id === 'agi_research_stage_bundle_v1');
      if (looksLikeBundle) {
        handleImportScanJsonText(JSON.stringify(data), sourcePath);
        const artifactPaths = [
          data?.artifacts?.dynamic_binding_json,
          data?.artifacts?.long_horizon_json,
          data?.artifacts?.local_credit_json,
          data?.artifacts?.triplet_targeted_json,
          data?.artifacts?.triplet_multiseed_json,
          data?.artifacts?.unified_decode_json,
        ].filter(Boolean);
        for (const apath of artifactPaths) {
          try {
            const ar = await fetch(`${MAIN_API_BASE}/api/main/scan_file?path=${encodeURIComponent(apath)}`);
            const aj = await ar.json();
            if (ar.ok && aj?.data) {
              handleImportScanJsonText(JSON.stringify(aj.data), aj.path || apath);
            }
          } catch (_err) {
            // Ignore missing optional artifact files.
          }
        }
        return;
      }

      const looksLikeFourTasks = Boolean(data?.suite_id === 'agi_four_tasks_suite_v1');
      if (looksLikeFourTasks) {
        handleImportScanJsonText(JSON.stringify(data), sourcePath);
        const joinPath = (dirPath, fileName) => {
          const base = String(dirPath || '').replace(/[\\/]+$/, '');
          if (!base) {
            return '';
          }
          return `${base}/${fileName}`;
        };
        const taskFiles = [
          joinPath(data?.artifacts?.task1, 'variable_binding_hard_verification.json'),
          joinPath(data?.artifacts?.task2, 'minimal_causal_circuit_search.json'),
          joinPath(data?.artifacts?.task3, 'unified_coordinate_system_test.json'),
          joinPath(data?.artifacts?.task4, 'concept_family_parallel_scale.json'),
        ].filter(Boolean);
        for (const p of taskFiles) {
          try {
            const r = await fetch(`${MAIN_API_BASE}/api/main/scan_file?path=${encodeURIComponent(p)}`);
            const j = await r.json();
            if (r.ok && j?.data) {
              handleImportScanJsonText(JSON.stringify(j.data), j.path || p);
            }
          } catch (_err) {
            // Ignore missing optional task file.
          }
        }
        return;
      }

      // 若导入的是多seed稳定性汇总，则自动加载其中一个 probe + ablation，避免“导入后看不到变化”。
      const looksLikeStability = Boolean(data?.runs && data?.aggregate && data?.aggregate?.specificity_margin_style);
      if (looksLikeStability) {
        const runs = Array.isArray(data?.runs) ? data.runs : [];
        const preferredRun = runs.find((r) => r?.seed === 505) || runs[runs.length - 1] || runs[0];
        const probePath = preferredRun?.probe_json;
        const ablationPath = preferredRun?.ablation_json;

        handleImportScanJsonText(JSON.stringify(data), sourcePath);

        if (probePath) {
          try {
            const probeRes = await fetch(`${MAIN_API_BASE}/api/main/scan_file?path=${encodeURIComponent(probePath)}`);
            const probePayload = await probeRes.json();
            if (probeRes.ok && probePayload?.data) {
              handleImportScanJsonText(JSON.stringify(probePayload.data), probePayload.path || probePath);
            }
          } catch (_e) {
            // ignore, keep stability summary import result
          }
        }
        if (ablationPath) {
          try {
            const abRes = await fetch(`${MAIN_API_BASE}/api/main/scan_file?path=${encodeURIComponent(ablationPath)}`);
            const abPayload = await abRes.json();
            if (abRes.ok && abPayload?.data) {
              handleImportScanJsonText(JSON.stringify(abPayload.data), abPayload.path || ablationPath);
            }
          } catch (_e) {
            // ignore, keep existing imported states
          }
        }
      } else {
        handleImportScanJsonText(JSON.stringify(data), sourcePath);
      }
    } catch (err) {
      setScanFileError(`导入失败: ${err?.message || err}`);
    } finally {
      setScanFileImporting(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>分析类型（四阶段）</div>
        <div style={{ padding: 12, border: '1px solid rgba(255,255,255,0.05)', borderRadius: 8 }}>
          {ANALYSIS_MODE_STAGE_GROUPS.map((group) => {
            const GroupIcon = group.icon || Activity;
            const stageModes = group.items
              .map((id) => modeMetaById[id])
              .filter(Boolean);
            if (stageModes.length === 0) {
              return null;
            }
            return (
              <div key={group.id} style={{ marginBottom: 8 }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  marginBottom: 6,
                  padding: '4px 8px',
                  background: 'rgba(255,255,255,0.03)',
                  borderRadius: 4,
                }}>
                  <GroupIcon size={14} color="#888" />
                  <span style={{ fontSize: 11, fontWeight: 600, color: '#888' }}>{group.label}</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 4 }}>
                  {stageModes.map((mode) => {
                    const active = analysisMode === mode.id;
                    const ModeIcon = ANALYSIS_MODE_ICONS[mode.id] || Activity;
                    return (
                      <button
                        key={mode.id}
                        type="button"
                        onClick={() => setAnalysisMode(mode.id)}
                        title={mode.desc}
                        style={{
                          padding: '8px 4px',
                          backgroundColor: active ? 'rgba(68, 136, 255, 0.2)' : 'transparent',
                          color: active ? '#4488ff' : '#666',
                          border: active ? '1px solid rgba(68, 136, 255, 0.4)' : '1px solid transparent',
                          borderRadius: 6,
                          cursor: 'pointer',
                          fontSize: 11,
                          fontWeight: 500,
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          gap: 4,
                        }}
                      >
                        <ModeIcon size={14} />
                        <span>{mode.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
        <div style={{ fontSize: 11, color: '#7f95bb', marginTop: 8, lineHeight: 1.6 }}>
          {modeMetaById[analysisMode]?.desc || ''}
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
            placeholder="输入上下文，例如：概念 是 一种 结构"
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
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>显示策略</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
          {[
            { id: 'auto', label: '自动聚焦', desc: '随分析类型切换重点' },
            { id: 'all', label: '全部显示', desc: '不过滤任何节点' },
            { id: 'manual', label: '手动筛选', desc: '按类别开关显示' },
          ].map((opt) => (
            <button
              key={`display-${opt.id}`}
              type="button"
              onClick={() => setDisplayStrategy(opt.id)}
              title={opt.desc}
              style={{
                borderRadius: 8,
                border: `1px solid ${displayStrategy === opt.id ? 'rgba(126, 224, 255, 0.75)' : 'rgba(122, 162, 255, 0.35)'}`,
                background: displayStrategy === opt.id ? 'rgba(24, 101, 134, 0.38)' : 'rgba(7, 12, 25, 0.82)',
                color: '#dbe9ff',
                fontSize: 11,
                padding: '7px 8px',
                cursor: 'pointer',
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
        {displayStrategy === 'manual' ? (
          <div style={{ marginTop: 8, display: 'grid', gap: 6 }}>
            {[
              { id: 'core', label: '核心/基础节点' },
              { id: 'query', label: '输入概念节点' },
              { id: 'multidim', label: '多维编码节点' },
              { id: 'hard', label: '硬伤实验节点' },
              { id: 'unified', label: '统一解码节点' },
              { id: 'background', label: '背景网络节点' },
            ].map((item) => (
              <label key={`manual-group-${item.id}`} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#9eb4dd' }}>
                <input
                  type="checkbox"
                  checked={manualDisplayGroups[item.id] !== false}
                  onChange={(e) => setManualDisplayGroups((prev) => ({ ...prev, [item.id]: e.target.checked }))}
                />
                <span>{item.label}</span>
              </label>
            ))}
          </div>
        ) : null}
        <div style={{ marginTop: 8, fontSize: 11, color: '#7ea2c9' }}>
          {displayStrategy === 'auto'
            ? '自动模式：因果类分析突出硬伤实验，结构类分析突出统一编码与多维节点。'
            : displayStrategy === 'all'
              ? '全部模式：所有节点同等显示，不做降噪。'
              : '手动模式：按勾选结果控制各类节点显示。'}
        </div>
      </div>

      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>Quick Concept Generator</div>
        <div style={{ display: 'grid', gap: 8 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>名称</div>
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
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>类别</div>
            <input
              value={queryCategoryInput}
              onChange={(e) => setQueryCategoryInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleGenerateQuery();
                }
              }}
              placeholder="输入类别，例如：动物 / 天体 / 抽象概念"
              style={inputStyle}
            />
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button type="button" onClick={handleGenerateQuery} style={smallActionButtonStyle}>
              Generate
            </button>
          </div>
        </div>
        <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid rgba(122, 162, 255, 0.2)', display: 'grid', gap: 8 }}>
          <div style={{ fontSize: 12, color: '#9eb4dd' }}>批量导入扫描结果</div>
          <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>导入数</div>
            <input
              type="number"
              min={1}
              max={120}
              value={scanImportLimit}
              onChange={(e) => setScanImportLimit(Number(e.target.value))}
              style={inputStyle}
            />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>每概念</div>
            <input
              type="number"
              min={4}
              max={64}
              value={scanImportTopK}
              onChange={(e) => setScanImportTopK(Number(e.target.value))}
              style={inputStyle}
            />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: `56px ${fixedFileControlWidth}px`, gap: 8, alignItems: 'center' }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>文件</div>
            <select
              value={selectedScanPath}
              onChange={(e) => setSelectedScanPath(e.target.value)}
              style={{
                ...inputStyle,
                width: fixedFileControlWidth,
                minWidth: fixedFileControlWidth,
                maxWidth: fixedFileControlWidth,
                boxSizing: 'border-box',
                display: 'block',
                whiteSpace: 'nowrap',
                textOverflow: 'ellipsis',
                overflow: 'hidden',
              }}
              disabled={scanFileLoading || filteredScanFileOptions.length === 0}
            >
              {filteredScanFileOptions.length === 0 ? (
                <option value="">{scanFileLoading ? '扫描中...' : '未发现可导入文件'}</option>
              ) : (
                filteredScanFileOptions.map((f) => (
                  <option key={f.path} value={f.path}>
                    {formatScanOptionLabel(f)}
                  </option>
                ))
              )}
            </select>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: `56px ${fixedFileControlWidth}px`, gap: 8, alignItems: 'center' }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>筛选</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
              {[
                { id: 'multidim', label: '多维编码' },
                { id: 'mass_noun', label: '名词扫描' },
                { id: 'hard_problem', label: '硬伤实验' },
                { id: 'four_tasks', label: '四任务' },
                { id: 'unified_decode', label: '统一解码' },
                { id: 'all', label: '全部' },
              ].map((opt) => (
                <button
                  key={`scan-filter-${opt.id}`}
                  type="button"
                  onClick={() => setScanFileFilter(opt.id)}
                  style={{
                    borderRadius: 8,
                    border: `1px solid ${scanFileFilter === opt.id ? 'rgba(126, 224, 255, 0.75)' : 'rgba(122, 162, 255, 0.35)'}`,
                    background: scanFileFilter === opt.id ? 'rgba(24, 101, 134, 0.38)' : 'rgba(7, 12, 25, 0.82)',
                    color: '#dbe9ff',
                    fontSize: 11,
                    padding: '6px 8px',
                    cursor: 'pointer',
                  }}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
          <div style={{ fontSize: 11, color: '#7ea2c9' }}>
            {`候选文件: ${filteredScanFileOptions.length}/${scanFileOptions.length}（当前筛选: ${scanFileFilterLabelMap[scanFileFilter] || scanFileFilter}）`}
          </div>
          <div
            style={{
              display: 'flex',
              width: fixedFileControlWidth + 56 + 8,
              maxWidth: fixedFileControlWidth + 56 + 8,
              minWidth: fixedFileControlWidth + 56 + 8,
              gap: 8,
              flexWrap: 'wrap',
            }}
          >
            <button
              type="button"
              onClick={refreshScanFileOptions}
              style={{ ...smallActionButtonStyle, flex: '1 1 120px', minWidth: 120 }}
              disabled={scanFileLoading}
            >
              {scanFileLoading ? '刷新中...' : '刷新列表'}
            </button>
            <button
              type="button"
              onClick={handleImportSelectedScanFile}
              style={{ ...smallActionButtonStyle, flex: '1 1 120px', minWidth: 120 }}
              disabled={scanFileImporting || !selectedScanPath}
            >
              {scanFileImporting ? '导入中...' : '导入选中文件'}
            </button>
          </div>
          {selectedScanPath ? (
            <div
              style={{
                width: fixedFileControlWidth + 56 + 8,
                maxWidth: fixedFileControlWidth + 56 + 8,
                minWidth: fixedFileControlWidth + 56 + 8,
                fontSize: 11,
                color: '#7ea2c9',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
              title={selectedScanPath}
            >
              {(() => {
                const meta = scanFileOptions.find((f) => f.path === selectedScanPath);
                if (!meta) {
                  const fallbackName = String(selectedScanPath).replace(/\\/g, '/').split('/').pop();
                  return `已选: ${fallbackName || 'scan.json'}`;
                }
                const kb = (Number(meta.size_bytes || 0) / 1024).toFixed(1);
                return `已选: ${meta.name} | ${kb} KB | ${meta.mtime_iso || ''}`;
              })()}
            </div>
          ) : null}
          {scanFileError ? (
            <div style={{ fontSize: 11, color: '#ff9fb0' }}>{scanFileError}</div>
          ) : null}
          {scanImportSummary ? (
            <div style={{ fontSize: 11, color: '#7eb8ff', lineHeight: 1.6 }}>
              {`来源: ${scanImportSummary.source} | 导入概念集: ${scanImportSummary.importedConcepts} | 类别: ${scanImportSummary.importedCategories} | 扫描名词总数: ${scanImportSummary.totalNouns} | 最小回路名词: ${scanImportSummary.minimalCircuitNouns || 0} | 反事实对: ${scanImportSummary.counterfactualPairs || 0}`}
            </div>
          ) : null}

          <div style={{ marginTop: 6, paddingTop: 8, borderTop: '1px solid rgba(122, 162, 255, 0.2)', display: 'grid', gap: 8 }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>三维编码（Style / Logic / Syntax）</div>
            <div style={{ fontSize: 11, color: '#7ea2c9' }}>
              {multidimProbeData
                ? `已导入探针，当前显示维度: ${DIMENSION_LABELS[multidimActiveDimension] || multidimActiveDimension}，层谱点数: ${multidimLayerProfile?.length || 0}`
                : '未导入三维探针 JSON（multidim_encoding_probe.json）'}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
              <div style={{ fontSize: 12, color: '#9eb4dd' }}>TopN</div>
              <input
                type="number"
                min={16}
                max={256}
                value={multidimTopN}
                onChange={(e) => setMultidimTopN(Number(e.target.value))}
                style={inputStyle}
              />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
              {['style', 'logic', 'syntax'].map((dim) => (
                <button
                  key={`dim-active-${dim}`}
                  type="button"
                  onClick={() => setMultidimActiveDimension(dim)}
                  style={{
                    borderRadius: 8,
                    border: `1px solid ${multidimActiveDimension === dim ? ROLE_COLORS[dim] : 'rgba(122,162,255,0.35)'}`,
                    background: multidimActiveDimension === dim ? 'rgba(42,71,132,0.82)' : 'rgba(7, 12, 25, 0.82)',
                    color: '#dbe9ff',
                    fontSize: 11,
                    padding: '6px 8px',
                    cursor: 'pointer',
                  }}
                >
                  {DIMENSION_LABELS[dim]}
                </button>
              ))}
            </div>
            <div style={{ display: 'grid', gap: 6 }}>
              {['style', 'logic', 'syntax'].map((dim) => (
                <label key={`dim-vis-${dim}`} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#9eb4dd' }}>
                  <input
                    type="checkbox"
                    checked={multidimVisible[dim] !== false}
                    onChange={(e) => setMultidimVisible((prev) => ({ ...prev, [dim]: e.target.checked }))}
                  />
                  <span style={{ color: ROLE_COLORS[dim] }}>●</span>
                  <span>{`${DIMENSION_LABELS[dim]}神经元`}</span>
                </label>
              ))}
            </div>
            <div style={{ fontSize: 11, color: '#7ea2c9', lineHeight: 1.6 }}>
              {multidimCausalData
                ? `对角优势 style=${toSafeNumber(multidimCausalData?.diagonal_advantage?.style, 0).toFixed(4)} / logic=${toSafeNumber(multidimCausalData?.diagonal_advantage?.logic, 0).toFixed(4)} / syntax=${toSafeNumber(multidimCausalData?.diagonal_advantage?.syntax, 0).toFixed(4)}`
                : '未导入三维因果消融 JSON（multidim_causal_ablation.json）'}
            </div>
          </div>
        </div>
        {queryFeedback ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#8fd4ff' }}>{queryFeedback}</div>
        ) : null}
        <div style={{ marginTop: 10, display: 'grid', gap: 6 }}>
          {querySets.length === 0 ? (
            <div style={{ fontSize: 11, color: '#6f84ad' }}>尚未生成概念神经元。</div>
          ) : (
            querySets.map((set) => (
              <div key={set.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 12, color: '#9eb4dd' }}>
                <span><span style={{ color: set.color }}>●</span>{` ${set.name} [${set.category}] (${set.nodes.length})`}</span>
                <button type="button" onClick={() => removeQuerySet(set.id)} style={{ ...smallActionButtonStyle, padding: '2px 8px', fontSize: 11 }}>Remove</button>
              </div>
            ))
          )}
        </div>
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
