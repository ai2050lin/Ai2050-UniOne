/**
 * 闈㈡澘閰嶇疆鏂囦欢
 * 缁熶竴绠＄悊鎵€鏈夐潰鏉跨殑浣嶇疆銆佹牱寮忋€佹爣绛鹃〉鍒嗙粍鍜屾暟鎹ā鏉?
 */

// 闈㈡澘浣嶇疆閰嶇疆
export const PANEL_POSITIONS = {
  inputPanel: {
    position: 'absolute',
    top: 60,
    left: 20,
    zIndex: 10,
    width: '360px',
    maxHeight: '85vh',
  },
  infoPanel: {
    position: 'absolute',
    top: 20,
    right: 20,
    zIndex: 100,
    minWidth: '320px',
    maxWidth: '400px',
    maxHeight: '80vh',
  },
  operationPanel: {
    position: 'absolute',
    bottom: 20,
    right: 20,
    zIndex: 10,
    minWidth: '320px',
    maxWidth: '400px',
    maxHeight: '60vh',
  },
  detailPanel: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    zIndex: 10,
    width: '380px',
    maxHeight: '50vh',
  },
};

// 闈㈡澘鍩虹鏍峰紡
export const PANEL_BASE_STYLE = {
  background: 'rgba(20, 20, 25, 0.95)',
  padding: '16px',
  borderRadius: '12px',
  backdropFilter: 'blur(10px)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
};

// 缁撴瀯鍒嗘瀽鏍囩椤靛垎缁勯厤缃紙浜岀骇鑿滃崟锛?
export const STRUCTURE_TABS_V2 = {
  groups: [
    {
      id: 'observation',
      label: '瑙傛祴',
      icon: 'Eye',
      color: '#00d2ff',
      description: '灞傞棿棰勬祴婕斿寲涓庢縺娲诲彲瑙嗗寲',
      items: [
        { id: 'logit_lens', label: '棰勬祴婕斿寲 (Logit)', desc: '灞傞棿棰勬祴婕斿寲', icon: 'BarChart2' },
        { id: 'glass_matrix', label: '鐜荤拑鐭╅樀 (Glass)', desc: '婵€娲荤煩闃靛彲瑙嗗寲', icon: 'Grid3x3' },
        { id: 'flow_tubes', label: '淇℃伅娴?(Flow)', desc: '淇℃伅娴佸姩杞ㄨ抗', icon: 'GitBranch' },
      ]
    },
    {
      id: 'analysis',
      label: '鍒嗘瀽',
      icon: 'Zap',
      color: '#ff9f43',
      description: '鍥犳灉鍥炶矾涓庣壒寰佹彁鍙?',
      items: [
        { id: 'circuit', label: '鍥炶矾 (Circuit)', desc: '鍥犳灉鍥炶矾鍙戠幇', icon: 'Share2' },
        { id: 'features', label: '鐗瑰緛 (Features)', desc: 'SAE绋€鐤忕壒寰?', icon: 'Sparkles' },
        { id: 'causal', label: '鍥犳灉 (Causal)', desc: '鍥犳灉涓粙鍒嗘瀽', icon: 'Target' },
        { id: 'manifold', label: '娴佸舰 (Manifold)', desc: '娴佸舰鍑犱綍鍒嗘瀽', icon: 'Globe2' },
        { id: 'compositional', label: '缁勫悎 (Compos)', desc: '缁勫悎娉涘寲', icon: 'Layers' },
      ]
    },
    {
      id: 'geometry',
      label: '鍑犱綍',
      icon: 'Hexagon',
      color: '#6c5ce7',
      description: '绾ょ淮涓涗笌鎷撴墤鍒嗘瀽',
      items: [
        { id: 'fibernet_v2', label: '绾ょ淮涓?(Fiber)', desc: '绾ょ淮涓涙嫇鎵?', icon: 'Network' },
        { id: 'rpt', label: '浼犺緭 (RPT)', desc: '榛庢浖骞宠浼犺緭', icon: 'ArrowRightLeft' },
        { id: 'curvature', label: '鏇茬巼 (Curv)', desc: '鏇茬巼鍦哄垎鏋?', icon: 'TrendingUp' },
        { id: 'tda', label: '鎷撴墤 (TDA)', desc: '鎷撴墤鏁版嵁鍒嗘瀽', icon: 'BarChart' },
        { id: 'global_topology', label: '鍏ㄥ眬鎷撴墤 (Topo)', desc: '鍏ㄥ眬鎷撴墤', icon: 'Globe' },
        { id: 'holonomy', label: '鍏ㄧ函 (Holo)', desc: '鍏ㄧ函鎵弿', icon: 'RefreshCw' },
      ]
    },
    {
      id: 'advanced',
      label: '楂樼骇',
      icon: 'FlaskConical',
      color: '#ff6b9d',
      description: 'AGI涓庡幓鍋忓垎鏋?',
      items: [
        { id: 'agi', label: '绁炵粡涓?(AGI)', desc: '绁炵粡绾ょ淮涓汚GI', icon: 'Brain' },
        { id: 'debias', label: '鍘诲亸 (Debias)', desc: '鍑犱綍鍘诲亸', icon: 'Scale' },
        { id: 'validity', label: '鏈夋晥鎬?(Valid)', desc: '鏈夋晥鎬ф楠?', icon: 'CheckCircle' },
        { id: 'training', label: '璁粌 (Training)', desc: '璁粌鍔ㄥ姏瀛?', icon: 'Activity' },
      ]
    }
  ]
};

// 鎵佸钩鍖栫殑鏍囩椤靛垪琛紙鍚戝悗鍏煎锛?
export const STRUCTURE_TABS = {
  observation: STRUCTURE_TABS_V2.groups.find(g => g.id === 'observation').items,
  analysis: STRUCTURE_TABS_V2.groups.find(g => g.id === 'analysis').items,
  geometry: STRUCTURE_TABS_V2.groups.find(g => g.id === 'geometry').items,
  advanced: STRUCTURE_TABS_V2.groups.find(g => g.id === 'advanced').items,
};

// 杈撳叆闈㈡澘鏍囩椤甸厤缃?
export const INPUT_PANEL_TABS = [
  { id: 'main', label: 'Main', color: '#38bdf8', description: 'Apple Neuron 3D Main' },
  { id: 'dnn', label: 'DNN', color: '#3a7bd5', description: '娣卞害绁炵粡缃戠粶' },
  { id: 'snn', label: 'SNN', color: '#4ecdc4', description: '鑴夊啿绁炵粡缃戠粶' },
  { id: 'fibernet', label: 'FiberNet', color: '#6c5ce7', description: '绾ょ淮涓涘疄楠屽' },
];

// 鏁版嵁灞曠ず妯℃澘閰嶇疆
export const ANALYSIS_DATA_TEMPLATES = {
  logit_lens: {
    title: 'Logit Lens 鍒嗘瀽',
    color: '#00d2ff',
    metrics: [
      { key: 'avg_confidence', label: '骞冲潎缃俊搴?', format: 'percent' },
      { key: 'entropy', label: '鐔靛€?', format: 'decimal' },
    ],
    sections: [
      { type: 'layer_list', title: '灞傞棿棰勬祴', source: 'logit_lens' },
    ]
  },
  circuit: {
    title: '鍥犳灉鍥炶矾鍙戠幇',
    color: '#ff6b6b',
    metrics: [
      { key: 'nodes', label: '鑺傜偣鏁?', format: 'number' },
      { key: 'edges', label: '杈规暟', format: 'number' },
      { key: 'density', label: '瀵嗗害', format: 'percent' },
    ],
    sections: [
      { type: 'graph_summary', title: '鍥剧粨鏋?', source: 'graph' },
    ]
  },
  features: {
    title: 'SAE 鐗瑰緛鎻愬彇',
    color: '#ffd93d',
    metrics: [
      { key: 'n_features', label: '鐗瑰緛鏁?', format: 'number' },
      { key: 'sparsity', label: '绋€鐤忓害', format: 'decimal' },
      { key: 'reconstruction_error', label: '閲嶆瀯璇樊', format: 'decimal' },
    ],
    sections: [
      { type: 'feature_table', title: 'Top鐗瑰緛', source: 'top_features' },
    ]
  },
  causal: {
    title: '鍥犳灉涓粙鍒嗘瀽',
    color: '#6c5ce7',
    metrics: [
      { key: 'n_components_analyzed', label: '鍒嗘瀽缁勪欢', format: 'number' },
      { key: 'n_important_components', label: '鍏抽敭缁勪欢', format: 'number' },
    ],
    sections: []
  },
  manifold: {
    title: '娴佸舰鍑犱綍鍒嗘瀽',
    color: '#4ecdc4',
    metrics: [
      { key: 'intrinsic_dim', label: '鍐呭湪缁村害', format: 'number' },
      { key: 'curvature', label: '鏇茬巼', format: 'decimal' },
    ],
    sections: []
  },
  fibernet_v2: {
    title: 'FiberNet V2',
    color: '#6c5ce7',
    metrics: [
      { key: 'base_dim', label: '搴曟祦褰㈢淮搴?', format: 'number' },
      { key: 'fiber_dim', label: '绾ょ淮缁村害', format: 'number' },
    ],
    sections: []
  },
  tda: {
    title: '鎷撴墤鏁版嵁鍒嗘瀽',
    color: '#e056fd',
    metrics: [
      { key: 'betti_0', label: '尾鈧€ 杩為€氬垎閲?', format: 'number' },
      { key: 'betti_1', label: '尾鈧?鐜?', format: 'number' },
      { key: 'betti_2', label: '尾鈧?绌鸿厰', format: 'number' },
    ],
    sections: []
  },
  rpt: {
    title: '榛庢浖骞宠浼犺緭',
    color: '#00d2ff',
    metrics: [
      { key: 'transport_distance', label: '浼犺緭璺濈', format: 'decimal' },
      { key: 'alignment', label: '瀵归綈搴?', format: 'percent' },
    ],
    sections: []
  },
  curvature: {
    title: '鏇茬巼鍦哄垎鏋?',
    color: '#ff9f43',
    metrics: [
      { key: 'scalar_curvature', label: '鏍囬噺鏇茬巼', format: 'decimal' },
      { key: 'ricci_curvature', label: 'Ricci鏇茬巼', format: 'decimal' },
    ],
    sections: []
  },
};

// 棰滆壊涓婚
export const COLORS = {
  primary: '#00d2ff',
  secondary: '#3a7bd5',
  accent: '#4ecdc4',
  warning: '#ff9f43',
  danger: '#ff4444',
  success: '#5ec962',
  purple: '#6c5ce7',
  pink: '#ff6b9d',
  bgDark: 'rgba(20, 20, 25, 0.95)',
  bgLight: 'rgba(255, 255, 255, 0.03)',
  bgBorder: 'rgba(255, 255, 255, 0.1)',
  textPrimary: '#ffffff',
  textSecondary: '#aaaaaa',
  textMuted: '#666666',
};

// 鎿嶄綔鍘嗗彶閰嶇疆
export const HISTORY_CONFIG = {
  maxItems: 50,
  storageKey: 'transformerlens_history',
};

// 榛樿闈㈡澘鍙鎬?
export const DEFAULT_PANEL_VISIBILITY = {
  inputPanel: true,
  infoPanel: true,
  operationPanel: true,
  detailPanel: false,
};


