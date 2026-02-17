/**
 * 面板配置文件
 * 统一管理所有面板的位置、样式、标签页分组和数据模板
 */

// 面板位置配置
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

// 面板基础样式
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

// 结构分析标签页分组配置（二级菜单）
export const STRUCTURE_TABS_V2 = {
  groups: [
    {
      id: 'observation',
      label: '观测',
      icon: 'Eye',
      color: '#00d2ff',
      description: '层间预测演化与激活可视化',
      items: [
        { id: 'logit_lens', label: '预测演化 (Logit)', desc: '层间预测演化', icon: 'BarChart2' },
        { id: 'glass_matrix', label: '玻璃矩阵 (Glass)', desc: '激活矩阵可视化', icon: 'Grid3x3' },
        { id: 'flow_tubes', label: '信息流 (Flow)', desc: '信息流动轨迹', icon: 'GitBranch' },
      ]
    },
    {
      id: 'analysis',
      label: '分析',
      icon: 'Zap',
      color: '#ff9f43',
      description: '因果回路与特征提取',
      items: [
        { id: 'circuit', label: '回路 (Circuit)', desc: '因果回路发现', icon: 'Share2' },
        { id: 'features', label: '特征 (Features)', desc: 'SAE稀疏特征', icon: 'Sparkles' },
        { id: 'causal', label: '因果 (Causal)', desc: '因果中介分析', icon: 'Target' },
        { id: 'manifold', label: '流形 (Manifold)', desc: '流形几何分析', icon: 'Globe2' },
        { id: 'compositional', label: '组合 (Compos)', desc: '组合泛化', icon: 'Layers' },
      ]
    },
    {
      id: 'geometry',
      label: '几何',
      icon: 'Hexagon',
      color: '#6c5ce7',
      description: '纤维丛与拓扑分析',
      items: [
        { id: 'fibernet_v2', label: '纤维丛 (Fiber)', desc: '纤维丛拓扑', icon: 'Network' },
        { id: 'rpt', label: '传输 (RPT)', desc: '黎曼平行传输', icon: 'ArrowRightLeft' },
        { id: 'curvature', label: '曲率 (Curv)', desc: '曲率场分析', icon: 'TrendingUp' },
        { id: 'tda', label: '拓扑 (TDA)', desc: '拓扑数据分析', icon: 'BarChart' },
        { id: 'global_topology', label: '全局拓扑 (Topo)', desc: '全局拓扑', icon: 'Globe' },
        { id: 'holonomy', label: '全纯 (Holo)', desc: '全纯扫描', icon: 'RefreshCw' },
      ]
    },
    {
      id: 'advanced',
      label: '高级',
      icon: 'FlaskConical',
      color: '#ff6b9d',
      description: 'AGI与去偏分析',
      items: [
        { id: 'agi', label: '神经丛 (AGI)', desc: '神经纤维丛AGI', icon: 'Brain' },
        { id: 'debias', label: '去偏 (Debias)', desc: '几何去偏', icon: 'Scale' },
        { id: 'validity', label: '有效性 (Valid)', desc: '有效性检验', icon: 'CheckCircle' },
        { id: 'training', label: '训练 (Training)', desc: '训练动力学', icon: 'Activity' },
      ]
    }
  ]
};

// 扁平化的标签页列表（向后兼容）
export const STRUCTURE_TABS = {
  observation: STRUCTURE_TABS_V2.groups.find(g => g.id === 'observation').items,
  analysis: STRUCTURE_TABS_V2.groups.find(g => g.id === 'analysis').items,
  geometry: STRUCTURE_TABS_V2.groups.find(g => g.id === 'geometry').items,
  advanced: STRUCTURE_TABS_V2.groups.find(g => g.id === 'advanced').items,
};

// 输入面板标签页配置
export const INPUT_PANEL_TABS = [
  { id: 'dnn', label: 'DNN', color: '#3a7bd5', description: '深度神经网络' },
  { id: 'snn', label: 'SNN', color: '#4ecdc4', description: '脉冲神经网络' },
  { id: 'fibernet', label: 'FiberNet', color: '#6c5ce7', description: '纤维丛实验室' },
];

// 数据展示模板配置
export const ANALYSIS_DATA_TEMPLATES = {
  logit_lens: {
    title: 'Logit Lens 分析',
    color: '#00d2ff',
    metrics: [
      { key: 'avg_confidence', label: '平均置信度', format: 'percent' },
      { key: 'entropy', label: '熵值', format: 'decimal' },
    ],
    sections: [
      { type: 'layer_list', title: '层间预测', source: 'logit_lens' },
    ]
  },
  circuit: {
    title: '因果回路发现',
    color: '#ff6b6b',
    metrics: [
      { key: 'nodes', label: '节点数', format: 'number' },
      { key: 'edges', label: '边数', format: 'number' },
      { key: 'density', label: '密度', format: 'percent' },
    ],
    sections: [
      { type: 'graph_summary', title: '图结构', source: 'graph' },
    ]
  },
  features: {
    title: 'SAE 特征提取',
    color: '#ffd93d',
    metrics: [
      { key: 'n_features', label: '特征数', format: 'number' },
      { key: 'sparsity', label: '稀疏度', format: 'decimal' },
      { key: 'reconstruction_error', label: '重构误差', format: 'decimal' },
    ],
    sections: [
      { type: 'feature_table', title: 'Top特征', source: 'top_features' },
    ]
  },
  causal: {
    title: '因果中介分析',
    color: '#6c5ce7',
    metrics: [
      { key: 'n_components_analyzed', label: '分析组件', format: 'number' },
      { key: 'n_important_components', label: '关键组件', format: 'number' },
    ],
    sections: []
  },
  manifold: {
    title: '流形几何分析',
    color: '#4ecdc4',
    metrics: [
      { key: 'intrinsic_dim', label: '内在维度', format: 'number' },
      { key: 'curvature', label: '曲率', format: 'decimal' },
    ],
    sections: []
  },
  fibernet_v2: {
    title: 'FiberNet V2',
    color: '#6c5ce7',
    metrics: [
      { key: 'base_dim', label: '底流形维度', format: 'number' },
      { key: 'fiber_dim', label: '纤维维度', format: 'number' },
    ],
    sections: []
  },
  tda: {
    title: '拓扑数据分析',
    color: '#e056fd',
    metrics: [
      { key: 'betti_0', label: 'β₀ 连通分量', format: 'number' },
      { key: 'betti_1', label: 'β₁ 环', format: 'number' },
      { key: 'betti_2', label: 'β₂ 空腔', format: 'number' },
    ],
    sections: []
  },
  rpt: {
    title: '黎曼平行传输',
    color: '#00d2ff',
    metrics: [
      { key: 'transport_distance', label: '传输距离', format: 'decimal' },
      { key: 'alignment', label: '对齐度', format: 'percent' },
    ],
    sections: []
  },
  curvature: {
    title: '曲率场分析',
    color: '#ff9f43',
    metrics: [
      { key: 'scalar_curvature', label: '标量曲率', format: 'decimal' },
      { key: 'ricci_curvature', label: 'Ricci曲率', format: 'decimal' },
    ],
    sections: []
  },
};

// 颜色主题
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

// 操作历史配置
export const HISTORY_CONFIG = {
  maxItems: 50,
  storageKey: 'transformerlens_history',
};

// 默认面板可见性
export const DEFAULT_PANEL_VISIBILITY = {
  inputPanel: true,
  infoPanel: true,
  operationPanel: true,
  detailPanel: false,
};
