/**
 * PuzzlePanel — 拼图面板
 * 显示完整的98格子拼图框架，以及每个模块的进展
 * 
 * 数据来源:
 *   1. 内嵌默认数据 (与PUZZLE_FRAMEWORK.md同步)
 *   2. 外部JSON加载 (type=puzzle_progress)
 */
import React, { useState, useMemo } from 'react';

// ==================== 默认拼图数据 (与PUZZLE_FRAMEWORK.md同步) ====================

const PUZZLE_CATEGORIES = [
  {
    id: 'KN', label: '知识网络', icon: '🧠', color: '#4ecdc4',
    description: '多层次系统化知识网络',
    subcategories: [
      {
        id: 'KN-1', label: '概念编码',
        cells: [
          { id: 'KN-1a', question: '概念在残差流中编码在哪些维度?', priority: 'P0', status: '◐', detail: 'eff_dim=17-22; PCA5探针0.98-1.00; 类别信号真实' },
          { id: 'KN-1b', question: '不同概念是否共享编码维度?', priority: 'P1', status: '◐', detail: '同类cos_gap=0.25-0.46(放大4.6-5.8x)' },
          { id: 'KN-1c', question: '概念编码的层级结构?', priority: 'P1', status: '◐', detail: 'separability 0.22→0.39; 属性信号随norm增强' },
          { id: 'KN-1d', question: '概念编码是稀疏还是稠密?', priority: 'P2', status: '◐', detail: '需关闭>100维才影响极性→超分散' },
        ]
      },
      {
        id: 'KN-2', label: '属性编码',
        cells: [
          { id: 'KN-2a', question: '属性是否独立于概念编码?', priority: 'P1', status: '◐', detail: '跨模板spec_cos=-0.15~-0.20; "not"正交扰动' },
          { id: 'KN-2b', question: '属性如何附加到概念上?', priority: 'P2', status: '◐', detail: '亚加性压缩; 交互高秩' },
          { id: 'KN-2c', question: '不同属性是否正交?', priority: 'P2', status: '□', detail: '' },
          { id: 'KN-2d', question: '属性信息的因果载体?', priority: 'P2', status: '□', detail: '' },
        ]
      },
      {
        id: 'KN-3', label: '抽象层级',
        cells: [
          { id: 'KN-3a', question: '抽象链如何在参数中表示?', priority: 'P0', status: '◐', detail: 'normed_dir_cos=-0.20(非嵌套); 层级间距离uniform' },
          { id: 'KN-3b', question: '抽象层级是嵌套还是正交?', priority: 'P1', status: '◐', detail: '非嵌套非正交, 方向cos≈-0.20' },
          { id: 'KN-3c', question: '泛化/特化是否有方向操作?', priority: 'P2', status: '□', detail: '' },
          { id: 'KN-3d', question: '交叉概念如何编码?', priority: 'P2', status: '□', detail: '' },
        ]
      },
      {
        id: 'KN-4', label: '知识网络拓扑',
        cells: [
          { id: 'KN-4a', question: '概念间关系是否有统一编码?', priority: 'P1', status: '□', detail: '' },
          { id: 'KN-4b', question: '知识图结构在参数中如何映射?', priority: 'P2', status: '□', detail: '' },
          { id: 'KN-4c', question: '知识检索如何实现?', priority: 'P2', status: '□', detail: '' },
          { id: 'KN-4d', question: '知识修改如何在参数中体现?', priority: 'P2', status: '□', detail: '' },
        ]
      },
    ]
  },
  {
    id: 'LG', label: '逻辑推理', icon: '⚡', color: '#a855f7',
    description: '逻辑体系与推理路径',
    subcategories: [
      {
        id: 'LG-1', label: '条件推理',
        cells: [
          { id: 'LG-1a', question: '条件推理的因果路径?', priority: 'P0', status: '□', detail: '' },
          { id: 'LG-1b', question: '推理信号在哪些层出现?', priority: 'P1', status: '◐', detail: '逻辑信号L0最强(0.70)逐步稀释' },
          { id: 'LG-1c', question: '逻辑信号稀释后去了哪里?', priority: 'P0', status: '□', detail: '' },
          { id: 'LG-1d', question: '逆向推理路径是否不同?', priority: 'P2', status: '□', detail: '' },
        ]
      },
      {
        id: 'LG-2', label: '深度思考',
        cells: [
          { id: 'LG-2a', question: '多步推理如何实现?', priority: 'P1', status: '□', detail: '' },
          { id: 'LG-2b', question: '推理链是串行还是并行?', priority: 'P2', status: '□', detail: '' },
          { id: 'LG-2c', question: '推理前沿是否对应参数空间结构?', priority: 'P2', status: '□', detail: '' },
          { id: 'LG-2d', question: '不同推理类型是否共享机制?', priority: 'P2', status: '□', detail: '' },
        ]
      },
      {
        id: 'LG-3', label: '翻译/转换',
        cells: [
          { id: 'LG-3a', question: '翻译的因果路径?', priority: 'P2', status: '□', detail: '' },
          { id: 'LG-3b', question: '不同语言共享什么编码?', priority: 'P1', status: '◐', detail: '否定跨语言一致; 时态不一致' },
          { id: 'LG-3c', question: '同义不同表达如何体现?', priority: 'P2', status: '□', detail: '' },
          { id: 'LG-3d', question: '计算能力的因果路径?', priority: 'P2', status: '□', detail: '' },
        ]
      },
      {
        id: 'LG-4', label: '逻辑×知识',
        cells: [
          { id: 'LG-4a', question: '逻辑如何调用知识网络?', priority: 'P0', status: '□', detail: '' },
          { id: 'LG-4b', question: '逻辑推理是否依赖特定知识路径?', priority: 'P2', status: '□', detail: '' },
          { id: 'LG-4c', question: '逻辑+知识的交互是否高秩?', priority: 'P1', status: '◐', detail: '交互高秩; 因果泄漏≈1.0' },
        ]
      },
    ]
  },
  {
    id: 'GR', label: '语法体系', icon: '📝', color: '#ffe66d',
    description: '语法编码与句式加工',
    subcategories: [
      {
        id: 'GR-1', label: '词性编码',
        cells: [
          { id: 'GR-1a', question: '词性信息如何编码?', priority: 'P0', status: '◐', detail: 'W_U⊥分类CV=100%; nsubj85%/dobj80%在W_U⊥' },
          { id: 'GR-1b', question: '词性信息是叠加还是分离?', priority: 'P1', status: '□', detail: '' },
          { id: 'GR-1c', question: '不同词性编码维度是否不同?', priority: 'P1', status: '□', detail: '' },
          { id: 'GR-1d', question: '词性的因果载体?', priority: 'P1', status: '◐', detail: '1维注入无效→词性可能高维因果' },
        ]
      },
      {
        id: 'GR-2', label: '句式模板',
        cells: [
          { id: 'GR-2a', question: '句式如何编码?', priority: 'P1', status: '◐', detail: '否定dim=1; 语法因果逐层增长' },
          { id: 'GR-2b', question: '句式模板选择机制?', priority: 'P2', status: '□', detail: '' },
          { id: 'GR-2c', question: '句式组合如何编码?', priority: 'P2', status: '◐', detail: '亚加性; 交互高秩' },
          { id: 'GR-2d', question: '句式信息的因果载体?', priority: 'P2', status: '□', detail: '' },
        ]
      },
      {
        id: 'GR-3', label: '层次结构',
        cells: [
          { id: 'GR-3a', question: '词→短语→句子的层次如何编码?', priority: 'P1', status: '□', detail: '' },
          { id: 'GR-3b', question: '层次是递归还是扁平?', priority: 'P2', status: '□', detail: '' },
          { id: 'GR-3c', question: '长距离依赖如何维持?', priority: 'P2', status: '□', detail: '' },
          { id: 'GR-3d', question: '层次结构信息在哪些层最强?', priority: 'P2', status: '□', detail: '' },
        ]
      },
    ]
  },
  {
    id: 'MG', label: '多维生成控制', icon: '🎛️', color: '#f97316',
    description: '生成时基于多个维度控制',
    subcategories: [
      {
        id: 'MG-1', label: '风格维度',
        cells: [
          { id: 'MG-1a', question: '风格在哪些层开始分化?', priority: 'P1', status: '□', detail: '' },
          { id: 'MG-1b', question: '风格是叠加还是独立通道?', priority: 'P1', status: '□', detail: '' },
          { id: 'MG-1c', question: '风格的因果载体?', priority: 'P1', status: '□', detail: '' },
          { id: 'MG-1d', question: '风格控制是全局还是局部?', priority: 'P2', status: '□', detail: '' },
        ]
      },
      {
        id: 'MG-2', label: '逻辑维度',
        cells: [
          { id: 'MG-2a', question: '上下文逻辑推理如何影响词选择?', priority: 'P1', status: '□', detail: '' },
          { id: 'MG-2b', question: '逻辑连贯性的因果载体?', priority: 'P2', status: '□', detail: '' },
          { id: 'MG-2c', question: '逻辑vs风格是否独立?', priority: 'P2', status: '□', detail: '' },
        ]
      },
      {
        id: 'MG-3', label: '语法维度',
        cells: [
          { id: 'MG-3a', question: '语法约束如何影响词选择?', priority: 'P1', status: '□', detail: '' },
          { id: 'MG-3b', question: '语法约束的因果路径?', priority: 'P1', status: '◐', detail: '语法因果逐层增长' },
          { id: 'MG-3c', question: '语法/逻辑/风格如何同时控制?', priority: 'P0', status: '□', detail: '' },
        ]
      },
      {
        id: 'MG-4', label: '全局选择机制 ★',
        cells: [
          { id: 'MG-4a', question: 'Unembedding如何将残差映射到词表?', priority: 'P0', status: '□', detail: '' },
          { id: 'MG-4b', question: 'softmax前logit分布形态?', priority: 'P0', status: '□', detail: '' },
          { id: 'MG-4c', question: '为什么4000+词中只选一个?', priority: 'P0', status: '□', detail: '' },
          { id: 'MG-4d', question: '多维度如何在logit层汇聚?', priority: 'P0', status: '□', detail: '' },
          { id: 'MG-4e', question: '全局唯一性的数学条件?', priority: 'P1', status: '◐', detail: 'SA<1.0给出约束' },
        ]
      },
    ]
  },
  {
    id: 'SE', label: '系统效率', icon: '⚙️', color: '#34d399',
    description: '系统层面的效率机制',
    subcategories: [
      {
        id: 'SE-1', label: '特征提取',
        cells: [
          { id: 'SE-1a', question: '多维度特征如何被提取?', priority: 'P2', status: '□', detail: '' },
          { id: 'SE-1b', question: '特征提取如何避免维度灾难?', priority: 'P1', status: '◐', detail: '亚加性压缩可能是机制之一' },
          { id: 'SE-1c', question: '特征是分布式还是局部提取?', priority: 'P1', status: '◐', detail: '注意力头分布式编码' },
        ]
      },
      {
        id: 'SE-2', label: '快速查找',
        cells: [
          { id: 'SE-2a', question: '知识检索的延迟有多小?', priority: 'P1', status: '□', detail: '' },
          { id: 'SE-2b', question: '什么结构允许O(1)检索?', priority: 'P2', status: '□', detail: '' },
          { id: 'SE-2c', question: '注意力如何实现查找?', priority: 'P1', status: '□', detail: '' },
        ]
      },
      {
        id: 'SE-3', label: '快速修改',
        cells: [
          { id: 'SE-3a', question: '上下文学习如何在参数中体现?', priority: 'P2', status: '□', detail: '' },
          { id: 'SE-3b', question: '修改一个知识是否影响其他?', priority: 'P1', status: '◐', detail: '因果泄漏≈1.0→修改可能影响全局' },
          { id: 'SE-3c', question: '修改的局部性如何保证?', priority: 'P2', status: '□', detail: '' },
        ]
      },
    ]
  },
  {
    id: 'WE', label: '词嵌入数学', icon: '🔢', color: '#ec4899',
    description: '词嵌入中的数学结构',
    subcategories: [
      {
        id: 'WE-1', label: '线性结构',
        cells: [
          { id: 'WE-1a', question: '线性结构有多普遍?', priority: 'P1', status: '◐', detail: '否定dim=1(部分验证)' },
          { id: 'WE-1b', question: '哪些关系是线性的?', priority: 'P1', status: '◐', detail: '时态dim=15(非线性); 否定dim=1(线性)' },
          { id: 'WE-1c', question: '线性结构是嵌入层还是整条链路?', priority: 'P2', status: '□', detail: '' },
          { id: 'WE-1d', question: '线性结构的因果效力?', priority: 'P1', status: '◐', detail: '1维注入几乎无效→因果效力弱' },
        ]
      },
      {
        id: 'WE-2', label: '数学结构类型',
        cells: [
          { id: 'WE-2a', question: '词嵌入空间有哪些对称性?', priority: 'P2', status: '□', detail: '' },
          { id: 'WE-2b', question: '词嵌入空间是否有度规结构?', priority: 'P2', status: '□', detail: '' },
          { id: 'WE-2c', question: '词嵌入空间是否有拓扑结构?', priority: 'P2', status: '□', detail: '' },
        ]
      },
    ]
  },
  {
    id: 'TD', label: '3D-1D映射', icon: '🧬', color: '#6366f1',
    description: '大脑3D结构→Transformer 1D展开',
    subcategories: [
      {
        id: 'TD-1', label: '结构映射',
        cells: [
          { id: 'TD-1a', question: '大脑6层↔Transformer 40层映射?', priority: 'P2', status: '□', detail: '' },
          { id: 'TD-1b', question: '柱状组↔注意力头/MLP?', priority: 'P2', status: '□', detail: '' },
          { id: 'TD-1c', question: '3D空间连接↔残差+注意力?', priority: 'P2', status: '□', detail: '' },
        ]
      },
      {
        id: 'TD-2', label: '效率映射',
        cells: [
          { id: 'TD-2a', question: '最小传送量原理↔?', priority: 'P2', status: '□', detail: '' },
          { id: 'TD-2b', question: '网格细胞编码效率↔?', priority: 'P2', status: '□', detail: '' },
          { id: 'TD-2c', question: '全局稳态+局部学习↔?', priority: 'P2', status: '□', detail: '' },
        ]
      },
    ]
  },
  {
    id: 'UN', label: '统合格子', icon: '🔗', color: '#ef4444',
    description: '不同能力的交叉验证 — 临界点来源',
    subcategories: [
      {
        id: 'UN-X', label: '交叉验证',
        cells: [
          { id: 'UN-1', question: '知识编码和逻辑推理是否共享编码?', priority: 'P0', status: '◐', detail: '"not"不否定属性方向, 正交扰动' },
          { id: 'UN-2', question: '语法编码和知识编码在同一空间?', priority: 'P1', status: '□', detail: '' },
          { id: 'UN-3', question: '全局选择如何同时满足三维?', priority: 'P0', status: '□', detail: '' },
          { id: 'UN-4', question: '分布式编码如何实现快速查找?', priority: 'P1', status: '□', detail: '' },
          { id: 'UN-5', question: '抽象层级和语法层次共享机制?', priority: 'P1', status: '□', detail: '' },
          { id: 'UN-6', question: '因果泄漏对知识修改意味着什么?', priority: 'P1', status: '□', detail: '' },
          { id: 'UN-7', question: '交互高秩对知识网络拓扑意味什么?', priority: 'P1', status: '□', detail: '' },
          { id: 'UN-8', question: 'PCA对齐和全局选择的关系?', priority: 'P0', status: '□', detail: '' },
        ]
      },
    ]
  },
];

// ==================== 已破解规律数据 ====================

const LAWS = [
  { id: 'R1', layer: '第0层', desc: 'FFN增益调制×内容方向分解', confidence: '⭐⭐⭐' },
  { id: 'R2', layer: '第0层', desc: '语义特异性在d_model不在n_inter', confidence: '⭐⭐⭐' },
  { id: 'R3', layer: '第0层', desc: '语义力线指数增长→W_U→Logit', confidence: '⭐⭐⭐' },
  { id: 'R4', layer: '第0层', desc: 'Attention是语义→Logit主通道', confidence: '⭐⭐⭐' },
  { id: 'R5', layer: '第0层', desc: '方向反射是训练习得的', confidence: '⭐⭐⭐' },
  { id: 'R6', layer: '第1层', desc: '1-3个SVD模式控制97%输出', confidence: '⭐⭐⭐' },
  { id: 'R7', layer: '第1层', desc: '增强对比(近更近远更远)', confidence: '⭐⭐⭐' },
  { id: 'R8', layer: '第1层', desc: '三组件几何角色', confidence: '⭐⭐⭐' },
  { id: 'R9', layer: '第1层', desc: '旋转>90°=维度诅咒', confidence: '⭐⭐⭐' },
  { id: 'R10', layer: '第1层', desc: '残差流主成分由类别驱动', confidence: '⭐⭐⭐' },
  { id: 'R11', layer: '第2层', desc: '概念是层依赖向量不是轨道', confidence: '⭐⭐' },
  { id: 'R12', layer: '第2层', desc: '暗物质承载75-87%steering效果', confidence: '⭐⭐' },
  { id: 'R13', layer: '第2层', desc: '语义流形≈变形单纯形', confidence: '⭐⭐' },
  { id: 'R14', layer: '第2层', desc: '同类正相关异类负相关', confidence: '⭐⭐' },
  { id: 'R15', layer: '第2层', desc: '传递性推理深层不成立', confidence: '⭐⭐' },
  { id: 'R16', layer: '第3层', desc: '语法信息编码在W_U⊥子空间', confidence: '⭐⭐⭐' },
  { id: 'R17', layer: '第3层', desc: '语法正交补空间理论', confidence: '⭐⭐⭐' },
  { id: 'R18', layer: '第3层', desc: 'W_U⊥操控翻转50-68%语法分类', confidence: '⭐⭐' },
];

const LAYER_LABELS = {
  '第0层': 'DNN组件编码',
  '第1层': '几何动力学',
  '第2层': '概念编码与传播',
  '第3层': '语法-语义解耦',
};

// ==================== 状态颜色映射 ====================

const STATUS_COLORS = {
  '✓': '#22c55e',    // 已填充 - 绿色
  '◐': '#f59e0b',    // 部分填充 - 黄色
  '□': '#334155',    // 未填充 - 灰色
};

const STATUS_BG = {
  '✓': 'rgba(34, 197, 94, 0.15)',
  '◐': 'rgba(245, 158, 11, 0.15)',
  '□': 'rgba(51, 65, 85, 0.3)',
};

const PRIORITY_COLORS = {
  'P0': '#ef4444',
  'P1': '#f59e0b',
  'P2': '#64748b',
};

// ==================== 子组件 ====================

function CellTile({ cell, onClick, isSelected }) {
  const borderColor = STATUS_COLORS[cell.status] || '#334155';
  const bgColor = STATUS_BG[cell.status] || 'transparent';
  const prioColor = PRIORITY_COLORS[cell.priority] || '#64748b';

  return (
    <div
      onClick={() => onClick(cell)}
      style={{
        padding: '6px 8px',
        borderRadius: 6,
        border: `1px solid ${isSelected ? '#60a5fa' : borderColor}`,
        background: isSelected ? 'rgba(96, 165, 250, 0.1)' : bgColor,
        cursor: 'pointer',
        fontSize: 11,
        lineHeight: 1.4,
        transition: 'all 0.15s',
        position: 'relative',
      }}
      title={cell.question}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
        <span style={{ color: borderColor, fontWeight: 'bold', fontSize: 10 }}>{cell.id}</span>
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
          <span style={{ fontSize: 9, color: prioColor, fontWeight: 'bold' }}>{cell.priority}</span>
          <span style={{ fontSize: 12, lineHeight: 1 }}>{cell.status}</span>
        </div>
      </div>
      <div style={{ color: '#94a3b8', fontSize: 10, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {cell.question}
      </div>
    </div>
  );
}

function CategorySection({ category, selectedCell, onSelectCell, collapsed, onToggleCollapse }) {
  // 统计
  const allCells = category.subcategories.flatMap(sc => sc.cells);
  const filled = allCells.filter(c => c.status === '✓').length;
  const partial = allCells.filter(c => c.status === '◐').length;
  const empty = allCells.filter(c => c.status === '□').length;
  const total = allCells.length;
  const fillRate = total > 0 ? ((filled + partial * 0.5) / total * 100).toFixed(0) : 0;

  return (
    <div style={{ marginBottom: 8 }}>
      {/* 分类标题 */}
      <div
        onClick={onToggleCollapse}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '8px 10px',
          background: 'rgba(30, 41, 59, 0.6)',
          borderRadius: 6,
          cursor: 'pointer',
          borderLeft: `3px solid ${category.color}`,
          marginBottom: collapsed ? 0 : 6,
        }}
      >
        <span style={{ fontSize: 14 }}>{category.icon}</span>
        <span style={{ fontSize: 12, fontWeight: 'bold', color: category.color, flex: 1 }}>
          {category.label}
        </span>
        {/* 进度条 */}
        <div style={{ width: 60, height: 6, background: '#1e293b', borderRadius: 3, overflow: 'hidden', marginRight: 8 }}>
          <div style={{ width: `${fillRate}%`, height: '100%', background: category.color, borderRadius: 3, transition: 'width 0.3s' }} />
        </div>
        <span style={{ fontSize: 10, color: '#64748b', minWidth: 30, textAlign: 'right' }}>{fillRate}%</span>
        <span style={{ fontSize: 10, color: '#475569', marginLeft: 4 }}>{collapsed ? '▶' : '▼'}</span>
      </div>

      {/* 子分类格子 */}
      {!collapsed && category.subcategories.map(sc => (
        <div key={sc.id} style={{ marginLeft: 12, marginBottom: 6 }}>
          <div style={{ fontSize: 10, color: '#64748b', marginBottom: 4, paddingLeft: 4 }}>
            {sc.label}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
            {sc.cells.map(cell => (
              <CellTile
                key={cell.id}
                cell={cell}
                onClick={onSelectCell}
                isSelected={selectedCell?.id === cell.id}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function LawsSection({ laws, collapsed, onToggleCollapse }) {
  if (collapsed) {
    return (
      <div
        onClick={onToggleCollapse}
        style={{
          padding: '8px 10px',
          background: 'rgba(30, 41, 59, 0.6)',
          borderRadius: 6,
          cursor: 'pointer',
          borderLeft: '3px solid #60a5fa',
          marginBottom: 6,
        }}
      >
        <span style={{ fontSize: 12, fontWeight: 'bold', color: '#60a5fa' }}>📐 18条已破解规律</span>
        <span style={{ float: 'right', fontSize: 10, color: '#475569' }}>▶</span>
      </div>
    );
  }

  // 按层分组
  const grouped = {};
  laws.forEach(l => {
    if (!grouped[l.layer]) grouped[l.layer] = [];
    grouped[l.layer].push(l);
  });

  return (
    <div style={{ marginBottom: 8 }}>
      <div
        onClick={onToggleCollapse}
        style={{
          padding: '8px 10px',
          background: 'rgba(30, 41, 59, 0.6)',
          borderRadius: 6,
          cursor: 'pointer',
          borderLeft: '3px solid #60a5fa',
          marginBottom: 6,
        }}
      >
        <span style={{ fontSize: 12, fontWeight: 'bold', color: '#60a5fa' }}>📐 18条已破解规律</span>
        <span style={{ float: 'right', fontSize: 10, color: '#475569' }}>▼</span>
      </div>
      {Object.entries(grouped).map(([layer, items]) => (
        <div key={layer} style={{ marginLeft: 12, marginBottom: 6 }}>
          <div style={{ fontSize: 10, color: '#64748b', marginBottom: 3, paddingLeft: 4 }}>
            {layer}: {LAYER_LABELS[layer]}
          </div>
          {items.map(law => (
            <div
              key={law.id}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '3px 6px',
                fontSize: 10,
                color: '#94a3b8',
              }}
            >
              <span style={{ color: '#60a5fa', fontWeight: 'bold', minWidth: 24 }}>{law.id}</span>
              <span style={{ flex: 1 }}>{law.desc}</span>
              <span style={{ fontSize: 8 }}>{law.confidence}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function CellDetail({ cell, category }) {
  if (!cell) return null;
  const statusColor = STATUS_COLORS[cell.status];
  const prioColor = PRIORITY_COLORS[cell.priority];

  return (
    <div style={{
      padding: 12,
      background: 'rgba(15, 23, 42, 0.8)',
      border: `1px solid ${statusColor}`,
      borderRadius: 8,
      marginTop: 8,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{ color: category?.color || '#60a5fa', fontWeight: 'bold', fontSize: 14 }}>{cell.id}</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ fontSize: 10, color: prioColor, fontWeight: 'bold', padding: '2px 6px', background: `${prioColor}22`, borderRadius: 4 }}>
            {cell.priority}
          </span>
          <span style={{ fontSize: 16 }}>{cell.status}</span>
        </div>
      </div>
      <div style={{ color: '#e2e8f0', fontSize: 12, marginBottom: 8, lineHeight: 1.5 }}>
        {cell.question}
      </div>
      {cell.detail && (
        <div style={{ padding: '8px 10px', background: 'rgba(30, 41, 59, 0.5)', borderRadius: 6, fontSize: 11, color: '#94a3b8', lineHeight: 1.6 }}>
          {cell.detail}
        </div>
      )}
      {category && (
        <div style={{ marginTop: 8, fontSize: 10, color: '#64748b' }}>
          <span style={{ color: category.color }}>{category.icon} {category.label}</span>
        </div>
      )}
    </div>
  );
}

// ==================== 总览统计 ====================

function OverviewBar({ categories }) {
  const allCells = categories.flatMap(c => c.subcategories.flatMap(sc => sc.cells));
  const filled = allCells.filter(c => c.status === '✓').length;
  const partial = allCells.filter(c => c.status === '◐').length;
  const empty = allCells.filter(c => c.status === '□').length;
  const total = allCells.length;
  const fillRate = ((filled + partial * 0.5) / total * 100).toFixed(0);
  const p0Cells = allCells.filter(c => c.priority === 'P0');
  const p0Filled = p0Cells.filter(c => c.status === '✓').length;
  const p0Partial = p0Cells.filter(c => c.status === '◐').length;
  const p0Rate = p0Cells.length > 0 ? ((p0Filled + p0Partial * 0.5) / p0Cells.length * 100).toFixed(0) : 0;

  return (
    <div style={{
      padding: '10px 12px',
      background: 'rgba(30, 41, 59, 0.8)',
      borderRadius: 8,
      marginBottom: 10,
      border: '1px solid #1e293b',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{ fontSize: 12, fontWeight: 'bold', color: '#e2e8f0' }}>拼图总览</span>
        <span style={{ fontSize: 14, fontWeight: 'bold', color: '#60a5fa' }}>{fillRate}%</span>
      </div>
      {/* 总进度条 */}
      <div style={{ height: 8, background: '#1e293b', borderRadius: 4, overflow: 'hidden', marginBottom: 6, display: 'flex' }}>
        <div style={{ width: `${filled / total * 100}%`, background: '#22c55e', transition: 'width 0.3s' }} />
        <div style={{ width: `${partial / total * 100}%`, background: '#f59e0b', transition: 'width 0.3s' }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#64748b' }}>
        <span><span style={{ color: '#22c55e' }}>■</span> 已填充✓ {filled}</span>
        <span><span style={{ color: '#f59e0b' }}>■</span> 部分填充◐ {partial}</span>
        <span><span style={{ color: '#475569' }}>■</span> 未填充□ {empty}</span>
      </div>
      <div style={{ marginTop: 6, fontSize: 10, color: '#64748b', borderTop: '1px solid #1e293b', paddingTop: 6 }}>
        P0格子: {p0Filled}✓ + {p0Partial}◐ / {p0Cells.length} = <span style={{ color: '#ef4444', fontWeight: 'bold' }}>{p0Rate}%</span>
      </div>
    </div>
  );
}

// ==================== 主组件 ====================

export default function PuzzlePanel({ puzzleData }) {
  const [selectedCell, setSelectedCell] = useState(null);
  const [collapsedCats, setCollapsedCats] = useState({});
  const [lawsCollapsed, setLawsCollapsed] = useState(true);
  const [tab, setTab] = useState('puzzle'); // puzzle | laws | theory

  // 合并外部数据 (如果有puzzle_progress类型的数据)
  const categories = useMemo(() => {
    if (!puzzleData) return PUZZLE_CATEGORIES;
    // TODO: 合并外部puzzle_progress数据
    return PUZZLE_CATEGORIES;
  }, [puzzleData]);

  const toggleCat = (id) => {
    setCollapsedCats(prev => ({ ...prev, [id]: !prev[id] }));
  };

  // 找到选中cell所属的category
  const selectedCategory = selectedCell
    ? categories.find(c => c.subcategories.some(sc => sc.cells.some(cell => cell.id === selectedCell.id)))
    : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', fontSize: 12 }}>
      {/* Tab切换 */}
      <div style={{ display: 'flex', borderBottom: '1px solid #1e293b', marginBottom: 8, flexShrink: 0 }}>
        {[
          { key: 'puzzle', label: '🧩 拼图' },
          { key: 'laws', label: '📐 规律' },
          { key: 'theory', label: '🔬 理论' },
        ].map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            style={{
              flex: 1, padding: '8px 4px',
              background: tab === t.key ? 'rgba(96, 165, 250, 0.1)' : 'transparent',
              border: 'none', borderBottom: tab === t.key ? '2px solid #60a5fa' : '2px solid transparent',
              color: tab === t.key ? '#60a5fa' : '#64748b',
              cursor: 'pointer', fontSize: 11, fontWeight: 'bold',
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* 内容区域 */}
      <div style={{ flex: 1, overflowY: 'auto', paddingRight: 4 }}>
        {tab === 'puzzle' && (
          <>
            <OverviewBar categories={categories} />
            {categories.map(cat => (
              <CategorySection
                key={cat.id}
                category={cat}
                selectedCell={selectedCell}
                onSelectCell={setSelectedCell}
                collapsed={collapsedCats[cat.id] !== false} // 默认展开
                onToggleCollapse={() => toggleCat(cat.id)}
              />
            ))}
            {/* 选中格子详情 */}
            <CellDetail cell={selectedCell} category={selectedCategory} />
          </>
        )}

        {tab === 'laws' && (
          <>
            {/* 四层规律体系 */}
            <div style={{ padding: '8px 10px', background: 'rgba(30, 41, 59, 0.6)', borderRadius: 6, marginBottom: 8, borderLeft: '3px solid #60a5fa' }}>
              <div style={{ fontSize: 12, fontWeight: 'bold', color: '#60a5fa', marginBottom: 4 }}>18条已破解规律 (四层体系)</div>
              <div style={{ fontSize: 10, color: '#64748b' }}>总解释力: ~18% (从2%提升)</div>
            </div>
            <LawsSection laws={LAWS} collapsed={lawsCollapsed} onToggleCollapse={() => setLawsCollapsed(!lawsCollapsed)} />

            {/* 28条不变量 */}
            <div style={{ padding: '8px 10px', background: 'rgba(30, 41, 59, 0.6)', borderRadius: 6, marginTop: 8, borderLeft: '3px solid #4ecdc4' }}>
              <div style={{ fontSize: 12, fontWeight: 'bold', color: '#4ecdc4', marginBottom: 4 }}>28条已确认不变量 (INV-1~28)</div>
              <div style={{ fontSize: 10, color: '#64748b', lineHeight: 1.8 }}>
                <div><span style={{ color: '#4ecdc4' }}>信号传播</span>(7): cos→0 / W_U能量转移 / 残差保留62-71% / L0→L1断裂 / 词嵌入正交 / 加法传播定律 / sigmoid映射</div>
                <div><span style={{ color: '#ffe66d' }}>概念编码</span>(8): PCA5探针0.98-1.00 / 同类正异类负 / eff_dim=17-22 / 抽象cos≈-0.20 / category驱动PCA / template热点 / switch_rate递减</div>
                <div><span style={{ color: '#a855f7' }}>逻辑编码</span>(4): "not"正交扰动 / 肯否定句cos≈0.8-0.9 / 逻辑2.7x属性 / 逻辑峰值L18</div>
                <div><span style={{ color: '#f97316' }}>因果干预</span>(4): 属性干预99.56% / 推理恢复率&lt;4% / 深层干预无效 / 功能信号线性可分</div>
                <div><span style={{ color: '#34d399' }}>频谱力学</span>(5): 频谱传播方程 / W_down决定h频谱 / 频谱不含语义 / 符号噪声源</div>
              </div>
            </div>

            {/* 11条推翻假说 */}
            <div style={{ padding: '8px 10px', background: 'rgba(127, 29, 29, 0.3)', borderRadius: 6, marginTop: 8, borderLeft: '3px solid #ef4444' }}>
              <div style={{ fontSize: 12, fontWeight: 'bold', color: '#ef4444', marginBottom: 4 }}>11条已推翻假说 (REF-1~11)</div>
              <div style={{ fontSize: 10, color: '#94a3b8', lineHeight: 1.6 }}>
                ratio=0.22恒定量✗ / 频谱力学预测logit✗ / 末端层gap高预测力✗ / 三模型统一涌现✗ / 平坦线性几何✗ / 逻辑是暗物质✗ / ICSPB统一变量✗ / 多模态假说✗ / 量子声学✗ / 概念编码是因果载体✗ / 残差流90%是模板✗
              </div>
            </div>
          </>
        )}

        {tab === 'theory' && (
          <>
            {/* 四层理论框架 */}
            <div style={{ padding: '10px 12px', background: 'rgba(30, 41, 59, 0.8)', borderRadius: 8, marginBottom: 8, border: '1px solid #1e293b' }}>
              <div style={{ fontSize: 13, fontWeight: 'bold', color: '#e2e8f0', marginBottom: 8 }}>
                语言数学原理 — 四层框架
              </div>
              <div style={{ fontSize: 11, color: '#60a5fa', marginBottom: 6, fontFamily: 'monospace' }}>
                Language = Geometry + Algebra + Dynamics + Information
              </div>

              {[
                { name: '几何层', desc: '信息位于哪些子空间? 维度? 流形?', color: '#4ecdc4', fill: '30%', icon: '📐' },
                { name: '代数层', desc: '子空间如何组合? 群/格/范畴?', color: '#a855f7', fill: '10%', icon: '🔢' },
                { name: '动力学层', desc: '信息如何传播? 因果链?', color: '#f97316', fill: '40%', icon: '⚡' },
                { name: '信息论层', desc: '信息如何分配? 冗余? 隐式?', color: '#ef4444', fill: '15%', icon: '📊' },
              ].map(layer => (
                <div key={layer.name} style={{ marginBottom: 6 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                    <span>{layer.icon}</span>
                    <span style={{ color: layer.color, fontWeight: 'bold', fontSize: 11 }}>{layer.name}</span>
                    <span style={{ flex: 1 }} />
                    <span style={{ fontSize: 10, color: '#64748b' }}>{layer.fill}</span>
                  </div>
                  <div style={{ height: 4, background: '#1e293b', borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{ width: layer.fill, height: '100%', background: layer.color, borderRadius: 2 }} />
                  </div>
                  <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>{layer.desc}</div>
                </div>
              ))}
            </div>

            {/* 子空间分解地图 */}
            <div style={{ padding: '10px 12px', background: 'rgba(30, 41, 59, 0.8)', borderRadius: 8, marginBottom: 8, border: '1px solid #1e293b' }}>
              <div style={{ fontSize: 12, fontWeight: 'bold', color: '#e2e8f0', marginBottom: 6 }}>子空间分解地图</div>
              <div style={{ fontFamily: 'monospace', fontSize: 10, color: '#94a3b8', lineHeight: 1.8, padding: '6px 8px', background: '#0f172a', borderRadius: 4 }}>
                <div><span style={{ color: '#4ecdc4' }}>V_WU∩V_sem</span> 语义(~14%)</div>
                <div><span style={{ color: '#ffe66d' }}>V_WU⊥∩V_syn</span> 语法(可分类)</div>
                <div><span style={{ color: '#f97316' }}>V_WU⊥∩V_dark</span> 暗物质(86-92%)</div>
                <div><span style={{ color: '#a855f7' }}>V_⊥logic</span> 逻辑(独立)</div>
                <div><span style={{ color: '#64748b' }}>V_residual</span> 余项</div>
              </div>
            </div>

            {/* 三条主线 */}
            <div style={{ padding: '10px 12px', background: 'rgba(30, 41, 59, 0.8)', borderRadius: 8, marginBottom: 8, border: '1px solid #1e293b' }}>
              <div style={{ fontSize: 12, fontWeight: 'bold', color: '#e2e8f0', marginBottom: 6 }}>三条研究主线</div>
              {[
                { name: 'A: 语法数学', desc: 'W_U⊥子空间完整刻画', color: '#ffe66d', steps: '9A精细结构 → 9B泄露机制 → 9C容量 → 9D因果链 → 变换群' },
                { name: 'B: 语义数学', desc: '概念流形与暗物质转导', color: '#4ecdc4', steps: '流形结构 → 暗物质转导★★★ → 属性绑定 → 深层锁定' },
                { name: 'C: 逻辑数学', desc: '逻辑操作的代数结构', color: '#a855f7', steps: '操作方向 → 组合律★★★ → 传递性机制 → 因果推理路径' },
              ].map(line => (
                <div key={line.name} style={{ marginBottom: 8, padding: '6px 8px', background: `${line.color}11`, borderRadius: 6, borderLeft: `2px solid ${line.color}` }}>
                  <div style={{ fontSize: 11, fontWeight: 'bold', color: line.color, marginBottom: 2 }}>{line.name}</div>
                  <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 2 }}>{line.desc}</div>
                  <div style={{ fontSize: 9, color: '#64748b' }}>{line.steps}</div>
                </div>
              ))}
            </div>

            {/* 临界点 */}
            <div style={{ padding: '10px 12px', background: 'rgba(127, 29, 29, 0.2)', borderRadius: 8, marginBottom: 8, border: '1px solid #7f1d1d' }}>
              <div style={{ fontSize: 12, fontWeight: 'bold', color: '#ef4444', marginBottom: 4 }}>🎯 临界点</div>
              <div style={{ fontSize: 10, color: '#94a3b8', lineHeight: 1.6 }}>
                条件: 填充率&gt;60% + 3个UN格子 + 数学结构自然涌现
                <br />当前: 填充率18% | P0填充率48% | 0个UN✓
                <br />解释力: 18% → 目标80%
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
