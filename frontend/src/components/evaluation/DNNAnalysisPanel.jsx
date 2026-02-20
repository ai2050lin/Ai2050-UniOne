/**
 * 深度神经网络数学结构分析模块
 * 展示从神经网络中还原数学结构的方案和进度
 */

import {
  Activity,
  Brain,
  CheckCircle2,
  CircleDot,
  Cpu,
  GitBranch,
  Network,
  Radio,
  Target,
  Zap,
} from 'lucide-react';
import { useMemo } from 'react';

// 分析阶段定义
const ANALYSIS_PHASES = [
  {
    id: 'L1',
    name: '表征几何分析',
    icon: Network,
    description: '提取激活空间的几何性质',
    status: 'done',
    progress: 100,
    items: [
      { name: '流形拓扑分析', status: 'done', metric: 'Betti数计算' },
      { name: '曲率估计', status: 'done', metric: 'Ricci曲率 ~0.01' },
      { name: '内在维度测量', status: 'done', metric: 'd_intrinsic ~128' },
    ],
  },
  {
    id: 'L2',
    name: '结构解耦分析',
    icon: GitBranch,
    description: '分离"结构"和"内容"',
    status: 'done',
    progress: 85,
    items: [
      { name: '底流形提取', status: 'done', metric: 'PCA/UMAP' },
      { name: '纤维结构识别', status: 'done', metric: '注意力头分析' },
      { name: '联络矩阵重建', status: 'partial', metric: '残差连接权重' },
    ],
  },
  {
    id: 'L3',
    name: '动力系统分析',
    icon: Zap,
    description: '理解推理如何发生',
    status: 'progress',
    progress: 60,
    items: [
      { name: '测地线提取', status: 'done', metric: '推理轨迹追踪' },
      { name: 'Ricci Flow重构', status: 'progress', metric: '训练过程演化' },
      { name: '吸引子识别', status: 'pending', metric: '稳定概念检测' },
    ],
  },
  {
    id: 'L4',
    name: '编码机制逆向',
    icon: Brain,
    description: '理解知识存储方式',
    status: 'progress',
    progress: 45,
    items: [
      { name: '稀疏编码字典学习', status: 'done', metric: '字典大小 100' },
      { name: '正交基底提取', status: 'progress', metric: 'SVD分解' },
      { name: '叠加模式分析', status: 'pending', metric: '全息编码验证' },
    ],
  },
];

// 数学结构映射
const STRUCTURE_MAPPING = [
  {
    nnComponent: '层激活',
    mathStructure: '流形切片',
    meaning: '知识表示空间',
    status: 'verified',
  },
  {
    nnComponent: '残差连接',
    mathStructure: '平行移动',
    meaning: '信息保真传输',
    status: 'verified',
  },
  {
    nnComponent: '注意力权重',
    mathStructure: '联络系数',
    meaning: '结构-内容关联',
    status: 'partial',
  },
  {
    nnComponent: '损失下降',
    mathStructure: '曲率演化',
    meaning: '知识流形平滑化',
    status: 'verified',
  },
  {
    nnComponent: '推理路径',
    mathStructure: '测地线',
    meaning: '最优推理轨迹',
    status: 'verified',
  },
  {
    nnComponent: '权重矩阵',
    mathStructure: '度量张量',
    meaning: '内积结构',
    status: 'partial',
  },
];

// 已实现工具
const IMPLEMENTED_TOOLS = [
  {
    name: 'global_topology_scanner.py',
    function: 'Betti数、持久同调',
    status: 'done',
    location: 'scripts/',
  },
  {
    name: 'ricci_flow.py',
    function: '曲率估计、演化',
    status: 'done',
    location: 'models/',
  },
  {
    name: 'geodesic_retrieval.py',
    function: '测地线搜索',
    status: 'done',
    location: 'models/',
  },
  {
    name: 'fibernet_service.py',
    function: '纤维丛分析',
    status: 'done',
    location: 'server/',
  },
  {
    name: 'geometric_intervention_test.py',
    function: '几何干预',
    status: 'done',
    location: 'scripts/',
  },
  {
    name: 'structure_analyzer.py',
    function: '结构分析集成',
    status: 'done',
    location: '根目录',
  },
];

// 统计数据
const STATISTICS = {
  modelsAnalyzed: 2,
  layersScanned: 36,
  geodesicsExtracted: 1000,
  curvatureMeasurements: 25000,
  bettiNumbers: { b0: 1, b1: 66, b2: 12 },
};

export default function DNNAnalysisPanel() {
  const overallProgress = useMemo(() => {
    const total = ANALYSIS_PHASES.reduce((sum, p) => sum + p.progress, 0);
    return Math.round(total / ANALYSIS_PHASES.length);
  }, []);

  const completedPhases = useMemo(
    () => ANALYSIS_PHASES.filter((p) => p.status === 'done').length,
    []
  );

  const totalItems = useMemo(() => {
    let count = 0;
    let done = 0;
    ANALYSIS_PHASES.forEach((p) => {
      p.items.forEach((item) => {
        count++;
        if (item.status === 'done') done++;
      });
    });
    return { total: count, done };
  }, []);

  return (
    <div className="space-y-6">
      {/* 标题和总体进度 */}
      <div className="bg-gradient-to-r from-violet-900/20 to-transparent p-4 rounded-xl border border-violet-500/20">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold flex items-center gap-2 text-white">
            <Brain className="text-violet-400" />
            深度神经网络数学结构分析
          </h3>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-400">总进度</span>
            <span className="text-xl font-bold text-violet-400">{overallProgress}%</span>
          </div>
        </div>
        <p className="text-sm text-zinc-400 mb-3">
          还原深度神经网络从人脑中学到的数学结构
        </p>
        <div className="w-full bg-zinc-800 rounded-full h-2 overflow-hidden">
          <div
            className="bg-gradient-to-r from-violet-500 to-purple-400 h-full transition-all duration-500"
            style={{ width: `${overallProgress}%` }}
          />
        </div>
        <div className="flex items-center justify-between mt-2 text-xs text-zinc-500">
          <span>
            {completedPhases}/{ANALYSIS_PHASES.length} 阶段完成
          </span>
          <span>
            {totalItems.done}/{totalItems.total} 项目完成
          </span>
        </div>
      </div>

      {/* 分析阶段卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {ANALYSIS_PHASES.map((phase) => {
          const Icon = phase.icon;
          const statusColors = {
            done: 'border-green-500/30 bg-green-900/10',
            progress: 'border-blue-500/30 bg-blue-900/10',
            pending: 'border-zinc-700 bg-zinc-900/30',
          };
          const progressColors = {
            done: 'bg-green-500',
            progress: 'bg-blue-500',
            pending: 'bg-zinc-600',
          };

          return (
            <div
              key={phase.id}
              className={`p-4 rounded-xl border transition-all ${statusColors[phase.status]}`}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Icon
                    className={`${
                      phase.status === 'done'
                        ? 'text-green-400'
                        : phase.status === 'progress'
                          ? 'text-blue-400'
                          : 'text-zinc-500'
                    }`}
                    size={18}
                  />
                  <span className="font-semibold text-white">
                    {phase.id}. {phase.name}
                  </span>
                </div>
                <span className="text-xs text-zinc-400">{phase.progress}%</span>
              </div>

              <p className="text-xs text-zinc-400 mb-3">{phase.description}</p>

              <div className="w-full bg-zinc-800 rounded-full h-1 mb-3 overflow-hidden">
                <div
                  className={`h-full transition-all ${progressColors[phase.status]}`}
                  style={{ width: `${phase.progress}%` }}
                />
              </div>

              <div className="space-y-2">
                {phase.items.map((item, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between text-xs"
                  >
                    <div className="flex items-center gap-2">
                      {item.status === 'done' ? (
                        <CheckCircle2 size={12} className="text-green-400" />
                      ) : item.status === 'partial' ? (
                        <CircleDot size={12} className="text-yellow-400" />
                      ) : (
                        <CircleDot size={12} className="text-zinc-500" />
                      )}
                      <span className="text-zinc-300">{item.name}</span>
                    </div>
                    <span className="text-zinc-500 font-mono">{item.metric}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* 数学结构映射表 */}
      <div className="bg-zinc-900/50 border border-white/5 rounded-xl p-4">
        <h4 className="text-sm font-semibold text-zinc-200 mb-3 flex items-center gap-2">
          <Target className="text-cyan-400" size={16} />
          神经网络 → 数学结构映射
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-2 px-2 text-zinc-500">神经网络组件</th>
                <th className="text-left py-2 px-2 text-zinc-500">数学结构</th>
                <th className="text-left py-2 px-2 text-zinc-500">物理意义</th>
                <th className="text-center py-2 px-2 text-zinc-500">验证状态</th>
              </tr>
            </thead>
            <tbody>
              {STRUCTURE_MAPPING.map((item, idx) => (
                <tr key={idx} className="border-b border-white/5">
                  <td className="py-2 px-2 text-zinc-300 font-mono">{item.nnComponent}</td>
                  <td className="py-2 px-2 text-violet-300">{item.mathStructure}</td>
                  <td className="py-2 px-2 text-zinc-400">{item.meaning}</td>
                  <td className="py-2 px-2 text-center">
                    <span
                      className={`px-2 py-0.5 rounded-full text-[10px] ${
                        item.status === 'verified'
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-yellow-500/20 text-yellow-400'
                      }`}
                    >
                      {item.status === 'verified' ? '已验证' : '部分验证'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 已实现工具 */}
      <div className="bg-zinc-900/50 border border-white/5 rounded-xl p-4">
        <h4 className="text-sm font-semibold text-zinc-200 mb-3 flex items-center gap-2">
          <Cpu className="text-orange-400" size={16} />
          已实现分析工具
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
          {IMPLEMENTED_TOOLS.map((tool, idx) => (
            <div
              key={idx}
              className="p-2 rounded-lg bg-zinc-800/50 border border-white/5 flex items-center gap-2"
            >
              <CheckCircle2 size={14} className="text-green-400 flex-shrink-0" />
              <div className="min-w-0">
                <div className="text-xs text-zinc-300 font-mono truncate">
                  {tool.name}
                </div>
                <div className="text-[10px] text-zinc-500 truncate">
                  {tool.function}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 统计数据 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="p-3 rounded-xl bg-zinc-900/50 border border-white/5 text-center">
          <div className="text-xs text-zinc-500 mb-1">已分析模型</div>
          <div className="text-xl font-bold text-blue-400">{STATISTICS.modelsAnalyzed}</div>
        </div>
        <div className="p-3 rounded-xl bg-zinc-900/50 border border-white/5 text-center">
          <div className="text-xs text-zinc-500 mb-1">扫描层数</div>
          <div className="text-xl font-bold text-green-400">{STATISTICS.layersScanned}</div>
        </div>
        <div className="p-3 rounded-xl bg-zinc-900/50 border border-white/5 text-center">
          <div className="text-xs text-zinc-500 mb-1">测地线提取</div>
          <div className="text-xl font-bold text-violet-400">
            {STATISTICS.geodesicsExtracted.toLocaleString()}
          </div>
        </div>
        <div className="p-3 rounded-xl bg-zinc-900/50 border border-white/5 text-center">
          <div className="text-xs text-zinc-500 mb-1">曲率测量点</div>
          <div className="text-xl font-bold text-cyan-400">
            {STATISTICS.curvatureMeasurements.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Betti 数统计 */}
      <div className="bg-zinc-900/50 border border-white/5 rounded-xl p-4">
        <h4 className="text-sm font-semibold text-zinc-200 mb-3 flex items-center gap-2">
          <Radio className="text-pink-400" size={16} />
          拓扑不变量 (Betti数)
        </h4>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-xs text-zinc-500 mb-1">β₀ (连通分量)</div>
            <div className="text-2xl font-bold text-pink-400">{STATISTICS.bettiNumbers.b0}</div>
            <div className="text-[10px] text-zinc-500 mt-1">独立概念簇</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-zinc-500 mb-1">β₁ (环路)</div>
            <div className="text-2xl font-bold text-orange-400">{STATISTICS.bettiNumbers.b1}</div>
            <div className="text-[10px] text-zinc-500 mt-1">循环依赖</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-zinc-500 mb-1">β₂ (空洞)</div>
            <div className="text-2xl font-bold text-cyan-400">{STATISTICS.bettiNumbers.b2}</div>
            <div className="text-[10px] text-zinc-500 mt-1">高阶结构</div>
          </div>
        </div>
      </div>

      {/* 核心洞见 */}
      <div className="bg-gradient-to-r from-amber-900/20 to-transparent p-4 rounded-xl border border-amber-500/20">
        <h4 className="text-sm font-semibold text-zinc-200 mb-2 flex items-center gap-2">
          <Activity className="text-amber-400" size={16} />
          核心洞见
        </h4>
        <ul className="text-xs text-zinc-400 space-y-1.5">
          <li className="flex items-start gap-2">
            <span className="text-green-400 mt-0.5">•</span>
            <span>
              <strong className="text-zinc-300">神经网络不是黑箱</strong> - 它是几何流形的编码器
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 mt-0.5">•</span>
            <span>
              <strong className="text-zinc-300">学习 = 曲率演化</strong> - Ricci Flow 平滑知识表示
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-violet-400 mt-0.5">•</span>
            <span>
              <strong className="text-zinc-300">推理 = 测地线运动</strong> - 最优路径上的状态转移
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-cyan-400 mt-0.5">•</span>
            <span>
              <strong className="text-zinc-300">知识 = 稀疏正交编码</strong> - 高效全息存储
            </span>
          </li>
        </ul>
      </div>
    </div>
  );
}
