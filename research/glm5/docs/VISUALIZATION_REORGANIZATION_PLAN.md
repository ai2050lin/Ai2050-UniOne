# 可视化项目整理方案

## 整理目标

基于三条核心研究路线，重新组织可视化项目，使其与研究方向对齐，方便查看和分析各路线的数据与进展。

---

## 当前问题分析

### 现状

```
frontend/src/
├── components/
│   ├── analysis/        # 分析组件
│   ├── evaluation/      # 评估组件
│   ├── intervention/    # 干预组件
│   ├── observation/     # 观察组件
│   ├── shared/         # 共享组件
│   ├── FiberNetPanel.jsx
│   └── WorkbenchLayout.jsx
│
├── 多个顶层组件文件
│   ├── BrainVis3D.jsx
│   ├── GlobalTopologyDashboard.jsx
│   ├── HLAIBlueprint.jsx
│   ├── StructureAnalysisPanel.jsx
│   ├── TDAVisualization3D.jsx
│   └── ... (20+文件)
│
└── App.jsx (主应用)
```

### 问题

1. **组件结构不清晰**: 顶层文件过多，缺乏组织
2. **与研究路线不对齐**: 未按三条路线组织
3. **缺乏数据集成**: 没有直接对接研究结果数据
4. **文档不足**: 缺少使用说明

---

## 整理方案

### 整体结构

```
frontend/src/
│
├── routes/                      # 按研究路线组织
│   ├── route1_dnn_analysis/    # 路线1: DNN结构分析
│   │   ├── FeatureExtraction/  # 特征提取可视化
│   │   ├── FourProperties/     # 四特性评估可视化
│   │   ├── SparsityAnalysis/   # 稀疏编码分析可视化
│   │   ├── LayerEvolution/     # 层级演化可视化
│   │   ├── index.jsx           # 路线入口
│   │   └── README.md           # 使用说明
│   │
│   ├── route2_brain_mechanism/ # 路线2: 大脑机制还原
│   │   ├── NeuroData/          # 神经数据可视化
│   │   ├── RSAComparison/      # RSA对比可视化
│   │   ├── Validation/         # 验证实验可视化
│   │   ├── index.jsx           # 路线入口
│   │   └── README.md           # 使用说明
│   │
│   ├── route3_fiber_net/       # 路线3: 纤维丛神经网络
│   │   ├── FiberBundle/        # 纤维丛可视化
│   │   ├── Manifold/           # 流形可视化
│   │   ├── Holonomy/           # 和乐群可视化
│   │   ├── EnergyEfficiency/   # 能效可视化
│   │   ├── index.jsx           # 路线入口
│   │   └── README.md           # 使用说明
│   │
│   └── overview/               # 总览面板
│       ├── ProgressDashboard/  # 进展仪表盘
│       ├── RoadmapView/        # 路线图视图
│       └── index.jsx
│
├── shared/                     # 共享组件
│   ├── 3d/                    # 3D可视化基础
│   │   ├── Scene.jsx
│   │   ├── Camera.jsx
│   │   ├── Controls.jsx
│   │   └── Lighting.jsx
│   │
│   ├── charts/                # 图表组件
│   │   ├── LineChart.jsx
│   │   ├── BarChart.jsx
│   │   ├── Heatmap.jsx
│   │   └── RadarChart.jsx
│   │
│   ├── layout/                # 布局组件
│   │   ├── Panel.jsx
│   │   ├── TabView.jsx
│   │   └── GridLayout.jsx
│   │
│   └── data/                  # 数据处理
│       ├── DataLoader.jsx
│       ├── DataTransform.jsx
│       └── hooks/
│
├── services/                  # 后端服务接口
│   ├── api/
│   │   ├── dnnAnalysis.js     # DNN分析API
│   │   ├── brainMechanism.js  # 大脑机制API
│   │   └── fiberNet.js        # 纤维丛API
│   │
│   └── websocket/
│       └── realTime.js        # 实时数据
│
├── config/                    # 配置
│   ├── routes.js              # 路由配置
│   ├── theme.js               # 主题配置
│   └── constants.js           # 常量
│
├── App.jsx                    # 主应用
└── index.jsx                  # 入口
```

---

## 路线1: DNN结构分析可视化

### 组件设计

```
route1_dnn_analysis/
│
├── FeatureExtraction/
│   ├── SAEVisualization.jsx      # SAE特征可视化
│   ├── ActivationHeatmap.jsx     # 激活热力图
│   ├── FeatureMatrix.jsx         # 特征矩阵
│   └── index.jsx
│
├── FourProperties/
│   ├── AbstractionView.jsx       # 高维抽象可视化
│   ├── PrecisionView.jsx         # 低维精确可视化
│   ├── SpecificityView.jsx       # 特异性可视化
│   ├── SystematicityView.jsx     # 系统性可视化
│   ├── RadarChart.jsx            # 雷达图总览
│   └── index.jsx
│
├── SparsityAnalysis/
│   ├── SparsityHeatmap.jsx       # 稀疏度热力图
│   ├── GiniChart.jsx             # Gini系数图表
│   ├── LayerSparsity.jsx         # 层级稀疏度
│   └── index.jsx
│
├── LayerEvolution/
│   ├── LayerComparison.jsx       # 层对比视图
│   ├── IDChart.jsx               # 内在维度图表
│   ├── EvolutionTimeline.jsx     # 演化时间线
│   └── index.jsx
│
├── data/                         # 示例数据
│   └── sample_results.json
│
└── README.md
```

### 数据对接

```javascript
// 自动加载最新分析结果
const loadLatestResults = async () => {
  const response = await fetch('/research/1_dnn_analysis/results/latest.json');
  return response.json();
};

// 数据映射
const mapResultsToProps = (results) => ({
  sparsity: results.extraction['11'].sparsity,
  fourProperties: results.evaluation['11'],
  // ...
});
```

---

## 路线2: 大脑机制还原可视化

### 组件设计

```
route2_brain_mechanism/
│
├── NeuroData/
│   ├── fMRIViewer.jsx            # fMRI数据查看器
│   ├── BrainActivation3D.jsx     # 大脑激活3D
│   ├── SingleCellViewer.jsx      # 单细胞数据查看
│   └── index.jsx
│
├── RSAComparison/
│   ├── RSAMatrix.jsx             # RSA矩阵
│   ├── CorrelationView.jsx       # 相关性视图
│   ├── DNNvsBrain.jsx            # DNN-大脑对比
│   └── index.jsx
│
├── Validation/
│   ├── HypothesisCard.jsx        # 假说卡片
│   ├── ExperimentTimeline.jsx    # 实验时间线
│   ├── ValidationResult.jsx      # 验证结果
│   └── index.jsx
│
└── README.md
```

### 与神经科学数据集成

```javascript
// 加载HCP fMRI数据
const loadHCPData = async (taskId) => {
  // 对接HCP API或本地数据
  return await neuroAPI.loadFmriData(taskId);
};

// RSA分析可视化
const RSAVisualization = ({ dnnActivations, brainActivations }) => {
  const rsaMatrix = computeRSA(dnnActivations, brainActivations);
  return <Heatmap data={rsaMatrix} />;
};
```

---

## 路线3: 纤维丛神经网络可视化

### 组件设计

```
route3_fiber_net/
│
├── FiberBundle/
│   ├── BundleVisualization.jsx    # 纤维丛可视化
│   ├── FiberGroup.jsx            # 纤维群
│   ├── ConnectionView.jsx        # 联络/连接
│   └── index.jsx
│
├── Manifold/
│   ├── Manifold3D.jsx            # 流形3D可视化
│   ├── CurvatureView.jsx         # 曲率视图
│   ├── GeodesicPath.jsx          # 测地线
│   └── index.jsx
│
├── Holonomy/
│   ├── HolonomyLoop.jsx          # 和乐群可视化
│   ├── LoopEvolution.jsx         # 环路演化
│   └── index.jsx
│
├── EnergyEfficiency/
│   ├── EnergyChart.jsx           # 能耗图表
│   ├── ComparisonView.jsx        # 架构对比
│   └── index.jsx
│
└── README.md
```

### 利用现有组件

```javascript
// 重构现有纤维丛组件
import { FiberBundleVisualization3D } from '../legacy/StructureAnalysisPanel';
import { ManifoldVisualization3D } from '../legacy/StructureAnalysisPanel';
import { HolonomyLoopVisualizer } from '../legacy/HolonomyLoopVisualizer';

// 封装为新接口
export const FiberBundleView = (props) => (
  <FiberBundleVisualization3D {...props} />
);
```

---

## 总览面板

### 进展仪表盘

```javascript
// routes/overview/ProgressDashboard/index.jsx
const ProgressDashboard = () => {
  return (
    <div className="dashboard">
      {/* 三条路线进度 */}
      <RouteProgress route="1" progress={40} />
      <RouteProgress route="2" progress={10} />
      <RouteProgress route="3" progress={5} />
      
      {/* 最新成果 */}
      <LatestResults />
      
      {/* 关键指标 */}
      <KeyMetrics />
    </div>
  );
};
```

### 路线图视图

```javascript
// routes/overview/RoadmapView/index.jsx
const RoadmapView = () => {
  return (
    <div className="roadmap">
      <Timeline milestones={milestones} />
      <RouteConnections />
      <NextSteps />
    </div>
  );
};
```

---

## 实施步骤

### 第一阶段：重构目录结构（1-2天）

```bash
# 创建新目录
mkdir -p src/routes/{route1_dnn_analysis,route2_brain_mechanism,route3_fiber_net,overview}
mkdir -p src/shared/{3d,charts,layout,data}
mkdir -p src/services/{api,websocket}
mkdir -p src/config

# 迁移现有组件
mv src/components/analysis/* src/routes/route1_dnn_analysis/
mv src/components/observation/* src/routes/route1_dnn_analysis/
```

### 第二阶段：创建路线可视化（3-5天）

**路线1**:
- [ ] 创建四特性评估雷达图
- [ ] 创建稀疏度热力图
- [ ] 创建层级演化图表
- [ ] 对接分析结果数据

**路线2**:
- [ ] 创建RSA矩阵可视化
- [ ] 创建DNN-大脑对比视图
- [ ] 创建验证实验卡片

**路线3**:
- [ ] 重构纤维丛3D可视化
- [ ] 创建能耗对比图表
- [ ] 优化流形可视化

### 第三阶段：集成与测试（2-3天）

- [ ] 更新路由配置
- [ ] 更新API接口
- [ ] 测试数据加载
- [ ] 编写使用文档

---

## 数据对接方案

### 后端API设计

```python
# server/api_routes.py

@app.route('/api/route1/results')
def get_dnn_analysis_results():
    """获取DNN分析结果"""
    results_dir = 'research/1_dnn_analysis/results/'
    latest_file = get_latest_file(results_dir)
    return send_file(latest_file)

@app.route('/api/route2/neurodata')
def get_neuro_data():
    """获取神经科学数据"""
    # 返回fMRI/单细胞数据
    
@app.route('/api/route3/fibernet')
def get_fibernet_status():
    """获取纤维丛网络状态"""
```

### 前端数据加载

```javascript
// services/api/dnnAnalysis.js
export const loadDNNResults = async () => {
  const response = await fetch('/api/route1/results');
  return response.json();
};

// 自动刷新
export const useAutoRefresh = (interval = 30000) => {
  const [results, setResults] = useState(null);
  
  useEffect(() => {
    const fetch = async () => {
      const data = await loadDNNResults();
      setResults(data);
    };
    
    fetch();
    const timer = setInterval(fetch, interval);
    return () => clearInterval(timer);
  }, [interval]);
  
  return results;
};
```

---

## 使用方式

### 启动可视化

```bash
# 启动后端
cd server
python server.py

# 启动前端
cd frontend
npm run dev

# 访问
http://localhost:5173
```

### 查看路线数据

```
1. 打开浏览器，进入总览面板
2. 点击路线卡片（如"路线1: DNN分析"）
3. 查看该路线的可视化组件
4. 点击具体组件查看详细数据
```

---

## 成功标准

```
✓ 三条路线有独立的可视化入口
✓ 可视化直接对接研究结果数据
✓ 总览面板显示整体进展
✓ 组件按路线清晰组织
✓ 有完整的使用文档
```

---

## 后续优化

### 实时更新

- [ ] WebSocket实时推送分析结果
- [ ] 进度条实时更新
- [ ] 新数据到达提示

### 交互增强

- [ ] 3D可视化交互
- [ ] 数据筛选与过滤
- [ ] 导出图表功能

### 移动端适配

- [ ] 响应式布局
- [ ] 移动端简化视图
- [ ] 触控手势支持
