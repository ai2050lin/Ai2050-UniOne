# 可视化项目整理 - 快速实施指南

## 一键开始

### 第一步：创建新目录结构

```bash
cd d:\ai2050\TransformerLens-Project\frontend\src

# 创建路线目录
mkdir routes
mkdir routes\route1_dnn_analysis
mkdir routes\route2_brain_mechanism
mkdir routes\route3_fiber_net
mkdir routes\overview

# 创建共享组件目录
mkdir shared
mkdir shared\3d
mkdir shared\charts
mkdir shared\layout
mkdir shared\data

# 创建服务目录
mkdir services
mkdir services\api
```

---

## 核心组件映射

### 路线1: DNN分析可视化

**目标**: 展示DNN特征提取、四特性评估、稀疏编码分析结果

**现有组件** → **新位置**:

| 现有组件 | 新位置 | 功能 |
|---------|-------|------|
| `components/analysis/*` | `routes/route1_dnn_analysis/` | 分析组件 |
| `components/observation/*` | `routes/route1_dnn_analysis/` | 观察组件 |
| `StructureAnalysisPanel.jsx` | `routes/route1_dnn_analysis/StructureAnalysis/` | 结构分析 |

**需要创建的新组件**:

```
route1_dnn_analysis/
├── FeatureHeatmap.jsx      # 特征热力图
├── FourPropertiesRadar.jsx # 四特性雷达图
├── SparsityChart.jsx       # 稀疏度图表
├── LayerEvolution.jsx      # 层级演化
├── DataLoader.js           # 数据加载器
└── index.jsx               # 入口
```

---

### 路线2: 大脑机制可视化

**目标**: 展示神经科学数据、DNN-大脑对比、验证实验

**需要创建**:

```
route2_brain_mechanism/
├── BrainActivation3D.jsx   # 大脑激活3D
├── RSAMatrix.jsx           # RSA矩阵
├── ComparisonView.jsx      # 对比视图
├── HypothesisCards.jsx     # 假说卡片
└── index.jsx
```

**数据源**:
- HCP fMRI数据
- DNN分析结果
- 验证实验数据

---

### 路线3: 纤维丛网络可视化

**目标**: 展示纤维丛结构、流形、能效分析

**现有组件** → **新位置**:

| 现有组件 | 新位置 |
|---------|-------|
| `FiberNetPanel.jsx` | `routes/route3_fiber_net/FiberBundle/` |
| `FiberNetV2Demo.jsx` | `routes/route3_fiber_net/FiberBundle/` |
| `StructureAnalysisPanel.jsx` (部分) | `routes/route3_fiber_net/Manifold/` |
| `HolonomyLoopVisualizer.jsx` | `routes/route3_fiber_net/Holonomy/` |

---

## 数据对接模板

### 创建数据加载器

```javascript
// shared/data/ResultLoader.js

/**
 * 加载DNN分析结果
 */
export const loadDNNResults = async () => {
  try {
    // 开发环境：从本地JSON加载
    const response = await fetch('/data/results/latest.json');
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Failed to load results:', error);
    return null;
  }
};

/**
 * 加载指定路线的数据
 */
export const loadRouteData = async (routeId) => {
  const routes = {
    1: '/research/1_dnn_analysis/results/latest.json',
    2: '/research/2_brain_mechanism/data/latest.json',
    3: '/research/3_fiber_net/experiments/latest.json'
  };
  
  const response = await fetch(routes[routeId]);
  return response.json();
};
```

### 创建数据Hook

```javascript
// shared/data/hooks/useAnalysisResults.js

import { useState, useEffect } from 'react';
import { loadDNNResults } from '../ResultLoader';

export const useAnalysisResults = () => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetch = async () => {
      try {
        const data = await loadDNNResults();
        setResults(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetch();
  }, []);

  return { results, loading, error };
};
```

---

## 快速创建第一个组件

### 四特性雷达图

```jsx
// routes/route1_dnn_analysis/FourPropertiesRadar.jsx

import { Radar } from 'react-chartjs-2';
import { useAnalysisResults } from '../../shared/data/hooks/useAnalysisResults';

export const FourPropertiesRadar = ({ layerId = '11' }) => {
  const { results, loading, error } = useAnalysisResults();

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!results) return <div>No data</div>;

  const layerResults = results.evaluation[layerId];

  const data = {
    labels: ['Abstraction', 'Precision', 'Specificity', 'Systematicity'],
    datasets: [{
      label: `Layer ${layerId}`,
      data: [
        layerResults.abstraction.ratio,
        layerResults.precision.accuracies['k=8'] * 100,
        layerResults.specificity.orthogonality * 100,
        layerResults.systematicity.accuracy * 100
      ],
      backgroundColor: 'rgba(54, 162, 235, 0.2)',
      borderColor: 'rgba(54, 162, 235, 1)'
    }]
  };

  return (
    <div className="four-properties-radar">
      <h3>Four Properties - Layer {layerId}</h3>
      <Radar data={data} />
    </div>
  );
};
```

---

## 更新App.jsx

### 新的路由结构

```jsx
// App.jsx

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { OverviewDashboard } from './routes/overview';
import { DNNAnalysisRoute } from './routes/route1_dnn_analysis';
import { BrainMechanismRoute } from './routes/route2_brain_mechanism';
import { FiberNetRoute } from './routes/route3_fiber_net';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/overview" />} />
        <Route path="/overview" element={<OverviewDashboard />} />
        <Route path="/route1/*" element={<DNNAnalysisRoute />} />
        <Route path="/route2/*" element={<BrainMechanismRoute />} />
        <Route path="/route3/*" element={<FiberNetRoute />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
```

---

## 后端API快速添加

### 添加数据端点

```python
# server/api_routes.py (新增文件)

from flask import Flask, jsonify, send_file
import os
import glob

app = Flask(__name__)

@app.route('/api/route1/results/latest')
def get_latest_dnn_results():
    """获取最新DNN分析结果"""
    results_dir = 'research/1_dnn_analysis/results/'
    files = glob.glob(f'{results_dir}*.json')
    if not files:
        return jsonify({'error': 'No results found'}), 404
    
    latest_file = max(files, key=os.path.getctime)
    return send_file(latest_file)

@app.route('/api/progress')
def get_progress():
    """获取研究进展"""
    return jsonify({
        'route1': {'progress': 40, 'status': 'active'},
        'route2': {'progress': 10, 'status': 'preparing'},
        'route3': {'progress': 5, 'status': 'planning'}
    })

# 在server.py中导入
# from api_routes import app
```

---

## 测试清单

### 功能测试

```bash
# 1. 启动后端
cd server
python server.py

# 2. 启动前端
cd frontend
npm run dev

# 3. 测试数据加载
curl http://localhost:5001/api/route1/results/latest

# 4. 访问前端
# http://localhost:5173
```

### 组件测试

- [ ] 总览面板显示三条路线
- [ ] 路线1数据加载成功
- [ ] 四特性雷达图显示正确
- [ ] 稀疏度图表显示正确
- [ ] 层级演化图表显示正确

---

## 一键命令汇总

### 创建所有目录

```bash
cd frontend/src && \
mkdir -p routes/{route1_dnn_analysis,route2_brain_mechanism,route3_fiber_net,overview} && \
mkdir -p shared/{3d,charts,layout,data/hooks} && \
mkdir -p services/api
```

### 复制现有组件

```bash
# 路线1
cp -r components/analysis/* routes/route1_dnn_analysis/
cp -r components/observation/* routes/route1_dnn_analysis/

# 路线3
cp components/FiberNetPanel.jsx routes/route3_fiber_net/
cp components/FiberNetV2Demo.jsx routes/route3_fiber_net/
```

---

## 下一步

1. **今天**: 创建目录结构 + 数据加载器
2. **明天**: 创建第一个可视化组件（四特性雷达图）
3. **后天**: 集成到App.jsx + 测试

---

## 获取帮助

- 完整方案: `VISUALIZATION_REORGANIZATION_PLAN.md`
- 项目进展: `research/PROGRESS.md`
- 数据位置: `research/1_dnn_analysis/results/`
