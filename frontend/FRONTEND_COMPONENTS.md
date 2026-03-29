# AGI可视化客户端 - 前端组件清单

## 已创建的React组件

### 核心组件

1. **AGIVisualizationDashboard.jsx** - 主Dashboard组件
   - 整合所有子组件
   - 管理状态传递
   - 布局协调

2. **AGIVisualizationApp.jsx** - 应用入口组件
   - 包裹Dashboard
   - 提供应用级上下文

### 左侧面板组件

3. **DataSourcePanel.jsx** - 数据源管理面板
   - 展示数据源列表
   - 显示数据状态和数量
   - 支持展开/折叠详情
   - 回调：onSelectDataSource

4. **DataPuzzleBrowser.jsx** - 数据拼图浏览面板
   - 展示数据拼图分类
   - 显示子分类和数量
   - 支持展开/折叠
   - 回调：onSelectCategory, onSelectSubcategory

5. **QuickSearchPanel.jsx** - 快速搜索面板
   - 关键词搜索
   - 类型过滤（全部/数据源/拼图）
   - 排序选项（时间/大小/类型）
   - 回调：onSearch

### 主区域组件

6. **MainVisualizationArea.jsx** - 主可视化区域
   - 根据选择的类别加载可视化数据
   - 集成Plotly.js图表
   - 生成示例数据
   - 刷新数据功能
   - Props: selectedDataSource, selectedCategory, selectedSubcategory

### 系统组件

7. **StatusBar.jsx** - 状态栏组件
   - API状态检查（在线/离线）
   - 数据加载状态
   - 内存使用情况
   - 系统状态显示

## 样式文件

1. **AGIDashboard.css** - Dashboard完整样式
   - 响应式设计
   - 渐变色主题
   - 动画效果
   - 加载状态样式
   - 错误和空状态样式

## 入口文件

1. **agi_visualization.jsx** - React应用入口
2. **agi_visualization.html** - HTML入口

## 组件依赖关系

```
AGIVisualizationApp
  └─ AGIVisualizationDashboard
      ├─ DataSourcePanel
      ├─ DataPuzzleBrowser
      ├─ QuickSearchPanel
      ├─ MainVisualizationArea
      └─ StatusBar
```

## 功能特性

### DataSourcePanel
- ✅ 数据源列表展示
- ✅ 状态指示（部分加载/未加载）
- ✅ 展开/折叠详情
- ✅ 显示数据数量和描述

### DataPuzzleBrowser
- ✅ 5大数据拼图分类
- ✅ 子分类展示
- ✅ 展开/折叠功能
- ✅ 选中状态反馈

### QuickSearchPanel
- ✅ 实时搜索
- ✅ 类型过滤
- ✅ 多种排序方式

### MainVisualizationArea
- ✅ 根据类别自动选择可视化类型
- ✅ Plotly.js图表集成
- ✅ 生成示例数据
- ✅ 加载状态显示
- ✅ 错误处理
- ✅ 空状态提示

### StatusBar
- ✅ API健康检查（30秒轮询）
- ✅ 数据加载状态
- ✅ 系统状态显示

## 技术栈

- React 19.2.0
- Plotly.js
- React-Plotly.js
- Axios（HTTP客户端）
- CSS（无框架）

## API调用

### 获取数据源
```javascript
axios.get('http://localhost:8000/api/data-sources')
```

### 获取数据拼图分类
```javascript
axios.get('http://localhost:8000/api/data-puzzle-categories')
```

### 获取可视化数据
```javascript
axios.post('http://localhost:8000/api/visualization/shared-bearing/heatmap', {
  family_type: 'cross_family',
  model: 'deepseek7b'
})
```

### 生成示例数据
```javascript
axios.post('http://localhost:8000/api/visualization/demo-data')
```

## 样式说明

### 颜色主题
- 主色调：#667eea（紫色渐变）
- 背景色：#f5f5f5
- 成功：#4caf50
- 警告：#ff9800
- 错误：#f44336

### 响应式断点
- >1200px：侧边栏350px
- 768px-1200px：侧边栏280px
- <768px：移动端布局

## 下一步开发

- [ ] 集成QuickSearchPanel到Dashboard
- [ ] 添加更多图表类型（3D视图）
- [ ] 实现数据导出功能
- [ ] 添加报告生成功能
- [ ] 实现多视图对比
- [ ] 添加用户自定义标记
- [ ] 性能优化（虚拟滚动、懒加载）
