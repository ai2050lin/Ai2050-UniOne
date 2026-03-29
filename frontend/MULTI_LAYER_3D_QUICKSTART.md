# 多层3D可视化关联机制 - 快速开始指南

## 时间戳
2026年3月28日 01:00

## 前置要求

- Python 3.8+
- Node.js 16+
- 现代浏览器（Chrome、Firefox、Edge等）

## 快速开始

### 1. 安装依赖

#### 后端依赖
```bash
cd tests/codex/api
pip install fastapi uvicorn numpy pydantic
```

#### 前端依赖
```bash
cd frontend
npm install
```

### 2. 启动服务

#### 启动后端API服务
```bash
cd tests/codex/api
python server.py
```

服务将在 `http://localhost:8000` 启动

#### 启动前端开发服务器
```bash
cd frontend
npm run dev
```

前端将在 `http://localhost:5173` 启动（或控制台显示的其他端口）

### 3. 使用3D可视化

#### 步骤1：打开应用
在浏览器中访问前端应用地址

#### 步骤2：切换到3D模式
点击主可视化区域右上角的"切换到3D"按钮

#### 步骤3：选择概念
在出现的下拉框中选择一个概念：
- 苹果 (Apple)
- 香蕉 (Banana)
- 橙子 (Orange)
- 狗 (Dog)
- 猫 (Cat)

#### 步骤4：探索3D场景
- **旋转场景**：按住鼠标左键拖动
- **缩放场景**：滚动鼠标滚轮
- **平移场景**：按住鼠标右键拖动

#### 步骤5：查看详细信息
- **层信息**：点击任意层的半透明立方体边框
- **节点信息**：点击任意节点（球体）
- **关联信息**：查看底部的关联性表格

#### 步骤6：控制显示
- 点击"显示/隐藏流向路径"按钮切换流向路径显示
- 点击"显示/隐藏跨层连接"按钮切换跨层连接显示

## 功能说明

### 多层表示
系统会显示以下5个神经网络层次：

1. **静态编码层**（红色） - 概念的基础静态表示
2. **动态路径层**（青色） - 概念的动态传播路径
3. **结果回收层**（蓝色） - 从激活状态恢复原始概念
4. **传播编码层**（绿色） - 跨层传播的编码信息
5. **语义角色层**（黄色） - 概念的语义角色和关系

### 可视化元素

#### 节点（球体）
- **大小**：表示激活度（越大越活跃）
- **颜色**：表示所属的层
- **发光效果**：基于激活度的发光强度

#### 层内边（线段）
- **颜色**：层特定颜色
- **透明度**：表示连接强度

#### 跨层边（橙色线）
- **颜色**：橙色
- **透明度**：表示跨层关联强度

#### 流向路径（绿色管道）
- **颜色**：绿色
- **形状**：3D管道
- **方向**：表示主要信息流向

### 关联性分析

系统会分析并显示：
- 层间关联强度
- 共享特征数量
- 信息流向（正向、反向、双向）
- 跨层连接详情

## API测试

### 测试单个概念分析
```bash
curl -X POST "http://localhost:8000/api/layer-association/analyze" \
  -H "Content-Type: application/json" \
  -d "apple"
```

### 测试概念比较
```bash
curl -X POST "http://localhost:8000/api/layer-association/compare" \
  -H "Content-Type: application/json" \
  -d '{"concept1": "apple", "concept2": "banana"}'
```

### 运行测试套件
```bash
cd tests/codex
python test_layer_association.py
```

## 常见问题

### Q: 3D场景无法显示？
A: 检查以下几点：
1. 确保后端服务正在运行（访问 http://localhost:8000/api/health）
2. 确保已选择概念
3. 检查浏览器控制台是否有错误信息
4. 尝试刷新页面

### Q: 性能较慢？
A: 可以尝试：
1. 关闭跨层连接显示
2. 关闭流向路径显示
3. 使用性能更好的设备或浏览器

### Q: 无法旋转或缩放？
A: 确保使用鼠标操作：
- 左键拖动：旋转
- 右键拖动：平移
- 滚轮：缩放

### Q: 节点太多看不清？
A: 可以尝试：
1. 缩小视图查看整体结构
2. 放大视图查看单个层
3. 点击层容器查看详细信息

## 扩展功能

### 添加新概念
在 `MainVisualizationArea.jsx` 中的概念选择下拉框添加：
```javascript
<option value="new_concept">新概念 (New Concept)</option>
```

### 添加新的层
在 `layer_association_analyzer.py` 中的 `LAYERS` 字典添加：
```python
'new_layer': {
    'name': '新层名称',
    'description': '层描述',
    'position': [x, y, z],
    'color': '#RRGGBB'
}
```

### 自定义颜色
修改 `layer_association_analyzer.py` 中各层的 `color` 值。

## 性能优化建议

1. **减少节点数量**：修改 `layer_association_analyzer.py` 中的节点生成逻辑
2. **限制跨层连接**：减少 `cross_layer_edges` 的数量
3. **简化渲染**：降低几何体复杂度
4. **使用缓存**：前端缓存已加载的数据

## 下一步

1. 接入真实的神经网络模型数据
2. 优化3D渲染性能
3. 添加更多交互功能
4. 实现时间序列动画
5. 支持多概念同时显示
6. 添加导出功能

## 技术支持

如有问题，请查看：
- 完整文档：`research/gpt5/docs/MULTI_LAYER_3D_ASSOCIATION.md`
- 测试脚本：`tests/codex/test_layer_association.py`
- API文档：访问 `http://localhost:8000/docs`

---

**版本：** v1.0.0
**更新时间：** 2026年3月28日 01:00
