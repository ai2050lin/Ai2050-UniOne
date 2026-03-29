# AGI可视化客户端安装和启动指南

## 系统要求

- Node.js 16+ 和 npm
- Python 3.8+
- pip

## 安装步骤

### 1. 安装Python依赖

```bash
cd d:/develop/TransformerLens-main
pip install fastapi uvicorn numpy pydantic
```

### 2. 启动API服务

```bash
cd tests/codex/api
python server.py
```

服务将在 http://localhost:8000 启动

### 3. 访问API文档

在浏览器中打开：http://localhost:8000/docs

### 4. 生成示例数据

在API文档页面中找到：
- POST `/api/visualization/demo-data`
- 点击 "Try it out"
- 点击 "Execute"

或使用curl：

```bash
curl -X POST "http://localhost:8000/api/visualization/demo-data"
```

### 5. 安装前端依赖

```bash
cd d:/develop/TransformerLens-main/frontend
npm install plotly.js react-plotly.js axios
```

### 6. 启动前端开发服务器

```bash
npm run dev
```

### 7. 访问可视化客户端

在浏览器中打开：
- http://localhost:5173/agi_visualization.html

## 功能说明

### 左侧面板

1. **数据源管理**
   - DNN参数激活（10,000+概念）
   - 跨模型对比（3个模型）
   - 因果干预（100个参数）
   - 脑桥接（fMRI/EEG）
   - 跨模态（图像/文本/音频）

2. **数据拼图浏览**
   - 共享承载机制（345条）
   - 偏置偏转机制（278条）
   - 逐层放大机制（156条）
   - 多空间映射（234条）
   - 跨模型验证（89条）

3. **快速搜索**
   - 关键词搜索
   - 类型过滤
   - 排序功能

### 主可视化区域

- **热图**：展示激活模式
- **散点图**：展示参数间关系
- **柱状图**：跨模型对比
- **折线图**：时间演化轨迹
- **网络图**：参数连接关系

### 交互控制

- **生成示例数据**：快速生成测试数据
- **刷新数据**：重新加载当前视图
- **缩放/平移**：Plotly.js内置交互
- **导出图表**：支持多种格式

## API接口

### 数据源管理

- `GET /api/data-sources` - 获取数据源列表
- `GET /api/data-puzzle-categories` - 获取数据拼图分类

### 可视化接口

- `POST /api/visualization/shared-bearing/heatmap` - 共享承载热图
- `POST /api/visualization/shared-bearing/scatter` - 承载机制散点图
- `POST /api/visualization/shared-bearing/network` - 承载关系网络图
- `POST /api/visualization/cross-model/comparison` - 跨模型对比
- `POST /api/visualization/temporal/trajectory` - 时间演化轨迹
- `POST /api/visualization/intervention/result` - 干预结果
- `POST /api/visualization/demo-data` - 生成示例数据
- `GET /api/visualization/chart-types` - 支持的图表类型

### 系统接口

- `GET /api/health` - 健康检查

## 数据积累流程

### 1. 运行测试脚本生成数据

```bash
cd tests/codex
python stage294_cross_family_shared.py
python stage295_bias_deflection.py
# ... 其他测试脚本
```

### 2. 查看可视化结果

- 打开可视化客户端
- 选择对应的数据拼图类别
- 查看生成的图表

### 3. 积累数据拼图

持续运行测试脚本，扩大数据覆盖范围：
- 扫描更多概念（目标：10,000+）
- 增加模型数量（目标：3+）
- 追踪时间演化（目标：10+checkpoint）

## 故障排除

### API服务无法启动

1. 检查端口8000是否被占用：
```bash
netstat -ano | findstr :8000
```

2. 检查Python依赖是否安装：
```bash
pip list | findstr fastapi
```

### 前端无法连接API

1. 确认API服务已启动
2. 检查浏览器控制台错误
3. 确认CORS配置正确

### 数据加载失败

1. 先生成示例数据
2. 检查数据目录权限
3. 查看API服务日志

## 下一步开发

- [ ] 集成真实测试数据（Stage294-298）
- [ ] 添加更多图表类型（3D视图等）
- [ ] 实现数据导出功能
- [ ] 添加报告生成功能
- [ ] 优化性能（分页加载、虚拟滚动）
- [ ] 添加用户自定义标记功能
- [ ] 实现多视图对比

## 理论研究支持

该可视化客户端支持以下研究方向：

1. **共享承载机制研究**
   - Stage294-298数据可视化
   - 跨家族/家族共享分析
   - 参数承载关系探索

2. **跨模型同构验证**
   - GPT-2、Qwen3、DeepSeek-7B对比
   - Stage141、150、159数据可视化
   - 模型间差异分析

3. **因果干预研究**
   - 参数消融效果可视化
   - 干预前后性能对比
   - 因果效应量化

4. **时间演化分析**
   - 概念在不同checkpoint的演化
   - 稳定性分析
   - 演化模式识别

## 避免预设理论

在使用可视化客户端时，请遵循以下原则：

1. **不要预设理论解释**
2. **使用中性、描述性的标签**
3. **让数据自己说话**
4. **保持开放态度，接受多种解释**
5. **诚实报告未知域**

## 联系方式

如有问题，请查看项目文档或联系开发团队。
