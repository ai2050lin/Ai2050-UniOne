# 前端客户端快速启动指南

## 问题诊断

### 原始问题
"为什么客户端无法启动"

### 问题分析
1. **端口被占用** - 之前有其他进程占用了5173端口（PID 6520）
2. **启动无输出** - Vite在后台启动，没有显示端口信息

### 解决方案
1. 终止占用端口的进程：`taskkill /F /PID 6520`
2. 重新启动服务：`cd frontend && npm run dev`
3. 验证服务状态：访问 http://localhost:5173

## 快速启动步骤

### 方法1：标准启动（推荐）

```bash
# 1. 进入前端目录
cd d:\develop\TransformerLens-main\frontend

# 2. 启动开发服务器
npm run dev
```

启动后，Vite会自动在 http://localhost:5173 上运行。

### 方法2：使用PowerShell（Windows）

```powershell
# 1. 检查端口是否被占用
netstat -ano | findstr ":5173"

# 2. 如果被占用，终止进程
taskkill /F /PID <进程ID>

# 3. 启动服务
cd d:\develop\TransformerLens-main\frontend
npm run dev
```

### 方法3：一键启动（推荐）

创建一个启动脚本 `start-frontend.ps1`:

```powershell
# 进入前端目录
cd $PSScriptRoot\..\frontend

# 检查端口占用
$portProcess = netstat -ano | Select-String ":5173" | Select-String "LISTENING"
if ($portProcess) {
    $pid = $portProcess.ToString().Split()[-1]
    Write-Host "端口5173已被占用 (PID: $pid)，正在终止..." -ForegroundColor Yellow
    taskkill /F /PID $pid | Out-Null
    Start-Sleep -Seconds 1
}

# 启动前端服务
Write-Host "启动前端服务..." -ForegroundColor Green
npm run dev
```

## 环境要求

### 必需的软件
- **Node.js**: v16.0.0 或更高版本
- **npm**: 8.0.0 或更高版本

### 当前环境
- Node.js: v22.14.0 ✅
- npm: 10.9.2 ✅

### 安装依赖（如果需要）

```bash
cd frontend
npm install
```

## 配置说明

### Vite配置 (vite.config.js)
```javascript
export default defineConfig({
  root: '.',
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',  // 允许外部访问
    port: 5173,        // 默认端口
  }
})
```

### 更改端口（如果5173被占用）

编辑 `frontend/vite.config.js`，修改端口号：

```javascript
server: {
  host: '0.0.0.0',
  port: 3000,  // 改为其他端口
}
```

## 访问应用

启动成功后，在浏览器中打开：

- **本地访问**: http://localhost:5173
- **局域网访问**: http://<你的IP地址>:5173

### 获取本地IP地址
```bash
ipconfig
```
找到"IPv4 地址"对应的IP。

## 常见问题排查

### 1. 端口被占用

**症状**: 运行 `npm run dev` 没有任何输出

**解决**:
```bash
# 查找占用端口的进程
netstat -ano | findstr ":5173"

# 终止进程
taskkill /F /PID <进程ID>
```

### 2. 依赖未安装

**症状**: 启动时报错找不到模块

**解决**:
```bash
cd frontend
npm install
```

### 3. Node.js版本过低

**症状**: 启动时报错 "Invalid Vite version"

**解决**:
从 https://nodejs.org/ 下载最新LTS版本

### 4. 无法访问

**症状**: 服务已启动但浏览器无法打开

**解决**:
- 检查防火墙设置
- 确认服务正在运行（`netstat -ano | findstr ":5173"`）
- 尝试访问 127.0.0.1:5173 而不是 localhost:5173

### 5. 热重载不工作

**症状**: 修改代码后页面不自动刷新

**解决**:
- 清除浏览器缓存
- 重启开发服务器
- 检查Vite配置

## 验证安装

运行以下命令验证环境：

```bash
# 检查Node.js版本
node --version

# 检查npm版本
npm --version

# 检查已安装的包
cd frontend
npm list

# 测试服务是否可访问
curl http://localhost:5173
```

## 构建生产版本

```bash
cd frontend
npm run build
```

构建产物会生成在 `dist` 目录。

## 预览生产版本

```bash
cd frontend
npm run preview
```

## 开发技巧

### 1. 使用多个终端

- 终端1: `npm run dev`（前端）
- 终端2: `npm run server`（后端，如果有）

### 2. 查看详细日志

```bash
# 设置日志级别
DEBUG=vite:* npm run dev
```

### 3. 清除缓存

```bash
# 清除Vite缓存
rm -rf node_modules/.vite

# 重新安装依赖
rm -rf node_modules package-lock.json
npm install
```

## 项目结构

```
frontend/
├── index.html              # HTML入口
├── package.json            # 项目配置
├── vite.config.js          # Vite配置
├── src/
│   ├── main.jsx            # React入口
│   ├── App.jsx             # 主应用组件
│   ├── components/         # 组件目录
│   └── css/               # 样式文件
└── dist/                   # 构建输出（运行npm run build后生成）
```

## 性能优化

### 1. 减少初始加载时间
- 使用代码分割
- 懒加载路由
- 压缩资源

### 2. 开发环境优化
- 使用Vite的快速HMR
- 禁用不必要的插件
- 使用更快的source map

## 总结

**当前状态**: ✅ 前端已成功运行在 http://localhost:5173

**验证方法**:
1. 打开浏览器访问 http://localhost:5173
2. 应该能看到应用界面
3. 打开浏览器控制台（F12），检查是否有错误

**下一步**:
- 探索DNN分析3D可视化功能
- 运行分析任务
- 测试3D视图切换功能

---

**注意**: 如果启动后仍然无法访问，请检查：
1. 浏览器控制台的错误信息
2. 终端的日志输出
3. 防火墙和网络设置
