import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// 3D可视化主界面（炫酷风格）
import App from './App.jsx'
// 新工作台架构备份: import App from './AppNew.jsx'
import './index.css'
import './css/DNNAnalysisControlPanel.css'
import ErrorBoundary from './ErrorBoundary.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
