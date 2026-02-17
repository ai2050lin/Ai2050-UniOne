/**
 * main_new.jsx - 新架构入口
 * 使用新的 Workbench 架构
 */
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './AppNew.jsx'
import './index.css'
import ErrorBoundary from './ErrorBoundary.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
