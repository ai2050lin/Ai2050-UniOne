/**
 * NeuralVis3D 独立入口
 * 访问 /neural-vis.html 直接打开3D可视化
 */
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import NeuralVis3DApp from './index.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <NeuralVis3DApp />
  </StrictMode>
);
