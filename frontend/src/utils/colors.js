/**
 * 颜色工具函数
 * 统一管理颜色生成和转换
 */

/**
 * 根据熵值生成颜色 (低熵=蓝色, 高熵=红色)
 * @param {number} value - 熵值 (0-6)
 * @returns {string} HSL颜色字符串
 */
export function getEntropyColor(value) {
  const norm = Math.min(value / 6, 1.0);
  const hue = 240 * (1 - norm);
  return `hsl(${hue}, 80%, 50%)`;
}

/**
 * 根据值生成渐变颜色
 * @param {number} value - 归一化值 (0-1)
 * @param {string} lowColor - 低值颜色
 * @param {string} highColor - 高值颜色
 * @returns {string} 颜色字符串
 */
export function getGradientColor(value, lowColor = '#3b82f6', highColor = '#ef4444') {
  const hue = (1 - value) * 240; // 从蓝到红
  return `hsl(${hue}, 80%, 50%)`;
}

/**
 * 根据索引生成彩虹颜色
 * @param {number} index - 索引
 * @param {number} total - 总数
 * @returns {string} HSL颜色字符串
 */
export function getRainbowColor(index, total) {
  const hue = (index / total) * 360;
  return `hsl(${hue}, 70%, 50%)`;
}

/**
 * 预定义颜色方案
 */
export const COLOR_SCHEMES = {
  // 主题色
  primary: '#00d2ff',
  secondary: '#3a7bd5',
  accent: '#ffaa00',
  danger: '#ff4444',
  success: '#10b981',
  warning: '#f59e0b',
  
  // 神经网络相关
  attention: '#4488ff',
  mlp: '#44ff88',
  residual: '#ff8844',
  embedding: '#ff44ff',
  
  // 几何相关
  manifold: '#00ffff',
  fiber: '#ff88ff',
  curvature: '#ffff00',
  geodesic: '#88ff88',
  
  // 状态相关
  active: '#00ff00',
  inactive: '#666666',
  error: '#ff0000',
  loading: '#ffaa00',
  
  // 背景色
  background: '#0a0a0c',
  surface: '#111111',
  surfaceLight: '#1a1a1a',
  border: '#333333',
};

/**
 * 根据层索引获取颜色
 * @param {number} layerIdx - 层索引
 * @param {number} totalLayers - 总层数
 * @returns {string} 颜色字符串
 */
export function getLayerColor(layerIdx, totalLayers = 12) {
  const colors = [
    '#ff6b6b', '#ffa502', '#ffd93d', '#6bcb77',
    '#4d96ff', '#9b59b6', '#e74c3c', '#3498db',
    '#1abc9c', '#f39c12', '#2ecc71', '#9b59b6'
  ];
  return colors[layerIdx % colors.length];
}

/**
 * 带透明度的颜色
 * @param {string} color - 基础颜色
 * @param {number} alpha - 透明度 (0-1)
 * @returns {string} rgba颜色字符串
 */
export function withAlpha(color, alpha) {
  // 转换 hex 到 rgb
  if (color.startsWith('#')) {
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  // 如果已经是 hsl/rgb，添加透明度
  if (color.startsWith('hsl')) {
    return color.replace('hsl', 'hsla').replace(')', `, ${alpha})`);
  }
  if (color.startsWith('rgb(')) {
    return color.replace('rgb', 'rgba').replace(')', `, ${alpha})`);
  }
  return color;
}

export default {
  getEntropyColor,
  getGradientColor,
  getRainbowColor,
  getLayerColor,
  withAlpha,
  COLOR_SCHEMES
};
