/**
 * API 配置文件
 * 统一管理所有 API 端点
 */

export const API_CONFIG = {
  // 主服务器端口
  main: 'http://localhost:5001',
  
  // 分析服务器端口 (与主服务器相同)
  analysis: 'http://localhost:5001',
  
  // 训练服务器端口
  training: 'http://localhost:8000',
  
  // 前端开发服务器
  frontend: 'http://localhost:5173'
};

// API 端点定义 - 基于实际后端路由
export const API_ENDPOINTS = {
  // 模型相关
  model: {
    load: `${API_CONFIG.main}/model/load`,
    info: `${API_CONFIG.main}/model/info`,
    generate: `${API_CONFIG.main}/generate`,
  },
  
  // 分析相关 - 使用实际存在的端点
  analysis: {
    attention: `${API_CONFIG.main}/nfb/topology`,  // 使用 topology 作为替代
    mlp: `${API_CONFIG.main}/nfb/topology`,         // 使用 topology 作为替代
    structure: `${API_CONFIG.main}/nfb/topology`,
    topology: `${API_CONFIG.main}/nfb/topology`,
    curvature: `${API_CONFIG.main}/nfb/topology`,   // 使用 topology 作为替代
    flow: `${API_CONFIG.main}/nfb/flux`,
  },
  
  // NFB 相关 - 实际端点
  nfb: {
    topology: `${API_CONFIG.main}/nfb/topology`,
    flux: `${API_CONFIG.main}/nfb/flux`,
    alignment: `${API_CONFIG.main}/nfb/alignment`,
    gwtStatus: `${API_CONFIG.main}/nfb/gwt/status`,
    evolutionStatus: `${API_CONFIG.main}/nfb/evolution/status`,
    multimodalAlign: `${API_CONFIG.main}/nfb/multimodal/align`,
  },
  
  // 训练相关
  training: {
    status: `${API_CONFIG.training}/training/status`,
    metrics: `${API_CONFIG.main}/toy_experiment/metrics`,
    ricci: `${API_CONFIG.main}/toy_experiment/ricci_metrics`,
  },
  
  // AGI 相关
  agi: {
    conscious: `${API_CONFIG.main}/nfb_ra/unified_conscious_field`,
    verify: `${API_CONFIG.main}/agi/verify`,
    test: `${API_CONFIG.main}/agi/test`,
  },
  
  // 结构提取器
  structureExtractor: {
    extract: `${API_CONFIG.main}/structure/extract`,
    compare: `${API_CONFIG.main}/structure/compare`,
  }
};

// 通用 fetch 封装 - 更优雅的错误处理
export async function apiCall(endpoint, options = {}) {
  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  try {
    const response = await fetch(endpoint, { ...defaultOptions, ...options });
    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    // 静默处理错误，让调用者决定如何处理
    // 不使用 console.error 避免污染控制台
    throw error;
  }
}

export default API_CONFIG;
