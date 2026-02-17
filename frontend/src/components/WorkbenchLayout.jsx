/**
 * WorkbenchLayout - AGI 研究工作台布局
 * 提供统一的导航和布局框架
 */
import React, { useState, createContext, useContext } from 'react';
import { COLOR_SCHEMES } from '../utils/colors';

// 工作台上下文
export const WorkbenchContext = createContext(null);

export function useWorkbench() {
  const context = useContext(WorkbenchContext);
  if (!context) {
    throw new Error('useWorkbench must be used within WorkbenchLayout');
  }
  return context;
}

// 导航配置
const NAV_ITEMS = [
  { 
    id: 'observe', 
    label: '观察台', 
    labelEn: 'Observe',
    icon: '🔍',
    description: '神经网络结构观察',
    subItems: [
      { id: 'layers', label: '层级视图', labelEn: 'Layers' },
      { id: 'activations', label: '激活视图', labelEn: 'Activations' },
      { id: 'geometry', label: '几何视图', labelEn: 'Geometry' }
    ]
  },
  { 
    id: 'analyze', 
    label: '分析台', 
    labelEn: 'Analyze',
    icon: '📊',
    description: '结构分析与提取',
    subItems: [
      { id: 'extract', label: '结构提取', labelEn: 'Extract' },
      { id: 'compare', label: '对比分析', labelEn: 'Compare' },
      { id: 'correlate', label: '关联分析', labelEn: 'Correlate' }
    ]
  },
  { 
    id: 'intervene', 
    label: '干预台', 
    labelEn: 'Intervene',
    icon: '🔧',
    description: '神经网络干预实验',
    subItems: [
      { id: 'activation', label: '激活干预', labelEn: 'Activation' },
      { id: 'geometric', label: '几何干预', labelEn: 'Geometric' },
      { id: 'safety', label: '安全干预', labelEn: 'Safety' }
    ]
  },
  { 
    id: 'evaluate', 
    label: '评估台', 
    labelEn: 'Evaluate',
    icon: '📈',
    description: 'AGI 能力评估',
    subItems: [
      { id: 'benchmark', label: '基准测试', labelEn: 'Benchmark' },
      { id: 'geometric', label: '几何测试', labelEn: 'Geometric' },
      { id: 'progress', label: '进度追踪', labelEn: 'Progress' }
    ]
  }
];

// 状态栏组件
function StatusBar({ model, layer, gpuUsage, latency }) {
  return (
    <div style={{
      height: '32px',
      background: 'rgba(0,0,0,0.3)',
      borderTop: '1px solid #333',
      display: 'flex',
      alignItems: 'center',
      padding: '0 16px',
      fontSize: '12px',
      color: '#888',
      gap: '24px'
    }}>
      <span>模型: <span style={{ color: COLOR_SCHEMES.primary }}>{model || 'GPT-2'}</span></span>
      <span>层: <span style={{ color: '#fff' }}>{layer || '0/12'}</span></span>
      <span>GPU: <span style={{ color: gpuUsage > 80 ? '#ef4444' : '#10b981' }}>{gpuUsage || 0}%</span></span>
      <span>延迟: <span style={{ color: '#fff' }}>{latency || '0'}ms</span></span>
      <div style={{ flex: 1 }} />
      <span style={{ color: '#666' }}>AGI Research Workbench v1.0</span>
    </div>
  );
}

// 主布局组件
export function WorkbenchLayout({ children, modelName = 'GPT-2' }) {
  const [activeSection, setActiveSection] = useState('observe');
  const [activeSubSection, setActiveSubSection] = useState('layers');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const contextValue = {
    activeSection,
    setActiveSection,
    activeSubSection,
    setActiveSubSection,
    modelName
  };

  const activeNav = NAV_ITEMS.find(n => n.id === activeSection);

  return (
    <WorkbenchContext.Provider value={contextValue}>
      <div style={{
        width: '100%',
        height: '100vh',
        background: COLOR_SCHEMES.background,
        color: '#fff',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden'
      }}>
        {/* 顶部导航栏 */}
        <header style={{
          height: '48px',
          background: 'rgba(255,255,255,0.02)',
          borderBottom: '1px solid #333',
          display: 'flex',
          alignItems: 'center',
          padding: '0 16px',
          gap: '8px'
        }}>
          {/* Logo */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            marginRight: '32px'
          }}>
            <span style={{ fontSize: '20px' }}>🧠</span>
            <span style={{ 
              fontWeight: 'bold', 
              fontSize: '14px',
              background: 'linear-gradient(45deg, #00d2ff, #3a7bd5)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}>
              AGI Research Workbench
            </span>
          </div>

          {/* 主导航 */}
          <nav style={{ display: 'flex', gap: '4px' }}>
            {NAV_ITEMS.map(item => (
              <button
                key={item.id}
                onClick={() => {
                  setActiveSection(item.id);
                  setActiveSubSection(item.subItems[0].id);
                }}
                style={{
                  padding: '8px 16px',
                  background: activeSection === item.id ? 'rgba(0, 210, 255, 0.1)' : 'transparent',
                  border: 'none',
                  borderRadius: '6px',
                  color: activeSection === item.id ? COLOR_SCHEMES.primary : '#888',
                  cursor: 'pointer',
                  fontSize: '13px',
                  fontWeight: activeSection === item.id ? '600' : '400',
                  transition: 'all 0.2s',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}
              >
                <span>{item.icon}</span>
                <span>{item.label}</span>
              </button>
            ))}
          </nav>

          <div style={{ flex: 1 }} />

          {/* 右侧工具 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <select style={{
              background: '#222',
              border: '1px solid #444',
              borderRadius: '4px',
              padding: '4px 8px',
              color: '#fff',
              fontSize: '12px'
            }}>
              <option>GPT-2</option>
              <option>Qwen3</option>
            </select>
          </div>
        </header>

        {/* 主内容区 */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {/* 侧边栏 */}
          <aside style={{
            width: sidebarCollapsed ? '48px' : '200px',
            background: 'rgba(255,255,255,0.02)',
            borderRight: '1px solid #333',
            transition: 'width 0.2s',
            display: 'flex',
            flexDirection: 'column'
          }}>
            {/* 折叠按钮 */}
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              style={{
                padding: '12px',
                background: 'transparent',
                border: 'none',
                color: '#888',
                cursor: 'pointer',
                textAlign: 'right'
              }}
            >
              {sidebarCollapsed ? '▶' : '◀'}
            </button>

            {/* 子导航 */}
            {activeNav && !sidebarCollapsed && (
              <div style={{ padding: '8px' }}>
                <div style={{
                  fontSize: '11px',
                  color: '#666',
                  padding: '8px',
                  textTransform: 'uppercase',
                  letterSpacing: '1px'
                }}>
                  {activeNav.description}
                </div>
                {activeNav.subItems.map(subItem => (
                  <button
                    key={subItem.id}
                    onClick={() => setActiveSubSection(subItem.id)}
                    style={{
                      width: '100%',
                      padding: '10px 12px',
                      background: activeSubSection === subItem.id 
                        ? 'rgba(0, 210, 255, 0.1)' 
                        : 'transparent',
                      border: 'none',
                      borderRadius: '6px',
                      color: activeSubSection === subItem.id ? COLOR_SCHEMES.primary : '#888',
                      cursor: 'pointer',
                      fontSize: '13px',
                      textAlign: 'left',
                      marginBottom: '4px',
                      transition: 'all 0.2s'
                    }}
                  >
                    {subItem.label}
                  </button>
                ))}
              </div>
            )}
          </aside>

          {/* 内容区 */}
          <main style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
            {children}
          </main>
        </div>

        {/* 状态栏 */}
        <StatusBar model={modelName} gpuUsage={35} latency={120} />
      </div>
    </WorkbenchContext.Provider>
  );
}

export default WorkbenchLayout;
