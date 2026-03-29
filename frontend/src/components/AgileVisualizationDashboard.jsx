import React, { useState } from 'react';
import SimplifiedControlPanel from './SimplifiedControlPanel';
import MainVisualizationArea from './MainVisualizationArea';
import StatusBar from './StatusBar';
import { Menu, X, Layers, Activity } from 'lucide-react';

/**
 * 敏捷可视化仪表板
 * 简化版的AGI数据可视化界面，提供快速访问常用功能
 */
const AgileVisualizationDashboard = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [currentMode, setCurrentMode] = useState('2d');
  const [searchQuery, setSearchQuery] = useState(null);

  const handleSearch = (term) => {
    setSearchQuery({ term, type: 'all', sortBy: 'time' });
    // 根据搜索内容自动切换到合适的模式
    if (['apple', 'banana', 'orange', 'dog', 'cat'].includes(term.toLowerCase())) {
      setCurrentMode('3d');
    } else {
      setCurrentMode('2d');
    }
  };

  const handleModeChange = (mode) => {
    setCurrentMode(mode);
  };

  return (
    <div className="agile-dashboard">
      {/* 顶部导航栏 */}
      <div className="dashboard-header">
        <div className="header-left">
          <button
            className="menu-toggle"
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            title={isSidebarOpen ? '收起侧边栏' : '展开侧边栏'}
          >
            {isSidebarOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
          <div className="header-title">
            <h1>AGI数据可视化</h1>
            <p className="header-subtitle">快速访问 | 简洁高效</p>
          </div>
        </div>
        <div className="header-right">
          <div className="mode-indicator">
            <Layers size={16} className="mode-icon" />
            <span>当前模式: {currentMode === '2d' ? '2D图表' : '3D可视化'}</span>
          </div>
        </div>
      </div>

      {/* 主内容区域 */}
      <div className="dashboard-content">
        {/* 侧边控制面板 */}
        {isSidebarOpen && (
          <div className="dashboard-sidebar">
            <SimplifiedControlPanel
              onSearch={handleSearch}
              onModeChange={handleModeChange}
              currentMode={currentMode}
            />
          </div>
        )}

        {/* 主可视化区域 */}
        <div className="dashboard-main">
          <MainVisualizationArea
            selectedDataSource={null}
            selectedCategory={null}
            selectedSubcategory={null}
            searchQuery={searchQuery}
            currentMode={currentMode}
          />
        </div>
      </div>

      {/* 状态栏 */}
      <StatusBar />
    </div>
  );
};

export default AgileVisualizationDashboard;
