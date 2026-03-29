import React, { useState } from 'react';
import { Search, Layers, Brain, Zap, TrendingUp, Settings, ChevronDown, ChevronUp } from 'lucide-react';

/**
 * 简化版控制面板
 * 将常用功能整合，提供快速访问
 */
const SimplifiedControlPanel = ({ onSearch, onModeChange, currentMode }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  // 可视化模式选项
  const visualizationModes = [
    { id: '2d', name: '2D图表', icon: TrendingUp, description: '查看传统图表分析' },
    { id: '3d', name: '3D可视化', icon: Layers, description: '查看多层3D关联' },
    { id: 'neuron', name: '神经元', icon: Brain, description: '查看神经元激活' },
    { id: 'flow', name: '数据流', icon: Zap, description: '查看数据流向' },
  ];

  // 常用概念快捷选择
  const quickConcepts = [
    { id: 'apple', name: '苹果', emoji: '🍎' },
    { id: 'banana', name: '香蕉', emoji: '🍌' },
    { id: 'orange', name: '橙子', emoji: '🍊' },
    { id: 'dog', name: '狗', emoji: '🐕' },
    { id: 'cat', name: '猫', emoji: '🐱' },
  ];

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchTerm.trim()) {
      onSearch(searchTerm.trim());
      setSearchTerm('');
    }
  };

  const handleQuickSelect = (conceptId) => {
    onSearch(conceptId);
  };

  return (
    <div className="simplified-control-panel">
      {/* 搜索栏 */}
      <div className="control-section">
        <div className="section-header">
          <Search className="section-icon" />
          <span className="section-title">快速搜索</span>
        </div>
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            className="search-input"
            placeholder="输入概念（如：apple、dog）..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <button type="submit" className="search-button">
            <Search size={18} />
          </button>
        </form>

        {/* 常用概念快捷选择 */}
        <div className="quick-concepts">
          <div className="quick-concepts-label">快速选择：</div>
          <div className="quick-concepts-grid">
            {quickConcepts.map(concept => (
              <button
                key={concept.id}
                className="quick-concept-button"
                onClick={() => handleQuickSelect(concept.id)}
              >
                <span className="concept-emoji">{concept.emoji}</span>
                <span className="concept-name">{concept.name}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 可视化模式选择 */}
      <div className="control-section">
        <div
          className="section-header clickable"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <Layers className="section-icon" />
          <span className="section-title">可视化模式</span>
          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>

        {isExpanded && (
          <div className="mode-selection">
            {visualizationModes.map(mode => (
              <button
                key={mode.id}
                className={`mode-button ${currentMode === mode.id ? 'active' : ''}`}
                onClick={() => onModeChange(mode.id)}
              >
                <mode.icon size={20} className="mode-icon" />
                <div className="mode-info">
                  <div className="mode-name">{mode.name}</div>
                  <div className="mode-description">{mode.description}</div>
                </div>
                {currentMode === mode.id && (
                  <div className="active-indicator">✓</div>
                )}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* 快速操作 */}
      <div className="control-section">
        <div className="section-header">
          <Zap className="section-icon" />
          <span className="section-title">快速操作</span>
        </div>
        <div className="quick-actions">
          <button className="action-button primary">
            刷新数据
          </button>
          <button className="action-button secondary">
            重置视图
          </button>
          <button className="action-button secondary">
            导出报告
          </button>
        </div>
      </div>
    </div>
  );
};

export default SimplifiedControlPanel;
