import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const DataSourcePanel = ({ onSelectDataSource }) => {
  const [dataSources, setDataSources] = useState({});
  const [loading, setLoading] = useState(true);
  const [expandedSource, setExpandedSource] = useState(null);

  useEffect(() => {
    fetchDataSources();
  }, []);

  const fetchDataSources = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/data-sources`);
      setDataSources(response.data);
      setLoading(false);
    } catch (error) {
      console.error('获取数据源失败:', error);
      setLoading(false);
    }
  };

  const toggleExpand = (sourceKey) => {
    setExpandedSource(expandedSource === sourceKey ? null : sourceKey);
  };

  if (loading) {
    return (
      <div className="panel-section">
        <h3>数据源管理</h3>
        <div className="loading">加载中...</div>
      </div>
    );
  }

  return (
    <div className="panel-section panel-compact">
      <h3 className="panel-title">数据源</h3>
      <div className="data-source-list">
        {Object.entries(dataSources).map(([key, source]) => (
          <div
            key={key}
            className={`data-source-item ${expandedSource === key ? 'expanded' : ''}`}
          >
            <div
              className="data-source-header"
              onClick={() => {
                toggleExpand(key);
                onSelectDataSource(key);
              }}
            >
              <span className="data-source-name">{source.name}</span>
              <span className="data-source-badge">{source.count}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DataSourcePanel;
