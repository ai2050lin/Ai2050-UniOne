/**
 * App.jsx - AGI Research Workbench 主入口
 * 新架构版本
 */
import React, { useState, useEffect } from 'react';
import { WorkbenchLayout, useWorkbench } from './components/WorkbenchLayout';
import { ObservationWorkbench } from './components/observation';
import { AnalysisWorkbench } from './components/analysis';
import { InterventionWorkbench } from './components/intervention';
import { EvaluationWorkbench } from './components/evaluation';
import { LoadingSpinner } from './components/shared/LoadingSpinner';
import { apiCall, API_ENDPOINTS } from './config/api';

// 主内容区组件
function MainContent({ modelData, selectedLayer, setSelectedLayer }) {
  const { activeSection } = useWorkbench();

  switch (activeSection) {
    case 'observe':
      return (
        <ObservationWorkbench 
          modelData={modelData}
          selectedLayer={selectedLayer}
          onLayerSelect={setSelectedLayer}
        />
      );
    
    case 'analyze':
      return (
        <AnalysisWorkbench 
          modelData={modelData}
          selectedLayer={selectedLayer}
        />
      );
    
    case 'intervene':
      return (
        <InterventionWorkbench 
          modelData={modelData}
          selectedLayer={selectedLayer}
        />
      );
    
    case 'evaluate':
      return (
        <EvaluationWorkbench 
          modelData={modelData}
          selectedLayer={selectedLayer}
        />
      );
    
    default:
      return (
        <ObservationWorkbench 
          modelData={modelData}
          selectedLayer={selectedLayer}
          onLayerSelect={setSelectedLayer}
        />
      );
  }
}

// 主应用组件
function App() {
  const [modelData, setModelData] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [loading, setLoading] = useState(true);
  const [modelName, setModelName] = useState('GPT-2');

  useEffect(() => {
    // 初始化模型数据
    const initModel = async () => {
      try {
        const data = await apiCall(API_ENDPOINTS.model.info);
        setModelData(data);
        setModelName(data.name || 'GPT-2');
      } catch (error) {
        // 静默使用默认数据
        setModelData({
          n_layers: 12,
          n_heads: 12,
          d_model: 768,
          total_params: 124000000
        });
      } finally {
        setLoading(false);
      }
    };

    initModel();
  }, []);

  if (loading) {
    return <LoadingSpinner message="初始化 AGI Research Workbench..." fullScreen />;
  }

  return (
    <WorkbenchLayout modelName={modelName}>
      <MainContent 
        modelData={modelData}
        selectedLayer={selectedLayer}
        setSelectedLayer={setSelectedLayer}
      />
    </WorkbenchLayout>
  );
}

export default App;
