/**
 * EvaluationWorkbench - 评估台主组件
 * 整合基准测试、几何测试、进度追踪三种功能
 */
import React from 'react';
import { useWorkbench } from '../WorkbenchLayout';
import { BenchmarkView } from './BenchmarkView';
import { GeometricTestView } from './GeometricTestView';
import { ProgressTracker } from './ProgressTracker';

export function EvaluationWorkbench({ modelData, selectedLayer }) {
  const { activeSubSection } = useWorkbench();

  // 根据子导航选择组件
  const renderContent = () => {
    switch (activeSubSection) {
      case 'benchmark':
        return (
          <BenchmarkView 
            modelData={modelData} 
            selectedLayer={selectedLayer}
          />
        );
      
      case 'geometric':
        return (
          <GeometricTestView 
            modelData={modelData} 
            selectedLayer={selectedLayer}
          />
        );
      
      case 'progress':
        return (
          <ProgressTracker 
            modelData={modelData} 
            selectedLayer={selectedLayer}
          />
        );
      
      default:
        return <BenchmarkView modelData={modelData} />;
    }
  };

  return (
    <div style={{ width: '100%', height: '100%' }}>
      {renderContent()}
    </div>
  );
}

// 导出所有子组件
export { BenchmarkView } from './BenchmarkView';
export { GeometricTestView } from './GeometricTestView';
export { ProgressTracker } from './ProgressTracker';

export default EvaluationWorkbench;
