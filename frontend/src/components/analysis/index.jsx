/**
 * AnalysisWorkbench - 分析台主组件
 * 整合结构提取、对比分析、关联分析三种功能
 */
import React from 'react';
import { useWorkbench } from '../WorkbenchLayout';
import { StructureExtractView } from './StructureExtractView';
import { CompareAnalysisView } from './CompareAnalysisView';
import { CorrelationView } from './CorrelationView';

export function AnalysisWorkbench({ modelData, selectedLayer }) {
  const { activeSubSection } = useWorkbench();

  // 根据子导航选择组件
  const renderContent = () => {
    switch (activeSubSection) {
      case 'extract':
        return (
          <StructureExtractView 
            modelData={modelData} 
            selectedLayer={selectedLayer}
          />
        );
      
      case 'compare':
        return (
          <CompareAnalysisView 
            modelData={modelData} 
            selectedLayer={selectedLayer}
          />
        );
      
      case 'correlate':
        return (
          <CorrelationView 
            modelData={modelData} 
            selectedLayer={selectedLayer}
          />
        );
      
      default:
        return <StructureExtractView modelData={modelData} />;
    }
  };

  return (
    <div style={{ width: '100%', height: '100%' }}>
      {renderContent()}
    </div>
  );
}

// 导出所有子组件
export { StructureExtractView } from './StructureExtractView';
export { CompareAnalysisView } from './CompareAnalysisView';
export { CorrelationView } from './CorrelationView';

export default AnalysisWorkbench;
