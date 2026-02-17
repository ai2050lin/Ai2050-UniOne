/**
 * ObservationWorkbench - 观察台主组件
 * 整合层级、激活、几何三种视图
 */
import React from 'react';
import { useWorkbench } from '../WorkbenchLayout';
import { LayerView } from './LayerView';
import { ActivationView } from './ActivationView';
import { GeometryView } from './GeometryView';

export function ObservationWorkbench({ modelData, onLayerSelect, selectedLayer }) {
  const { activeSubSection } = useWorkbench();

  // 根据子导航选择组件
  const renderContent = () => {
    switch (activeSubSection) {
      case 'layers':
        return (
          <LayerView 
            modelData={modelData} 
            onLayerSelect={onLayerSelect}
            selectedLayer={selectedLayer}
          />
        );
      
      case 'activations':
        return (
          <ActivationView 
            modelData={modelData} 
            selectedLayer={selectedLayer}
          />
        );
      
      case 'geometry':
        return (
          <GeometryView 
            modelData={modelData} 
            selectedLayer={selectedLayer}
          />
        );
      
      default:
        return <LayerView modelData={modelData} />;
    }
  };

  return (
    <div style={{ width: '100%', height: '100%' }}>
      {renderContent()}
    </div>
  );
}

// 导出所有子组件
export { LayerView } from './LayerView';
export { ActivationView } from './ActivationView';
export { GeometryView } from './GeometryView';

export default ObservationWorkbench;
