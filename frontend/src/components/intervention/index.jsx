/**
 * InterventionWorkbench - 干预台主组件
 * 整合激活干预、几何干预、安全干预三种功能
 */
import React from 'react';
import { useWorkbench } from '../WorkbenchLayout';
import { ActivationIntervention } from './ActivationIntervention';
import { GeometricIntervention } from './GeometricIntervention';
import { SafetyIntervention } from './SafetyIntervention';

export function InterventionWorkbench({ modelData, selectedLayer }) {
  const { activeSubSection } = useWorkbench();

  // 根据子导航选择组件
  const renderContent = () => {
    switch (activeSubSection) {
      case 'activation':
        return (
          <ActivationIntervention 
            modelData={modelData} 
            selectedLayer={selectedLayer}
          />
        );
      
      case 'geometric':
        return (
          <GeometricIntervention 
            modelData={modelData} 
            selectedLayer={selectedLayer}
          />
        );
      
      case 'safety':
        return (
          <SafetyIntervention 
            modelData={modelData} 
            selectedLayer={selectedLayer}
          />
        );
      
      default:
        return <ActivationIntervention modelData={modelData} />;
    }
  };

  return (
    <div style={{ width: '100%', height: '100%' }}>
      {renderContent()}
    </div>
  );
}

// 导出所有子组件
export { ActivationIntervention } from './ActivationIntervention';
export { GeometricIntervention } from './GeometricIntervention';
export { SafetyIntervention } from './SafetyIntervention';

export default InterventionWorkbench;
