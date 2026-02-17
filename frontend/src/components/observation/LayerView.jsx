/**
 * LayerView - 层级视图
 * 展示神经网络的层级结构
 */
import React, { useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import { MetricCard, MetricGrid } from '../shared/MetricCard';
import { LoadingSpinner } from '../shared/LoadingSpinner';
import { API_ENDPOINTS, apiCall } from '../../config/api';
import { getLayerColor, COLOR_SCHEMES } from '../../utils/colors';

// 3D 层节点组件
function LayerNode({ position, layerIdx, totalLayers, isActive, onClick }) {
  const color = getLayerColor(layerIdx, totalLayers);
  
  return (
    <group position={position} onClick={onClick}>
      {/* 层立方体 */}
      <mesh>
        <boxGeometry args={[2, 0.3, 2]} />
        <meshStandardMaterial 
          color={color}
          emissive={color}
          emissiveIntensity={isActive ? 0.5 : 0.2}
          transparent
          opacity={isActive ? 1 : 0.7}
        />
      </mesh>
      
      {/* 层标签 */}
      <Text
        position={[0, 0.3, 0]}
        fontSize={0.2}
        color="#fff"
        anchorX="center"
        anchorY="bottom"
      >
        Layer {layerIdx}
      </Text>
    </group>
  );
}

// 3D 连接线组件
function LayerConnection({ start, end, color = '#333' }) {
  const points = [start, end];
  const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
  
  return (
    <line geometry={lineGeometry}>
      <lineBasicMaterial color={color} transparent opacity={0.3} />
    </line>
  );
}

// 导入 THREE
import * as THREE from 'three';

// 主组件
export function LayerView({ modelData, onLayerSelect, selectedLayer }) {
  const [layerInfo, setLayerInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 加载层级信息
    const fetchLayerInfo = async () => {
      try {
        const data = await apiCall(API_ENDPOINTS.model.info);
        setLayerInfo(data);
      } catch (error) {
        // 静默使用默认数据
        setLayerInfo({
          n_layers: 12,
          n_heads: 12,
          d_model: 768,
          total_params: 124000000
        });
      } finally {
        setLoading(false);
      }
    };
    
    fetchLayerInfo();
  }, []);

  if (loading) {
    return <LoadingSpinner message="加载层级结构..." />;
  }

  const nLayers = layerInfo?.n_layers || 12;

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex' }}>
      {/* 3D 视图 */}
      <div style={{ flex: 1, position: 'relative' }}>
        <Canvas>
          <PerspectiveCamera makeDefault position={[15, 10, 15]} fov={50} />
          <OrbitControls enableDamping dampingFactor={0.05} />
          
          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={0.8} />
          
          {/* 渲染层级 */}
          {Array.from({ length: nLayers }).map((_, idx) => (
            <LayerNode
              key={idx}
              position={[0, idx * 1.2 - (nLayers * 0.6), 0]}
              layerIdx={idx}
              totalLayers={nLayers}
              isActive={selectedLayer === idx}
              onClick={() => onLayerSelect?.(idx)}
            />
          ))}
          
          {/* 连接线 */}
          {Array.from({ length: nLayers - 1 }).map((_, idx) => (
            <LayerConnection
              key={`conn-${idx}`}
              start={[0, idx * 1.2 - (nLayers * 0.6) + 0.15, 0]}
              end={[0, (idx + 1) * 1.2 - (nLayers * 0.6) - 0.15, 0]}
            />
          ))}
          
          {/* 网格 */}
          <gridHelper args={[20, 20, '#222', '#111']} position={[0, -nLayers * 0.6 - 1, 0]} />
        </Canvas>
        
        {/* 选中层信息面板 */}
        {selectedLayer !== null && (
          <div style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: 'rgba(0,0,0,0.8)',
            borderRadius: '12px',
            padding: '16px',
            minWidth: '200px',
            border: `1px solid ${getLayerColor(selectedLayer, nLayers)}40`
          }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: '14px', color: getLayerColor(selectedLayer, nLayers) }}>
              Layer {selectedLayer}
            </h3>
            <div style={{ fontSize: '12px', color: '#888' }}>
              <div>注意力头: {layerInfo?.n_heads || 12}</div>
              <div>隐藏维度: {layerInfo?.d_model || 768}</div>
            </div>
          </div>
        )}
      </div>

      {/* 右侧信息面板 */}
      <div style={{
        width: '280px',
        background: 'rgba(255,255,255,0.02)',
        borderLeft: '1px solid #333',
        padding: '16px',
        overflowY: 'auto'
      }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', color: '#fff' }}>
          模型信息
        </h3>
        
        <MetricGrid columns={1}>
          <MetricCard title="总层数" value={nLayers} color={COLOR_SCHEMES.primary} />
          <MetricCard title="注意力头" value={layerInfo?.n_heads || 12} color={COLOR_SCHEMES.attention} />
          <MetricCard title="隐藏维度" value={layerInfo?.d_model || 768} color={COLOR_SCHEMES.mlp} />
          <MetricCard 
            title="参数量" 
            value={`${((layerInfo?.total_params || 124000000) / 1e6).toFixed(1)}M`}
            color={COLOR_SCHEMES.accent}
          />
        </MetricGrid>
        
        <div style={{ marginTop: '24px' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: '12px', color: '#666' }}>
            操作说明
          </h3>
          <div style={{ fontSize: '11px', color: '#888', lineHeight: '1.6' }}>
            <p>• 点击层节点查看详情</p>
            <p>• 拖动旋转3D视图</p>
            <p>• 滚轮缩放</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LayerView;
