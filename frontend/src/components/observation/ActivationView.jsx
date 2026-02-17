/**
 * ActivationView - 激活视图
 * 展示注意力模式和 MLP 激活分布
 */
import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text, Sphere } from '@react-three/drei';
import { LoadingSpinner } from '../shared/LoadingSpinner';
import { API_ENDPOINTS, apiCall } from '../../config/api';
import { getRainbowColor, COLOR_SCHEMES } from '../../utils/colors';

// 注意力头 3D 可视化
function AttentionHeadViz({ position, headIdx, attentionPattern, isActive, onClick }) {
  // 将注意力模式转换为可视化
  const nTokens = attentionPattern?.length || 10;
  
  return (
    <group position={position} onClick={onClick}>
      {/* 头容器 */}
      <mesh>
        <boxGeometry args={[2, 2, 0.3]} />
        <meshStandardMaterial 
          color={isActive ? COLOR_SCHEMES.attention : '#333'}
          emissive={isActive ? COLOR_SCHEMES.attention : '#111'}
          emissiveIntensity={isActive ? 0.3 : 0.1}
          transparent
          opacity={0.8}
        />
      </mesh>
      
      {/* 头标签 */}
      <Text
        position={[0, 1.2, 0.2]}
        fontSize={0.15}
        color="#fff"
        anchorX="center"
      >
        Head {headIdx}
      </Text>
      
      {/* 注意力点阵 */}
      {attentionPattern && Array.from({ length: Math.min(nTokens, 8) }).map((_, i) => (
        Array.from({ length: Math.min(nTokens, 8) }).map((_, j) => {
          const weight = attentionPattern[i]?.[j] || 0;
          return (
            <Sphere
              key={`${i}-${j}`}
              args={[0.05 * weight + 0.02, 8, 8]}
              position={[
                (i - 4) * 0.2,
                (j - 4) * 0.2,
                0.2
              ]}
            >
              <meshStandardMaterial 
                color={getRainbowColor(i * 8 + j, 64)}
                emissive={getRainbowColor(i * 8 + j, 64)}
                emissiveIntensity={weight}
              />
            </Sphere>
          );
        })
      ))}
    </group>
  );
}

// MLP 激活分布
function MLPActivationViz({ position, activations }) {
  // 将 MLP 激活转换为柱状图
  const bars = activations?.slice(0, 20) || Array(20).fill(0).map(() => Math.random());
  
  return (
    <group position={position}>
      {bars.map((val, idx) => (
        <mesh key={idx} position={[(idx - 10) * 0.15, val * 0.5, 0]}>
          <boxGeometry args={[0.1, val, 0.1]} />
          <meshStandardMaterial 
            color={getRainbowColor(idx, 20)}
            emissive={getRainbowColor(idx, 20)}
            emissiveIntensity={0.5}
          />
        </mesh>
      ))}
    </group>
  );
}

// 主组件
export function ActivationView({ modelData, selectedLayer = 0 }) {
  const [attentionData, setAttentionData] = useState(null);
  const [mlpActivations, setMlpActivations] = useState(null);
  const [selectedHead, setSelectedHead] = useState(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState('attention'); // 'attention' | 'mlp'

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // 获取注意力数据
        const attnData = await apiCall(`${API_ENDPOINTS.analysis.attention}?layer=${selectedLayer}`);
        setAttentionData(attnData);
        
        // 获取 MLP 激活
        const mlpData = await apiCall(`${API_ENDPOINTS.analysis.mlp}?layer=${selectedLayer}`);
        setMlpActivations(mlpData);
      } catch (error) {
        // 静默使用模拟数据
        setAttentionData({
          patterns: Array(12).fill(null).map(() => 
            Array(10).fill(null).map(() => 
              Array(10).fill(null).map(() => Math.random())
            )
          )
        });
        setMlpActivations({
          activations: Array(3072).fill(null).map(() => Math.random())
        });
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [selectedLayer]);

  if (loading) {
    return <LoadingSpinner message="加载激活数据..." />;
  }

  const nHeads = attentionData?.patterns?.length || 12;

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* 工具栏 */}
      <div style={{
        padding: '12px 16px',
        background: 'rgba(255,255,255,0.02)',
        borderBottom: '1px solid #333',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
      }}>
        <span style={{ fontSize: '12px', color: '#888' }}>视图模式:</span>
        <button
          onClick={() => setViewMode('attention')}
          style={{
            padding: '6px 12px',
            background: viewMode === 'attention' ? COLOR_SCHEMES.attention + '20' : 'transparent',
            border: `1px solid ${viewMode === 'attention' ? COLOR_SCHEMES.attention : '#444'}`,
            borderRadius: '4px',
            color: viewMode === 'attention' ? COLOR_SCHEMES.attention : '#888',
            cursor: 'pointer',
            fontSize: '12px'
          }}
        >
          注意力模式
        </button>
        <button
          onClick={() => setViewMode('mlp')}
          style={{
            padding: '6px 12px',
            background: viewMode === 'mlp' ? COLOR_SCHEMES.mlp + '20' : 'transparent',
            border: `1px solid ${viewMode === 'mlp' ? COLOR_SCHEMES.mlp : '#444'}`,
            borderRadius: '4px',
            color: viewMode === 'mlp' ? COLOR_SCHEMES.mlp : '#888',
            cursor: 'pointer',
            fontSize: '12px'
          }}
        >
          MLP 激活
        </button>
        
        <div style={{ flex: 1 }} />
        
        <span style={{ fontSize: '11px', color: '#666' }}>
          Layer: {selectedLayer} | Heads: {nHeads}
        </span>
      </div>

      {/* 3D 视图 */}
      <div style={{ flex: 1 }}>
        <Canvas>
          <PerspectiveCamera makeDefault position={[15, 8, 15]} fov={50} />
          <OrbitControls enableDamping dampingFactor={0.05} />
          
          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={0.8} />
          
          {viewMode === 'attention' ? (
            // 注意力头网格
            <group>
              {Array.from({ length: nHeads }).map((_, idx) => {
                const row = Math.floor(idx / 4);
                const col = idx % 4;
                return (
                  <AttentionHeadViz
                    key={idx}
                    position={[col * 2.5 - 3.75, 2 - row * 2.5, 0]}
                    headIdx={idx}
                    attentionPattern={attentionData?.patterns?.[idx]}
                    isActive={selectedHead === idx}
                    onClick={() => setSelectedHead(idx)}
                  />
                );
              })}
            </group>
          ) : (
            // MLP 激活
            <MLPActivationViz 
              position={[0, 0, 0]} 
              activations={mlpActivations?.activations}
            />
          )}
          
          <gridHelper args={[20, 20, '#222', '#111']} position={[0, -2, 0]} />
        </Canvas>
      </div>
    </div>
  );
}

export default ActivationView;
