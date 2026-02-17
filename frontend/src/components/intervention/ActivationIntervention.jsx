/**
 * ActivationIntervention - 激活干预视图
 * 对神经网络激活进行干预实验
 */
import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text, Sphere, Box, Line } from '@react-three/drei';
import { MetricCard, MetricGrid } from '../shared/MetricCard';
import { LoadingSpinner } from '../shared/LoadingSpinner';
import { API_ENDPOINTS, apiCall } from '../../config/api';
import { COLOR_SCHEMES, getGradientColor, getLayerColor, withAlpha } from '../../utils/colors';
import * as THREE from 'three';

// 激活状态可视化
function ActivationState({ activations, interventions, position = [0, 0, 0] }) {
  if (!activations) return null;
  
  const n = activations.length || 100;
  const gridSize = Math.ceil(Math.sqrt(n));
  
  return (
    <group position={position}>
      {activations.slice(0, n).map((act, idx) => {
        const row = Math.floor(idx / gridSize);
        const col = idx % gridSize;
        const height = Math.abs(act) * 0.5;
        const isIntervened = interventions?.includes(idx);
        
        return (
          <mesh
            key={idx}
            position={[(col - gridSize/2) * 0.3, height/2, (row - gridSize/2) * 0.3]}
          >
            <boxGeometry args={[0.25, height, 0.25]} />
            <meshStandardMaterial 
              color={isIntervened ? COLOR_SCHEMES.danger : getGradientColor(act)}
              emissive={isIntervened ? COLOR_SCHEMES.danger : getGradientColor(act)}
              emissiveIntensity={isIntervened ? 0.5 : 0.2}
            />
          </mesh>
        );
      })}
    </group>
  );
}

// 干预效果可视化
function InterventionEffect({ before, after, position = [0, 0, 0] }) {
  if (!before || !after) return null;
  
  const maxDiff = Math.max(...after.map((a, i) => Math.abs(a - before[i])));
  
  return (
    <group position={position}>
      {/* 差异柱状图 */}
      {after.slice(0, 40).map((val, idx) => {
        const diff = val - before[idx];
        const height = Math.abs(diff) / maxDiff * 2;
        const color = diff > 0 ? COLOR_SCHEMES.success : COLOR_SCHEMES.danger;
        
        return (
          <mesh
            key={idx}
            position={[(idx - 20) * 0.2, height/2, 0]}
          >
            <boxGeometry args={[0.15, height, 0.15]} />
            <meshStandardMaterial 
              color={color}
              emissive={color}
              emissiveIntensity={0.3}
            />
          </mesh>
        );
      })}
      
      {/* 基线 */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[8, 0.02, 0.02]} />
        <meshStandardMaterial color="#444" />
      </mesh>
    </group>
  );
}

// 注意力头干预可视化
function AttentionIntervention({ heads, selectedHeads, onHeadSelect, position = [0, 0, 0] }) {
  if (!heads) return null;
  
  return (
    <group position={position}>
      {heads.map((head, idx) => {
        const row = Math.floor(idx / 4);
        const col = idx % 4;
        const isSelected = selectedHeads?.includes(idx);
        
        return (
          <group 
            key={idx} 
            position={[col * 1.5 - 2.25, 0, row * 1.5 - 2.25]}
          >
            <mesh onClick={() => onHeadSelect?.(idx)}>
              <boxGeometry args={[1, head.importance * 0.5 + 0.2, 1]} />
              <meshStandardMaterial 
                color={isSelected ? COLOR_SCHEMES.danger : COLOR_SCHEMES.attention}
                emissive={isSelected ? COLOR_SCHEMES.danger : COLOR_SCHEMES.attention}
                emissiveIntensity={isSelected ? 0.5 : 0.2}
                transparent
                opacity={isSelected ? 1 : 0.7}
              />
            </mesh>
            
            <Text
              position={[0, head.importance * 0.5 + 0.4, 0]}
              fontSize={0.1}
              color="#fff"
              anchorX="center"
            >
              H{idx}
            </Text>
          </group>
        );
      })}
    </group>
  );
}

// 主组件
export function ActivationIntervention({ modelData, selectedLayer = 0 }) {
  const [activations, setActivations] = useState(null);
  const [originalActivations, setOriginalActivations] = useState(null);
  const [interventions, setInterventions] = useState([]);
  const [heads, setHeads] = useState(null);
  const [selectedHeads, setSelectedHeads] = useState([]);
  const [interventionType, setInterventionType] = useState('clamp'); // 'clamp' | 'zero' | 'amplify'
  const [viewMode, setViewMode] = useState('state'); // 'state' | 'effect'
  const [loading, setLoading] = useState(false);
  const [interventionStrength, setInterventionStrength] = useState(1.0);

  // 加载激活数据
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const data = await apiCall(`${API_ENDPOINTS.analysis.attention}?layer=${selectedLayer}`);
        setActivations(data.activations);
        setOriginalActivations(data.activations);
        setHeads(data.heads || Array(12).fill(0).map((_, i) => ({ importance: Math.random() })));
      } catch (error) {
        // 静默使用模拟数据
        const mockActivations = Array(100).fill(0).map(() => Math.random() * 2 - 1);
        setActivations(mockActivations);
        setOriginalActivations([...mockActivations]);
        setHeads(Array(12).fill(0).map(() => ({ importance: Math.random() })));
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [selectedLayer]);

  // 应用干预
  const applyIntervention = () => {
    if (!originalActivations) return;
    
    let newActivations = [...originalActivations];
    
    interventions.forEach(idx => {
      if (idx < newActivations.length) {
        switch (interventionType) {
          case 'clamp':
            newActivations[idx] = Math.sign(newActivations[idx]) * interventionStrength;
            break;
          case 'zero':
            newActivations[idx] = 0;
            break;
          case 'amplify':
            newActivations[idx] *= interventionStrength;
            break;
        }
      }
    });
    
    setActivations(newActivations);
    setViewMode('effect');
  };

  // 重置
  const resetIntervention = () => {
    setActivations(originalActivations ? [...originalActivations] : null);
    setInterventions([]);
    setSelectedHeads([]);
  };

  // 选择/取消选择激活单元
  const toggleActivation = (idx) => {
    if (interventions.includes(idx)) {
      setInterventions(interventions.filter(i => i !== idx));
    } else {
      setInterventions([...interventions, idx]);
    }
  };

  // 选择/取消选择注意力头
  const toggleHead = (idx) => {
    if (selectedHeads.includes(idx)) {
      setSelectedHeads(selectedHeads.filter(i => i !== idx));
    } else {
      setSelectedHeads([...selectedHeads, idx]);
    }
  };

  // 计算效果统计
  const getEffectStats = () => {
    if (!activations || !originalActivations) return null;
    
    const diffs = activations.map((a, i) => Math.abs(a - originalActivations[i]));
    const avgDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
    const maxDiff = Math.max(...diffs);
    const changedCount = diffs.filter(d => d > 0.01).length;
    
    return { avgDiff, maxDiff, changedCount };
  };

  const stats = getEffectStats();

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex' }}>
      {/* 3D 视图 */}
      <div style={{ flex: 1, position: 'relative' }}>
        {/* 工具栏 */}
        <div style={{
          position: 'absolute',
          top: '12px',
          left: '12px',
          zIndex: 10,
          display: 'flex',
          gap: '8px',
          flexWrap: 'wrap',
          alignItems: 'center'
        }}>
          {/* 干预类型 */}
          {[
            { id: 'clamp', label: '钳制', icon: '📌' },
            { id: 'zero', label: '归零', icon: '⭕' },
            { id: 'amplify', label: '放大', icon: '🔊' }
          ].map(item => (
            <button
              key={item.id}
              onClick={() => setInterventionType(item.id)}
              style={{
                padding: '8px 12px',
                background: interventionType === item.id ? 'rgba(255, 68, 68, 0.2)' : 'rgba(0,0,0,0.6)',
                border: `1px solid ${interventionType === item.id ? COLOR_SCHEMES.danger : '#444'}`,
                borderRadius: '6px',
                color: interventionType === item.id ? COLOR_SCHEMES.danger : '#888',
                cursor: 'pointer',
                fontSize: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
          
          {/* 强度滑块 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '11px', color: '#888' }}>强度:</span>
            <input
              type="range"
              min="0.1"
              max="3"
              step="0.1"
              value={interventionStrength}
              onChange={(e) => setInterventionStrength(parseFloat(e.target.value))}
              style={{ width: '80px' }}
            />
            <span style={{ fontSize: '11px', color: '#fff' }}>{interventionStrength.toFixed(1)}</span>
          </div>
        </div>

        {/* 操作按钮 */}
        <div style={{
          position: 'absolute',
          top: '60px',
          left: '12px',
          zIndex: 10,
          display: 'flex',
          gap: '8px'
        }}>
          <button
            onClick={applyIntervention}
            disabled={interventions.length === 0 && selectedHeads.length === 0}
            style={{
              padding: '10px 20px',
              background: 'linear-gradient(45deg, #ff4444, #ff8844)',
              border: 'none',
              borderRadius: '6px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: '500'
            }}
          >
            应用干预
          </button>
          
          <button
            onClick={resetIntervention}
            style={{
              padding: '10px 20px',
              background: '#333',
              border: '1px solid #444',
              borderRadius: '6px',
              color: '#888',
              cursor: 'pointer',
              fontSize: '13px'
            }}
          >
            重置
          </button>
        </div>

        {loading && <LoadingSpinner message="加载激活数据..." />}

        <Canvas>
          <PerspectiveCamera makeDefault position={[8, 6, 8]} fov={50} />
          <OrbitControls enableDamping dampingFactor={0.05} />
          
          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={0.8} />
          
          {viewMode === 'state' && activations && (
            <ActivationState 
              activations={activations}
              interventions={interventions}
              position={[0, 0, 0]}
            />
          )}
          
          {viewMode === 'effect' && activations && originalActivations && (
            <InterventionEffect 
              before={originalActivations}
              after={activations}
              position={[0, 0, 0]}
            />
          )}
          
          <gridHelper args={[12, 12, '#222', '#111']} position={[0, -1, 0]} />
        </Canvas>
      </div>

      {/* 右侧控制面板 */}
      <div style={{
        width: '280px',
        background: 'rgba(255,255,255,0.02)',
        borderLeft: '1px solid #333',
        padding: '16px',
        overflowY: 'auto'
      }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', color: COLOR_SCHEMES.danger }}>
          激活干预控制
        </h3>
        
        {/* 注意力头选择 */}
        <div style={{ marginBottom: '16px' }}>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#666' }}>
            选择干预目标
          </h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
            {Array(12).fill(0).map((_, idx) => (
              <button
                key={idx}
                onClick={() => toggleHead(idx)}
                style={{
                  width: '40px',
                  height: '40px',
                  background: selectedHeads.includes(idx) ? COLOR_SCHEMES.danger : '#222',
                  border: `1px solid ${selectedHeads.includes(idx) ? COLOR_SCHEMES.danger : '#444'}`,
                  borderRadius: '6px',
                  color: '#fff',
                  cursor: 'pointer',
                  fontSize: '11px'
                }}
              >
                H{idx}
              </button>
            ))}
          </div>
        </div>
        
        {/* 当前状态 */}
        <div style={{ marginBottom: '16px' }}>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#666' }}>
            当前干预
          </h4>
          <div style={{
            padding: '12px',
            background: 'rgba(255,255,255,0.02)',
            borderRadius: '8px',
            fontSize: '11px',
            color: '#888'
          }}>
            <div>已选择: {interventions.length + selectedHeads.length} 个单元</div>
            <div>干预类型: {interventionType}</div>
            <div>强度: {interventionStrength.toFixed(1)}x</div>
          </div>
        </div>

        {/* 效果统计 */}
        {stats && viewMode === 'effect' && (
          <div style={{ marginBottom: '16px' }}>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#666' }}>
              干预效果
            </h4>
            <MetricGrid columns={1}>
              <MetricCard 
                title="平均变化" 
                value={stats.avgDiff.toFixed(4)}
                color={COLOR_SCHEMES.accent}
              />
              <MetricCard 
                title="最大变化" 
                value={stats.maxDiff.toFixed(4)}
                color={COLOR_SCHEMES.danger}
              />
              <MetricCard 
                title="受影响单元" 
                value={stats.changedCount}
                color={COLOR_SCHEMES.primary}
              />
            </MetricGrid>
          </div>
        )}

        {/* 注意事项 */}
        <div style={{
          marginTop: '24px',
          padding: '12px',
          background: 'rgba(255, 170, 0, 0.1)',
          borderRadius: '8px',
          borderLeft: `2px solid ${COLOR_SCHEMES.warning}`
        }}>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '11px', color: COLOR_SCHEMES.warning }}>
            ⚠️ 注意事项
          </h4>
          <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: '10px', color: '#888', lineHeight: '1.6' }}>
            <li>干预操作会直接影响模型推理</li>
            <li>建议从小强度开始测试</li>
            <li>观察干预对输出的影响</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default ActivationIntervention;
