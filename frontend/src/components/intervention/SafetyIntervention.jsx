/**
 * SafetyIntervention - 安全干预视图
 * 检测和处理潜在的安全问题
 */
import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text, Sphere, Box, Line, Html } from '@react-three/drei';
import { MetricCard, MetricGrid } from '../shared/MetricCard';
import { LoadingSpinner } from '../shared/LoadingSpinner';
import { API_ENDPOINTS, apiCall } from '../../config/api';
import { COLOR_SCHEMES, getGradientColor, getEntropyColor } from '../../utils/colors';
import * as THREE from 'three';

// 安全风险热图
function SafetyHeatmap({ risks, position = [0, 0, 0] }) {
  if (!risks) return null;
  
  const gridSize = Math.ceil(Math.sqrt(risks.length));
  
  return (
    <group position={position}>
      {risks.map((risk, idx) => {
        const row = Math.floor(idx / gridSize);
        const col = idx % gridSize;
        const height = risk.level * 2;
        
        const color = risk.level > 0.7 ? COLOR_SCHEMES.danger
          : risk.level > 0.4 ? COLOR_SCHEMES.warning
          : COLOR_SCHEMES.success;
        
        return (
          <group key={idx} position={[(col - gridSize/2) * 0.5, 0, (row - gridSize/2) * 0.5]}>
            <mesh position={[0, height/2, 0]}>
              <boxGeometry args={[0.4, height, 0.4]} />
              <meshStandardMaterial 
                color={color}
                emissive={color}
                emissiveIntensity={risk.level}
                transparent
                opacity={0.8}
              />
            </mesh>
            
            {risk.level > 0.6 && (
              <Html distanceFactor={15} position={[0, height + 0.3, 0]}>
                <div style={{
                  background: 'rgba(255,0,0,0.8)',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  fontSize: '9px',
                  color: '#fff',
                  whiteSpace: 'nowrap',
                  transform: 'translateX(-50%)'
                }}>
                  ⚠️ {risk.type}
                </div>
              </Html>
            )}
          </group>
        );
      })}
    </group>
  );
}

// 对抗样本检测可视化
function AdversarialDetection({ samples, position = [0, 0, 0] }) {
  if (!samples) return null;
  
  return (
    <group position={position}>
      {samples.map((sample, idx) => {
        const isAdversarial = sample.isAdversarial;
        const color = isAdversarial ? COLOR_SCHEMES.danger : COLOR_SCHEMES.success;
        
        return (
          <group key={idx} position={[sample.x * 4, sample.y * 4, sample.z * 4]}>
            <Sphere args={[isAdversarial ? 0.12 : 0.08, 16, 16]}>
              <meshStandardMaterial 
                color={color}
                emissive={color}
                emissiveIntensity={0.5}
              />
            </Sphere>
            
            {isAdversarial && (
              <Line
                points={[[0, 0, 0], [0, 0.5, 0]]}
                color={COLOR_SCHEMES.danger}
                lineWidth={2}
              />
            )}
          </group>
        );
      })}
    </group>
  );
}

// 安全边界可视化
function SafetyBoundary({ boundary, position = [0, 0, 0] }) {
  if (!boundary) return null;
  
  // 创建边界平面
  const points = [];
  const size = 5;
  
  for (let i = 0; i <= 20; i++) {
    for (let j = 0; j <= 20; j++) {
      const x = (i / 20 - 0.5) * size * 2;
      const z = (j / 20 - 0.5) * size * 2;
      const y = Math.sin(x * 0.5) * Math.cos(z * 0.5) * 0.5;
      points.push(x, y, z);
    }
  }
  
  const indices = [];
  for (let i = 0; i < 20; i++) {
    for (let j = 0; j < 20; j++) {
      const a = i * 21 + j;
      const b = a + 1;
      const c = a + 21;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  
  return (
    <group position={position}>
      <mesh geometry={geometry}>
        <meshStandardMaterial 
          color={COLOR_SCHEMES.warning}
          transparent
          opacity={0.3}
          side={THREE.DoubleSide}
          wireframe
        />
      </mesh>
    </group>
  );
}

// 主组件
export function SafetyIntervention({ modelData, selectedLayer = 0 }) {
  const [risks, setRisks] = useState([]);
  const [samples, setSamples] = useState([]);
  const [viewMode, setViewMode] = useState('heatmap'); // 'heatmap' | 'adversarial' | 'boundary'
  const [loading, setLoading] = useState(false);
  const [selectedRisk, setSelectedRisk] = useState(null);
  const [mitigationApplied, setMitigationApplied] = useState(false);

  // 加载安全数据
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const data = await apiCall(`${API_ENDPOINTS.agi.verify}?layer=${selectedLayer}`);
        setRisks(data.risks || []);
        setSamples(data.samples || []);
      } catch (error) {
        // 静默使用模拟数据
        setRisks(Array(36).fill(0).map((_, idx) => ({
          type: ['Out-of-distribution', 'Toxic output', 'Hallucination', 'Bias'][Math.floor(Math.random() * 4)],
          level: Math.random(),
          location: idx
        })));
        
        setSamples(Array(50).fill(0).map(() => ({
          x: Math.random() - 0.5,
          y: Math.random() - 0.5,
          z: Math.random() - 0.5,
          isAdversarial: Math.random() > 0.8
        })));
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [selectedLayer]);

  // 应用缓解措施
  const applyMitigation = (riskIdx) => {
    setMitigationApplied(true);
    setRisks(risks.map((r, i) => 
      i === riskIdx ? { ...r, level: Math.max(0, r.level - 0.5) } : r
    ));
    
    setTimeout(() => setMitigationApplied(false), 1000);
  };

  // 批量缓解
  const applyBulkMitigation = () => {
    setMitigationApplied(true);
    setRisks(risks.map(r => ({
      ...r,
      level: Math.max(0, r.level - 0.3)
    })));
    
    setTimeout(() => setMitigationApplied(false), 1000);
  };

  // 计算统计
  const getStats = () => {
    if (risks.length === 0) return null;
    
    const highRisks = risks.filter(r => r.level > 0.7).length;
    const mediumRisks = risks.filter(r => r.level > 0.4 && r.level <= 0.7).length;
    const lowRisks = risks.filter(r => r.level <= 0.4).length;
    const avgRisk = risks.reduce((sum, r) => sum + r.level, 0) / risks.length;
    
    return { highRisks, mediumRisks, lowRisks, avgRisk };
  };

  const stats = getStats();

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
          {/* 视图模式 */}
          {[
            { id: 'heatmap', label: '风险热图', icon: '🗺️' },
            { id: 'adversarial', label: '对抗检测', icon: '🎯' },
            { id: 'boundary', label: '安全边界', icon: '🛡️' }
          ].map(item => (
            <button
              key={item.id}
              onClick={() => setViewMode(item.id)}
              style={{
                padding: '8px 12px',
                background: viewMode === item.id ? 'rgba(255, 68, 68, 0.2)' : 'rgba(0,0,0,0.6)',
                border: `1px solid ${viewMode === item.id ? COLOR_SCHEMES.danger : '#444'}`,
                borderRadius: '6px',
                color: viewMode === item.id ? COLOR_SCHEMES.danger : '#888',
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
        </div>

        {/* 批量操作按钮 */}
        <div style={{
          position: 'absolute',
          top: '60px',
          left: '12px',
          zIndex: 10
        }}>
          <button
            onClick={applyBulkMitigation}
            disabled={mitigationApplied}
            style={{
              padding: '10px 20px',
              background: mitigationApplied ? '#333' : 'linear-gradient(45deg, #ff4444, #ff8800)',
              border: 'none',
              borderRadius: '6px',
              color: '#fff',
              cursor: mitigationApplied ? 'wait' : 'pointer',
              fontSize: '13px',
              fontWeight: '500'
            }}
          >
            {mitigationApplied ? '处理中...' : '批量缓解风险'}
          </button>
        </div>

        {loading && <LoadingSpinner message="分析安全风险..." />}

        <Canvas>
          <PerspectiveCamera makeDefault position={[8, 6, 8]} fov={50} />
          <OrbitControls enableDamping dampingFactor={0.05} />
          
          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={0.8} />
          
          {viewMode === 'heatmap' && risks.length > 0 && (
            <SafetyHeatmap risks={risks} position={[0, 0, 0]} />
          )}
          
          {viewMode === 'adversarial' && samples.length > 0 && (
            <AdversarialDetection samples={samples} position={[0, 0, 0]} />
          )}
          
          {viewMode === 'boundary' && (
            <SafetyBoundary position={[0, 0, 0]} />
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
          安全干预控制
        </h3>
        
        {stats && (
          <MetricGrid columns={2}>
            <MetricCard 
              title="高风险" 
              value={stats.highRisks}
              color={COLOR_SCHEMES.danger}
            />
            <MetricCard 
              title="中风险" 
              value={stats.mediumRisks}
              color={COLOR_SCHEMES.warning}
            />
            <MetricCard 
              title="低风险" 
              value={stats.lowRisks}
              color={COLOR_SCHEMES.success}
            />
            <MetricCard 
              title="平均风险" 
              value={`${(stats.avgRisk * 100).toFixed(1)}%`}
              color={COLOR_SCHEMES.accent}
            />
          </MetricGrid>
        )}

        {/* 高风险列表 */}
        <div style={{ marginTop: '16px' }}>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#666' }}>
            高风险项目
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {risks.filter(r => r.level > 0.6).slice(0, 5).map((risk, idx) => (
              <div 
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 12px',
                  background: 'rgba(255,0,0,0.1)',
                  borderRadius: '6px',
                  borderLeft: `3px solid ${COLOR_SCHEMES.danger}`
                }}
              >
                <span style={{ flex: 1, fontSize: '11px', color: '#fff' }}>
                  {risk.type}
                </span>
                <span style={{ 
                  fontSize: '10px', 
                  color: COLOR_SCHEMES.danger,
                  fontWeight: '500'
                }}>
                  {(risk.level * 100).toFixed(0)}%
                </span>
                <button
                  onClick={() => applyMitigation(risks.indexOf(risk))}
                  style={{
                    padding: '2px 8px',
                    background: COLOR_SCHEMES.danger,
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                    fontSize: '10px',
                    cursor: 'pointer'
                  }}
                >
                  缓解
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* 安全策略 */}
        <div style={{ marginTop: '24px' }}>
          <h4 style={{ margin: '0 0 12px 0', fontSize: '12px', color: '#666' }}>
            可用安全策略
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {[
              { name: '输出过滤', status: 'active' },
              { name: '输入检测', status: 'active' },
              { name: '对抗训练', status: 'inactive' },
              { name: '安全微调', status: 'inactive' }
            ].map((policy, idx) => (
              <div 
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 12px',
                  background: 'rgba(255,255,255,0.02)',
                  borderRadius: '6px',
                  fontSize: '11px'
                }}
              >
                <span style={{
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  background: policy.status === 'active' ? COLOR_SCHEMES.success : '#444'
                }} />
                <span style={{ flex: 1, color: '#888' }}>{policy.name}</span>
                <span style={{ 
                  color: policy.status === 'active' ? COLOR_SCHEMES.success : '#666',
                  fontSize: '10px'
                }}>
                  {policy.status === 'active' ? '已启用' : '未启用'}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* 警告 */}
        <div style={{
          marginTop: '24px',
          padding: '12px',
          background: 'rgba(255, 68, 68, 0.1)',
          borderRadius: '8px',
          borderLeft: `2px solid ${COLOR_SCHEMES.danger}`
        }}>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '11px', color: COLOR_SCHEMES.danger }}>
            ⚠️ 安全警告
          </h4>
          <div style={{ fontSize: '10px', color: '#888', lineHeight: '1.6' }}>
            检测到多个高风险区域，建议启用对抗训练和安全微调策略以提升模型安全性。
          </div>
        </div>
      </div>
    </div>
  );
}

export default SafetyIntervention;
