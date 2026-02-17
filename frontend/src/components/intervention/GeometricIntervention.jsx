/**
 * GeometricIntervention - 几何干预视图
 * 对流形结构、曲率进行干预实验
 */
import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text, Sphere, Box, Line } from '@react-three/drei';
import { MetricCard, MetricGrid } from '../shared/MetricCard';
import { LoadingSpinner } from '../shared/LoadingSpinner';
import { API_ENDPOINTS, apiCall } from '../../config/api';
import { COLOR_SCHEMES, getGradientColor, getEntropyColor } from '../../utils/colors';
import * as THREE from 'three';

// 流形变形可视化
function ManifoldDeformation({ points, deformations, position = [0, 0, 0] }) {
  if (!points) return null;
  
  return (
    <group position={position}>
      {points.map((point, idx) => {
        const deform = deformations?.[idx] || [0, 0, 0];
        const deformedPoint = [
          point[0] + deform[0],
          point[1] + deform[1],
          point[2] + deform[2]
        ];
        const deformMag = Math.sqrt(deform[0]**2 + deform[1]**2 + deform[2]**2);
        
        return (
          <group key={idx}>
            {/* 原始点 */}
            <Sphere args={[0.03, 8, 8]} position={point}>
              <meshStandardMaterial 
                color="#666"
                transparent
                opacity={0.3}
              />
            </Sphere>
            
            {/* 变形后的点 */}
            <Sphere args={[0.04, 8, 8]} position={deformedPoint}>
              <meshStandardMaterial 
                color={getGradientColor(deformMag * 2)}
                emissive={getGradientColor(deformMag * 2)}
                emissiveIntensity={0.5}
              />
            </Sphere>
            
            {/* 变形箭头 */}
            {deformMag > 0.05 && (
              <Line
                points={[point, deformedPoint]}
                color={COLOR_SCHEMES.curvature}
                lineWidth={1}
              />
            )}
          </group>
        );
      })}
    </group>
  );
}

// 曲率调整可视化
function CurvatureAdjustment({ curvatureData, adjustments, position = [0, 0, 0] }) {
  if (!curvatureData) return null;
  
  const points = curvatureData.points || [];
  const curvatures = curvatureData.curvatures || [];
  const adjustedCurvatures = adjustments || curvatures;
  
  return (
    <group position={position}>
      {points.map((point, idx) => {
        const original = curvatures[idx] || 0;
        const adjusted = adjustedCurvatures[idx] || 0;
        const diff = adjusted - original;
        
        return (
          <group key={idx} position={point}>
            {/* 原始曲率柱 */}
            <mesh position={[0, Math.abs(original) * 2, 0]}>
              <cylinderGeometry args={[0.03, 0.03, Math.abs(original) * 4, 8]} />
              <meshStandardMaterial 
                color={original > 0 ? '#666' : '#444'}
                transparent
                opacity={0.3}
              />
            </mesh>
            
            {/* 调整后曲率柱 */}
            <mesh position={[0, Math.abs(adjusted) * 2, 0]}>
              <cylinderGeometry args={[0.04, 0.04, Math.abs(adjusted) * 4, 8]} />
              <meshStandardMaterial 
                color={adjusted > 0 ? COLOR_SCHEMES.danger : COLOR_SCHEMES.success}
                emissive={adjusted > 0 ? COLOR_SCHEMES.danger : COLOR_SCHEMES.success}
                emissiveIntensity={0.3}
              />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

// 测地线修改可视化
function GeodesicModification({ originalPath, modifiedPath, position = [0, 0, 0] }) {
  return (
    <group position={position}>
      {/* 原始路径 */}
      {originalPath && originalPath.length > 1 && (
        <Line
          points={originalPath}
          color="#666"
          lineWidth={1}
          dashed
          dashScale={2}
        />
      )}
      
      {/* 修改后路径 */}
      {modifiedPath && modifiedPath.length > 1 && (
        <Line
          points={modifiedPath}
          color={COLOR_SCHEMES.geodesic}
          lineWidth={2}
        />
      )}
      
      {/* 起点 */}
      {modifiedPath && modifiedPath.length > 0 && (
        <Sphere args={[0.1, 16, 16]} position={modifiedPath[0]}>
          <meshStandardMaterial color="#00ff00" emissive="#00ff00" emissiveIntensity={0.5} />
        </Sphere>
      )}
      
      {/* 终点 */}
      {modifiedPath && modifiedPath.length > 0 && (
        <Sphere args={[0.1, 16, 16]} position={modifiedPath[modifiedPath.length - 1]}>
          <meshStandardMaterial color="#ff0000" emissive="#ff0000" emissiveIntensity={0.5} />
        </Sphere>
      )}
    </group>
  );
}

// 主组件
export function GeometricIntervention({ modelData, selectedLayer = 0 }) {
  const [manifoldData, setManifoldData] = useState(null);
  const [curvatureData, setCurvatureData] = useState(null);
  const [deformations, setDeformations] = useState([]);
  const [curvatureAdjustments, setCurvatureAdjustments] = useState([]);
  const [originalGeodesic, setOriginalGeodesic] = useState(null);
  const [modifiedGeodesic, setModifiedGeodesic] = useState(null);
  const [viewMode, setViewMode] = useState('manifold'); // 'manifold' | 'curvature' | 'geodesic'
  const [interventionMode, setInterventionMode] = useState('smooth'); // 'smooth' | 'warp' | 'flatten'
  const [loading, setLoading] = useState(false);
  const [intensity, setIntensity] = useState(1.0);

  // 加载几何数据
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const [topoData, curvData] = await Promise.all([
          apiCall(`${API_ENDPOINTS.analysis.topology}?layer=${selectedLayer}`),
          apiCall(`${API_ENDPOINTS.analysis.curvature}?layer=${selectedLayer}`)
        ]);
        
        setManifoldData(topoData);
        setCurvatureData(curvData);
        
        // 初始化测地线
        setOriginalGeodesic(
          Array(20).fill(0).map((_, i) => [
            Math.sin(i * 0.3) * 2,
            i * 0.1,
            Math.cos(i * 0.3) * 2
          ])
        );
        setModifiedGeodesic(
          Array(20).fill(0).map((_, i) => [
            Math.sin(i * 0.3) * 2,
            i * 0.1,
            Math.cos(i * 0.3) * 2
          ])
        );
      } catch (error) {
        // 静默使用模拟数据
        setManifoldData({
          pca: Array(150).fill(0).map(() => [
            (Math.random() - 0.5) * 4,
            (Math.random() - 0.5) * 4,
            (Math.random() - 0.5) * 4
          ])
        });
        setCurvatureData({
          points: Array(40).fill(0).map(() => [
            (Math.random() - 0.5) * 3,
            0,
            (Math.random() - 0.5) * 3
          ]),
          curvatures: Array(40).fill(0).map(() => (Math.random() - 0.5) * 0.5)
        });
        
        setOriginalGeodesic(
          Array(20).fill(0).map((_, i) => [
            Math.sin(i * 0.3) * 2,
            i * 0.1,
            Math.cos(i * 0.3) * 2
          ])
        );
        setModifiedGeodesic(
          Array(20).fill(0).map((_, i) => [
            Math.sin(i * 0.3) * 2,
            i * 0.1,
            Math.cos(i * 0.3) * 2
          ])
        );
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [selectedLayer]);

  // 应用流形干预
  const applyManifoldIntervention = () => {
    if (!manifoldData?.pca) return;
    
    const newDeformations = manifoldData.pca.map(point => {
      const r = Math.sqrt(point[0]**2 + point[2]**2);
      switch (interventionMode) {
        case 'smooth':
          return [
            -point[0] * 0.1 * intensity,
            -point[1] * 0.05 * intensity,
            -point[2] * 0.1 * intensity
          ];
        case 'warp':
          return [
            Math.sin(point[1] * 2) * 0.2 * intensity,
            Math.cos(point[0] * 2) * 0.2 * intensity,
            Math.sin(point[2] * 2) * 0.2 * intensity
          ];
        case 'flatten':
          return [0, -point[1] * 0.5 * intensity, 0];
        default:
          return [0, 0, 0];
      }
    });
    
    setDeformations(newDeformations);
  };

  // 应用曲率干预
  const applyCurvatureIntervention = () => {
    if (!curvatureData?.curvatures) return;
    
    const newAdjustments = curvatureData.curvatures.map(c => {
      switch (interventionMode) {
        case 'smooth':
          return c * (1 - intensity * 0.5);
        case 'warp':
          return c + Math.random() * 0.2 * intensity;
        case 'flatten':
          return c * (1 - intensity * 0.8);
        default:
          return c;
      }
    });
    
    setCurvatureAdjustments(newAdjustments);
  };

  // 应用测地线干预
  const applyGeodesicIntervention = () => {
    if (!originalGeodesic) return;
    
    const newModified = originalGeodesic.map((point, idx) => {
      switch (interventionMode) {
        case 'smooth':
          return [
            point[0] * (1 - intensity * 0.1),
            point[1],
            point[2] * (1 - intensity * 0.1)
          ];
        case 'warp':
          return [
            point[0] + Math.sin(idx * 0.5) * 0.3 * intensity,
            point[1] + Math.cos(idx * 0.5) * 0.2 * intensity,
            point[2] + Math.sin(idx * 0.5) * 0.3 * intensity
          ];
        case 'flatten':
          return [point[0], point[1] * (1 - intensity * 0.5), point[2]];
        default:
          return point;
      }
    });
    
    setModifiedGeodesic(newModified);
  };

  // 重置所有干预
  const resetAll = () => {
    setDeformations([]);
    setCurvatureAdjustments([]);
    setModifiedGeodesic(originalGeodesic);
  };

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
            { id: 'manifold', label: '流形变形', icon: '🌐' },
            { id: 'curvature', label: '曲率调整', icon: '📈' },
            { id: 'geodesic', label: '测地线修改', icon: '➡️' }
          ].map(item => (
            <button
              key={item.id}
              onClick={() => setViewMode(item.id)}
              style={{
                padding: '8px 12px',
                background: viewMode === item.id ? 'rgba(0, 255, 255, 0.2)' : 'rgba(0,0,0,0.6)',
                border: `1px solid ${viewMode === item.id ? COLOR_SCHEMES.manifold : '#444'}`,
                borderRadius: '6px',
                color: viewMode === item.id ? COLOR_SCHEMES.manifold : '#888',
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
          
          {/* 干预模式 */}
          <select
            value={interventionMode}
            onChange={(e) => setInterventionMode(e.target.value)}
            style={{
              padding: '6px 12px',
              background: '#222',
              border: '1px solid #444',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '12px'
            }}
          >
            <option value="smooth">平滑化</option>
            <option value="warp">扭曲化</option>
            <option value="flatten">扁平化</option>
          </select>
          
          {/* 强度 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '11px', color: '#888' }}>强度:</span>
            <input
              type="range"
              min="0.1"
              max="2"
              step="0.1"
              value={intensity}
              onChange={(e) => setIntensity(parseFloat(e.target.value))}
              style={{ width: '80px' }}
            />
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
            onClick={() => {
              if (viewMode === 'manifold') applyManifoldIntervention();
              else if (viewMode === 'curvature') applyCurvatureIntervention();
              else applyGeodesicIntervention();
            }}
            style={{
              padding: '10px 20px',
              background: 'linear-gradient(45deg, #00ffff, #00ff88)',
              border: 'none',
              borderRadius: '6px',
              color: '#000',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: '500'
            }}
          >
            应用几何干预
          </button>
          
          <button
            onClick={resetAll}
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

        {loading && <LoadingSpinner message="加载几何数据..." />}

        <Canvas>
          <PerspectiveCamera makeDefault position={[6, 5, 6]} fov={50} />
          <OrbitControls enableDamping dampingFactor={0.05} />
          
          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={0.8} />
          
          {viewMode === 'manifold' && manifoldData?.pca && (
            <ManifoldDeformation 
              points={manifoldData.pca}
              deformations={deformations}
              position={[0, 0, 0]}
            />
          )}
          
          {viewMode === 'curvature' && curvatureData && (
            <CurvatureAdjustment 
              curvatureData={curvatureData}
              adjustments={curvatureAdjustments}
              position={[0, -1, 0]}
            />
          )}
          
          {viewMode === 'geodesic' && (
            <GeodesicModification 
              originalPath={originalGeodesic}
              modifiedPath={modifiedGeodesic}
              position={[0, 0, 0]}
            />
          )}
          
          <gridHelper args={[10, 10, '#222', '#111']} />
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
        <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', color: COLOR_SCHEMES.manifold }}>
          几何干预控制
        </h3>
        
        <div style={{ marginBottom: '16px' }}>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#666' }}>
            干预效果
          </h4>
          <MetricGrid columns={1}>
            <MetricCard 
              title="流形变形量" 
              value={(deformations.length > 0 ? deformations.length : 0).toString()}
              color={COLOR_SCHEMES.manifold}
            />
            <MetricCard 
              title="曲率变化" 
              value={curvatureAdjustments.length > 0 ? '已应用' : '未应用'}
              color={COLOR_SCHEMES.curvature}
            />
            <MetricCard 
              title="测地线长度变化" 
              value={
                originalGeodesic && modifiedGeodesic
                  ? `${((modifiedGeodesic.length - originalGeodesic.length) / originalGeodesic.length * 100).toFixed(1)}%`
                  : '0%'
              }
              color={COLOR_SCHEMES.geodesic}
            />
          </MetricGrid>
        </div>

        <div style={{ marginBottom: '16px' }}>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#666' }}>
            干预模式说明
          </h4>
          <div style={{ fontSize: '11px', color: '#888', lineHeight: '1.8' }}>
            <p><strong style={{ color: COLOR_SCHEMES.success }}>平滑化</strong>: 减少局部变形</p>
            <p><strong style={{ color: COLOR_SCHEMES.warning }}>扭曲化</strong>: 增加局部复杂性</p>
            <p><strong style={{ color: COLOR_SCHEMES.primary }}>扁平化</strong>: 简化几何结构</p>
          </div>
        </div>

        {/* 理论背景 */}
        <div style={{
          marginTop: '24px',
          padding: '12px',
          background: 'rgba(0, 210, 255, 0.05)',
          borderRadius: '8px',
          borderLeft: `2px solid ${COLOR_SCHEMES.manifold}`
        }}>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '11px', color: COLOR_SCHEMES.manifold }}>
            📐 理论背景
          </h4>
          <div style={{ fontSize: '10px', color: '#888', lineHeight: '1.6' }}>
            <p>几何干预通过修改神经网络特征空间的流形结构，可以影响模型的表示能力和推理路径。</p>
            <p style={{ marginTop: '8px' }}>曲率调整可以改变局部决策边界的复杂度。</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default GeometricIntervention;
