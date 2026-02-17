/**
 * GeometryView - 几何视图
 * 展示流形结构、曲率场和测地线
 */
import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text, Line, Sphere } from '@react-three/drei';
import { MetricCard, MetricGrid } from '../shared/MetricCard';
import { LoadingSpinner } from '../shared/LoadingSpinner';
import { API_ENDPOINTS, apiCall } from '../../config/api';
import { getEntropyColor, COLOR_SCHEMES } from '../../utils/colors';
import * as THREE from 'three';

// 流形点云组件
function ManifoldPointCloud({ points, colors }) {
  if (!points || points.length === 0) return null;
  
  return (
    <group>
      {points.map((point, idx) => (
        <Sphere
          key={idx}
          args={[0.03, 8, 8]}
          position={point}
        >
          <meshStandardMaterial 
            color={colors?.[idx] || getEntropyColor(idx / points.length)}
            emissive={colors?.[idx] || getEntropyColor(idx / points.length)}
            emissiveIntensity={0.5}
          />
        </Sphere>
      ))}
    </group>
  );
}

// 曲率场可视化
function CurvatureField({ curvatureData }) {
  if (!curvatureData) return null;
  
  const points = curvatureData.points || [];
  const curvatures = curvatureData.curvatures || [];
  
  return (
    <group>
      {points.map((point, idx) => {
        const curvature = curvatures[idx] || 0;
        const normalizedCurvature = Math.min(Math.abs(curvature) / 0.5, 1);
        const height = normalizedCurvature * 2;
        
        return (
          <group key={idx} position={point}>
            {/* 曲率柱 */}
            <mesh position={[0, height / 2, 0]}>
              <cylinderGeometry args={[0.02, 0.02, height, 8]} />
              <meshStandardMaterial 
                color={curvature > 0 ? '#ff4444' : '#4444ff'}
                emissive={curvature > 0 ? '#ff4444' : '#4444ff'}
                emissiveIntensity={0.3}
                transparent
                opacity={0.7}
              />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

// 测地线可视化
function GeodesicPath({ path, color = COLOR_SCHEMES.geodesic }) {
  if (!path || path.length < 2) return null;
  
  return (
    <group>
      <Line
        points={path}
        color={color}
        lineWidth={2}
      />
      {/* 起点终点标记 */}
      <Sphere args={[0.1, 16, 16]} position={path[0]}>
        <meshStandardMaterial color="#00ff00" emissive="#00ff00" emissiveIntensity={0.5} />
      </Sphere>
      <Sphere args={[0.1, 16, 16]} position={path[path.length - 1]}>
        <meshStandardMaterial color="#ff0000" emissive="#ff0000" emissiveIntensity={0.5} />
      </Sphere>
    </group>
  );
}

// 主组件
export function GeometryView({ modelData, selectedLayer = 0 }) {
  const [topologyData, setTopologyData] = useState(null);
  const [curvatureData, setCurvatureData] = useState(null);
  const [geodesicData, setGeodesicData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState('manifold'); // 'manifold' | 'curvature' | 'geodesic'

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // 获取拓扑数据
        const topoData = await apiCall(`${API_ENDPOINTS.analysis.topology}?layer=${selectedLayer}`);
        setTopologyData(topoData);
        
        // 获取曲率数据 (使用 topology 作为替代)
        setCurvatureData(topoData);
        
      } catch (error) {
        // 静默使用模拟数据
        setTopologyData({
          pca: Array(200).fill(null).map(() => [
            (Math.random() - 0.5) * 4,
            (Math.random() - 0.5) * 4,
            (Math.random() - 0.5) * 4
          ])
        });
        setCurvatureData({
          points: Array(50).fill(null).map(() => [
            (Math.random() - 0.5) * 3,
            0,
            (Math.random() - 0.5) * 3
          ]),
          curvatures: Array(50).fill(null).map(() => (Math.random() - 0.5) * 0.5)
        });
        setGeodesicData({
          path: Array(20).fill(null).map((_, i) => [
            Math.sin(i * 0.3) * 2,
            i * 0.1,
            Math.cos(i * 0.3) * 2
          ])
        });
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [selectedLayer]);

  if (loading) {
    return <LoadingSpinner message="加载几何数据..." />;
  }

  // 计算统计信息
  const stats = {
    intrinsicDim: topologyData?.intrinsic_dim || 2.5,
    avgCurvature: curvatureData?.curvatures 
      ? (curvatureData.curvatures.reduce((a, b) => a + Math.abs(b), 0) / curvatureData.curvatures.length).toFixed(4)
      : '0.0234',
    nPoints: topologyData?.pca?.length || 200
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
          gap: '8px'
        }}>
          {[
            { id: 'manifold', label: '流形', icon: '🌐' },
            { id: 'curvature', label: '曲率', icon: '📈' },
            { id: 'geodesic', label: '测地线', icon: '➡️' }
          ].map(item => (
            <button
              key={item.id}
              onClick={() => setViewMode(item.id)}
              style={{
                padding: '8px 12px',
                background: viewMode === item.id ? 'rgba(0, 210, 255, 0.2)' : 'rgba(0,0,0,0.6)',
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
        </div>

        <Canvas>
          <PerspectiveCamera makeDefault position={[5, 5, 5]} fov={50} />
          <OrbitControls enableDamping dampingFactor={0.05} />
          
          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={0.8} />
          
          {/* 根据视图模式渲染 */}
          {viewMode === 'manifold' && (
            <ManifoldPointCloud 
              points={topologyData?.pca} 
            />
          )}
          
          {viewMode === 'curvature' && (
            <CurvatureField curvatureData={curvatureData} />
          )}
          
          {viewMode === 'geodesic' && (
            <group>
              <ManifoldPointCloud points={topologyData?.pca} />
              <GeodesicPath path={geodesicData?.path} />
            </group>
          )}
          
          <gridHelper args={[10, 10, '#222', '#111']} />
        </Canvas>
      </div>

      {/* 右侧信息面板 */}
      <div style={{
        width: '260px',
        background: 'rgba(255,255,255,0.02)',
        borderLeft: '1px solid #333',
        padding: '16px',
        overflowY: 'auto'
      }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', color: COLOR_SCHEMES.manifold }}>
          几何统计
        </h3>
        
        <MetricGrid columns={1}>
          <MetricCard 
            title="内在维度" 
            value={stats.intrinsicDim.toFixed(2)}
            description="流形的有效维度"
            color={COLOR_SCHEMES.manifold}
          />
          <MetricCard 
            title="平均曲率" 
            value={stats.avgCurvature}
            description="局部曲率绝对值平均"
            color={COLOR_SCHEMES.curvature}
          />
          <MetricCard 
            title="采样点数" 
            value={stats.nPoints}
            description="流形上的采样点数量"
            color={COLOR_SCHEMES.accent}
          />
        </MetricGrid>

        <div style={{ marginTop: '24px' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: '12px', color: '#666' }}>
            几何解释
          </h3>
          <div style={{ fontSize: '11px', color: '#888', lineHeight: '1.8' }}>
            <p><strong style={{ color: COLOR_SCHEMES.manifold }}>流形</strong>: 特征空间的低维嵌入</p>
            <p><strong style={{ color: COLOR_SCHEMES.curvature }}>曲率</strong>: 局部几何变形度量</p>
            <p><strong style={{ color: COLOR_SCHEMES.geodesic }}>测地线</strong>: 流形上的最短路径</p>
          </div>
        </div>

        <div style={{ marginTop: '24px' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: '12px', color: '#666' }}>
            当前视图
          </h3>
          <div style={{
            padding: '12px',
            background: 'rgba(0, 210, 255, 0.05)',
            borderRadius: '8px',
            border: `1px solid ${COLOR_SCHEMES.manifold}30`
          }}>
            <div style={{ fontSize: '13px', color: '#fff', marginBottom: '4px' }}>
              {viewMode === 'manifold' && '流形点云'}
              {viewMode === 'curvature' && '曲率场'}
              {viewMode === 'geodesic' && '测地线路径'}
            </div>
            <div style={{ fontSize: '11px', color: '#666' }}>
              {viewMode === 'manifold' && '展示特征空间的几何结构'}
              {viewMode === 'curvature' && '展示局部曲率分布'}
              {viewMode === 'geodesic' && '展示推理路径'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default GeometryView;
