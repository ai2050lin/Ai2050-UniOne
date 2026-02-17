/**
 * ProgressTracker - 进度追踪视图
 * 追踪 AGI 研究和模型训练的进度
 */
import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text, Box, Line, Sphere, Html } from '@react-three/drei';
import { MetricCard, MetricGrid } from '../shared/MetricCard';
import { LoadingSpinner } from '../shared/LoadingSpinner';
import { API_ENDPOINTS, apiCall } from '../../config/api';
import { COLOR_SCHEMES, getGradientColor, getLayerColor } from '../../utils/colors';
import * as THREE from 'three';

// 进度路线图 3D
function ProgressRoadmap({ milestones, position = [0, 0, 0] }) {
  if (!milestones) return null;
  
  return (
    <group position={position}>
      {/* 主线 */}
      <Line
        points={milestones.map((_, i) => [i * 2, 0, 0])}
        color="#333"
        lineWidth={3}
      />
      
      {milestones.map((milestone, idx) => {
        const isCompleted = milestone.status === 'completed';
        const isInProgress = milestone.status === 'in_progress';
        const color = isCompleted ? COLOR_SCHEMES.success
          : isInProgress ? COLOR_SCHEMES.warning
          : '#444';
        
        return (
          <group key={idx} position={[idx * 2, 0, 0]}>
            {/* 节点 */}
            <Sphere args={[0.3, 16, 16]}>
              <meshStandardMaterial 
                color={color}
                emissive={color}
                emissiveIntensity={isCompleted || isInProgress ? 0.5 : 0}
              />
            </Sphere>
            
            {/* 进度柱 (进行中) */}
            {isInProgress && milestone.progress && (
              <mesh position={[0, -milestone.progress * 0.5 - 0.5, 0]}>
                <cylinderGeometry args={[0.15, 0.15, milestone.progress, 16]} />
                <meshStandardMaterial 
                  color={COLOR_SCHEMES.warning}
                  emissive={COLOR_SCHEMES.warning}
                  emissiveIntensity={0.3}
                />
              </mesh>
            )}
            
            {/* 标签 */}
            <Html distanceFactor={15} position={[0, 0.8, 0]}>
              <div style={{
                textAlign: 'center',
                width: '80px',
                transform: 'translateX(-50%)'
              }}>
                <div style={{
                  fontSize: '9px',
                  color: isCompleted ? COLOR_SCHEMES.success : isInProgress ? COLOR_SCHEMES.warning : '#666',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis'
                }}>
                  {milestone.name}
                </div>
                {isInProgress && (
                  <div style={{ fontSize: '8px', color: '#888' }}>
                    {(milestone.progress * 100).toFixed(0)}%
                  </div>
                )}
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
}

// 研究领域雷达图
function ResearchRadar({ areas, position = [0, 0, 0] }) {
  if (!areas) return null;
  
  const n = areas.length;
  const radius = 2;
  
  const getVertex = (angle, value) => [
    Math.cos(angle) * radius * value,
    Math.sin(angle) * radius * value,
    0
  ];
  
  return (
    <group position={position}>
      {/* 背景网格 */}
      {[0.25, 0.5, 0.75, 1.0].map((scale, idx) => (
        <Line
          key={idx}
          points={Array.from({ length: n + 1 }).map((_, i) => 
            getVertex((i * 2 * Math.PI) / n, scale)
          )}
          color="#222"
          lineWidth={1}
        />
      ))}
      
      {/* 轴线 */}
      {Array.from({ length: n }).map((_, i) => (
        <Line
          key={i}
          points={[[0, 0, 0], getVertex((i * 2 * Math.PI) / n, 1)]}
          color="#333"
          lineWidth={1}
        />
      ))}
      
      {/* 进度区域 */}
      <Line
        points={areas.map((area, i) => 
          getVertex((i * 2 * Math.PI) / n, area.progress)
        ).concat([getVertex(0, areas[0].progress)])}
        color={COLOR_SCHEMES.primary}
        lineWidth={2}
      />
      
      {/* 填充面 */}
      <mesh>
        <shapeGeometry args={[
          new THREE.Shape(areas.map((area, i) => {
            const vertex = getVertex((i * 2 * Math.PI) / n, area.progress);
            return new THREE.Vector2(vertex[0], vertex[1]);
          }))
        ]} />
        <meshBasicMaterial 
          color={COLOR_SCHEMES.primary}
          transparent
          opacity={0.1}
        />
      </mesh>
      
      {/* 标签 */}
      {areas.map((area, i) => {
        const pos = getVertex((i * 2 * Math.PI) / n, 1.2);
        return (
          <Text
            key={i}
            position={[pos[0], pos[1], 0]}
            fontSize={0.12}
            color="#888"
            anchorX="center"
          >
            {area.name}
          </Text>
        );
      })}
    </group>
  );
}

// 时间线可视化
function TimelineView({ events, position = [0, 0, 0] }) {
  if (!events) return null;
  
  return (
    <group position={position}>
      {events.map((event, idx) => {
        const y = -idx * 0.8;
        const isRecent = idx < 3;
        
        return (
          <group key={idx} position={[0, y, 0]}>
            {/* 时间点 */}
            <Sphere args={[isRecent ? 0.1 : 0.06, 8, 8]} position={[-3, 0, 0]}>
              <meshStandardMaterial 
                color={isRecent ? COLOR_SCHEMES.primary : '#444'}
                emissive={isRecent ? COLOR_SCHEMES.primary : '#111'}
                emissiveIntensity={isRecent ? 0.5 : 0}
              />
            </Sphere>
            
            {/* 连接线 */}
            <Line
              points={[[-3, 0, 0], [3, 0, 0]]}
              color={isRecent ? '#333' : '#222'}
              lineWidth={1}
            />
            
            {/* 事件内容 */}
            <Html distanceFactor={15} position={[0, 0, 0]}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                width: '200px'
              }}>
                <span style={{
                  fontSize: '9px',
                  color: '#666',
                  width: '50px'
                }}>
                  {event.date}
                </span>
                <span style={{
                  fontSize: '10px',
                  color: isRecent ? '#fff' : '#888'
                }}>
                  {event.title}
                </span>
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
}

// 主组件
export function ProgressTracker({ modelData, selectedLayer = 0 }) {
  const [milestones, setMilestones] = useState([]);
  const [researchAreas, setResearchAreas] = useState([]);
  const [events, setEvents] = useState([]);
  const [viewMode, setViewMode] = useState('roadmap'); // 'roadmap' | 'radar' | 'timeline'
  const [loading, setLoading] = useState(false);

  // 加载进度数据
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const data = await apiCall(`${API_ENDPOINTS.training.status}`);
        setMilestones(data.milestones || []);
        setResearchAreas(data.researchAreas || []);
        setEvents(data.events || []);
      } catch (error) {
        // 静默使用模拟数据
        setMilestones([
          { name: '结构提取', status: 'completed', progress: 1 },
          { name: '几何分析', status: 'completed', progress: 1 },
          { name: '干预实验', status: 'in_progress', progress: 0.65 },
          { name: '能力评估', status: 'pending', progress: 0 },
          { name: 'AGI验证', status: 'pending', progress: 0 }
        ]);
        
        setResearchAreas([
          { name: '观察', progress: 0.9 },
          { name: '分析', progress: 0.75 },
          { name: '干预', progress: 0.65 },
          { name: '评估', progress: 0.4 },
          { name: '理论', progress: 0.8 },
          { name: '应用', progress: 0.3 }
        ]);
        
        setEvents([
          { date: '2026-02-16', title: '完成前端重组 Phase 2' },
          { date: '2026-02-15', title: '实现结构提取算法' },
          { date: '2026-02-14', title: '发现新的几何特征' },
          { date: '2026-02-10', title: '完成观察台开发' },
          { date: '2026-02-05', title: '项目启动' }
        ]);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, []);

  // 计算总体进度
  const getOverallProgress = () => {
    if (milestones.length === 0) return 0;
    
    const completed = milestones.filter(m => m.status === 'completed').length;
    const inProgress = milestones.filter(m => m.status === 'in_progress');
    const partialProgress = inProgress.reduce((sum, m) => sum + (m.progress || 0), 0);
    
    return (completed + partialProgress) / milestones.length;
  };

  const overallProgress = getOverallProgress();

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
            { id: 'roadmap', label: '路线图', icon: '🗺️' },
            { id: 'radar', label: '研究雷达', icon: '🎯' },
            { id: 'timeline', label: '时间线', icon: '📅' }
          ].map(item => (
            <button
              key={item.id}
              onClick={() => setViewMode(item.id)}
              style={{
                padding: '8px 12px',
                background: viewMode === item.id ? 'rgba(255, 170, 0, 0.2)' : 'rgba(0,0,0,0.6)',
                border: `1px solid ${viewMode === item.id ? COLOR_SCHEMES.accent : '#444'}`,
                borderRadius: '6px',
                color: viewMode === item.id ? COLOR_SCHEMES.accent : '#888',
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

        {loading && <LoadingSpinner message="加载进度数据..." />}

        <Canvas>
          <PerspectiveCamera 
            makeDefault 
            position={viewMode === 'timeline' ? [0, 5, 10] : [8, 5, 8]} 
            fov={50} 
          />
          <OrbitControls enableDamping dampingFactor={0.05} />
          
          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={0.8} />
          
          {viewMode === 'roadmap' && milestones.length > 0 && (
            <ProgressRoadmap milestones={milestones} position={[-4, 0, 0]} />
          )}
          
          {viewMode === 'radar' && researchAreas.length > 0 && (
            <ResearchRadar areas={researchAreas} position={[0, 0, 0]} />
          )}
          
          {viewMode === 'timeline' && events.length > 0 && (
            <TimelineView events={events} position={[0, 3, 0]} />
          )}
          
          <gridHelper args={[12, 12, '#222', '#111']} />
        </Canvas>
      </div>

      {/* 右侧信息面板 */}
      <div style={{
        width: '280px',
        background: 'rgba(255,255,255,0.02)',
        borderLeft: '1px solid #333',
        padding: '16px',
        overflowY: 'auto'
      }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', color: COLOR_SCHEMES.accent }}>
          研究进度追踪
        </h3>
        
        {/* 总体进度 */}
        <div style={{
          padding: '16px',
          background: 'linear-gradient(135deg, rgba(255, 170, 0, 0.2), rgba(0, 210, 255, 0.2))',
          borderRadius: '12px',
          marginBottom: '16px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '11px', color: '#888' }}>总体进度</span>
            <span style={{ fontSize: '14px', color: '#fff', fontWeight: 'bold' }}>
              {(overallProgress * 100).toFixed(0)}%
            </span>
          </div>
          <div style={{
            width: '100%',
            height: '6px',
            background: '#222',
            borderRadius: '3px',
            overflow: 'hidden'
          }}>
            <div style={{
              width: `${overallProgress * 100}%`,
              height: '100%',
              background: `linear-gradient(90deg, ${COLOR_SCHEMES.accent}, ${COLOR_SCHEMES.primary})`,
              borderRadius: '3px'
            }} />
          </div>
        </div>
        
        {/* 里程碑状态 */}
        <div style={{ marginBottom: '16px' }}>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#666' }}>
            里程碑状态
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {milestones.map((milestone, idx) => (
              <div 
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 12px',
                  background: 'rgba(255,255,255,0.02)',
                  borderRadius: '6px',
                  borderLeft: `3px solid ${
                    milestone.status === 'completed' ? COLOR_SCHEMES.success
                    : milestone.status === 'in_progress' ? COLOR_SCHEMES.warning
                    : '#444'
                  }`
                }}
              >
                <span style={{ flex: 1, fontSize: '11px', color: '#fff' }}>
                  {milestone.name}
                </span>
                <span style={{ 
                  fontSize: '10px',
                  color: milestone.status === 'completed' ? COLOR_SCHEMES.success
                    : milestone.status === 'in_progress' ? COLOR_SCHEMES.warning
                    : '#666'
                }}>
                  {milestone.status === 'completed' ? '✓ 完成'
                    : milestone.status === 'in_progress' ? `${(milestone.progress * 100).toFixed(0)}%`
                    : '待开始'}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* 研究统计 */}
        {researchAreas.length > 0 && (
          <div style={{ marginBottom: '16px' }}>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#666' }}>
              研究领域进度
            </h4>
            <MetricGrid columns={2}>
              {researchAreas.slice(0, 4).map((area, idx) => (
                <MetricCard 
                  key={idx}
                  title={area.name}
                  value={`${(area.progress * 100).toFixed(0)}%`}
                  color={area.progress > 0.7 ? COLOR_SCHEMES.success
                    : area.progress > 0.4 ? COLOR_SCHEMES.warning
                    : '#666'}
                />
              ))}
            </MetricGrid>
          </div>
        )}

        {/* 下一步计划 */}
        <div style={{
          marginTop: '24px',
          padding: '12px',
          background: 'rgba(255, 170, 0, 0.1)',
          borderRadius: '8px',
          borderLeft: `2px solid ${COLOR_SCHEMES.accent}`
        }}>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '11px', color: COLOR_SCHEMES.accent }}>
            📋 下一步计划
          </h4>
          <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: '10px', color: '#888', lineHeight: '1.8' }}>
            <li>完成干预实验验证</li>
            <li>开发能力评估模块</li>
            <li>设计 AGI 验证测试</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default ProgressTracker;
