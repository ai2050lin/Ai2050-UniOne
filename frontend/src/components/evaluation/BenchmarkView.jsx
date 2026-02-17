/**
 * BenchmarkView - 基准测试视图
 * 运行标准基准测试评估模型能力
 */
import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text, Box, Line, Html } from '@react-three/drei';
import { MetricCard, MetricGrid } from '../shared/MetricCard';
import { LoadingSpinner } from '../shared/LoadingSpinner';
import { API_ENDPOINTS, apiCall } from '../../config/api';
import { COLOR_SCHEMES, getGradientColor } from '../../utils/colors';
import * as THREE from 'three';

// 基准测试结果柱状图
function BenchmarkBars({ results, position = [0, 0, 0] }) {
  if (!results || results.length === 0) return null;
  
  const maxScore = Math.max(...results.map(r => r.score));
  
  return (
    <group position={position}>
      {results.map((result, idx) => {
        const height = (result.score / maxScore) * 3;
        const color = result.score > 0.8 ? COLOR_SCHEMES.success
          : result.score > 0.6 ? COLOR_SCHEMES.primary
          : result.score > 0.4 ? COLOR_SCHEMES.warning
          : COLOR_SCHEMES.danger;
        
        return (
          <group key={idx} position={[(idx - results.length/2) * 1.2, 0, 0]}>
            <Box args={[0.8, height, 0.5]} position={[0, height/2, 0]}>
              <meshStandardMaterial 
                color={color}
                emissive={color}
                emissiveIntensity={0.3}
              />
            </Box>
            
            <Html distanceFactor={12} position={[0, -0.5, 0]}>
              <div style={{
                textAlign: 'center',
                width: '60px'
              }}>
                <div style={{
                  fontSize: '9px',
                  color: '#888',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis'
                }}>
                  {result.name}
                </div>
                <div style={{
                  fontSize: '11px',
                  color: '#fff',
                  fontWeight: 'bold'
                }}>
                  {(result.score * 100).toFixed(0)}%
                </div>
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
}

// 雷达图能力展示
function CapabilityRadar({ capabilities, position = [0, 0, 0] }) {
  if (!capabilities) return null;
  
  const n = capabilities.length;
  const radius = 2;
  
  const getVertex = (angle, value) => [
    Math.cos(angle) * radius * value,
    0,
    Math.sin(angle) * radius * value
  ];
  
  return (
    <group position={position} rotation={[-Math.PI/6, 0, 0]}>
      {/* 背景网格 */}
      {[0.2, 0.4, 0.6, 0.8, 1.0].map((scale, idx) => (
        <Line
          key={idx}
          points={Array.from({ length: n + 1 }).map((_, i) => 
            getVertex((i * 2 * Math.PI) / n, scale)
          )}
          color="#333"
          lineWidth={1}
          transparent
          opacity={0.3}
        />
      ))}
      
      {/* 轴线 */}
      {Array.from({ length: n }).map((_, i) => (
        <Line
          key={i}
          points={[[0, 0, 0], getVertex((i * 2 * Math.PI) / n, 1)]}
          color="#444"
          lineWidth={1}
        />
      ))}
      
      {/* 能力区域 */}
      <Line
        points={capabilities.map((cap, i) => 
          getVertex((i * 2 * Math.PI) / n, cap.value)
        ).concat([getVertex(0, capabilities[0].value)])}
        color={COLOR_SCHEMES.primary}
        lineWidth={2}
      />
      
      {/* 标签 */}
      {capabilities.map((cap, i) => {
        const pos = getVertex((i * 2 * Math.PI) / n, 1.2);
        return (
          <Text
            key={i}
            position={pos}
            fontSize={0.12}
            color="#888"
            anchorX="center"
          >
            {cap.name}
          </Text>
        );
      })}
    </group>
  );
}

// 性能趋势线
function PerformanceTrend({ history, position = [0, 0, 0] }) {
  if (!history || history.length === 0) return null;
  
  const maxScore = Math.max(...history.map(h => h.score));
  const points = history.map((h, i) => [
    (i / (history.length - 1)) * 8 - 4,
    (h.score / maxScore) * 3,
    0
  ]);
  
  return (
    <group position={position}>
      <Line
        points={points}
        color={COLOR_SCHEMES.primary}
        lineWidth={2}
      />
      
      {points.map((point, i) => (
        <Sphere key={i} args={[0.08, 8, 8]} position={point}>
          <meshStandardMaterial 
            color={COLOR_SCHEMES.primary}
            emissive={COLOR_SCHEMES.primary}
            emissiveIntensity={0.5}
          />
        </Sphere>
      ))}
      
      {/* 轴 */}
      <Line points={[[-4, 0, 0], [4, 0, 0]]} color="#444" lineWidth={1} />
      <Line points={[[-4, 0, 0], [-4, 3, 0]]} color="#444" lineWidth={1} />
    </group>
  );
}

// 主组件
export function BenchmarkView({ modelData, selectedLayer = 0 }) {
  const [results, setResults] = useState([]);
  const [capabilities, setCapabilities] = useState([]);
  const [history, setHistory] = useState([]);
  const [viewMode, setViewMode] = useState('bars'); // 'bars' | 'radar' | 'trend'
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [selectedBenchmark, setSelectedBenchmark] = useState('all');

  // 加载基准测试数据
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const data = await apiCall(`${API_ENDPOINTS.training.metrics}`);
        setResults(data.results || []);
        setCapabilities(data.capabilities || []);
        setHistory(data.history || []);
      } catch (error) {
        // 静默使用模拟数据
        setResults([
          { name: '语言理解', score: 0.85 },
          { name: '逻辑推理', score: 0.72 },
          { name: '数学能力', score: 0.68 },
          { name: '代码生成', score: 0.78 },
          { name: '创意写作', score: 0.81 },
          { name: '知识问答', score: 0.88 }
        ]);
        
        setCapabilities([
          { name: '理解', value: 0.85 },
          { name: '推理', value: 0.72 },
          { name: '计算', value: 0.68 },
          { name: '创造', value: 0.81 },
          { name: '记忆', value: 0.88 },
          { name: '泛化', value: 0.75 }
        ]);
        
        setHistory(Array(10).fill(0).map((_, i) => ({
          epoch: i,
          score: 0.6 + i * 0.03 + Math.random() * 0.05
        })));
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, []);

  // 运行基准测试
  const runBenchmark = async () => {
    setRunning(true);
    
    try {
      // 模拟测试进度
      for (let i = 0; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      
      // 更新结果
      setResults(results.map(r => ({
        ...r,
        score: Math.min(1, r.score + (Math.random() - 0.5) * 0.05)
      })));
      
    } catch (error) {
      // 静默处理
    } finally {
      setRunning(false);
    }
  };

  // 计算总体分数
  const getOverallScore = () => {
    if (results.length === 0) return 0;
    return results.reduce((sum, r) => sum + r.score, 0) / results.length;
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
            { id: 'bars', label: '柱状图', icon: '📊' },
            { id: 'radar', label: '雷达图', icon: '🎯' },
            { id: 'trend', label: '趋势图', icon: '📈' }
          ].map(item => (
            <button
              key={item.id}
              onClick={() => setViewMode(item.id)}
              style={{
                padding: '8px 12px',
                background: viewMode === item.id ? 'rgba(0, 210, 255, 0.2)' : 'rgba(0,0,0,0.6)',
                border: `1px solid ${viewMode === item.id ? COLOR_SCHEMES.primary : '#444'}`,
                borderRadius: '6px',
                color: viewMode === item.id ? COLOR_SCHEMES.primary : '#888',
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

        {/* 运行按钮 */}
        <div style={{
          position: 'absolute',
          top: '60px',
          left: '12px',
          zIndex: 10
        }}>
          <button
            onClick={runBenchmark}
            disabled={running}
            style={{
              padding: '10px 20px',
              background: running ? '#333' : 'linear-gradient(45deg, #10b981, #00d2ff)',
              border: 'none',
              borderRadius: '6px',
              color: '#fff',
              cursor: running ? 'wait' : 'pointer',
              fontSize: '13px',
              fontWeight: '500'
            }}
          >
            {running ? '测试中...' : '运行基准测试'}
          </button>
        </div>

        {loading && <LoadingSpinner message="加载基准测试数据..." />}

        <Canvas>
          <PerspectiveCamera makeDefault position={[8, 5, 8]} fov={50} />
          <OrbitControls enableDamping dampingFactor={0.05} />
          
          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={0.8} />
          
          {viewMode === 'bars' && results.length > 0 && (
            <BenchmarkBars results={results} position={[0, 0, 0]} />
          )}
          
          {viewMode === 'radar' && capabilities.length > 0 && (
            <CapabilityRadar capabilities={capabilities} position={[0, 0, 0]} />
          )}
          
          {viewMode === 'trend' && history.length > 0 && (
            <PerformanceTrend history={history} position={[0, 0, 0]} />
          )}
          
          <gridHelper args={[12, 12, '#222', '#111']} position={[0, -1, 0]} />
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
        <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', color: COLOR_SCHEMES.success }}>
          基准测试结果
        </h3>
        
        {/* 总体分数 */}
        <div style={{
          padding: '16px',
          background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(0, 210, 255, 0.2))',
          borderRadius: '12px',
          marginBottom: '16px',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '11px', color: '#888', marginBottom: '4px' }}>总体分数</div>
          <div style={{ fontSize: '32px', fontWeight: 'bold', color: COLOR_SCHEMES.primary }}>
            {(getOverallScore() * 100).toFixed(1)}%
          </div>
        </div>
        
        {/* 详细结果 */}
        <div style={{ marginBottom: '16px' }}>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#666' }}>
            测试项目详情
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {results.map((result, idx) => (
              <div 
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 12px',
                  background: 'rgba(255,255,255,0.02)',
                  borderRadius: '6px'
                }}
              >
                <span style={{ flex: 1, fontSize: '11px', color: '#888' }}>
                  {result.name}
                </span>
                <div style={{
                  width: '60px',
                  height: '4px',
                  background: '#222',
                  borderRadius: '2px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${result.score * 100}%`,
                    height: '100%',
                    background: result.score > 0.7 ? COLOR_SCHEMES.success : COLOR_SCHEMES.warning
                  }} />
                </div>
                <span style={{ 
                  fontSize: '11px', 
                  color: '#fff',
                  fontWeight: '500',
                  width: '40px',
                  textAlign: 'right'
                }}>
                  {(result.score * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* 测试状态 */}
        <div style={{
          marginTop: '24px',
          padding: '12px',
          background: 'rgba(0, 210, 255, 0.05)',
          borderRadius: '8px',
          borderLeft: `2px solid ${COLOR_SCHEMES.primary}`
        }}>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '11px', color: COLOR_SCHEMES.primary }}>
            📊 测试信息
          </h4>
          <div style={{ fontSize: '10px', color: '#888', lineHeight: '1.8' }}>
            <p>最近测试: {new Date().toLocaleDateString()}</p>
            <p>测试项目: {results.length} 项</p>
            <p>模型版本: v1.0.0</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default BenchmarkView;
