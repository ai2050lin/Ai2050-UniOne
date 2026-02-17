

import { Line, OrbitControls, Text } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { Activity, Layers, Network, Scissors, Zap } from 'lucide-react';
import { useEffect, useState } from 'react';
import * as THREE from 'three';

const API_BASE = 'http://localhost:5001';

const FiberNetV2Demo = ({ t }) => {

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  /* Animation State */
  const [animState, setAnimState] = useState({
      injecting: false,
      transporting: false,
      constraining: false
  });

  const [dataMode, setDataMode] = useState('mock'); // 'mock' | 'real'

  // Fetch simulation data from backend
  const fetchData = (mode = dataMode) => {
      setLoading(true);
      const endpoint = mode === 'real' ? '/nfb_ra/data' : '/fibernet_v2/demo';
      fetch(`${API_BASE}${endpoint}`)
          .then(res => res.json())
          .then(d => {
              if (d.error || d.detail) {
                  console.warn("Backend Error:", d.error || d.detail);
                  setDataMode('mock'); // Auto-revert
                  return;
              }
              setData(d);
              setLoading(false);
          })
          .catch(err => {
              console.error(err);
              setLoading(false);
              setDataMode('mock'); // Fallback on network error
          });
  };

  useEffect(() => {
      fetchData(dataMode);
  }, [dataMode]);

  /* Surgery State */
  const [surgeryMode, setSurgeryMode] = useState(false);
  const [surgeryTool, setSurgeryTool] = useState('graft'); // 'graft' | 'ablate'
  const [selection, setSelection] = useState([]); // [source_id, target_id]

  /* Surgery Handlers */
  const handleSurgeryClick = (nodeId) => {
      if (!surgeryMode) return;

      if (surgeryTool === 'ablate') {
          // Immediate Ablation Confirmation
          if (confirm(`Confirm Ablation of Concept Node ${nodeId}?`)) {
              performSurgery('ablate', nodeId);
          }
      } else if (surgeryTool === 'graft') {
          // Source-Target Selection
          if (selection.length === 0) {
              setSelection([nodeId]);
          } else if (selection.length === 1) {
              if (selection[0] === nodeId) return; // Ignore self-select
              if (confirm(`Graft connection from ${selection[0]} to ${nodeId}?`)) {
                  performSurgery('graft', selection[0], nodeId);
              }
              setSelection([]); // Reset
          }
      }
  };

  const performSurgery = (action, src, tgt = null) => {
      // Validation: Mock Mode
      if (dataMode === 'mock') {
          alert("模拟模式: 手术仅用于演示。请切换到真实数据模式以应用实际修改。");
          return;
      }

      // Parse IDs safely
      const s_id = parseInt(src);
      const t_id = tgt ? parseInt(tgt) : null;

      // Validation: Invalid IDs
      if (isNaN(s_id) || (action === 'graft' && isNaN(t_id))) {
          console.error("Invalid Node IDs for Surgery:", { src, tgt, s_id, t_id });
          alert("错误: 无效的节点ID");
          return;
      }
      
      const payload = {
          action: action,
          source_id: action === 'graft' ? s_id : null,
          target_id: action === 'graft' ? t_id : s_id,
          layer: 6,
          strength: 1.5
      };

      console.log("Sending Surgery Payload:", payload);

      fetch(`${API_BASE}/nfb_ra/surgery`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
      })
      .then(async res => {
          const d = await res.json();
          if (!res.ok) throw new Error(d.detail || res.statusText);
          return d;
      })
      .then(d => {
          alert(`手术成功: ${d.message}`);
          fetchData(); 
      })
      .catch(err => {
          console.error("Surgery Error:", err);
          alert("手术失败: " + err.message);
      });
  };

  // Controls
  const handleInject = () => {
      setAnimState(prev => ({ ...prev, injecting: true }));
      setTimeout(() => setAnimState(prev => ({ ...prev, injecting: false })), 2000);
      if (dataMode === 'mock') fetchData(); 
  };

  const handleTransport = () => {
      setAnimState(prev => ({ ...prev, transporting: true }));
      setTimeout(() => setAnimState(prev => ({ ...prev, transporting: false })), 3000);
  };

  const handleConstraint = () => {
      setAnimState(prev => ({ ...prev, constraining: true }));
      setTimeout(() => setAnimState(prev => ({ ...prev, constraining: false })), 2000);
  };

  if (loading) return <div style={{color: '#888', textAlign: 'center', padding: '20px', fontSize: '12px'}}>加载神经纤维仿真数据...</div>;

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        
        {/* 简介说明 */}
        <div style={{ 
          background: 'rgba(78, 205, 196, 0.1)', 
          padding: '10px', 
          borderRadius: '6px', 
          border: '1px solid rgba(78, 205, 196, 0.2)',
          fontSize: '11px',
          color: '#aaa',
          lineHeight: '1.5'
        }}>
          <div style={{ color: '#4ecdc4', fontWeight: 'bold', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Network size={14} /> Neural Fiber Bundle 仿真
          </div>
          <div>探索基于神经纤维丛理论的表示空间几何结构。</div>
        </div>

        {/* 数据源切换 */}
        <div style={{ display: 'flex', gap: '6px' }}>
          <button
              onClick={() => setDataMode('mock')}
              style={{ 
                  flex: 1,
                  background: dataMode === 'mock' ? '#4488ff' : '#333', 
                  color: '#fff', 
                  border: 'none', 
                  padding: '8px', 
                  borderRadius: '4px', 
                  fontSize: '11px',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
              }}
          >
              演示模式
          </button>
          <button
              onClick={() => setDataMode('real')}
              style={{ 
                  flex: 1,
                  background: dataMode === 'real' ? '#4ecdc4' : '#333', 
                  color: '#fff', 
                  border: 'none', 
                  padding: '8px', 
                  borderRadius: '4px', 
                  fontSize: '11px',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
              }}
          >
              真实数据
          </button>
        </div>

        {/* 动画控制 */}
        <div style={{ 
          background: 'rgba(255,255,255,0.03)', 
          padding: '10px', 
          borderRadius: '6px',
          border: '1px solid rgba(255,255,255,0.05)'
        }}>
          <div style={{ fontSize: '11px', color: '#888', marginBottom: '8px', fontWeight: 'bold' }}>
            动画演示
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px' }}>
              <button 
                onClick={handleInject} 
                title="向纤维丛注入信息"
                style={{ 
                  background: animState.injecting ? '#5ec962' : 'rgba(255,255,255,0.05)', 
                  color: animState.injecting ? '#000' : '#aaa', 
                  border: '1px solid rgba(255,255,255,0.1)', 
                  padding: '8px 4px', 
                  borderRadius: '4px', 
                  fontSize: '10px',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '4px'
                }}
              >
                  <Zap size={14} />
                  注入
              </button>
              <button 
                onClick={handleTransport} 
                title="展示信息传输路径"
                style={{ 
                  background: animState.transporting ? '#4488ff' : 'rgba(255,255,255,0.05)', 
                  color: animState.transporting ? '#fff' : '#aaa', 
                  border: '1px solid rgba(255,255,255,0.1)', 
                  padding: '8px 4px', 
                  borderRadius: '4px', 
                  fontSize: '10px',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '4px'
                }}
              >
                  <Activity size={14} />
                  传输
              </button>
              <button 
                onClick={handleConstraint} 
                title="展示流形约束"
                style={{ 
                  background: animState.constraining ? '#ff6b6b' : 'rgba(255,255,255,0.05)', 
                  color: animState.constraining ? '#fff' : '#aaa', 
                  border: '1px solid rgba(255,255,255,0.1)', 
                  padding: '8px 4px', 
                  borderRadius: '4px', 
                  fontSize: '10px',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '4px'
                }}
              >
                  <Layers size={14} />
                  流形
              </button>
          </div>
        </div>

        {/* 手术模式 */}
        <div style={{ 
          background: surgeryMode ? 'rgba(255,107,107,0.1)' : 'rgba(255,255,255,0.03)', 
          padding: '10px', 
          borderRadius: '6px',
          border: surgeryMode ? '1px solid rgba(255,107,107,0.3)' : '1px solid rgba(255,255,255,0.05)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <div style={{ fontSize: '11px', color: surgeryMode ? '#ff6b6b' : '#888', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Scissors size={14} /> 流形手术
            </div>
            <button 
              onClick={() => setSurgeryMode(!surgeryMode)}
              style={{ 
                background: surgeryMode ? '#ff4444' : 'transparent', 
                color: surgeryMode ? '#fff' : '#888', 
                border: `1px solid ${surgeryMode ? '#ff4444' : 'rgba(255,255,255,0.2)'}`, 
                padding: '4px 8px', 
                borderRadius: '4px', 
                fontSize: '10px',
                cursor: 'pointer'
              }}
            >
              {surgeryMode ? '关闭' : '启用'}
            </button>
          </div>
          
          {surgeryMode && (
            <div style={{ display: 'flex', gap: '6px', marginTop: '8px' }}>
              <button 
                onClick={() => setSurgeryTool('graft')}
                style={{ 
                  flex: 1,
                  background: surgeryTool === 'graft' ? '#4488ff' : '#222', 
                  color: '#fff', 
                  border: '1px solid #4488ff', 
                  padding: '6px', 
                  borderRadius: '4px', 
                  fontSize: '10px', 
                  cursor: 'pointer' 
                }}
              >
                连接 (Graft)
              </button>
              <button 
                onClick={() => setSurgeryTool('ablate')}
                style={{ 
                  flex: 1,
                  background: surgeryTool === 'ablate' ? '#ff4444' : '#222', 
                  color: '#fff', 
                  border: '1px solid #ff4444', 
                  padding: '6px', 
                  borderRadius: '4px', 
                  fontSize: '10px', 
                  cursor: 'pointer' 
                }}
              >
                切除 (Ablate)
              </button>
            </div>
          )}

          {surgeryMode && (
            <div style={{ fontSize: '10px', color: '#666', marginTop: '8px', fontStyle: 'italic' }}>
              {surgeryTool === 'graft' ? (selection.length === 0 ? "选择源节点..." : "选择目标节点...") : "点击节点以切除"}
            </div>
          )}
        </div>

        {/* 图例说明 */}
        <div style={{ 
          background: 'rgba(255,255,255,0.02)', 
          padding: '8px', 
          borderRadius: '4px',
          fontSize: '10px',
          color: '#666'
        }}>
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <span><span style={{color: '#00ffff'}}>●</span> 流形节点</span>
            <span><span style={{color: '#ffaa00'}}>●</span> 概念节点</span>
            <span><span style={{color: '#fff'}}>│</span> 纤维向量</span>
          </div>
        </div>

        {/* 3D Scene */}
        <div style={{ flex: 1, minHeight: '200px', background: '#0a0a0a', borderRadius: '6px', overflow: 'hidden' }}>
            <Canvas camera={{ position: [5, 5, 8], fov: 45 }}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={1} />
                <OrbitControls />

                {/* Grid Helper for Manifold */}
                <gridHelper args={[10, 10, 0x444444, 0x222222]} position={[0, -0.1, 0]} />

                {/* Manifold Constraint Plane */}
                <mesh rotation={[-Math.PI/2, 0, 0]} position={[0, -0.15, 0]}>
                    <planeGeometry args={[10, 10]} />
                    <meshBasicMaterial color="#00ffff" transparent opacity={0.1} side={THREE.DoubleSide} />
                </mesh>

                {/* Data Points */}
                {data && (
                    <group>
                        {/* Manifold Point Cloud */}
                        {data.manifold_points?.map((pt, i) => (
                            <mesh key={`pt-${i}`} position={pt.pos}>
                                <sphereGeometry args={[pt.type === 'concept' ? 0.08 : 0.03, 8, 8]} />
                                <meshBasicMaterial 
                                    color={pt.type === 'concept' ? '#ffaa00' : '#888888'} 
                                    transparent 
                                    opacity={pt.type === 'concept' ? 1.0 : 0.3} 
                                />
                                {pt.type === 'concept' && (
                                     <Text position={[0, 0.2, 0]} fontSize={0.12} color="#ffaa00" anchorX="center" anchorY="bottom">
                                         {pt.text.length > 15 ? pt.text.substring(0, 15) + '...' : pt.text}
                                     </Text>
                                )}
                            </mesh>
                        ))}

                        {/* Manifold Nodes */}
                        {data.manifold_nodes?.map((node, i) => {
                            const isSelected = selection.includes(node.id);
                            const isSource = selection[0] === node.id;
                            
                            return (
                                <mesh 
                                    key={node.id} 
                                    position={node.pos}
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleSurgeryClick(node.id);
                                    }}
                                    onPointerOver={(e) => {
                                        e.stopPropagation();
                                        if (surgeryMode) document.body.style.cursor = 'crosshair';
                                    }}
                                    onPointerOut={(e) => {
                                        document.body.style.cursor = 'default';
                                    }}
                                >
                                    <sphereGeometry args={[isSelected ? 0.12 : 0.08, 12, 12]} />
                                    <meshStandardMaterial 
                                        color={isSelected ? (isSource ? "#44ff44" : "#ffaa00") : (animState.constraining ? "#ff0000" : "#00ffff")} 
                                        emissive={isSelected ? (isSource ? "#44ff44" : "#ffaa00") : (animState.constraining ? "#ff0000" : "#0044aa")}
                                        emissiveIntensity={isSelected ? 1.0 : 0.5}
                                    />
                                </mesh>
                            );
                        })}

                        {/* Fibers */}
                        {data.fibers?.map((fiber, i) => {
                            const parent = data.manifold_nodes?.find(n => n.id === fiber.parent_id);
                            if (!parent) return null;
                            const isInjectTarget = i === data.fibers.length - 1 && animState.injecting;
                            
                            return (
                                <group key={i} position={parent.pos}>
                                    <mesh position={[0, fiber.height/2, 0]}>
                                        <cylinderGeometry args={[0.015, 0.015, fiber.height, 8]} />
                                        <meshStandardMaterial 
                                            color={isInjectTarget ? "#44ff44" : "#ffffff"} 
                                            emissive={isInjectTarget ? "#44ff44" : "#ffffff"} 
                                            emissiveIntensity={0.5} 
                                        />
                                    </mesh>
                                    <mesh position={[0, fiber.height, 0]}>
                                        <sphereGeometry args={[0.1, 12, 12]} />
                                        <meshStandardMaterial 
                                            color={`hsl(${fiber.color_intensity * 360}, 80%, 50%)`} 
                                            emissive={`hsl(${fiber.color_intensity * 360}, 80%, 50%)`}
                                            emissiveIntensity={0.8}
                                        />
                                    </mesh>
                                    {isInjectTarget && (
                                        <mesh position={[0, fiber.height, 0]}>
                                            <sphereGeometry args={[0.3, 12, 12]} />
                                            <meshBasicMaterial color="#44ff44" transparent opacity={0.3} wireframe />
                                        </mesh>
                                    )}
                                </group>
                            );
                        })}

                        {/* Transport Links */}
                        {data.connections?.map((conn, i) => {
                            const src = data.manifold_nodes?.find(n => n.id === conn.source);
                            const tgt = data.manifold_nodes?.find(n => n.id === conn.target);
                            if (!src || !tgt || !animState.transporting) return null;

                            const mid = [
                                (src.pos[0] + tgt.pos[0]) / 2,
                                (src.pos[1] + tgt.pos[1]) / 2 + 1.5,
                                (src.pos[2] + tgt.pos[2]) / 2
                            ];
                            
                            const curve = new THREE.QuadraticBezierCurve3(
                                new THREE.Vector3(...src.pos),
                                new THREE.Vector3(...mid),
                                new THREE.Vector3(...tgt.pos)
                            );
                            
                            const points = curve.getPoints(20);

                            return (
                                <group key={i}>
                                    <Line points={points} color="#4488ff" lineWidth={2} opacity={0.8} transparent />
                                    <mesh position={mid}>
                                        <sphereGeometry args={[0.08, 8, 8]} />
                                        <meshBasicMaterial color="#ffffff" />
                                    </mesh>
                                </group>
                            );
                        })}
                    </group>
                )}
            </Canvas>
        </div>
    </div>
  );
};

export default FiberNetV2Demo;
