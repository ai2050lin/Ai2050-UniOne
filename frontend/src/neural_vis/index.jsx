import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Stars, Text, Line, Html } from '@react-three/drei';
import * as THREE from 'three';

// ==================== 常量 ====================
const LAYER_GAP = 3.5;
const PLANE_SIZE = 18;
const SPHERE_BASE_SIZE = 0.2;
const TRAJECTORY_LINE_WIDTH = 3;

// ==================== 颜色方案 ====================
const CATEGORY_COLORS = {
  fruit: '#ff6b6b', animal: '#4ecdc4', vehicle: '#ffe66d', tool: '#a855f7',
  nature: '#34d399', food: '#f97316', person: '#ec4899', abstract: '#6366f1',
};

const LAYER_FUNC_COLORS = {
  lexical: '#ff6b6b', semantic: '#4ecdc4', syntactic: '#ffe66d', decision: '#a855f7',
};

function deltaCosToColor(deltaCos) {
  const r = Math.max(0, Math.min(1, deltaCos));
  let red, green, blue;
  if (r > 0.5) {
    const t = (r - 0.5) * 2;
    red = Math.round(239 * t + 245 * (1 - t));
    green = Math.round(68 * t + 158 * (1 - t));
    blue = Math.round(68 * t + 11 * (1 - t));
  } else {
    const t = r * 2;
    red = Math.round(245 * t + 59 * (1 - t));
    green = Math.round(158 * t + 130 * (1 - t));
    blue = Math.round(11 * t + 246 * (1 - t));
  }
  return `#${red.toString(16).padStart(2,'0')}${green.toString(16).padStart(2,'0')}${blue.toString(16).padStart(2,'0')}`;
}

// ==================== TrajectoryRenderer ====================
function TrajectoryRenderer({ trajectory, animated, animationProgress, onHoverToken }) {
  const points = trajectory.points || [];
  const visibleCount = animated
    ? Math.max(1, Math.floor(points.length * animationProgress))
    : points.length;
  const visiblePoints = points.slice(0, visibleCount);

  if (visiblePoints.length < 2) return null;

  const linePoints = visiblePoints.map(p => [p.x, p.y, p.z]);

  return (
    <group>
      {/* 轨迹线 */}
      <Line
        points={linePoints}
        color={trajectory.color || '#ffffff'}
        lineWidth={TRAJECTORY_LINE_WIDTH}
        transparent
        opacity={0.7}
      />
      {/* 逐层标记球 */}
      {visiblePoints.map((pt, i) => {
        const color = deltaCosToColor(pt.delta_cos ?? 0.5);
        const size = SPHERE_BASE_SIZE + (pt.norm || 10) * 0.0008;
        const isCorrection = trajectory.correction_layers?.includes(pt.layer);
        return (
          <group key={i} position={[pt.x, pt.y, pt.z]}>
            <mesh
              onPointerOver={(e) => {
                e.stopPropagation();
                onHoverToken?.({
                  token: trajectory.token,
                  source: trajectory.source_token,
                  layer: pt.layer,
                  delta_cos: pt.delta_cos,
                  cos_with_target: pt.cos_with_target,
                  norm: pt.norm,
                  isCorrection,
                });
              }}
            >
              <sphereGeometry args={[size, 16, 16]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.5}
                roughness={0.3}
                metalness={0.6}
              />
            </mesh>
            {/* 纠正层环标记 */}
            {isCorrection && (
              <mesh rotation={[Math.PI / 2, 0, 0]}>
                <torusGeometry args={[size + 0.15, 0.04, 8, 32]} />
                <meshStandardMaterial
                  color="#fbbf24"
                  emissive="#fbbf24"
                  emissiveIntensity={2}
                />
              </mesh>
            )}
            {/* 层标签 (仅关键层显示) */}
            {(i === 0 || i === visiblePoints.length - 1 || isCorrection) && (
              <Text
                position={[0, size + 0.4, 0]}
                fontSize={0.35}
                color="#e2e8f0"
                anchorX="center"
                anchorY="bottom"
                outlineWidth={0.05}
                outlineColor="#000000"
              >
                L{pt.layer}
              </Text>
            )}
          </group>
        );
      })}
    </group>
  );
}

// ==================== PointCloudRenderer ====================
function PointCloudRenderer({ pointCloud, onHoverToken }) {
  const points = pointCloud.points || [];
  const catColors = pointCloud.categories || CATEGORY_COLORS;

  return (
    <group>
      {points.map((pt, i) => {
        const color = catColors[pt.category] || '#888888';
        const size = SPHERE_BASE_SIZE + (pt.norm || 10) * 0.0006;
        return (
          <group key={i} position={[pt.x, pt.y, pt.z]}>
            <mesh
              onPointerOver={(e) => {
                e.stopPropagation();
                onHoverToken?.({
                  token: pt.token,
                  category: pt.category,
                  layer: pointCloud.layer,
                  norm: pt.norm,
                  activation: pt.activation,
                });
              }}
            >
              <sphereGeometry args={[size, 12, 12]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.3}
                roughness={0.4}
                metalness={0.5}
              />
            </mesh>
            {/* 悬停标签通过外部tooltip实现 */}
          </group>
        );
      })}
    </group>
  );
}

// ==================== LayerStackRenderer ====================
function LayerStackRenderer({ layerStack, selectedLayers, trajectoryData }) {
  const layers = (layerStack?.layers || []).filter(
    l => !selectedLayers || selectedLayers.includes(l.layer)
  );

  return (
    <group>
      {layers.map((layer, i) => {
        const yPos = i * LAYER_GAP;
        const color = layer.color || LAYER_FUNC_COLORS[layer.function] || '#4ecdc4';
        return (
          <group key={layer.layer} position={[0, yPos, 0]}>
            {/* 透明层板 */}
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
              <planeGeometry args={[PLANE_SIZE, PLANE_SIZE]} />
              <meshStandardMaterial
                color={color}
                transparent
                opacity={0.06}
                side={THREE.DoubleSide}
                depthWrite={false}
              />
            </mesh>
            {/* 层边框 */}
            <Line
              points={[
                [-PLANE_SIZE/2, 0, -PLANE_SIZE/2],
                [PLANE_SIZE/2, 0, -PLANE_SIZE/2],
                [PLANE_SIZE/2, 0, PLANE_SIZE/2],
                [-PLANE_SIZE/2, 0, PLANE_SIZE/2],
                [-PLANE_SIZE/2, 0, -PLANE_SIZE/2],
              ]}
              color={color}
              lineWidth={1}
              transparent
              opacity={0.3}
            />
            {/* 层标签 */}
            <Text
              position={[-PLANE_SIZE / 2 - 1.5, 0, 0]}
              fontSize={0.45}
              color={color}
              anchorX="right"
              anchorY="middle"
            >
              L{layer.layer} {layer.label || ''}
            </Text>
            {/* 指标摘要 */}
            {layer.metrics && (
              <Text
                position={[PLANE_SIZE / 2 + 0.5, 0, 0]}
                fontSize={0.25}
                color="#94a3b8"
                anchorX="left"
                anchorY="middle"
              >
                {`δ=${(layer.metrics.avg_delta_cos ?? 0).toFixed(2)} sw=${(layer.metrics.switch_rate ?? 0).toFixed(2)}`}
              </Text>
            )}
          </group>
        );
      })}
    </group>
  );
}

// ==================== Heatmap3DRenderer ====================
function Heatmap3DRenderer({ heatmap }) {
  const cells = heatmap?.cells || [];
  const xValues = heatmap?.x_axis?.values || [];
  const yValues = heatmap?.y_axis?.values || [];
  const zRange = heatmap?.z_axis?.range || [0, 1];

  return (
    <group>
      {cells.map((cell, i) => {
        const height = ((cell.value - zRange[0]) / (zRange[1] - zRange[0] + 1e-10)) * 5;
        const xPos = (cell.x - xValues.length / 2) * 1.2;
        const zPos = (cell.y - yValues.length / 2) * 1.2;
        const color = deltaCosToColor(cell.value);
        return (
          <group key={i} position={[xPos, height / 2, zPos]}>
            <mesh>
              <boxGeometry args={[0.9, Math.max(0.05, height), 0.9]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.3}
                roughness={0.5}
              />
            </mesh>
          </group>
        );
      })}
      {/* X轴标签 */}
      {xValues.map((xv, i) => (
        <Text
          key={`x${i}`}
          position={[(i - xValues.length / 2) * 1.2, -0.5, -(yValues.length / 2 + 1) * 1.2]}
          fontSize={0.3}
          color="#94a3b8"
          anchorX="center"
        >
          {String(xv)}
        </Text>
      ))}
    </group>
  );
}

// ==================== FlowRenderer ====================
function FlowRenderer({ flow, animated, animationProgress }) {
  const flows = flow?.flows || [];
  const nodes = flow?.node_positions || [];

  // 构建node位置映射
  const nodeMap = {};
  nodes.forEach(n => { nodeMap[n.id] = n; });

  return (
    <group>
      {/* 节点 */}
      {nodes.map((node, i) => (
        <group key={`node${i}`} position={[node.x, node.y, node.z]}>
          <mesh>
            <sphereGeometry args={[0.4, 16, 16]} />
            <meshStandardMaterial color="#4ecdc4" emissive="#4ecdc4" emissiveIntensity={0.5} />
          </mesh>
          <Text position={[0, 0.7, 0]} fontSize={0.3} color="#e2e8f0" anchorX="center">
            {node.token}
          </Text>
        </group>
      ))}
      {/* 注意力弧线 */}
      {flows.map((f, i) => {
        const src = nodeMap[f.source];
        const tgt = nodeMap[f.target];
        if (!src || !tgt) return null;
        
        // 弧线: 从source向上弯到target
        const midY = Math.max(src.y || 0, tgt.y || 0) + 1 + f.weight * 2;
        const curvePoints = [];
        const segments = 20;
        for (let s = 0; s <= segments; s++) {
          const t = s / segments;
          const x = (src.x || 0) * (1 - t) + (tgt.x || 0) * t;
          const y = ((src.y || 0) * (1 - t) + (tgt.y || 0) * t) + Math.sin(t * Math.PI) * midY;
          const z = (src.z || 0) * (1 - t) + (tgt.z || 0) * t;
          curvePoints.push([x, y, z]);
        }
        
        return (
          <Line
            key={`flow${i}`}
            points={curvePoints}
            color={f.color || '#4ecdc4'}
            lineWidth={1 + f.weight * 4}
            transparent
            opacity={0.3 + f.weight * 0.5}
          />
        );
      })}
    </group>
  );
}

// ==================== SceneHelpers ====================
function SceneHelpers({ nLayers = 36 }) {
  return (
    <group>
      {/* 地面网格 */}
      <gridHelper args={[40, 40, '#1e293b', '#0f172a']} position={[0, -1, 0]} />
      {/* Y轴标尺 (层号) */}
      {Array.from({ length: Math.min(nLayers, 37) }, (_, i) => (
        i % 6 === 0 ? (
          <Text
            key={`y${i}`}
            position={[-PLANE_SIZE / 2 - 3, i * LAYER_GAP, 0]}
            fontSize={0.3}
            color="#64748b"
            anchorX="right"
          >
            L{i}
          </Text>
        ) : null
      ))}
    </group>
  );
}

// ==================== Tooltip ====================
function HoverTooltip({ data }) {
  if (!data) return null;
  return (
    <Html style={{ pointerEvents: 'none' }}>
      <div style={{
        background: 'rgba(15, 23, 42, 0.95)',
        border: '1px solid #334155',
        borderRadius: '8px',
        padding: '8px 12px',
        color: '#e2e8f0',
        fontSize: '12px',
        fontFamily: 'monospace',
        whiteSpace: 'nowrap',
        boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
      }}>
        {data.token && <div style={{ color: '#60a5fa', fontWeight: 'bold' }}>{data.token}</div>}
        {data.source && <div style={{ color: '#94a3b8' }}>from: {data.source}</div>}
        {data.layer !== undefined && <div>Layer: {data.layer}</div>}
        {data.delta_cos !== undefined && <div>δ_cos: {data.delta_cos.toFixed(4)}</div>}
        {data.cos_with_target !== undefined && <div>cos(target): {data.cos_with_target.toFixed(4)}</div>}
        {data.norm !== undefined && <div>norm: {data.norm.toFixed(1)}</div>}
        {data.category && <div style={{ color: CATEGORY_COLORS[data.category] || '#888' }}>cat: {data.category}</div>}
        {data.isCorrection && <div style={{ color: '#fbbf24', fontWeight: 'bold' }}>⚡ Correction Layer</div>}
      </div>
    </Html>
  );
}

// ==================== Data Loading Hook ====================
function useVisData() {
  const [dataFiles, setDataFiles] = useState([]);
  const [activeData, setActiveData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadDataManifest = useCallback(async () => {
    try {
      const resp = await fetch('/vis_data/manifest.json');
      const manifest = await resp.json();
      setDataFiles(manifest.files || []);
    } catch {
      setDataFiles([]);
    }
  }, []);

  const loadDataFile = useCallback(async (filepath) => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`/vis_data/${filepath}`);
      const data = await resp.json();
      if (data.schema_version !== '1.0') {
        throw new Error(`Unsupported schema: ${data.schema_version}`);
      }
      setActiveData(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadLocalFile = useCallback((file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        if (data.schema_version !== '1.0') {
          throw new Error(`Unsupported schema: ${data.schema_version}`);
        }
        setActiveData(data);
      } catch (err) {
        setError(err.message);
      }
    };
    reader.readAsText(file);
  }, []);

  return { dataFiles, activeData, loading, error, loadDataManifest, loadDataFile, loadLocalFile };
}

// ==================== 主组件 ====================
export default function NeuralVis3DApp() {
  const { dataFiles, activeData, loading, error, loadDataManifest, loadDataFile, loadLocalFile } = useVisData();
  const [animProgress, setAnimProgress] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [hoveredInfo, setHoveredInfo] = useState(null);
  const [selectedLayers, setSelectedLayers] = useState(null);
  const [viewMode, setViewMode] = useState('all'); // all | trajectory | point_cloud | heatmap | flow
  const fileInputRef = useRef();

  useEffect(() => { loadDataManifest(); }, [loadDataManifest]);

  const visualizations = activeData?.visualizations || [];
  const trajectories = visualizations.filter(v => v.type === 'trajectory' && (viewMode === 'all' || viewMode === 'trajectory'));
  const pointClouds = visualizations.filter(v => v.type === 'point_cloud' && (viewMode === 'all' || viewMode === 'point_cloud'));
  const heatmaps = visualizations.filter(v => v.type === 'heatmap_3d' && (viewMode === 'all' || viewMode === 'heatmap'));
  const flows = visualizations.filter(v => v.type === 'flow' && (viewMode === 'all' || viewMode === 'flow'));
  const layerStacks = visualizations.filter(v => v.type === 'layer_stack');

  // 动画循环
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      setAnimProgress(prev => {
        if (prev >= 1) { setPlaying(false); return 1; }
        return prev + 0.01;
      });
    }, 50);
    return () => clearInterval(timer);
  }, [playing]);

  return (
    <div style={{ display: 'flex', height: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: 'system-ui, sans-serif' }}>
      {/* 左侧面板 */}
      <div style={{ width: 280, padding: 16, borderRight: '1px solid #1e293b', overflowY: 'auto', flexShrink: 0 }}>
        <h2 style={{ fontSize: 16, margin: '0 0 16px 0', color: '#60a5fa' }}>Neural Vis 3D</h2>

        {/* 数据源 */}
        <div style={{ marginBottom: 20 }}>
          <h3 style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>数据源</h3>
          <button
            onClick={() => fileInputRef.current?.click()}
            style={{ width: '100%', padding: '8px', background: '#1e293b', border: '1px solid #334155', borderRadius: 6, color: '#e2e8f0', cursor: 'pointer', marginBottom: 8 }}
          >
            📂 加载本地JSON文件
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            style={{ display: 'none' }}
            onChange={(e) => e.target.files[0] && loadLocalFile(e.target.files[0])}
          />
          <button
            onClick={loadDataManifest}
            style={{ width: '100%', padding: '8px', background: '#1e293b', border: '1px solid #334155', borderRadius: 6, color: '#e2e8f0', cursor: 'pointer' }}
          >
            🔄 刷新文件列表
          </button>
          {dataFiles.length > 0 && (
            <div style={{ marginTop: 8 }}>
              {dataFiles.map((f, i) => (
                <button
                  key={i}
                  onClick={() => loadDataFile(f.filename)}
                  style={{ display: 'block', width: '100%', padding: '6px 8px', background: '#0f172a', border: '1px solid #1e293b', borderRadius: 4, color: '#94a3b8', cursor: 'pointer', textAlign: 'left', fontSize: 11, marginBottom: 4 }}
                >
                  {f.label || f.filename}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* 可视化模式 */}
        <div style={{ marginBottom: 20 }}>
          <h3 style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>可视化模式</h3>
          {['all', 'trajectory', 'point_cloud', 'heatmap', 'flow'].map(mode => (
            <button
              key={mode}
              onClick={() => setViewMode(mode)}
              style={{
                display: 'block', width: '100%', padding: '6px 8px', marginBottom: 4,
                background: viewMode === mode ? '#1e40af' : '#1e293b',
                border: viewMode === mode ? '1px solid #3b82f6' : '1px solid #334155',
                borderRadius: 4, color: viewMode === mode ? '#bfdbfe' : '#94a3b8',
                cursor: 'pointer', textAlign: 'left', fontSize: 12,
              }}
            >
              {mode === 'all' ? '🔍 全部' : mode === 'trajectory' ? '📈 轨迹' : mode === 'point_cloud' ? '⚪ 点云' : mode === 'heatmap' ? '📊 热力图' : '🔀 信息流'}
              <span style={{ float: 'right', color: '#64748b' }}>
                {mode === 'all' ? visualizations.length : mode === 'trajectory' ? trajectories.length : mode === 'point_cloud' ? pointClouds.length : mode === 'heatmap' ? heatmaps.length : flows.length}
              </span>
            </button>
          ))}
        </div>

        {/* 动画控制 */}
        <div style={{ marginBottom: 20 }}>
          <h3 style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>动画控制</h3>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={() => { setAnimProgress(0); setPlaying(true); }}
              style={{ flex: 1, padding: '6px', background: '#1e293b', border: '1px solid #334155', borderRadius: 4, color: '#e2e8f0', cursor: 'pointer', fontSize: 12 }}
            >
              ▶ 播放
            </button>
            <button
              onClick={() => setPlaying(false)}
              style={{ flex: 1, padding: '6px', background: '#1e293b', border: '1px solid #334155', borderRadius: 4, color: '#e2e8f0', cursor: 'pointer', fontSize: 12 }}
            >
              ⏸ 暂停
            </button>
            <button
              onClick={() => { setAnimProgress(1); setPlaying(false); }}
              style={{ flex: 1, padding: '6px', background: '#1e293b', border: '1px solid #334155', borderRadius: 4, color: '#e2e8f0', cursor: 'pointer', fontSize: 12 }}
            >
              ⏹ 重置
            </button>
          </div>
          <div style={{ marginTop: 8, fontSize: 11, color: '#64748b' }}>
            进度: {(animProgress * 100).toFixed(0)}%
          </div>
        </div>

        {/* 数据摘要 */}
        {activeData && (
          <div>
            <h3 style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>数据摘要</h3>
            <div style={{ fontSize: 11, lineHeight: 1.6 }}>
              <div>Phase: <span style={{ color: '#60a5fa' }}>{activeData.phase}</span></div>
              <div>Model: <span style={{ color: '#4ecdc4' }}>{activeData.model}</span></div>
              <div>Exp: <span style={{ color: '#ffe66d' }}>{activeData.experiment}</span></div>
              <div>Layers: {activeData.model_info?.n_layers}</div>
              <div>d_model: {activeData.model_info?.d_model}</div>
              <div>时间: {activeData.timestamp}</div>
            </div>
          </div>
        )}

        {/* 颜色图例 */}
        <div style={{ marginTop: 20 }}>
          <h3 style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>δ_cos 颜色图例</h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            {[1, 0.75, 0.5, 0.25, 0].map(v => (
              <div key={v} style={{ textAlign: 'center' }}>
                <div style={{ width: 24, height: 12, background: deltaCosToColor(v), borderRadius: 2 }} />
                <div style={{ fontSize: 9, color: '#64748b' }}>{v.toFixed(2)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 中央3D画布 */}
      <div style={{ flex: 1, position: 'relative' }}>
        {loading && (
          <div style={{ position: 'absolute', top: 20, left: 20, zIndex: 10, background: '#1e293b', padding: '8px 16px', borderRadius: 6, fontSize: 13 }}>
            加载中...
          </div>
        )}
        {error && (
          <div style={{ position: 'absolute', top: 20, left: 20, zIndex: 10, background: '#7f1d1d', padding: '8px 16px', borderRadius: 6, fontSize: 13 }}>
            错误: {error}
          </div>
        )}
        {!activeData && (
          <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 10, textAlign: 'center', color: '#64748b' }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>🧠</div>
            <div style={{ fontSize: 16 }}>请加载可视化数据文件</div>
            <div style={{ fontSize: 12, marginTop: 8 }}>点击左侧"加载本地JSON文件"或选择已有数据</div>
          </div>
        )}
        <Canvas
          camera={{ position: [25, 20, 25], fov: 50 }}
          gl={{ antialias: true, alpha: true }}
          style={{ background: '#0f172a' }}
        >
          <PerspectiveCamera makeDefault position={[25, 20, 25]} fov={50} />
          <OrbitControls enableDamping dampingFactor={0.1} minDistance={5} maxDistance={100} />

          {/* 灯光 */}
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 20, 10]} intensity={0.8} />
          <pointLight position={[-10, -10, -10]} intensity={0.3} />

          {/* 背景 */}
          <Stars radius={100} depth={50} count={1500} factor={4} fade speed={0.5} />

          {/* 场景辅助 */}
          <SceneHelpers nLayers={activeData?.model_info?.n_layers} />

          {/* 渲染各类型 */}
          {layerStacks.map(ls => (
            <LayerStackRenderer key={ls.id} layerStack={ls} selectedLayers={selectedLayers} />
          ))}
          {trajectories.map(traj => (
            <TrajectoryRenderer
              key={traj.id}
              trajectory={traj}
              animated={animProgress < 1}
              animationProgress={animProgress}
              onHoverToken={setHoveredInfo}
            />
          ))}
          {pointClouds.map(pc => (
            <PointCloudRenderer key={pc.id} pointCloud={pc} onHoverToken={setHoveredInfo} />
          ))}
          {heatmaps.map(hm => (
            <Heatmap3DRenderer key={hm.id} heatmap={hm} />
          ))}
          {flows.map(fl => (
            <FlowRenderer key={fl.id} flow={fl} animated={animProgress < 1} animationProgress={animProgress} />
          ))}
        </Canvas>
      </div>

      {/* 右侧详情面板 */}
      <div style={{ width: 260, padding: 16, borderLeft: '1px solid #1e293b', overflowY: 'auto', flexShrink: 0 }}>
        <h3 style={{ fontSize: 13, color: '#94a3b8', marginBottom: 12 }}>详情</h3>
        {hoveredInfo ? (
          <div style={{ fontSize: 12, lineHeight: 1.8 }}>
            {hoveredInfo.token && <div style={{ color: '#60a5fa', fontWeight: 'bold', fontSize: 14 }}>{hoveredInfo.token}</div>}
            {hoveredInfo.source && <div><span style={{ color: '#94a3b8' }}>from:</span> {hoveredInfo.source}</div>}
            {hoveredInfo.layer !== undefined && <div><span style={{ color: '#94a3b8' }}>Layer:</span> {hoveredInfo.layer}</div>}
            {hoveredInfo.delta_cos !== undefined && (
              <div>
                <span style={{ color: '#94a3b8' }}>δ_cos:</span>{' '}
                <span style={{ color: deltaCosToColor(hoveredInfo.delta_cos) }}>{hoveredInfo.delta_cos.toFixed(4)}</span>
              </div>
            )}
            {hoveredInfo.cos_with_target !== undefined && <div><span style={{ color: '#94a3b8' }}>cos(target):</span> {hoveredInfo.cos_with_target.toFixed(4)}</div>}
            {hoveredInfo.norm !== undefined && <div><span style={{ color: '#94a3b8' }}>norm:</span> {hoveredInfo.norm.toFixed(1)}</div>}
            {hoveredInfo.category && (
              <div>
                <span style={{ color: '#94a3b8' }}>category:</span>{' '}
                <span style={{ color: CATEGORY_COLORS[hoveredInfo.category] || '#888' }}>{hoveredInfo.category}</span>
              </div>
            )}
            {hoveredInfo.isCorrection && <div style={{ color: '#fbbf24', fontWeight: 'bold' }}>⚡ 纠正层</div>}
          </div>
        ) : (
          <div style={{ fontSize: 12, color: '#64748b' }}>悬停3D对象查看详情</div>
        )}

        {/* 当前数据的可视化对象列表 */}
        {visualizations.length > 0 && (
          <div style={{ marginTop: 20 }}>
            <h3 style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>可视化对象 ({visualizations.length})</h3>
            {visualizations.map((v, i) => (
              <div key={i} style={{ padding: '4px 8px', fontSize: 11, color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>
                <span style={{ color: '#60a5fa' }}>{v.type}</span> {v.label || v.id}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
