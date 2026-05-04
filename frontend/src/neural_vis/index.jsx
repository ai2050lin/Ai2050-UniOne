/**
 * NeuralVis3DApp — 3D可视化主组件
 * 支持 Schema v1.0 + v2.0
 * 
 * 可视化类型:
 *   v1.0: trajectory, point_cloud, heatmap_3d, flow, layer_stack
 *   v2.0: subspace_decomposition, force_line, grammar_role_matrix, causal_chain, dark_matter_flow
 */
import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Stars, Text } from '@react-three/drei';
import * as THREE from 'three';

// 模块化导入
import TrajectoryRenderer from './renderers/TrajectoryRenderer';
import PointCloudRenderer from './renderers/PointCloudRenderer';
import LayerStackRenderer from './renderers/LayerStackRenderer';
import Heatmap3DRenderer from './renderers/Heatmap3DRenderer';
import FlowRenderer from './renderers/FlowRenderer';
import SubspaceRenderer from './renderers/SubspaceRenderer';
import ForceLineRenderer from './renderers/ForceLineRenderer';
import GrammarRoleMatrixRenderer from './renderers/GrammarRoleMatrixRenderer';
import CausalChainRenderer from './renderers/CausalChainRenderer';
import DarkMatterFlowRenderer from './renderers/DarkMatterFlowRenderer';
import SceneHelpers from './components/SceneHelpers';
import HoverTooltip from './components/HoverTooltip';
import useVisData from './hooks/useVisData';
import { CATEGORY_COLORS, deltaCosToColor, cosWuToColor, SUBSPACE_COLORS } from './utils/constants';

// ==================== 视图模式定义 ====================
const VIEW_MODES = [
  { key: 'all', label: '🔍 全部', icon: '🔍' },
  { key: 'trajectory', label: '📈 轨迹', icon: '📈' },
  { key: 'point_cloud', label: '⚪ 点云', icon: '⚪' },
  { key: 'heatmap', label: '📊 热力图', icon: '📊' },
  { key: 'flow', label: '🔀 信息流', icon: '🔀' },
  { key: 'subspace', label: '🧬 子空间', icon: '🧬' },
  { key: 'force_line', label: '⚡ 力线', icon: '⚡' },
  { key: 'grammar', label: '📝 语法矩阵', icon: '📝' },
  { key: 'causal', label: '🔗 因果链', icon: '🔗' },
  { key: 'dark_matter', label: '🌑 暗物质', icon: '🌑' },
];

// ==================== 主组件 ====================
export default function NeuralVis3DApp() {
  const { dataFiles, activeData, loading, error, loadDataManifest, loadDataFile, loadLocalFile } = useVisData();
  const [animProgress, setAnimProgress] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [hoveredInfo, setHoveredInfo] = useState(null);
  const [selectedLayers, setSelectedLayers] = useState(null);
  const [viewMode, setViewMode] = useState('all');
  const fileInputRef = useRef();

  useEffect(() => { loadDataManifest(); }, [loadDataManifest]);

  const visualizations = activeData?.visualizations || [];
  const schemaVersion = activeData?.schema_version || '1.0';

  // 按类型分类可视化对象
  const byType = {
    trajectory: visualizations.filter(v => v.type === 'trajectory'),
    point_cloud: visualizations.filter(v => v.type === 'point_cloud'),
    heatmap_3d: visualizations.filter(v => v.type === 'heatmap_3d'),
    flow: visualizations.filter(v => v.type === 'flow'),
    layer_stack: visualizations.filter(v => v.type === 'layer_stack'),
    subspace_decomposition: visualizations.filter(v => v.type === 'subspace_decomposition'),
    force_line: visualizations.filter(v => v.type === 'force_line'),
    grammar_role_matrix: visualizations.filter(v => v.type === 'grammar_role_matrix'),
    causal_chain: visualizations.filter(v => v.type === 'causal_chain'),
    dark_matter_flow: visualizations.filter(v => v.type === 'dark_matter_flow'),
  };

  // 根据视图模式过滤
  const filterByMode = (type) => viewMode === 'all' || viewMode === type;
  const trajectories = byType.trajectory.filter(() => filterByMode('trajectory'));
  const pointClouds = byType.point_cloud.filter(() => filterByMode('point_cloud'));
  const heatmaps = byType.heatmap_3d.filter(() => filterByMode('heatmap'));
  const flows = byType.flow.filter(() => filterByMode('flow'));
  const layerStacks = byType.layer_stack;
  const subspaceDecomps = byType.subspace_decomposition.filter(() => filterByMode('subspace'));
  const forceLines = byType.force_line.filter(() => filterByMode('force_line'));
  const grammarMatrices = byType.grammar_role_matrix.filter(() => filterByMode('grammar'));
  const causalChains = byType.causal_chain.filter(() => filterByMode('causal'));
  const darkMatterFlows = byType.dark_matter_flow.filter(() => filterByMode('dark_matter'));

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

  // 获取每个模式的数量
  const getCount = (key) => {
    if (key === 'all') return visualizations.length;
    if (key === 'heatmap') return byType.heatmap_3d.length;
    if (key === 'subspace') return byType.subspace_decomposition.length;
    if (key === 'force_line') return byType.force_line.length;
    if (key === 'grammar') return byType.grammar_role_matrix.length;
    if (key === 'causal') return byType.causal_chain.length;
    if (key === 'dark_matter') return byType.dark_matter_flow.length;
    return byType[key]?.length || 0;
  };

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
          {VIEW_MODES.map(mode => {
            const count = getCount(mode.key);
            const isActive = viewMode === mode.key;
            return (
              <button
                key={mode.key}
                onClick={() => setViewMode(mode.key)}
                style={{
                  display: 'block', width: '100%', padding: '6px 8px', marginBottom: 4,
                  background: isActive ? '#1e40af' : '#1e293b',
                  border: isActive ? '1px solid #3b82f6' : '1px solid #334155',
                  borderRadius: 4, color: isActive ? '#bfdbfe' : '#94a3b8',
                  cursor: 'pointer', textAlign: 'left', fontSize: 12,
                }}
              >
                {mode.label}
                <span style={{ float: 'right', color: '#64748b' }}>{count}</span>
              </button>
            );
          })}
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
              <div>Schema: <span style={{ color: '#60a5fa' }}>v{schemaVersion}</span></div>
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

        {/* cos(W_U) 颜色图例 */}
        <div style={{ marginTop: 12 }}>
          <h3 style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>cos(W_U) 图例</h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            {[1, 0.75, 0.5, 0.25, 0].map(v => (
              <div key={v} style={{ textAlign: 'center' }}>
                <div style={{ width: 24, height: 12, background: cosWuToColor(v), borderRadius: 2 }} />
                <div style={{ fontSize: 9, color: '#64748b' }}>{v.toFixed(2)}</div>
              </div>
            ))}
          </div>
          <div style={{ fontSize: 10, color: '#64748b', marginTop: 4 }}>
            <span style={{ color: SUBSPACE_COLORS.w_u }}>■</span> W_U对齐 → <span style={{ color: SUBSPACE_COLORS.w_u_perp }}>■</span> W_U⊥
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

          {/* v1.0 渲染器 */}
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

          {/* v2.0 新增渲染器 */}
          {subspaceDecomps.map(sd => (
            <SubspaceRenderer
              key={sd.id}
              subspaceDecomp={sd}
              animated={animProgress < 1}
              animationProgress={animProgress}
              onHoverToken={setHoveredInfo}
            />
          ))}
          {forceLines.map(fl => (
            <ForceLineRenderer
              key={fl.id}
              forceLine={fl}
              animated={animProgress < 1}
              animationProgress={animProgress}
              onHoverToken={setHoveredInfo}
            />
          ))}
          {grammarMatrices.map(gm => (
            <GrammarRoleMatrixRenderer
              key={gm.id}
              grammarMatrix={gm}
              onHoverToken={setHoveredInfo}
            />
          ))}
          {causalChains.map(cc => (
            <CausalChainRenderer
              key={cc.id}
              causalChain={cc}
              animated={animProgress < 1}
              animationProgress={animProgress}
              onHoverToken={setHoveredInfo}
            />
          ))}
          {darkMatterFlows.map(dmf => (
            <DarkMatterFlowRenderer
              key={dmf.id}
              darkMatterFlow={dmf}
              animated={animProgress < 1}
              animationProgress={animProgress}
              onHoverToken={setHoveredInfo}
            />
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
            {hoveredInfo.cos_with_wu !== undefined && (
              <div>
                <span style={{ color: '#94a3b8' }}>cos(W_U):</span>{' '}
                <span style={{ color: cosWuToColor(hoveredInfo.cos_with_wu) }}>{hoveredInfo.cos_with_wu.toFixed(4)}</span>
              </div>
            )}
            {hoveredInfo.norm !== undefined && <div><span style={{ color: '#94a3b8' }}>norm:</span> {hoveredInfo.norm.toFixed(1)}</div>}
            {hoveredInfo.category && (
              <div>
                <span style={{ color: '#94a3b8' }}>category:</span>{' '}
                <span style={{ color: CATEGORY_COLORS[hoveredInfo.category] || '#888' }}>{hoveredInfo.category}</span>
              </div>
            )}
            {hoveredInfo.subspace && (
              <div>
                <span style={{ color: '#94a3b8' }}>subspace:</span>{' '}
                <span style={{ color: SUBSPACE_COLORS[hoveredInfo.subspace] || '#888' }}>
                  {hoveredInfo.subspace === 'w_u' ? 'W_U' : 'W_U⊥'}
                </span>
              </div>
            )}
            {hoveredInfo.isCorrection && <div style={{ color: '#fbbf24', fontWeight: 'bold' }}>⚡ 纠正层</div>}
            {hoveredInfo.growth_rate !== undefined && <div><span style={{ color: '#94a3b8' }}>growth_rate:</span> {hoveredInfo.growth_rate.toFixed(3)}</div>}
            {hoveredInfo.role_pair && <div><span style={{ color: '#94a3b8' }}>角色对:</span> <span style={{ color: '#ffe66d' }}>{hoveredInfo.role_pair}</span></div>}
            {hoveredInfo.cosine !== undefined && <div><span style={{ color: '#94a3b8' }}>cosine:</span> {hoveredInfo.cosine.toFixed(4)}</div>}
            {hoveredInfo.kl_divergence !== undefined && <div><span style={{ color: '#94a3b8' }}>KL:</span> {hoveredInfo.kl_divergence.toFixed(2)}</div>}
            {hoveredInfo.classification_flip !== undefined && <div><span style={{ color: '#94a3b8' }}>flip:</span> {(hoveredInfo.classification_flip * 100).toFixed(1)}%</div>}
            {hoveredInfo.w_u_signal !== undefined && <div><span style={{ color: SUBSPACE_COLORS.w_u }}>W_U:</span> {(hoveredInfo.w_u_signal * 100).toFixed(0)}%</div>}
            {hoveredInfo.w_u_perp_signal !== undefined && <div><span style={{ color: SUBSPACE_COLORS.w_u_perp }}>W_U⊥:</span> {(hoveredInfo.w_u_perp_signal * 100).toFixed(0)}%</div>}
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

        {/* 类型统计 */}
        {visualizations.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <h3 style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>类型统计</h3>
            <div style={{ fontSize: 11, lineHeight: 1.8 }}>
              {Object.entries(byType).filter(([_, arr]) => arr.length > 0).map(([type, arr]) => (
                <div key={type}>
                  <span style={{ color: '#60a5fa' }}>{type}</span>: {arr.length}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
