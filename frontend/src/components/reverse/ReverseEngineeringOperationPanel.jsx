/**
 * 逆向工程操作面板（右下操作面板）
 * 显示配置摘要、模型对比、拼图进度、实验历史
 */
import { useMemo, useState } from 'react';
import { BarChart2, BookOpen, FlaskConical, RotateCcw, Save } from 'lucide-react';
import { LANGUAGE_DIMENSIONS, countSelectedDims, getSelectedDimIds } from '../../config/languageDimensions';
import { DNN_FEATURES, findSubFeature } from '../../config/dnnFeatures';
import { TEST_PRESETS, STATUS_CONFIG, getPuzzleProgressSummary } from '../../config/testPresets';
import { DIMENSION_COLORS, FEATURE_COLORS, PUZZLE_STATUS_COLORS } from '../../config/reverseColorMaps';
import PuzzleProgressView from './PuzzleProgressView';
import ModelComparisonView from './ModelComparisonView';

const cardStyle = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: '8px',
  padding: '10px',
};

export default function ReverseEngineeringOperationPanel({ workspace }) {
  const reverseState = workspace?.reverseEngineeringState;
  const selectedDims = reverseState?.selectedLanguageDims || {};
  const selectedFeature = reverseState?.selectedDNNFeature || 'A3';
  const selectedCategory = reverseState?.selectedDNNCategory || 'activation';
  const viewMode = reverseState?.viewMode || 'structure';
  const activePreset = reverseState?.activePreset;

  const selectedCount = useMemo(() => countSelectedDims(selectedDims), [selectedDims]);
  const selectedDimIds = useMemo(() => getSelectedDimIds(selectedDims), [selectedDims]);
  const featureInfo = useMemo(() => findSubFeature(selectedFeature), [selectedFeature]);

  const [experimentHistory, setExperimentHistory] = useState([]);
  const [showModelCompare, setShowModelCompare] = useState(false);

  const handleRunAnalysis = () => {
    const entry = {
      time: new Date().toLocaleTimeString(),
      dims: selectedDimIds.join(','),
      feature: selectedFeature,
      viewMode,
    };
    setExperimentHistory((prev) => [entry, ...prev.slice(0, 4)]);
  };

  const handleReset = () => {
    if (workspace?.setReverseEngineeringState) {
      workspace.setReverseEngineeringState({
        selectedLanguageDims: {
          syntax: { S1: false, S2: false, S3: false, S4: false, S5: false, S6: false, S7: false, S8: false },
          semantic: { M1: false, M2: false, M3: false, M4: false, M5: false, M6: false, M7: false, M8: false },
          logic: { L1: false, L2: false, L3: false, L4: false, L5: false, L6: false, L7: false, L8: false },
          pragmatic: { P1: false, P2: false, P3: false, P4: false, P5: false, P6: false },
          morphological: { F1: false, F2: false, F3: false, F4: false, F5: false },
        },
        selectedDNNFeature: 'A3',
        selectedDNNCategory: 'activation',
        viewMode: 'structure',
        activePreset: null,
      });
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {/* Current Config Summary */}
      <div style={cardStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
          <BookOpen size={13} color="#ec4899" />
          <span style={{ fontSize: '11px', fontWeight: 700, color: '#dfe8ff' }}>当前配置</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '10px' }}>
          <div>
            <span style={{ color: '#7f95bb' }}>模型:</span>{' '}
            <span style={{ color: '#eef7ff' }}>DeepSeek-7B</span>
          </div>
          <div>
            <span style={{ color: '#7f95bb' }}>层范围:</span>{' '}
            <span style={{ color: '#eef7ff' }}>L0-L31</span>
          </div>
          <div>
            <span style={{ color: '#7f95bb' }}>语言维度:</span>{' '}
            <span style={{ color: '#4facfe' }}>{selectedCount}/35</span>
          </div>
          <div>
            <span style={{ color: '#7f95bb' }}>DNN特征:</span>{' '}
            <span style={{ color: FEATURE_COLORS[selectedCategory] || '#eef7ff' }}>{selectedFeature}</span>
          </div>
          <div>
            <span style={{ color: '#7f95bb' }}>视角:</span>{' '}
            <span style={{ color: '#eef7ff' }}>{viewMode}</span>
          </div>
          <div>
            <span style={{ color: '#7f95bb' }}>预设:</span>{' '}
            <span style={{ color: activePreset ? '#fbbf24' : '#666' }}>{activePreset || '无'}</span>
          </div>
        </div>
      </div>

      {/* Model Comparison View */}
      <div style={cardStyle}>
        <div
          style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', marginBottom: showModelCompare ? '6px' : 0 }}
          onClick={() => setShowModelCompare(!showModelCompare)}
        >
          <BarChart2 size={13} color="#38bdf8" />
          <span style={{ fontSize: '11px', fontWeight: 700, color: '#dfe8ff' }}>跨模型对比</span>
          <span style={{ fontSize: '9px', color: '#666', marginLeft: 'auto' }}>
            {showModelCompare ? '收起 ▲' : '展开 ▼'}
          </span>
        </div>
        {showModelCompare && <ModelComparisonView selectedFeature={selectedFeature} />}
      </div>

      {/* Puzzle Progress View */}
      <div style={cardStyle}>
        <PuzzleProgressView activePreset={activePreset} />
      </div>

      {/* Experiment History */}
      {experimentHistory.length > 0 && (
        <div style={cardStyle}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
            <FlaskConical size={13} color="#6c5ce7" />
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#dfe8ff' }}>实验历史</span>
          </div>
          {experimentHistory.map((entry, idx) => (
            <div key={idx} style={{
              fontSize: '10px', color: '#999', padding: '3px 0',
              borderBottom: '1px solid rgba(255,255,255,0.04)',
            }}>
              <span style={{ color: '#7f95bb' }}>{entry.time}</span>{' '}
              维度[{entry.dims || 'none'}] {entry.feature} {entry.viewMode}
            </div>
          ))}
        </div>
      )}

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
        <button onClick={handleRunAnalysis} style={{
          padding: '6px 12px', borderRadius: '6px',
          border: '1px solid rgba(236, 72, 153, 0.35)',
          background: 'rgba(236, 72, 153, 0.12)',
          color: '#f9a8d4', fontSize: '10px', fontWeight: 600,
          cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px',
        }}>
          <FlaskConical size={11} /> 运行分析
        </button>
        <button onClick={() => console.log('Export data:', reverseState)} style={{
          padding: '6px 12px', borderRadius: '6px',
          border: '1px solid rgba(255,255,255,0.12)',
          background: 'rgba(255,255,255,0.04)',
          color: '#d8e6ff', fontSize: '10px', fontWeight: 600,
          cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px',
        }}>
          <Save size={11} /> 导出数据
        </button>
        <button onClick={handleReset} style={{
          padding: '6px 12px', borderRadius: '6px',
          border: '1px solid rgba(255,255,255,0.12)',
          background: 'rgba(255,255,255,0.04)',
          color: '#d8e6ff', fontSize: '10px', fontWeight: 600,
          cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px',
        }}>
          <RotateCcw size={11} /> 重置配置
        </button>
      </div>
    </div>
  );
}
