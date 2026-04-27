/**
 * 逆向工程数据面板（右上信息面板）
 * 显示选中维度概览、交叉矩阵、特征详情、拼图进度
 */
import { useMemo } from 'react';
import { Crosshair, TrendingUp } from 'lucide-react';
import { LANGUAGE_DIMENSIONS } from '../../config/languageDimensions';
import { DNN_FEATURES, findSubFeature } from '../../config/dnnFeatures';
import { DIMENSION_COLORS, FEATURE_COLORS, heatmapColor } from '../../config/reverseColorMaps';
import { countSelectedDims, getSelectedDimIds } from '../../config/languageDimensions';
import CrossDimensionMatrix from './CrossDimensionMatrix';
import FeatureDetailView from './FeatureDetailView';
import { getPuzzleProgressSummary } from '../../config/testPresets';

const cardStyle = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: '8px',
  padding: '10px',
};

export default function ReverseEngineeringDataPanel({ workspace, hoveredInfo, displayInfo }) {
  const reverseState = workspace?.reverseEngineeringState;
  const selectedDims = reverseState?.selectedLanguageDims || {};
  const selectedFeature = reverseState?.selectedDNNFeature || 'A3';
  const selectedCategory = reverseState?.selectedDNNCategory || 'activation';
  const viewMode = reverseState?.viewMode || 'structure';

  const selectedCount = useMemo(() => countSelectedDims(selectedDims), [selectedDims]);
  const selectedDimIds = useMemo(() => getSelectedDimIds(selectedDims), [selectedDims]);

  // Find active sub-feature info
  const featureInfo = useMemo(() => findSubFeature(selectedFeature), [selectedFeature]);
  const featureColor = FEATURE_COLORS[selectedCategory] || '#4ecdc4';

  // Build key metrics based on selected dimensions and feature
  const keyMetrics = useMemo(() => {
    const metrics = [];
    if (selectedCount === 0) {
      return [
        { label: '状态', value: '未选择维度', color: '#666' },
        { label: '提示', value: '请在左侧选择语言维度', color: '#888' },
      ];
    }

    // Orthogonality metric
    metrics.push({
      label: '正交性',
      value: selectedCount > 3 ? '待验证' : 'cos<0.2',
      color: '#4facfe',
    });

    // PC1 ratio
    metrics.push({
      label: 'PC1占比',
      value: viewMode === 'structure' ? '86.5%' : '-',
      color: '#22c55e',
    });

    // Causal effect
    metrics.push({
      label: '因果效应',
      value: featureInfo?.id?.startsWith('C') ? '递增' : '-',
      color: '#ff6b6b',
    });

    // R²
    metrics.push({
      label: 'R²',
      value: featureInfo?.id?.startsWith('I') ? '0.96' : '-',
      color: '#ffd93d',
    });

    return metrics;
  }, [selectedCount, viewMode, featureInfo]);

  // Puzzle progress summary
  const puzzleSummary = useMemo(() => getPuzzleProgressSummary(), []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      {/* Selected Dimension Summary */}
      <div style={cardStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
          <Crosshair size={13} color="#ec4899" />
          <span style={{ fontSize: '12px', fontWeight: 700, color: '#dfe8ff' }}>
            维度概览
          </span>
          <span style={{ fontSize: '10px', color: '#7f95bb', marginLeft: 'auto' }}>
            {selectedCount}/35 已选
          </span>
        </div>

        {/* Active dimensions summary */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '8px' }}>
          {Object.entries(LANGUAGE_DIMENSIONS).map(([dimId, dim]) => {
            const groupCount = Object.values(selectedDims[dimId] || {}).filter(Boolean).length;
            if (groupCount === 0) return null;
            return (
              <span key={dimId} style={{
                fontSize: '10px',
                padding: '2px 6px',
                borderRadius: '3px',
                background: dim.color + '20',
                color: dim.color,
                border: `1px solid ${dim.color}40`,
                fontWeight: 600,
              }}>
                {dim.label}({groupCount})
              </span>
            );
          })}
          {selectedCount === 0 && (
            <span style={{ fontSize: '10px', color: '#666' }}>暂未选择语言维度</span>
          )}
        </div>

        {/* Key Metric Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px' }}>
          {keyMetrics.map((metric, idx) => (
            <div key={idx} style={{
              padding: '6px 8px',
              borderRadius: '5px',
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}>
              <div style={{ fontSize: '10px', color: '#7f95bb', marginBottom: '2px' }}>{metric.label}</div>
              <div style={{ fontSize: '13px', fontWeight: 700, color: metric.color }}>{metric.value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Cross Dimension Matrix */}
      <div style={cardStyle}>
        <CrossDimensionMatrix
          selectedDims={selectedDims}
          selectedFeature={selectedFeature}
          selectedCategory={selectedCategory}
        />
      </div>

      {/* Feature Detail View */}
      <div style={cardStyle}>
        <FeatureDetailView
          featureInfo={featureInfo}
          selectedDims={selectedDims}
          viewMode={viewMode}
          hoveredInfo={hoveredInfo}
        />
      </div>

      {/* Puzzle Progress Summary */}
      <div style={cardStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
          <TrendingUp size={13} color="#ffd93d" />
          <span style={{ fontSize: '12px', fontWeight: 700, color: '#dfe8ff' }}>拼图进度</span>
        </div>
        <div style={{ display: 'flex', gap: '8px', fontSize: '10px' }}>
          <span style={{ color: '#10b981' }}>✅ {puzzleSummary.confirmed}</span>
          <span style={{ color: '#f59e0b' }}>🔶 {puzzleSummary.partial}</span>
          <span style={{ color: '#6b7280' }}>⬜ {puzzleSummary.pending}</span>
          <span style={{ color: '#ef4444' }}>❌ {puzzleSummary.missing}</span>
        </div>
        <div style={{
          marginTop: '6px', height: '4px', borderRadius: '2px',
          background: 'rgba(255,255,255,0.06)', overflow: 'hidden',
        }}>
          <div style={{
            height: '100%', width: `${(puzzleSummary.confirmed / 10) * 100}%`,
            background: 'linear-gradient(90deg, #10b981, #22c55e)',
            borderRadius: '2px',
            transition: 'width 0.3s',
          }} />
        </div>
      </div>

      {/* Hover Detail */}
      {hoveredInfo && (
        <div style={{ ...cardStyle, fontSize: '10px', color: '#aaa', lineHeight: 1.6 }}>
          {typeof hoveredInfo === 'string' ? hoveredInfo : JSON.stringify(hoveredInfo).slice(0, 200)}
        </div>
      )}
    </div>
  );
}
