/**
 * 跨模型对比视图
 * 显示不同模型在同一特征上的指标对比
 */
import { useMemo } from 'react';
import { FEATURE_COLORS } from '../../config/reverseColorMaps';

const MODELS = [
  { id: 'deepseek-7b', label: 'DeepSeek-7B', color: '#4facfe' },
  { id: 'qwen3-8b', label: 'Qwen3-8B', color: '#22c55e' },
  { id: 'llama3-8b', label: 'LLaMA3-8B', color: '#f97316' },
];

// Mock comparison data
const MOCK_COMPARE_DATA = {
  A1: { 'deepseek-7b': 0.87, 'qwen3-8b': 0.82, 'llama3-8b': 0.78 },
  A2: { 'deepseek-7b': 0.86, 'qwen3-8b': 0.79, 'llama3-8b': 0.72 },
  A3: { 'deepseek-7b': 0.92, 'qwen3-8b': 0.88, 'llama3-8b': 0.85 },
  A5: { 'deepseek-7b': 0.75, 'qwen3-8b': 0.70, 'llama3-8b': 0.65 },
  C1: { 'deepseek-7b': 0.72, 'qwen3-8b': 0.68, 'llama3-8b': 0.60 },
  C2: { 'deepseek-7b': 0.88, 'qwen3-8b': 0.75, 'llama3-8b': 0.70 },
  I3: { 'deepseek-7b': 0.65, 'qwen3-8b': 0.60, 'llama3-8b': 0.55 },
  I6: { 'deepseek-7b': 0.96, 'qwen3-8b': 0.91, 'llama3-8b': 0.88 },
};

export default function ModelComparisonView({ selectedFeature }) {
  const compareData = MOCK_COMPARE_DATA[selectedFeature] || {
    'deepseek-7b': 0.50, 'qwen3-8b': 0.45, 'llama3-8b': 0.40,
  };

  const maxValue = Math.max(...Object.values(compareData));

  return (
    <div>
      <div style={{ fontSize: '10px', color: '#7f95bb', marginBottom: '6px' }}>
        特征 {selectedFeature} 跨模型对比
      </div>
      {MODELS.map((model) => {
        const value = compareData[model.id] || 0;
        const isMax = value === maxValue;
        return (
          <div key={model.id} style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
            <span style={{ fontSize: '9px', color: model.color, width: '70px', fontWeight: 600 }}>
              {model.label}
            </span>
            <div style={{ flex: 1, height: '8px', borderRadius: '4px', background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
              <div style={{
                width: `${value * 100}%`, height: '100%',
                background: isMax ? model.color : model.color + '60',
                borderRadius: '4px',
                transition: 'width 0.3s',
              }} />
            </div>
            <span style={{
              fontSize: '10px', color: isMax ? model.color : '#888',
              fontWeight: isMax ? 700 : 400, width: '35px', textAlign: 'right',
            }}>
              {value.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
