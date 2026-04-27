/**
 * 语言×DNN交叉矩阵迷你热力图
 * Canvas2D绘制的5×5缩略热力图
 */
import { useRef, useEffect } from 'react';
import { DIMENSION_COLORS, FEATURE_COLORS } from '../../config/reverseColorMaps';

const DIM_IDS = ['syntax', 'semantic', 'logic', 'pragmatic', 'morphological'];
const FEAT_IDS = ['weight', 'activation', 'causal', 'information', 'dynamics'];

// Mock correlation data for visualization
const MOCK_CORRELATIONS = [
  [0.85, 0.72, 0.45, 0.30, 0.55], // syntax
  [0.60, 0.90, 0.50, 0.65, 0.35], // semantic
  [0.35, 0.55, 0.88, 0.70, 0.40], // logic
  [0.25, 0.40, 0.60, 0.75, 0.20], // pragmatic
  [0.50, 0.30, 0.25, 0.35, 0.82], // morphological
];

export default function CrossDimensionMatrix({ selectedDims, selectedFeature, selectedCategory }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const cellSize = Math.min((width - 40) / 5, (height - 40) / 5);
    const offsetX = 36;
    const offsetY = 4;

    // Draw cells
    DIM_IDS.forEach((dimId, row) => {
      FEAT_IDS.forEach((featId, col) => {
        const x = offsetX + col * cellSize;
        const y = offsetY + row * cellSize;
        const value = MOCK_CORRELATIONS[row][col];

        // Highlight if selected
        const isDimSelected = Object.values(selectedDims[dimId] || {}).some(Boolean);
        const isFeatSelected = featId === selectedCategory;
        const alpha = (isDimSelected || isFeatSelected) ? 1.0 : 0.4;

        ctx.fillStyle = getCorrelationColor(value, alpha);
        ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);

        // Value text
        ctx.fillStyle = alpha > 0.5 ? '#fff' : '#666';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(value.toFixed(2), x + cellSize / 2, y + cellSize / 2);
      });
    });

    // Row labels (language dims)
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    DIM_IDS.forEach((dimId, row) => {
      const dimLabel = dimId.slice(0, 3).toUpperCase();
      const isDimSelected = Object.values(selectedDims[dimId] || {}).some(Boolean);
      ctx.fillStyle = isDimSelected ? DIMENSION_COLORS[dimId] : '#666';
      ctx.fillText(dimLabel, offsetX - 3, offsetY + row * cellSize + cellSize / 2);
    });

    // Col labels (DNN features)
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    FEAT_IDS.forEach((featId, col) => {
      const featLabel = featId.slice(0, 3).toUpperCase();
      ctx.fillStyle = featId === selectedCategory ? FEATURE_COLORS[featId] : '#666';
      ctx.save();
      ctx.translate(offsetX + col * cellSize + cellSize / 2, offsetY + 5 * cellSize + 3);
      ctx.fillText(featLabel, 0, 0);
      ctx.restore();
    });
  }, [selectedDims, selectedFeature, selectedCategory]);

  return (
    <div>
      <div style={{ fontSize: '11px', fontWeight: 600, color: '#dfe8ff', marginBottom: '6px' }}>
        语言×DNN 交叉矩阵
      </div>
      <canvas
        ref={canvasRef}
        width={220}
        height={160}
        style={{ width: '100%', maxWidth: '220px', imageRendering: 'auto' }}
      />
      <div style={{ fontSize: '9px', color: '#666', marginTop: '4px' }}>
        行=语言维度 | 列=DNN特征 | 值=相关性
      </div>
    </div>
  );
}

function getCorrelationColor(value, alpha = 1.0) {
  const v = Math.max(0, Math.min(1, value));
  const r = Math.round(v * 255);
  const g = Math.round((1 - Math.abs(v - 0.5) * 2) * 180);
  const b = Math.round((1 - v) * 255);
  return `rgba(${r}, ${g}, ${b}, ${alpha * 0.85})`;
}
