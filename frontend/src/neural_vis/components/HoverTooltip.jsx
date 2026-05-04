/**
 * HoverTooltip — 悬停提示
 * 增强: 支持新增可视化类型的详情显示
 */
import React from 'react';
import { Html } from '@react-three/drei';
import { CATEGORY_COLORS, SUBSPACE_COLORS, GRAMMAR_ROLE_COLORS } from '../utils/constants';
import { deltaCosToColor, cosWuToColor } from '../utils/constants';

export default function HoverTooltip({ data }) {
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
        {data.cos_with_wu !== undefined && (
          <div>
            <span style={{ color: '#94a3b8' }}>cos(W_U):</span>{' '}
            <span style={{ color: cosWuToColor(data.cos_with_wu) }}>{data.cos_with_wu.toFixed(4)}</span>
          </div>
        )}
        {data.norm !== undefined && <div>norm: {data.norm.toFixed(1)}</div>}
        {data.category && <div style={{ color: CATEGORY_COLORS[data.category] || '#888' }}>cat: {data.category}</div>}
        {data.subspace && (
          <div style={{ color: SUBSPACE_COLORS[data.subspace] || '#888' }}>
            subspace: {data.subspace === 'w_u' ? 'W_U' : 'W_U⊥'}
          </div>
        )}
        {data.isCorrection && <div style={{ color: '#fbbf24', fontWeight: 'bold' }}>⚡ Correction Layer</div>}
        {data.growth_rate !== undefined && <div>growth_rate: {data.growth_rate.toFixed(3)}</div>}
        {data.role_pair && <div style={{ color: '#ffe66d' }}>{data.role_pair}</div>}
        {data.cosine !== undefined && <div>cosine: {data.cosine.toFixed(4)}</div>}
        {data.kl_divergence !== undefined && <div>KL: {data.kl_divergence.toFixed(2)}</div>}
        {data.classification_flip !== undefined && <div>flip: {(data.classification_flip * 100).toFixed(1)}%</div>}
        {data.w_u_signal !== undefined && <div style={{ color: SUBSPACE_COLORS.w_u }}>W_U: {(data.w_u_signal * 100).toFixed(0)}%</div>}
        {data.w_u_perp_signal !== undefined && <div style={{ color: SUBSPACE_COLORS.w_u_perp }}>W_U⊥: {(data.w_u_perp_signal * 100).toFixed(0)}%</div>}
      </div>
    </Html>
  );
}
