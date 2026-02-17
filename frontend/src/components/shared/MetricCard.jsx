/**
 * MetricCard - 指标卡片组件
 * 用于显示各种分析指标
 */
import React from 'react';
import { COLOR_SCHEMES } from '../../utils/colors';

export function MetricCard({ 
  title, 
  value, 
  unit = '', 
  description = '', 
  color = COLOR_SCHEMES.primary,
  trend = null, // 'up' | 'down' | null
  onClick = null
}) {
  return (
    <div 
      className="metric-card"
      style={{
        background: 'rgba(255,255,255,0.02)',
        borderRadius: '12px',
        padding: '16px',
        border: `1px solid ${color}20`,
        cursor: onClick ? 'pointer' : 'default',
        transition: 'all 0.2s ease'
      }}
      onClick={onClick}
    >
      <div style={{ 
        fontSize: '11px', 
        color: '#888', 
        marginBottom: '8px',
        textTransform: 'uppercase',
        letterSpacing: '0.5px'
      }}>
        {title}
      </div>
      
      <div style={{ 
        fontSize: '24px', 
        fontWeight: 'bold', 
        color: color,
        display: 'flex',
        alignItems: 'baseline',
        gap: '4px'
      }}>
        {typeof value === 'number' ? value.toFixed(3) : value}
        {unit && (
          <span style={{ fontSize: '12px', color: '#666' }}>
            {unit}
          </span>
        )}
        {trend && (
          <span style={{ 
            fontSize: '12px',
            color: trend === 'up' ? '#10b981' : '#ef4444'
          }}>
            {trend === 'up' ? '↑' : '↓'}
          </span>
        )}
      </div>
      
      {description && (
        <div style={{ 
          fontSize: '11px', 
          color: '#666', 
          marginTop: '8px' 
        }}>
          {description}
        </div>
      )}
    </div>
  );
}

/**
 * MetricGrid - 指标网格组件
 */
export function MetricGrid({ children, columns = 4 }) {
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: `repeat(${columns}, 1fr)`,
      gap: '16px'
    }}>
      {children}
    </div>
  );
}

export default MetricCard;
