/**
 * 特征详情视图
 * 根据DNN子特征显示不同类型的图表
 */
import { useMemo } from 'react';
import { findSubFeature } from '../../config/dnnFeatures';
import { BAND_COLORS, BAND_LABELS } from '../../config/reverseColorMaps';

const cardStyle = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.06)',
  borderRadius: '5px',
  padding: '6px 8px',
};

export default function FeatureDetailView({ featureInfo, selectedDims, viewMode, hoveredInfo }) {
  if (!featureInfo) {
    return (
      <div>
        <div style={{ fontSize: '11px', fontWeight: 600, color: '#dfe8ff', marginBottom: '6px' }}>
          特征详情
        </div>
        <div style={{ fontSize: '10px', color: '#666' }}>请选择一个DNN子特征</div>
      </div>
    );
  }

  const featureId = featureInfo.id;
  const catId = featureInfo.parentId;

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
        <span style={{ fontSize: '11px', fontWeight: 600, color: '#dfe8ff' }}>
          特征详情
        </span>
        <span style={{
          fontSize: '10px', padding: '1px 5px', borderRadius: '3px',
          background: featureInfo.parentColor + '20',
          color: featureInfo.parentColor,
          border: `1px solid ${featureInfo.parentColor}40`,
        }}>
          {featureId}
        </span>
        <span style={{ fontSize: '10px', color: '#aaa' }}>{featureInfo.label}</span>
      </div>

      {/* Feature description */}
      <div style={{ ...cardStyle, fontSize: '10px', color: '#9bb3de', marginBottom: '8px' }}>
        {featureInfo.description}
      </div>

      {/* Conditional content based on feature category */}
      {catId === 'activation' && (
        <ActivationDetailView featureId={featureId} viewMode={viewMode} description={featureInfo.description} />
      )}
      {catId === 'causal' && (
        <CausalDetailView featureId={featureId} viewMode={viewMode} />
      )}
      {catId === 'information' && (
        <InformationDetailView featureId={featureId} viewMode={viewMode} />
      )}
      {catId === 'weight' && (
        <WeightDetailView featureId={featureId} viewMode={viewMode} />
      )}
      {catId === 'dynamics' && (
        <DynamicsDetailView featureId={featureId} viewMode={viewMode} />
      )}
    </div>
  );
}

function ActivationDetailView({ featureId, viewMode, description }) {
  // Band decomposition for A5
  if (featureId === 'A5') {
    return (
      <div>
        <div style={{ fontSize: '10px', fontWeight: 600, color: '#ccc', marginBottom: '4px' }}>频段分工</div>
        {[1, 2, 3, 4, 5].map((band) => (
          <div key={band} style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '3px' }}>
            <div style={{ width: '10px', height: '10px', borderRadius: '2px', background: BAND_COLORS[band], flexShrink: 0 }} />
            <span style={{ fontSize: '9px', color: '#aaa', width: '60px' }}>{BAND_LABELS[band]}</span>
            <div style={{ flex: 1, height: '6px', borderRadius: '3px', background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
              <div style={{ width: `${[65, 70, 80, 55, 45][band - 1]}%`, height: '100%', background: BAND_COLORS[band], borderRadius: '3px' }} />
            </div>
          </div>
        ))}
      </div>
    );
  }

  // Differential vector for A1
  if (featureId === 'A1') {
    return (
      <div>
        <div style={{ fontSize: '10px', fontWeight: 600, color: '#ccc', marginBottom: '4px' }}>差分向量</div>
        <div style={cardStyle}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '10px' }}>
            <div><span style={{ color: '#7f95bb' }}>方向:</span> <span style={{ color: '#4facfe' }}>语法↗</span></div>
            <div><span style={{ color: '#7f95bb' }}>范数:</span> <span style={{ color: '#eef7ff' }}>0.42</span></div>
            <div><span style={{ color: '#7f95bb' }}>一致性:</span> <span style={{ color: '#22c55e' }}>0.87</span></div>
            <div><span style={{ color: '#7f95bb' }}>PC1:</span> <span style={{ color: '#fbbf24' }}>0.86</span></div>
          </div>
        </div>
      </div>
    );
  }

  // PC1 compression for A2
  if (featureId === 'A2') {
    return (
      <div>
        <div style={{ fontSize: '10px', fontWeight: 600, color: '#ccc', marginBottom: '4px' }}>PC1压缩率</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
          {['L0-L8', 'L8-L16', 'L16-L24', 'L24-L31'].map((range, i) => (
            <div key={range} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <span style={{ fontSize: '9px', color: '#888', width: '45px' }}>{range}</span>
              <div style={{ flex: 1, height: '6px', borderRadius: '3px', background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                <div style={{ width: `${[90, 72, 55, 40][i]}%`, height: '100%', background: `hsl(${[120, 90, 50, 20][i]}, 75%, 50%)`, borderRadius: '3px' }} />
              </div>
              <span style={{ fontSize: '9px', color: '#aaa', width: '30px' }}>{[90, 72, 55, 40][i]}%</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Default activation detail
  return (
    <div style={cardStyle}>
      <div style={{ fontSize: '10px', color: '#9bb3de' }}>
        {description || '激活空间分析详情'}
      </div>
    </div>
  );
}

function CausalDetailView({ featureId }) {
  return (
    <div>
      <div style={{ fontSize: '10px', fontWeight: 600, color: '#ccc', marginBottom: '4px' }}>因果效应</div>
      <div style={cardStyle}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '10px' }}>
          <div><span style={{ color: '#7f95bb' }}>效应类型:</span> <span style={{ color: '#ff6b6b' }}>{featureId === 'C1' ? 'Patching' : featureId === 'C2' ? 'Interchange' : featureId}</span></div>
          <div><span style={{ color: '#7f95bb' }}>强度:</span> <span style={{ color: '#eef7ff' }}>0.72</span></div>
          <div><span style={{ color: '#7f95bb' }}>方向:</span> <span style={{ color: '#4facfe' }}>因果链</span></div>
          <div><span style={{ color: '#7f95bb' }}>1D流形:</span> <span style={{ color: '#22c55e' }}>{featureId === 'C8' ? '验证中' : '-'}</span></div>
        </div>
      </div>
    </div>
  );
}

function InformationDetailView({ featureId }) {
  return (
    <div>
      <div style={{ fontSize: '10px', fontWeight: 600, color: '#ccc', marginBottom: '4px' }}>信息论指标</div>
      <div style={cardStyle}>
        <div style={{ fontSize: '10px', color: '#9bb3de', marginBottom: '6px' }}>
          编码方程: logit = Σα_band × band_logit + β
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '10px' }}>
          <div><span style={{ color: '#7f95bb' }}>R²:</span> <span style={{ color: '#ffd93d' }}>{featureId === 'I6' ? '0.96' : '-'}</span></div>
          <div><span style={{ color: '#7f95bb' }}>效率:</span> <span style={{ color: '#22c55e' }}>0.82</span></div>
          <div><span style={{ color: '#7f95bb' }}>瓶颈:</span> <span style={{ color: '#f97316' }}>L8-L16</span></div>
          <div><span style={{ color: '#7f95bb' }}>互信息:</span> <span style={{ color: '#eef7ff' }}>0.68</span></div>
        </div>
      </div>
    </div>
  );
}

function WeightDetailView({ featureId }) {
  return (
    <div style={cardStyle}>
      <div style={{ fontSize: '10px', color: '#9bb3de' }}>
        权重结构分析 - {featureId}
      </div>
    </div>
  );
}

function DynamicsDetailView({ featureId }) {
  return (
    <div style={cardStyle}>
      <div style={{ fontSize: '10px', color: '#9bb3de' }}>
        动力学分析 - {featureId}
      </div>
    </div>
  );
}
