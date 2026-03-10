import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/qwen3_deepseek7b_real_model_structure_atlas_sample.json';

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function isValidPayload(payload) {
  return Boolean(payload && payload.models && payload.headline_metrics && payload.hypotheses);
}

const MODEL_LABELS = {
  qwen3_4b: 'Qwen3-4B',
  deepseek_7b: 'DeepSeek-7B',
};

const LAYER_COLORS = {
  concept_biased: '#38bdf8',
  relation_biased: '#f97316',
  balanced: '#94a3b8',
};

export default function Qwen3DeepSeekRealModelStructureAtlasDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const orientationRows = useMemo(() => {
    return Object.entries(payload?.models || {}).map(([key, row]) => ({
      model: MODEL_LABELS[key] || key,
      预测取向: Number(row?.global_summary?.predicted_orientation || 0),
      实际取向: Number(row?.global_summary?.actual_orientation || 0),
      取向落差: Number(row?.global_summary?.orientation_gap_abs || 0),
    }));
  }, [payload]);

  const bridgeRows = useMemo(() => {
    return Object.entries(payload?.models || {}).map(([key, row]) => ({
      model: MODEL_LABELS[key] || key,
      机制桥接: Number(row?.global_summary?.mechanism_bridge_score || 0),
      结构收益: Number(row?.global_summary?.mean_behavior_gain || 0),
      软层重合: Number(row?.global_summary?.soft_layer_overlap || 0),
    }));
  }, [payload]);

  const layerRows = useMemo(() => {
    const rows = [];
    for (const [modelKey, row] of Object.entries(payload?.models || {})) {
      for (const layerRow of row?.layer_atlas || []) {
        rows.push({
          id: `${modelKey}_L${layerRow.layer}`,
          model: MODEL_LABELS[modelKey] || modelKey,
          layer: `L${layerRow.layer}`,
          共享支撑: Number(layerRow.shared_support || 0),
          概念支撑: Number(layerRow.concept_support || 0),
          关系支撑: Number(layerRow.relation_support || 0),
          supportStage: layerRow.support_stage || 'balanced',
          isTarget: Boolean(layerRow.is_targeted_band),
        });
      }
    }
    return rows;
  }, [payload]);

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 models / headline_metrics / hypotheses 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`真实模型 atlas JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setPayload(samplePayload);
    setSource('内置样例');
    setError('');
  }

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(14,165,233,0.16), transparent 28%), radial-gradient(circle at bottom right, rgba(249,115,22,0.12), transparent 26%), rgba(2,6,23,0.68)',
        border: '1px solid rgba(14,165,233,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#f8fafc', fontSize: '15px', fontWeight: 'bold' }}>五点五十一、真实模型结构普查 Atlas</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px', maxWidth: '920px' }}>
            这一步把真实模型里的共享层带、阶段取向、定向消融和结构任务收益收成一张统一 atlas。重点不是只看正结果，而是把“预测成立”和“真实干预不自动成立”的落差一起外部化。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <label style={{ color: '#e2e8f0', fontSize: '11px', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px', cursor: 'pointer' }}>
            导入 JSON
            <input type="file" accept="application/json" onChange={onUpload} style={{ display: 'none' }} />
          </label>
          <button
            type="button"
            onClick={resetAll}
            style={{ color: '#e2e8f0', fontSize: '11px', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px', background: 'transparent', cursor: 'pointer' }}
          >
            重置
          </button>
        </div>
      </div>

      <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px' }}>{`数据源: ${source}`}</div>
      {error && <div style={{ marginTop: '8px', color: '#fca5a5', fontSize: '11px' }}>{error}</div>}

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Qwen 取向落差</div>
          <div style={{ color: '#38bdf8', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.headline_metrics?.qwen_orientation_gap_abs)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek 取向落差</div>
          <div style={{ color: '#f97316', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.headline_metrics?.deepseek_orientation_gap_abs)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Qwen 结构收益</div>
          <div style={{ color: '#22c55e', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.headline_metrics?.qwen_behavior_gain)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek 机制桥接</div>
          <div style={{ color: '#eab308', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.headline_metrics?.deepseek_mechanism_bridge)}</div>
        </div>
      </div>

      <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {Object.entries(payload?.hypotheses || {}).map(([key, value]) => (
          <div
            key={key}
            style={{
              borderRadius: '999px',
              padding: '6px 10px',
              fontSize: '11px',
              color: value ? '#dcfce7' : '#fee2e2',
              background: value ? 'rgba(34,197,94,0.16)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(34,197,94,0.30)' : 'rgba(248,113,113,0.30)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '未成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 预测取向 vs 实际取向</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={orientationRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="预测取向" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="实际取向" fill="#f97316" radius={[4, 4, 0, 0]} />
              <Bar dataKey="取向落差" fill="#eab308" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 桥接强度与结构收益</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={bridgeRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="机制桥接" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="结构收益" fill="#a855f7" radius={[4, 4, 0, 0]} />
              <Bar dataKey="软层重合" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '12px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 层带 Atlas</div>
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={layerRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="id" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.35)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
            <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
            <Legend />
            <Bar dataKey="共享支撑" radius={[4, 4, 0, 0]}>
              {layerRows.map((row) => (
                <Cell
                  key={row.id}
                  fill={row.isTarget ? '#facc15' : (LAYER_COLORS[row.supportStage] || '#94a3b8')}
                />
              ))}
            </Bar>
            <Bar dataKey="概念支撑" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            <Bar dataKey="关系支撑" fill="#f97316" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <div style={{ color: '#94a3b8', fontSize: '11px', marginTop: '8px' }}>
          黄色表示真实定向消融使用的目标层带；蓝色偏概念，橙色偏关系，灰色表示相对平衡。
        </div>
      </div>
    </div>
  );
}
