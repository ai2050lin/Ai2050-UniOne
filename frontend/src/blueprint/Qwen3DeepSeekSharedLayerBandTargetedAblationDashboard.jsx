import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/qwen3_deepseek7b_shared_layer_band_targeted_ablation_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.models && payload.headline_metrics && payload.hypotheses);
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

export default function Qwen3DeepSeekSharedLayerBandTargetedAblationDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const orientationRows = useMemo(
    () => [
      {
        model: 'Qwen3-4B',
        预测取向: Number(payload?.models?.qwen3_4b?.global_summary?.predicted_orientation || 0),
        实际取向: Number(payload?.models?.qwen3_4b?.global_summary?.actual_targeted_orientation || 0),
      },
      {
        model: 'DeepSeek-7B',
        预测取向: Number(payload?.models?.deepseek_7b?.global_summary?.predicted_orientation || 0),
        实际取向: Number(payload?.models?.deepseek_7b?.global_summary?.actual_targeted_orientation || 0),
      },
    ],
    [payload]
  );

  const summaryRows = useMemo(
    () => [
      {
        model: 'Qwen3-4B',
        概念因果边际: Number(payload?.headline_metrics?.qwen_concept_causal_margin || 0),
        关系因果边际: Number(payload?.headline_metrics?.qwen_relation_causal_margin || 0),
      },
      {
        model: 'DeepSeek-7B',
        概念因果边际: Number(payload?.headline_metrics?.deepseek_concept_causal_margin || 0),
        关系因果边际: Number(payload?.headline_metrics?.deepseek_relation_causal_margin || 0),
      },
    ],
    [payload]
  );

  const conceptRows = useMemo(() => {
    const rows = [];
    for (const [modelName, modelRow] of Object.entries(payload?.models || {})) {
      for (const item of modelRow.probe_concepts || []) {
        rows.push({
          item: `${modelName}:${item.concept}`,
          因果边际: Number(item.causal_margin || 0),
        });
      }
    }
    return rows;
  }, [payload]);

  const relationRows = useMemo(() => {
    const rows = [];
    for (const [modelName, modelRow] of Object.entries(payload?.models || {})) {
      for (const item of modelRow.probe_relations || []) {
        rows.push({
          item: `${modelName}:${item.relation}`,
          因果边际: Number(item.causal_margin || 0),
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
      setError(`共享层带定向消融 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(239,68,68,0.12), transparent 26%), radial-gradient(circle at bottom right, rgba(14,165,233,0.10), transparent 24%), rgba(2,6,23,0.66)',
        border: '1px solid rgba(239,68,68,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>五点三十九终、Qwen3 / DeepSeek7B 共享层带定向消融</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            直接对高共享层带做真实消融，把“预测取向”和“实际伤害方向”放在一起，检查共享层带相关性是否已经升级成因果证据。
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Qwen3 预测 / 实际</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>
            {`${payload?.models?.qwen3_4b?.global_summary?.predicted_orientation_label || '-'} / ${payload?.models?.qwen3_4b?.global_summary?.actual_orientation_label || '-'}`}
          </div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek 预测 / 实际</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>
            {`${payload?.models?.deepseek_7b?.global_summary?.predicted_orientation_label || '-'} / ${payload?.models?.deepseek_7b?.global_summary?.actual_orientation_label || '-'}`}
          </div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek 相对 Qwen 实际取向差</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.gains?.deepseek_minus_qwen_actual_orientation)}</div>
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
              background: value ? 'rgba(34,197,94,0.18)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(34,197,94,0.35)' : 'rgba(248,113,113,0.3)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '不成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 预测取向 vs 实际取向</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={orientationRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[-0.6, 0.3]} />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="预测取向" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="实际取向" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 概念侧 vs 关系侧真实因果边际</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={summaryRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[-0.6, 0.2]} />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="概念因果边际" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="关系因果边际" fill="#f97316" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 概念 probe 因果边际</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={conceptRows} margin={{ top: 10, right: 12, left: 0, bottom: 44 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-18} textAnchor="end" height={58} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[-0.25, 0.2]} />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="因果边际" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 关系 probe 因果边际</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={relationRows} margin={{ top: 10, right: 12, left: 0, bottom: 44 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-18} textAnchor="end" height={58} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[-0.9, 0.1]} />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="因果边际" fill="#f97316" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>5. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>{payload?.project_readout?.summary || '-'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>6. 下一步</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>{payload?.project_readout?.next_question || '-'}</div>
        </div>
      </div>
    </div>
  );
}
