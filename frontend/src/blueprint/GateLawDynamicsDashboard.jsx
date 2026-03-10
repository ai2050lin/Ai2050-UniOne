import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/gate_law_dynamics_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.models && typeof payload.models === 'object');
}

function fmt(v, digits = 4) {
  return Number(v || 0).toFixed(digits);
}

function modelOptions(payload) {
  return Object.keys(payload?.models || {});
}

export default function GateLawDynamicsDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');
  const models = useMemo(() => modelOptions(payload), [payload]);
  const [selectedModel, setSelectedModel] = useState(models[0] || 'gpt2');

  const modelRow = payload?.models?.[selectedModel];
  const summary = modelRow?.global_summary || {};
  const transitions = Array.isArray(modelRow?.transition_reports) ? modelRow.transition_reports : [];

  const lineData = useMemo(
    () =>
      transitions.map((row) => ({
        transition: row.transition,
        factor_only_r2: Number(row.mean_factor_only_r2 || 0),
        full_r2: Number(row.mean_full_r2 || 0),
        recurrence_gain: Number(row.mean_recurrence_gain || 0),
      })),
    [transitions]
  );

  const topGainRows = useMemo(
    () => [...transitions].sort((a, b) => Number(b.mean_recurrence_gain || 0) - Number(a.mean_recurrence_gain || 0)).slice(0, 8),
    [transitions]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 models 字段');
      }
      const nextModels = modelOptions(parsed);
      setPayload(parsed);
      setSelectedModel(nextModels[0] || '');
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`G 门控律动态 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    const nextModels = modelOptions(samplePayload);
    setPayload(samplePayload);
    setSelectedModel(nextModels[0] || '');
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
          'radial-gradient(circle at top left, rgba(16,185,129,0.12), transparent 28%), radial-gradient(circle at top right, rgba(96,165,250,0.1), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(52,211,153,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>G 门控律层间递推</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            比较“只用因素信息”和“因素信息 + 上一层门控状态”对下一层门控的预测力，判断 `G` 是否具有可学习的层间递推。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <select
            value={selectedModel}
            onChange={(event) => setSelectedModel(event.target.value)}
            style={{ background: 'rgba(15,23,42,0.9)', color: '#e2e8f0', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px' }}
          >
            {models.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Factor-only 平均 R²</div>
          <div style={{ color: '#93c5fd', fontSize: '20px', fontWeight: 'bold' }}>{fmt(summary.mean_factor_only_r2)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>递推模型平均 R²</div>
          <div style={{ color: '#86efac', fontSize: '20px', fontWeight: 'bold' }}>{fmt(summary.mean_full_r2)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均递推增益</div>
          <div style={{ color: '#fcd34d', fontSize: '20px', fontWeight: 'bold' }}>{fmt(summary.mean_recurrence_gain)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>正增益层迁移数</div>
          <div style={{ color: '#67e8f9', fontSize: '20px', fontWeight: 'bold' }}>{summary.positive_gain_transition_count ?? '-'}</div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1.15fr) minmax(320px, 0.85fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 各层迁移的递推增益曲线</div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={lineData} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="transition" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-35} textAnchor="end" height={72} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="factor_only_r2" name="factor-only R²" stroke="#60a5fa" strokeWidth={2.2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="full_r2" name="递推模型 R²" stroke="#34d399" strokeWidth={2.2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="recurrence_gain" name="递推增益" stroke="#fbbf24" strokeWidth={2.2} dot={{ r: 2.5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 增益最高的层迁移</div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={topGainRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="transition" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-25} textAnchor="end" height={56} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="mean_recurrence_gain" name="递推增益" fill="#fbbf24" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div style={{ marginTop: '8px', display: 'grid', gap: '8px' }}>
            {topGainRows.slice(0, 4).map((row) => (
              <div key={row.transition} style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '10px', padding: '8px 10px', color: '#cbd5e1', fontSize: '11px', lineHeight: '1.7' }}>
                <div style={{ color: '#fde68a', fontWeight: 'bold' }}>{row.transition}</div>
                <div>{`gain=${fmt(row.mean_recurrence_gain)}, full=${fmt(row.mean_full_r2)}, factor=${fmt(row.mean_factor_only_r2)}`}</div>
                <div>{`正增益头数=${row.positive_gain_head_count}`}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
