import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/qwen3_deepseek7b_protocol_field_boundary_atlas_sample.json';

const MODEL_LABELS = {
  qwen3_4b: 'Qwen3-4B',
  deepseek_7b: 'DeepSeek-7B',
};

function isValidPayload(payload) {
  return Boolean(payload && payload.models && typeof payload.models === 'object');
}

function fmt(v, digits = 4) {
  return Number(v || 0).toFixed(digits);
}

function modelOptions(payload) {
  return Object.keys(payload?.models || {});
}

function heatColor(value) {
  if (value === null || value === undefined) {
    return 'rgba(248,113,113,0.18)';
  }
  const normalized = Math.max(0, Math.min(1, Number(value) / 32));
  const alpha = 0.12 + normalized * 0.55;
  return `rgba(34,197,94,${alpha.toFixed(3)})`;
}

export default function Qwen3DeepSeekProtocolFieldBoundaryAtlasDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');
  const models = useMemo(() => modelOptions(payload), [payload]);
  const [selectedModel, setSelectedModel] = useState(models[0] || 'qwen3_4b');

  const modelRow = payload?.models?.[selectedModel];
  const summary = modelRow?.global_summary || {};
  const concepts = modelRow?.concepts || {};

  const histogramData = useMemo(
    () =>
      Object.entries(summary?.minimal_boundary_histogram || {}).map(([key, value]) => ({
        k: key,
        count: Number(value || 0),
      })),
    [summary]
  );

  const fieldBars = useMemo(
    () =>
      Object.entries(summary?.boundary_histogram_by_true_field || {}).map(([field, value]) => ({
        field,
        boundaryable_count: Object.entries(value || {}).reduce((acc, [k, count]) => acc + (k === 'none' ? 0 : Number(count || 0)), 0),
      })),
    [summary]
  );

  const conceptRows = useMemo(
    () =>
      Object.entries(concepts).map(([concept, entry]) => {
        const mass = entry?.field_scores?.[entry?.true_field]?.mass_summary || {};
        return {
          concept,
          true_field: entry?.true_field,
          preferred_field: entry?.preferred_field,
          match: Boolean(entry?.preferred_field_matches_truth),
          boundary_k: entry?.boundary_summary?.minimal_boundary_k ?? null,
          heads50: Number(mass?.heads_for_50pct_mass || 0),
          heads80: Number(mass?.heads_for_80pct_mass || 0),
        };
      }),
    [concepts]
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
      setError(`协议场边界图谱 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(251,191,36,0.12), transparent 28%), radial-gradient(circle at top right, rgba(248,113,113,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(251,191,36,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>Qwen3 / DeepSeek7B 协议场边界图谱</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            展示更大概念集合上的 `k*(c, tau)` 分布、协议场匹配率，以及调用质量覆盖所需的头数规模。
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
                {MODEL_LABELS[item] || item}
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>协议场匹配率</div>
          <div style={{ color: '#86efac', fontSize: '20px', fontWeight: 'bold' }}>{fmt(summary.preferred_field_match_rate, 3)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均 50% 质量头数</div>
          <div style={{ color: '#fde68a', fontSize: '20px', fontWeight: 'bold' }}>{fmt(summary.mean_heads_for_50pct_mass, 2)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均 80% 质量头数</div>
          <div style={{ color: '#f59e0b', fontSize: '20px', fontWeight: 'bold' }}>{fmt(summary.mean_heads_for_80pct_mass, 2)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>边界直方图</div>
          <div style={{ color: '#93c5fd', fontSize: '16px', fontWeight: 'bold' }}>
            {Object.entries(summary.minimal_boundary_histogram || {}).map(([k, v]) => `${k}:${v}`).join(' / ') || '-'}
          </div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(320px, 0.85fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 概念级边界热表</div>
          <div style={{ display: 'grid', gap: '8px' }}>
            {conceptRows.map((row) => (
              <div
                key={row.concept}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '110px 90px 90px 90px 100px 100px',
                  gap: '8px',
                  alignItems: 'center',
                  border: '1px solid rgba(148,163,184,0.18)',
                  borderRadius: '10px',
                  padding: '8px 10px',
                  background: 'rgba(15,23,42,0.35)',
                  color: '#cbd5e1',
                  fontSize: '11px',
                }}
              >
                <div>
                  <div style={{ color: '#f8fafc', fontWeight: 'bold' }}>{row.concept}</div>
                  <div style={{ color: '#94a3b8' }}>{row.true_field}</div>
                </div>
                <div style={{ color: row.match ? '#86efac' : '#fca5a5' }}>{row.match ? '匹配' : '偏移'}</div>
                <div
                  style={{
                    textAlign: 'center',
                    borderRadius: '999px',
                    padding: '4px 0',
                    background: heatColor(row.boundary_k),
                    color: '#f8fafc',
                    fontWeight: 'bold',
                  }}
                >
                  {row.boundary_k === null ? 'none' : `k=${row.boundary_k}`}
                </div>
                <div>{`50%=${row.heads50}`}</div>
                <div>{`80%=${row.heads80}`}</div>
                <div style={{ color: '#94a3b8' }}>{`pref=${row.preferred_field}`}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ display: 'grid', gap: '12px' }}>
          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 最小边界直方图</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={histogramData} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
                <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
                <XAxis dataKey="k" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
                <Bar dataKey="count" name="概念数" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 各协议场的可边界化数量</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={fieldBars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
                <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
                <XAxis dataKey="field" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
                <Bar dataKey="boundaryable_count" name="有边界概念数" fill="#22c55e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
