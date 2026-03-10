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
import samplePayload from './data/qwen3_deepseek7b_attention_topology_basis_sample.json';

const MODEL_LABELS = {
  qwen3_4b: 'Qwen3-4B',
  deepseek_7b: 'DeepSeek-7B',
};

const FAMILY_LABELS = {
  fruit: 'fruit',
  animal: 'animal',
  abstract: 'abstract',
};

function isValidPayload(payload) {
  return Boolean(payload && payload.models && typeof payload.models === 'object');
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

export default function Qwen3DeepSeekAttentionTopologyDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const models = payload?.models || {};

  const familyRows = useMemo(
    () =>
      Object.keys(FAMILY_LABELS).map((family) => ({
        family: FAMILY_LABELS[family],
        qwen3: Number(models?.qwen3_4b?.family_summary?.[family]?.mean_topology_residual_ratio || 0),
        deepseek7b: Number(models?.deepseek_7b?.family_summary?.[family]?.mean_topology_residual_ratio || 0),
      })),
    [models]
  );

  const entropyRows = useMemo(
    () =>
      Object.keys(FAMILY_LABELS).map((family) => ({
        family: FAMILY_LABELS[family],
        qwen3: Number(models?.qwen3_4b?.family_summary?.[family]?.mean_last_token_entropy || 0),
        deepseek7b: Number(models?.deepseek_7b?.family_summary?.[family]?.mean_last_token_entropy || 0),
      })),
    [models]
  );

  const probeRows = useMemo(() => {
    const words = ['apple', 'cat', 'truth'];
    const rows = [];
    for (const modelKey of ['qwen3_4b', 'deepseek_7b']) {
      for (const word of words) {
        const probe = models?.[modelKey]?.probe_fits?.[word];
        if (!probe) continue;
        rows.push({
          model: MODEL_LABELS[modelKey],
          word,
          family_fit: Number(probe.fit?.[probe.family]?.residual_ratio || 0),
          best_wrong_fit: Number(
            Math.min(
              ...Object.entries(probe.fit || {})
                .filter(([family]) => family !== probe.family)
                .map(([, row]) => Number(row.residual_ratio || 0))
            )
          ),
          supported: probe.supports_family_topology_basis ? 1 : 0,
        });
      }
    }
    return rows;
  }, [models]);

  const supportRows = useMemo(
    () =>
      Object.entries(models).map(([modelKey, row]) => ({
        model: MODEL_LABELS[modelKey] || modelKey,
        support_rate:
          ['apple', 'cat', 'truth'].reduce(
            (acc, word) => acc + (row?.probe_fits?.[word]?.supports_family_topology_basis ? 1 : 0),
            0
          ) / 3.0,
      })),
    [models]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 models 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`Qwen3 / DeepSeek7B 拓扑直测 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(14,165,233,0.12), transparent 28%), radial-gradient(circle at top right, rgba(34,197,94,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(14,165,233,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>Qwen3 / DeepSeek7B 拓扑直测</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            直接比较两模型的 attention-topology family basis 与 `apple / cat / truth` 探针残差，确认 `T` 是否已经进入对称直测状态。
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
        {supportRows.map((row) => (
          <div key={row.model} style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
            <div style={{ color: '#94a3b8', fontSize: '11px' }}>{row.model}</div>
            <div style={{ color: '#38bdf8', fontSize: '18px', fontWeight: 'bold' }}>{pct(row.support_rate)}</div>
            <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>probe 支持率</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. family residual 对比</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={familyRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="family" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="qwen3" name="Qwen3-4B" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="deepseek7b" name="DeepSeek-7B" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. family entropy 对比</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={entropyRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="family" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="qwen3" name="Qwen3-4B" fill="#60a5fa" radius={[4, 4, 0, 0]} />
              <Bar dataKey="deepseek7b" name="DeepSeek-7B" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. probe residual 排序</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={probeRows} margin={{ top: 10, right: 12, left: 0, bottom: 48 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="word" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value, _name, props) => [fmt(value), `${props?.payload?.model} / ${props?.payload?.word}`]} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="family_fit" name="真实 family residual" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="best_wrong_fit" name="最优错误 family residual" fill="#f87171" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div style={{ marginTop: '10px', color: '#cbd5e1', fontSize: '12px', lineHeight: '1.7' }}>
            如果绿色柱 consistently 低于红色柱，并且 `probe 支持率 = 100%`，就说明 `apple / cat / truth` 已经在两模型上进入了同协议拓扑 family basis。
          </div>
        </div>
      </div>
    </div>
  );
}
