import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/qwen3_deepseek7b_attention_topology_atlas_sample.json';

const MODEL_LABELS = {
  qwen3_4b: 'Qwen3-4B',
  deepseek_7b: 'DeepSeek-7B',
};

const FAMILY_COLORS = {
  fruit: '#38bdf8',
  animal: '#22c55e',
  abstract: '#f59e0b',
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

export default function Qwen3DeepSeekAttentionTopologyAtlasDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const models = payload?.models || {};

  const summaryRows = useMemo(
    () =>
      Object.entries(models).map(([name, row]) => ({
        model: MODEL_LABELS[name] || name,
        match_rate: Number(row?.global_summary?.preferred_family_match_rate || 0),
        margin: Number(row?.global_summary?.mean_margin_vs_best_wrong || 0),
        true_fit: Number(row?.global_summary?.mean_true_family_residual || 0),
        wrong_fit: Number(row?.global_summary?.mean_best_wrong_residual || 0),
      })),
    [models]
  );

  const familyRows = useMemo(
    () =>
      Object.entries(models).flatMap(([name, row]) =>
        Object.entries(row?.global_summary?.family_match_rate || {}).map(([family, value]) => ({
          model: MODEL_LABELS[name] || name,
          family,
          match_rate: Number(value || 0),
        }))
      ),
    [models]
  );

  const conceptRows = useMemo(
    () =>
      Object.entries(models).flatMap(([name, row]) =>
        Object.entries(row?.concepts || {}).map(([word, item]) => ({
          model: MODEL_LABELS[name] || name,
          word,
          family: item.true_family,
          true_fit: Number(item.family_fit?.[item.true_family]?.residual_ratio || 0),
          wrong_fit: Number(
            Math.min(
              ...Object.entries(item.family_fit || {})
                .filter(([family]) => family !== item.true_family)
                .map(([, fit]) => Number(fit.residual_ratio || 0))
            )
          ),
          margin: Number(item.summary?.margin_vs_best_wrong || 0),
        }))
      ),
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
      setError(`拓扑图谱 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(56,189,248,0.12), transparent 28%), radial-gradient(circle at top right, rgba(245,158,11,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(56,189,248,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>Qwen3 / DeepSeek7B 拓扑图谱</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把直测 `T` 从 `apple / cat / truth` 扩到完整概念集，直接看两模型在更大概念域里是否仍保持稳定的 family-basis 拓扑结构。
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: '10px' }}>
        {summaryRows.map((row) => (
          <div key={row.model} style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
            <div style={{ color: '#94a3b8', fontSize: '11px' }}>{row.model}</div>
            <div style={{ color: '#22c55e', fontSize: '18px', fontWeight: 'bold' }}>{pct(row.match_rate)}</div>
            <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`平均 margin ${fmt(row.margin)}`}</div>
            <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '2px' }}>{`真实 residual ${fmt(row.true_fit)} / 错误 residual ${fmt(row.wrong_fit)}`}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 模型级总览</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={summaryRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="match_rate" name="匹配率" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="true_fit" name="真实 residual" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="wrong_fit" name="错误 residual" fill="#f87171" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. family 匹配率</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={familyRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="family" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="match_rate" name="family 匹配率" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 概念级 residual 散点图</div>
          <ResponsiveContainer width="100%" height={340}>
            <ScatterChart margin={{ top: 10, right: 12, left: 0, bottom: 20 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis type="number" dataKey="true_fit" name="真实 residual" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <YAxis type="number" dataKey="wrong_fit" name="错误 residual" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                formatter={(value) => fmt(value)}
                labelFormatter={(_value, payload) => {
                  const row = payload?.[0]?.payload;
                  return row ? `${row.model} / ${row.word} / ${row.family}` : '';
                }}
                contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }}
              />
              <Legend />
              {Object.keys(FAMILY_COLORS).map((family) => (
                <Scatter
                  key={family}
                  name={family}
                  data={conceptRows.filter((row) => row.family === family)}
                  fill={FAMILY_COLORS[family]}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
          <div style={{ marginTop: '10px', color: '#cbd5e1', fontSize: '12px', lineHeight: '1.7' }}>
            如果点整体落在对角线下方，说明真实 family residual 系统性低于错误 family residual，也就是 `T` 的 family-basis 不是个别词现象，而是更广概念域里的稳定结构。
          </div>
        </div>
      </div>
    </div>
  );
}
