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
import samplePayload from './data/semantic_4d_brain_candidate_coverage_expansion_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.best_config && payload.light_mix_best && payload.baseline_semantic_4d_vector);
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function MetricCard({ label, value }) {
  return (
    <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
      <div style={{ color: '#94a3b8', fontSize: '10px' }}>{label}</div>
      <div style={{ color: '#f8fafc', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{value}</div>
    </div>
  );
}

export default function Semantic4DBrainCandidateCoverageDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const compareRows = useMemo(() => ([
    {
      item: 'brain',
      baseline: Number(payload?.baseline_semantic_4d_vector?.brain_held_out_gap || 0),
      light: Number(payload?.light_mix_best?.brain_held_out_gap || 0),
      coverage: Number(payload?.best_config?.brain_held_out_gap || 0),
    },
    {
      item: 'mean',
      baseline: Number(payload?.baseline_semantic_4d_vector?.held_out_mean_gap || 0),
      light: Number(payload?.light_mix_best?.mean_held_out_gap || 0),
      coverage: Number(payload?.best_config?.mean_held_out_gap || 0),
    },
  ]), [payload]);

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 best_config / light_mix_best / baseline 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`脑侧候选覆盖扩展 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setPayload(samplePayload);
    setSource('内置样例');
    setError('');
  }

  const best = payload?.best_config || {};

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(5,150,105,0.16), transparent 30%), radial-gradient(circle at top right, rgba(59,130,246,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(5,150,105,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>语义 4D + 3D 的脑侧候选覆盖面扩展</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            在上一轮轻量有效区上，只增量加入少量候选覆盖锚点，测试能否继续压低脑侧误差，而不重新掉回全量过约束。
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
        <MetricCard label="最优 coverage 模式" value={best.coverage_mode || '-'} />
        <MetricCard label="coverage 权重" value={best.coverage_weight ?? '-'} />
        <MetricCard label="brain 相对轻量混合改善" value={fmt(best.brain_gap_improvement_vs_light_best)} />
        <MetricCard label="coverage 锚点数" value={fmt(best.mean_coverage_count, 2)} />
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三阶段对比</div>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={compareRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip
              formatter={(value) => [fmt(value), '留一法误差']}
              contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }}
            />
            <Bar dataKey="baseline" name="基线 4D + 3D" fill="#334155" radius={[4, 4, 0, 0]} />
            <Bar dataKey="light" name="轻量混合区" fill="#16a34a" radius={[4, 4, 0, 0]} />
            <Bar dataKey="coverage" name="轻量混合 + 覆盖扩展" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 最优配置</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.9' }}>
            <div>{`topk = ${best.topk}`}</div>
            <div>{`alpha = ${best.alpha}`}</div>
            <div>{`coverage_mode = ${best.coverage_mode}`}</div>
            <div>{`coverage_topm = ${best.coverage_topm}`}</div>
            <div>{`brain held-out gap = ${fmt(best.brain_held_out_gap)}`}</div>
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            脑侧候选覆盖面确实可以继续扩，但必须保持“极小锚点 + 低权重”的方式。当前最优不是全量覆盖，而是轻量混合区上再加两个低权重 `mean_anchor`。
          </div>
          <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
            下一步问题：{payload?.project_readout?.next_question || '-'}
          </div>
        </div>
      </div>
    </div>
  );
}
