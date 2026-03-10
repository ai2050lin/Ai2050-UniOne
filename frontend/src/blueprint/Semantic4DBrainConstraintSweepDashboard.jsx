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
import samplePayload from './data/semantic_4d_brain_constraint_expansion_sweep_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.best_config && Array.isArray(payload.rows));
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

export default function Semantic4DBrainConstraintSweepDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const topRows = useMemo(() => {
    const rows = [...(payload?.rows || [])];
    rows.sort((a, b) => a.brain_held_out_gap - b.brain_held_out_gap);
    return rows.slice(0, 6).map((row, idx) => ({
      rank: `#${idx + 1}`,
      brain_gap: Number(row.brain_held_out_gap || 0),
      mean_gap: Number(row.mean_held_out_gap || 0),
      label: `topk=${row.topk}, agg=${row.include_aggregate ? 'Y' : 'N'}, α=${row.alpha}`,
    }));
  }, [payload]);

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 best_config 或 rows 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`脑侧约束混合扫描 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setPayload(samplePayload);
    setSource('内置样例');
    setError('');
  }

  const best = payload?.best_config || {};
  const baseline = payload?.baseline_semantic_4d_vector || {};

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(22,163,74,0.16), transparent 30%), radial-gradient(circle at top right, rgba(14,165,233,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(34,197,94,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>语义 4D + 3D 的脑侧候选约束混合扫描</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            回答“脑侧约束扩展是不是越多越好”。这里直接扫描轻量混合区，寻找既能压低脑侧误差、又不明显伤害 D 和真实任务的可用配置。
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
        <MetricCard label="最优 topk" value={best.topk ?? '-'} />
        <MetricCard label="最优 α" value={best.alpha ?? '-'} />
        <MetricCard label="脑侧误差改善" value={fmt(best.brain_gap_improvement)} />
        <MetricCard label="整体相关改善" value={fmt(best.corr_improvement)} />
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>1. 最优配置与基线对比</div>
        <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.9' }}>
          <div>{`基线 brain held-out gap: ${fmt(baseline.brain_held_out_gap)}`}</div>
          <div>{`最优配置 brain held-out gap: ${fmt(best.brain_held_out_gap)}`}</div>
          <div>{`最优配置 mean held-out gap: ${fmt(best.mean_held_out_gap)}`}</div>
          <div>{`最优配置 constraint count: ${fmt(best.mean_constraint_count, 2)}`}</div>
          <div>{`是否使用 aggregate: ${best.include_aggregate ? '是' : '否'}`}</div>
        </div>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 前六个配置的脑侧误差</div>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={topRows} margin={{ top: 10, right: 12, left: 0, bottom: 40 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="rank" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip
              formatter={(value, name) => [fmt(value), name === 'brain_gap' ? 'brain held-out gap' : 'mean held-out gap']}
              labelFormatter={(label, items) => {
                const row = items?.[0]?.payload;
                return row ? `${label} | ${row.label}` : label;
              }}
              contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }}
            />
            <Bar dataKey="brain_gap" name="brain_gap" fill="#16a34a" radius={[4, 4, 0, 0]} />
            <Bar dataKey="mean_gap" name="mean_gap" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 当前判断</div>
        <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
          全量结构化脑侧约束会过约束，但轻量混合区能明显降低脑侧留一法误差。当前更合理的策略不是“继续堆更多约束”，而是保持约束面足够薄、足够有针对性。
        </div>
        <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
          下一步问题：{payload?.project_readout?.next_question || '-'}
        </div>
      </div>
    </div>
  );
}
