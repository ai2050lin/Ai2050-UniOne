import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/d_problem_atlas_summary_sample.json';

const MODEL_LABELS = {
  gpt2: 'GPT-2',
  qwen3_4b: 'Qwen3-4B',
  deepseek_7b: 'DeepSeek-7B',
};

function isValidPayload(payload) {
  return Boolean(payload && Array.isArray(payload.models) && payload.scan && payload.global_summary);
}

function fmt(v, digits = 4) {
  return Number(v || 0).toFixed(digits);
}

function gainColor(x, y) {
  if (x > 0 && y > 0) return '#22c55e';
  if (x > 0 || y > 0) return '#f59e0b';
  return '#f87171';
}

function MetricCard({ label, value }) {
  return (
    <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
      <div style={{ color: '#94a3b8', fontSize: '10px' }}>{label}</div>
      <div style={{ color: '#f8fafc', fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>{value}</div>
    </div>
  );
}

function CandidateLines({ title, rows }) {
  return (
    <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
      <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>{title}</div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={rows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
          <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
          <XAxis dataKey="rank" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
          <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
          <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
          <Legend />
          <Line type="monotone" dataKey="novel_gain" name="novel_gain" stroke="#60a5fa" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="retention_gain" name="retention_gain" stroke="#34d399" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="overall_gain" name="overall_gain" stroke="#f97316" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function BestCard({ title, payload }) {
  const entries = Object.entries(payload || {});
  return (
    <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
      <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>{title}</div>
      <div style={{ display: 'grid', gap: '8px' }}>
        {entries.length === 0 && <div style={{ color: '#94a3b8', fontSize: '11px' }}>当前没有正区域。</div>}
        {entries.map(([key, value]) => (
          <div key={key} style={{ border: '1px solid rgba(148,163,184,0.16)', borderRadius: '10px', padding: '8px 10px', color: '#cbd5e1', fontSize: '11px' }}>
            <div style={{ color: '#94a3b8', marginBottom: '4px' }}>{key}</div>
            <div style={{ color: '#f8fafc', fontWeight: 'bold' }}>{typeof value === 'number' ? fmt(value) : String(value)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function DProblemAtlasDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const modelBars = useMemo(
    () =>
      (payload?.models || []).map((row) => ({
        model: MODEL_LABELS[row.model] || row.model,
        direct_grounding: Number(row.direct_grounding || 0),
        raw_shared_grounding: Number(row.raw_shared_grounding || 0),
        geometry_grounding: Number(row.geometry_grounding || 0),
      })),
    [payload]
  );

  const scatterRows = useMemo(
    () =>
      (payload?.models || []).flatMap((row) => [
        {
          id: `${row.model}-raw`,
          label: `${MODEL_LABELS[row.model] || row.model} / raw_shared`,
          x: Number(row.raw_shared_novel_gain || 0),
          y: Number(row.raw_shared_retention_gain || 0),
          fill: gainColor(Number(row.raw_shared_novel_gain || 0), Number(row.raw_shared_retention_gain || 0)),
        },
        {
          id: `${row.model}-geometry`,
          label: `${MODEL_LABELS[row.model] || row.model} / geometry_dual_store`,
          x: Number(row.geometry_novel_gain || 0),
          y: Number(row.geometry_retention_gain || 0),
          fill: gainColor(Number(row.geometry_novel_gain || 0), Number(row.geometry_retention_gain || 0)),
        },
      ]),
    [payload]
  );

  const scanRows = useMemo(
    () =>
      (payload?.scan?.top_candidates || []).slice(0, 12).map((row, index) => ({
        rank: index + 1,
        overall_gain: Number(row.overall_gain || 0),
        novel_gain: Number(row.novel_gain || 0),
        retention_gain: Number(row.retention_gain || 0),
      })),
    [payload]
  );

  const learnedRows = useMemo(
    () =>
      (payload?.learned_controller?.top_overall || []).slice(0, 8).map((row, index) => ({
        rank: index + 1,
        overall_gain: Number(row.overall_gain || 0),
        novel_gain: Number(row.novel_gain || 0),
        retention_gain: Number(row.retention_gain || 0),
      })),
    [payload]
  );

  const baseOffsetRows = useMemo(
    () =>
      (payload?.base_offset_consolidation?.top_overall || []).slice(0, 8).map((row, index) => ({
        rank: index + 1,
        overall_gain: Number(row.overall_gain || 0),
        novel_gain: Number(row.novel_gain || 0),
        retention_gain: Number(row.retention_gain || 0),
      })),
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) throw new Error('缺少 models / scan / global_summary 字段');
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`D Problem Atlas JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(251,191,36,0.12), transparent 28%), radial-gradient(circle at top right, rgba(14,165,233,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(251,191,36,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>D Problem Atlas</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            汇总 D 的关键证据链：三模型高维接地桥、旧 dual-store 扫描、residual-gate、Bayesian、learned controller、
            two-phase、three-phase，以及最新的 base+offset 统一整合律。
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
        <MetricCard label="三模型是否同时突破 novel + retention" value={payload?.global_summary?.all_models_fail_novel_and_retention ? '未突破' : '已突破'} />
        <MetricCard label="residual-gate dual-positive 区域" value={payload?.residual_gate?.dual_positive_count ?? 0} />
        <MetricCard label="Bayesian dual-positive 区域" value={payload?.bayes_consolidation?.dual_positive_count ?? 0} />
        <MetricCard label="base+offset dual-positive 区域" value={payload?.base_offset_consolidation?.dual_positive_count ?? 0} />
        <MetricCard label="offset-stabilization dual-positive 区域" value={payload?.offset_stabilization?.dual_positive_count ?? 0} />
        <MetricCard label="multistage 最优 overall_gain" value={fmt(payload?.global_summary?.multistage_best_overall_gain ?? 0)} />
        <MetricCard label="learned controller 最优 overall_gain" value={fmt(payload?.global_summary?.learned_controller_best_overall_gain ?? 0)} />
        <MetricCard label="base+offset 最优 overall_gain" value={fmt(payload?.global_summary?.base_offset_best_overall_gain ?? 0)} />
        <MetricCard label="offset-stabilization 最优 overall_gain" value={fmt(payload?.global_summary?.offset_stabilization_best_overall_gain ?? 0)} />
        <MetricCard label="base+offset 最优 novel_gain" value={fmt(payload?.global_summary?.base_offset_best_novel_gain ?? 0)} />
        <MetricCard label="base+offset 最优 retention_gain" value={fmt(payload?.global_summary?.base_offset_best_retention_gain ?? 0)} />
        <MetricCard label="multistage 最优 retention_gain" value={fmt(payload?.global_summary?.multistage_best_retention_gain ?? 0)} />
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三模型接地总分对比</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={modelBars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="direct_grounding" name="direct" fill="#94a3b8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="raw_shared_grounding" name="raw_shared" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="geometry_grounding" name="geometry_dual_store" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. novel_gain vs retention_gain</div>
          <ResponsiveContainer width="100%" height={280}>
            <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="novel_gain" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis type="number" dataKey="y" name="retention_gain" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                formatter={(value) => fmt(value)}
                labelFormatter={(_, payloadRows) => payloadRows?.[0]?.payload?.label || ''}
                contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }}
              />
              <Scatter data={scatterRows}>
                {scatterRows.map((row) => (
                  <Cell key={row.id} fill={row.fill} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <CandidateLines title="3. 旧 dual-store 扫描前 12 名" rows={scanRows} />
        <CandidateLines title="4. learned controller 前 8 名" rows={learnedRows} />
        <CandidateLines title="5. base+offset 统一律前 8 名" rows={baseOffsetRows} />
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '12px' }}>
        <BestCard title="6. residual-gate 最优 dual-positive" payload={payload?.residual_gate?.best_dual_positive} />
        <BestCard title="7. Bayesian 最优 dual-positive" payload={payload?.bayes_consolidation?.best_dual_positive} />
        <BestCard title="8. two-phase 最优 overall" payload={payload?.two_phase_consolidation?.best_overall} />
        <BestCard title="9. three-phase 最优 overall" payload={payload?.three_phase_consolidation?.best_overall} />
        <BestCard title="10. base+offset 最优 overall" payload={payload?.base_offset_consolidation?.top_overall?.[0]} />
        <BestCard title="11. offset-stabilization 最优 overall" payload={payload?.offset_stabilization?.top_overall?.[0]} />
        <BestCard title="12. multistage 最优 overall" payload={payload?.multistage_stabilization?.top_overall?.[0]} />
        <BestCard title="13. 三模态原型增益" payload={payload?.multimodal_proto?.gains_vs_direct} />
      </div>
    </div>
  );
}
