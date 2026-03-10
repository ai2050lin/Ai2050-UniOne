import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/toy_grounding_credit_continual_benchmark_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && typeof payload.systems === 'object');
}

function fmt(v, digits = 4) {
  return Number(v || 0).toFixed(digits);
}

function pct(v) {
  return `${(Number(v || 0) * 100).toFixed(1)}%`;
}

function metricMean(row, key) {
  return Number(row?.[key]?.mean || 0);
}

export default function ToyGroundingCreditContinualDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const plain = payload?.systems?.plain_local || {};
  const trace = payload?.systems?.trace_gated_local || {};
  const improvements = payload?.improvements || {};
  const hypotheses = payload?.hypotheses || {};

  const stageRows = useMemo(
    () => [
      { metric: '接地阶段一', plain: metricMean(plain, 'grounding_phase1_accuracy'), trace: metricMean(trace, 'grounding_phase1_accuracy') },
      { metric: '延迟阶段一', plain: metricMean(plain, 'delayed_phase1_accuracy'), trace: metricMean(trace, 'delayed_phase1_accuracy') },
      { metric: '接地阶段二', plain: metricMean(plain, 'grounding_phase2_accuracy'), trace: metricMean(trace, 'grounding_phase2_accuracy') },
      { metric: '延迟阶段二', plain: metricMean(plain, 'delayed_phase2_accuracy'), trace: metricMean(trace, 'delayed_phase2_accuracy') },
      { metric: '总体接地', plain: metricMean(plain, 'overall_grounding_accuracy'), trace: metricMean(trace, 'overall_grounding_accuracy') },
      { metric: '总体延迟', plain: metricMean(plain, 'overall_delayed_accuracy'), trace: metricMean(trace, 'overall_delayed_accuracy') },
    ],
    [plain, trace]
  );

  const retentionRows = useMemo(
    () => [
      { metric: '阶段二后保留率', plain: metricMean(plain, 'retention_after_phase2'), trace: metricMean(trace, 'retention_after_phase2') },
      { metric: '遗忘下降量', plain: metricMean(plain, 'retention_drop'), trace: metricMean(trace, 'retention_drop') },
    ],
    [plain, trace]
  );

  const radarRows = useMemo(
    () => [
      { metric: '总体接地', plain: metricMean(plain, 'overall_grounding_accuracy'), trace: metricMean(trace, 'overall_grounding_accuracy') },
      { metric: '总体延迟', plain: metricMean(plain, 'overall_delayed_accuracy'), trace: metricMean(trace, 'overall_delayed_accuracy') },
      { metric: '阶段二后保留', plain: metricMean(plain, 'retention_after_phase2'), trace: metricMean(trace, 'retention_after_phase2') },
    ],
    [plain, trace]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 systems 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`toy 闭环基准 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(16,185,129,0.12), transparent 28%), radial-gradient(circle at top right, rgba(248,113,113,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(16,185,129,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>toy 接地-信用-持续学习闭环</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            对比 `plain_local` 与 `trace_gated_local`，看局部 trace、稳定化和少量回放是否真的改善接地、延迟信用分配和持续学习。
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>总体接地提升</div>
          <div style={{ color: '#86efac', fontSize: '20px', fontWeight: 'bold' }}>{fmt(improvements.overall_grounding_accuracy)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>总体延迟提升</div>
          <div style={{ color: '#93c5fd', fontSize: '20px', fontWeight: 'bold' }}>{fmt(improvements.overall_delayed_accuracy)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>阶段二后保留提升</div>
          <div style={{ color: '#fcd34d', fontSize: '20px', fontWeight: 'bold' }}>{fmt(improvements.retention_after_phase2)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>遗忘下降削减</div>
          <div style={{ color: '#f87171', fontSize: '20px', fontWeight: 'bold' }}>{fmt(improvements.retention_drop_reduction)}</div>
        </div>
      </div>

      <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {Object.entries(hypotheses).map(([key, value]) => (
          <div
            key={key}
            style={{
              borderRadius: '999px',
              padding: '6px 10px',
              fontSize: '11px',
              color: value ? '#dcfce7' : '#fee2e2',
              background: value ? 'rgba(34,197,94,0.18)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(34,197,94,0.3)' : 'rgba(248,113,113,0.3)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '不成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1.1fr) minmax(320px, 0.9fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 阶段性表现对比</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={stageRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="metric" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-20} textAnchor="end" height={56} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="plain" name="plain_local" fill="#60a5fa" radius={[4, 4, 0, 0]} />
              <Bar dataKey="trace" name="trace_gated_local" fill="#34d399" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ display: 'grid', gap: '12px' }}>
          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 核心闭环能力雷达</div>
            <ResponsiveContainer width="100%" height={240}>
              <RadarChart data={radarRows}>
                <PolarGrid stroke="rgba(148,163,184,0.28)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#cbd5e1', fontSize: 11 }} />
                <PolarRadiusAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <Radar name="plain_local" dataKey="plain" stroke="#60a5fa" fill="#60a5fa" fillOpacity={0.2} />
                <Radar name="trace_gated_local" dataKey="trace" stroke="#34d399" fill="#34d399" fillOpacity={0.2} />
                <Legend />
                <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 持续学习与遗忘</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={retentionRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
                <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
                <XAxis dataKey="metric" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
                <Legend />
                <Bar dataKey="plain" name="plain_local" fill="#60a5fa" radius={[4, 4, 0, 0]} />
                <Bar dataKey="trace" name="trace_gated_local" fill="#34d399" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
