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
import samplePayload from './data/real_multistep_agi_closure_benchmark_sample.json';

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

export default function RealMultistepAgiClosureDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const plain = payload?.systems?.plain_local || {};
  const trace = payload?.systems?.trace_gated_local || {};
  const improvements = payload?.improvements || {};
  const score = payload?.real_closure_score || {};
  const hypotheses = payload?.hypotheses || {};

  const stageRows = useMemo(
    () => [
      { metric: 'phase1 工具', plain: metricMean(plain, 'phase1_tool_accuracy'), trace: metricMean(trace, 'phase1_tool_accuracy') },
      { metric: 'phase1 路径', plain: metricMean(plain, 'phase1_route_accuracy'), trace: metricMean(trace, 'phase1_route_accuracy') },
      { metric: 'phase1 终步', plain: metricMean(plain, 'phase1_final_accuracy'), trace: metricMean(trace, 'phase1_final_accuracy') },
      { metric: 'phase1 成功', plain: metricMean(plain, 'phase1_episode_success'), trace: metricMean(trace, 'phase1_episode_success') },
      { metric: 'phase2 成功', plain: metricMean(plain, 'phase2_episode_success'), trace: metricMean(trace, 'phase2_episode_success') },
      { metric: '总体成功', plain: metricMean(plain, 'overall_episode_success'), trace: metricMean(trace, 'overall_episode_success') },
    ],
    [plain, trace]
  );

  const retentionRows = useMemo(
    () => [
      { metric: 'phase2 后保留', plain: metricMean(plain, 'retention_after_phase2'), trace: metricMean(trace, 'retention_after_phase2') },
      { metric: '遗忘下降量', plain: metricMean(plain, 'retention_drop'), trace: metricMean(trace, 'retention_drop') },
      { metric: '总体路径准确', plain: metricMean(plain, 'overall_route_accuracy'), trace: metricMean(trace, 'overall_route_accuracy') },
      { metric: '总体终步准确', plain: metricMean(plain, 'overall_final_accuracy'), trace: metricMean(trace, 'overall_final_accuracy') },
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
      setError(`真实多步闭环 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(34,197,94,0.12), transparent 28%), radial-gradient(circle at top right, rgba(14,165,233,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(34,197,94,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>真实多步 AGI 闭环</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            三步序列任务、末端统一监督、phase 切换与 replay/stability 同时在场，用来检验机制是否真的转化成真实多步收益。
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>真实闭环分数</div>
          <div style={{ color: '#86efac', fontSize: '20px', fontWeight: 'bold' }}>{fmt(score.trace_gated_local)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>分数增益</div>
          <div style={{ color: '#93c5fd', fontSize: '20px', fontWeight: 'bold' }}>{fmt(score.score_gain)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>回合成功增益</div>
          <div style={{ color: '#fcd34d', fontSize: '20px', fontWeight: 'bold' }}>{fmt(improvements.overall_episode_success)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>保留收益</div>
          <div style={{ color: '#f87171', fontSize: '20px', fontWeight: 'bold' }}>{fmt(improvements.retention_after_phase2)}</div>
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
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. phase 表现与总体成功</div>
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

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. retention 与后续步骤质量</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={retentionRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="metric" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-18} textAnchor="end" height={54} />
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
  );
}
