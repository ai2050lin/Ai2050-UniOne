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
import samplePayload from './data/real_multistep_memory_segment_summary_scan_sample.json';

const SYSTEM_META = {
  single_anchor_beta_086: {
    label: '单锚点 beta=0.86',
    color: '#94a3b8',
  },
  gated_triple_tau_joint_long_horizon: {
    label: '联合温度三锚点',
    color: '#22c55e',
  },
  gated_triple_tau_joint_long_horizon_segment_summary: {
    label: '联合温度三锚点 + 段级摘要',
    color: '#38bdf8',
  },
  gated_triple_tau_joint_ultra_oracle: {
    label: '超长程强化三锚点',
    color: '#f59e0b',
  },
  gated_triple_tau_joint_ultra_oracle_segment_summary: {
    label: '超长程强化三锚点 + 段级摘要',
    color: '#f472b6',
  },
};

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && Array.isArray(payload.ranking));
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function labelOfSystem(systemName) {
  return SYSTEM_META[systemName]?.label || systemName || '-';
}

export default function RealMultistepSegmentSummaryDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const systems = payload?.systems || {};
  const ranking = Array.isArray(payload?.ranking) ? payload.ranking : [];
  const gains = payload?.gains || {};
  const best = payload?.best || {};
  const hypotheses = payload?.hypotheses || {};

  const curveRows = useMemo(() => {
    const single = systems?.single_anchor_beta_086?.global_summary || {};
    const joint = systems?.gated_triple_tau_joint_long_horizon?.global_summary || {};
    const jointSegment = systems?.gated_triple_tau_joint_long_horizon_segment_summary?.global_summary || {};
    const ultra = systems?.gated_triple_tau_joint_ultra_oracle?.global_summary || {};
    const ultraSegment = systems?.gated_triple_tau_joint_ultra_oracle_segment_summary?.global_summary || {};
    const lengths = Array.isArray(single?.lengths) ? single.lengths : [];
    return lengths.map((length, idx) => ({
      length: `L=${length}`,
      single: Number(single?.closure_curve?.[idx] || 0),
      joint: Number(joint?.closure_curve?.[idx] || 0),
      joint_segment: Number(jointSegment?.closure_curve?.[idx] || 0),
      ultra: Number(ultra?.closure_curve?.[idx] || 0),
      ultra_segment: Number(ultraSegment?.closure_curve?.[idx] || 0),
    }));
  }, [systems]);

  const rankingRows = useMemo(
    () =>
      ranking.map((row) => ({
        system: labelOfSystem(row.system),
        mean_closure: Number(row.mean_closure_score || 0),
        max_length: Number(row.max_length_score || 0),
        mean_retention: Number(row.mean_retention_score || 0),
      })),
    [ranking]
  );

  const gainRows = useMemo(
    () =>
      Object.entries(gains?.per_length || {}).map(([length, row]) => ({
        length: `L=${length}`,
        joint_segment_vs_joint: Number(row.joint_segment_vs_joint || 0),
        ultra_segment_vs_ultra: Number(row.ultra_segment_vs_ultra || 0),
        best_segment_vs_single: Number(row.best_segment_vs_single || 0),
      })),
    [gains]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 systems 或 ranking 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`段级摘要状态 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(14,165,233,0.12), transparent 28%), radial-gradient(circle at top right, rgba(244,114,182,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(56,189,248,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>段级摘要状态</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            在超长程链上显式加入段级摘要变量 <code>s_t</code>，检查状态压缩能否帮助联合温度律重新抬高闭环分数，尤其是 <code>L=32</code> 末端表现。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <label
            style={{
              color: '#e2e8f0',
              fontSize: '11px',
              border: '1px solid rgba(148,163,184,0.35)',
              borderRadius: '999px',
              padding: '6px 10px',
              cursor: 'pointer',
            }}
          >
            导入 JSON
            <input type="file" accept="application/json" onChange={onUpload} style={{ display: 'none' }} />
          </label>
          <button
            type="button"
            onClick={resetAll}
            style={{
              color: '#e2e8f0',
              fontSize: '11px',
              border: '1px solid rgba(148,163,184,0.35)',
              borderRadius: '999px',
              padding: '6px 10px',
              background: 'transparent',
              cursor: 'pointer',
            }}
          >
            重置
          </button>
        </div>
      </div>

      <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px' }}>{`数据源: ${source}`}</div>
      {error && <div style={{ marginTop: '8px', color: '#fca5a5', fontSize: '11px' }}>{error}</div>}

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最佳平均摘要系统</div>
          <div style={{ color: '#38bdf8', fontSize: '16px', fontWeight: 'bold' }}>{labelOfSystem(best?.best_mean_segment_system?.system)}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`平均闭环 ${fmt(best?.best_mean_segment_system?.mean_closure_score)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最佳末端摘要系统</div>
          <div style={{ color: '#f472b6', fontSize: '16px', fontWeight: 'bold' }}>{labelOfSystem(best?.best_max_segment_system?.system)}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`L=32 ${fmt(best?.best_max_segment_system?.max_length_score)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>联合温度 + 段级摘要</div>
          <div style={{ color: '#22c55e', fontSize: '16px', fontWeight: 'bold' }}>{fmt(gains?.joint_segment_mean_vs_joint)}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`平均增益，相对无摘要版本`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>超长程强化 + 段级摘要</div>
          <div style={{ color: '#f59e0b', fontSize: '16px', fontWeight: 'bold' }}>{fmt(gains?.ultra_segment_max_vs_ultra)}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`L=32 末端增益，相对无摘要版本`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>摘要系统相对单锚点</div>
          <div style={{ color: gains?.best_segment_max_vs_single >= 0 ? '#22c55e' : '#f87171', fontSize: '16px', fontWeight: 'bold' }}>
            {fmt(gains?.best_segment_max_vs_single)}
          </div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`L=32 最佳摘要系统相对单锚点`}</div>
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 超长程闭环曲线</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={curveRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="single" name="单锚点" stroke={SYSTEM_META.single_anchor_beta_086.color} strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="joint" name="联合温度三锚点" stroke={SYSTEM_META.gated_triple_tau_joint_long_horizon.color} strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="joint_segment" name="联合温度 + 段级摘要" stroke={SYSTEM_META.gated_triple_tau_joint_long_horizon_segment_summary.color} strokeWidth={2.2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="ultra" name="超长程强化三锚点" stroke={SYSTEM_META.gated_triple_tau_joint_ultra_oracle.color} strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="ultra_segment" name="超长程强化 + 段级摘要" stroke={SYSTEM_META.gated_triple_tau_joint_ultra_oracle_segment_summary.color} strokeWidth={2.2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 系统总分对比</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={rankingRows} margin={{ top: 10, right: 12, left: 0, bottom: 40 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="system" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-16} textAnchor="end" height={66} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="mean_closure" name="平均闭环" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="max_length" name="L=32 闭环" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="mean_retention" name="平均保留" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 段级摘要带来的增益与缺口</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={gainRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="joint_segment_vs_joint" name="联合温度摘要增益" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="ultra_segment_vs_ultra" name="超长程强化摘要增益" fill="#f472b6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="best_segment_vs_single" name="最佳摘要系统相对单锚点" fill="#f87171" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
