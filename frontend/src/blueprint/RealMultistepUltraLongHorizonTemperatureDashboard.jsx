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
import samplePayload from './data/real_multistep_memory_ultra_long_horizon_temperature_scan_sample.json';

const POLICY_META = {
  none: {
    label: '单锚点基线',
    desc: '单一慢记忆锚点，不使用温度门控。',
  },
  fixed_100: {
    label: '固定 tau=1.0',
    desc: '超长程区间的静态门控基线。',
  },
  joint_long_horizon: {
    label: '联合长程温度律',
    desc: '长度、阶段、剩余步数联合调度。',
  },
  joint_ultra_oracle: {
    label: '超长程强化调度',
    desc: '在更长保持链上进一步强化 route 软化与 tail 硬化。',
  },
  joint_ultra_tail_focus: {
    label: '超长尾段聚焦',
    desc: '把更多温度预算压到 final 收束阶段。',
  },
};

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && Array.isArray(payload.ranking));
}

function fmt(v, digits = 4) {
  return Number(v || 0).toFixed(digits);
}

function pct(v) {
  return `${(Number(v || 0) * 100).toFixed(1)}%`;
}

function labelOfPolicy(policy) {
  return POLICY_META[policy]?.label || policy || '-';
}

export default function RealMultistepUltraLongHorizonTemperatureDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const systems = payload?.systems || {};
  const ranking = Array.isArray(payload?.ranking) ? payload.ranking : [];
  const dynamicRows = Array.isArray(payload?.dynamic_rows) ? payload.dynamic_rows : [];
  const best = payload?.best || {};
  const gains = payload?.gains || {};
  const hypotheses = payload?.hypotheses || {};

  const curveRows = useMemo(() => {
    const single = systems?.single_anchor_beta_086?.global_summary || {};
    const fixed = systems?.gated_triple_tau_100?.global_summary || {};
    const longJoint = systems?.gated_triple_tau_joint_long_horizon?.global_summary || {};
    const ultra = systems?.gated_triple_tau_joint_ultra_oracle?.global_summary || {};
    const lengths = Array.isArray(single?.lengths) ? single.lengths : [];
    return lengths.map((length, idx) => ({
      length: `L=${length}`,
      single: Number(single?.closure_curve?.[idx] || 0),
      fixed: Number(fixed?.closure_curve?.[idx] || 0),
      long_joint: Number(longJoint?.closure_curve?.[idx] || 0),
      ultra: Number(ultra?.closure_curve?.[idx] || 0),
    }));
  }, [systems]);

  const rankingRows = useMemo(
    () =>
      ranking.map((row) => ({
        policy: labelOfPolicy(row.policy),
        mean_closure: Number(row.mean_closure_score || 0),
        max_length: Number(row.max_length_score || 0),
        mean_retention: Number(row.mean_retention_score || 0),
      })),
    [ranking]
  );

  const lengthBestRows = useMemo(
    () =>
      Object.entries(gains?.per_length_best_dynamic || {}).map(([length, row]) => ({
        length: `L=${length}`,
        best_score: Number(row.best_score || 0),
        gain_vs_fixed: Number(row.gain_vs_fixed_tau_100 || 0),
        gain_vs_single: Number(row.gain_vs_single_anchor || 0),
      })),
    [gains]
  );

  const dynamicShapeRows = useMemo(
    () =>
      dynamicRows.map((row) => ({
        policy: labelOfPolicy(row.policy),
        entropy: Number(row.mean_gate_entropy || 0),
        peak: Number(row.mean_gate_peak || 0),
        drop: Number(row.closure_relative_drop || 0),
      })),
    [dynamicRows]
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
      setError(`超长程温度律 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(244,114,182,0.12), transparent 28%), radial-gradient(circle at top right, rgba(14,165,233,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(244,114,182,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>超长程温度律</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把保持链继续推到 `L=24/28/32`，直接看联合温度律还能保留多少优势，以及优势在哪个区间开始失守。
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均最优动态策略</div>
          <div style={{ color: '#22c55e', fontSize: '16px', fontWeight: 'bold' }}>{labelOfPolicy(best?.best_mean_dynamic?.policy)}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`平均闭环 ${fmt(best?.best_mean_dynamic?.mean_closure_score)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最长任务最优动态策略</div>
          <div style={{ color: '#60a5fa', fontSize: '16px', fontWeight: 'bold' }}>{labelOfPolicy(best?.best_max_dynamic?.policy)}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`L_max ${fmt(best?.best_max_dynamic?.max_length_score)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>相对固定 tau=1.0</div>
          <div style={{ color: '#f59e0b', fontSize: '16px', fontWeight: 'bold' }}>{fmt(gains?.best_dynamic_max_vs_fixed_tau_100)}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`平均 ${fmt(gains?.best_dynamic_mean_vs_fixed_tau_100)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>相对单锚点</div>
          <div style={{ color: '#f472b6', fontSize: '16px', fontWeight: 'bold' }}>{fmt(gains?.best_dynamic_max_vs_single_anchor)}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`平均 ${fmt(gains?.best_dynamic_mean_vs_single_anchor)}`}</div>
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
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 超长程退化曲线</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={curveRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="single" name="单锚点" stroke="#94a3b8" strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="fixed" name="固定 tau=1.0" stroke="#f59e0b" strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="long_joint" name="联合长程温度律" stroke="#22c55e" strokeWidth={2.1} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="ultra" name="超长程强化调度" stroke="#f472b6" strokeWidth={2.2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 策略总分对比</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={rankingRows} margin={{ top: 10, right: 12, left: 0, bottom: 40 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="policy" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-18} textAnchor="end" height={60} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="mean_closure" name="平均闭环" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="max_length" name="最长任务" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="mean_retention" name="平均保留" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 各长度最佳动态策略</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={lengthBestRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="best_score" name="最佳闭环分数" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="gain_vs_fixed" name="相对固定增益" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="gain_vs_single" name="相对单锚点增益" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 动态门控形状</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={dynamicShapeRows} margin={{ top: 10, right: 12, left: 0, bottom: 40 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="policy" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-18} textAnchor="end" height={60} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="entropy" name="gate entropy" fill="#a855f7" radius={[4, 4, 0, 0]} />
              <Bar dataKey="peak" name="gate peak" fill="#ef4444" radius={[4, 4, 0, 0]} />
              <Bar dataKey="drop" name="相对衰减" fill="#94a3b8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(320px, 0.9fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>5. 当前结论</div>
          <div style={{ color: '#e2e8f0', fontSize: '12px', lineHeight: '1.8' }}>
            到 `L=32` 这个区间，联合策略仍然可以
            <span style={{ color: '#38bdf8', fontWeight: 'bold' }}> 超过固定 tau=1.0 </span>
            ，但已经不能稳定超过单锚点。也就是说，长程门控优势还在，但系统已经进入新的退化区。
          </div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>6. 含义</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.8' }}>
            这说明温度律并没有失效，但在超长保持链上，光靠门控调度已经不够，还需要更强的状态压缩、层级记忆或显式阶段表示来继续压退化。
          </div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gap: '8px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px' }}>7. 策略说明</div>
        {ranking.map((row) => (
          <div key={row.system} style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
            <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}>
              {`${labelOfPolicy(row.policy)} | 平均闭环 ${fmt(row.mean_closure_score)} | L_max ${fmt(row.max_length_score)}`}
            </div>
            <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
              {POLICY_META[row.policy]?.desc || '无说明'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
