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
import samplePayload from './data/real_multistep_memory_dynamic_temperature_scan_sample.json';

const POLICY_META = {
  none: {
    label: '单锚点基线',
    desc: '不使用动态温度，只保留单一慢记忆锚点。',
  },
  fixed_100: {
    label: '固定 tau=1.0',
    desc: '固定门控温度作为当前最强静态基线。',
  },
  length_adaptive: {
    label: '长度自适应',
    desc: '短任务更软，长任务回到 tau=1.0。',
  },
  phase_adaptive: {
    label: '阶段自适应',
    desc: '按 tool / route / final 三阶段分配不同温度。',
  },
  uncertainty_adaptive: {
    label: '不确定性自适应',
    desc: '按记忆离散度和隐藏态强度调温度。',
  },
  length_phase_adaptive: {
    label: '长度+阶段自适应',
    desc: '同时考虑任务长度和当前阶段。',
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

export default function RealMultistepDynamicTemperatureDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const ranking = Array.isArray(payload?.ranking) ? payload.ranking : [];
  const dynamicRows = Array.isArray(payload?.dynamic_rows) ? payload.dynamic_rows : [];
  const best = payload?.best || {};
  const gains = payload?.gains || {};
  const hypotheses = payload?.hypotheses || {};

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

  const dynamicGateRows = useMemo(
    () =>
      dynamicRows.map((row) => ({
        policy: labelOfPolicy(row.policy),
        gate_entropy: Number(row.mean_gate_entropy || 0),
        gate_peak: Number(row.mean_gate_peak || 0),
        relative_drop: Number(row.closure_relative_drop || 0),
      })),
    [dynamicRows]
  );

  const lengthRows = useMemo(
    () =>
      Object.entries(gains?.per_length_best_dynamic || {}).map(([length, row]) => ({
        length: `L=${length}`,
        gain_vs_fixed: Number(row.gain_vs_fixed_tau_100 || 0),
        best_score: Number(row.best_score || 0),
        policy: labelOfPolicy(row.best_policy),
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
      setError(`动态温度扫描 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(16,185,129,0.14), transparent 28%), radial-gradient(circle at top right, rgba(56,189,248,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(16,185,129,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>动态门控温度策略</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            对比固定 `tau=1.0` 和几种动态温度律，直接看哪种策略真的提升平均闭环、保留率和长程稳定性。
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最佳动态策略</div>
          <div style={{ color: '#22c55e', fontSize: '16px', fontWeight: 'bold' }}>{labelOfPolicy(best?.best_mean_dynamic?.policy)}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`平均闭环 ${fmt(best?.best_mean_dynamic?.mean_closure_score)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>相对固定 tau=1.0</div>
          <div style={{ color: '#60a5fa', fontSize: '16px', fontWeight: 'bold' }}>{fmt(gains?.best_dynamic_mean_vs_fixed_tau_100)}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`最长任务 ${fmt(gains?.best_dynamic_max_vs_fixed_tau_100)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>相对单锚点</div>
          <div style={{ color: '#f59e0b', fontSize: '16px', fontWeight: 'bold' }}>{fmt(gains?.best_dynamic_mean_vs_single_anchor)}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`最佳动态保留 ${fmt(best?.best_retention_dynamic?.mean_retention_score)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>结论方向</div>
          <div style={{ color: '#10b981', fontSize: '16px', fontWeight: 'bold' }}>
            {hypotheses?.H1_some_dynamic_beats_fixed_tau_100_on_average ? '动态温度有效' : '动态温度未胜出'}
          </div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>
            {hypotheses?.H2_some_dynamic_beats_fixed_tau_100_at_max_length ? '最长任务也提升' : '最长任务暂未继续抬高'}
          </div>
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
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 策略闭环对比</div>
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

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 动态策略的门控形状</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={dynamicGateRows} margin={{ top: 10, right: 12, left: 0, bottom: 40 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="policy" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-18} textAnchor="end" height={60} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="gate_entropy" name="gate entropy" fill="#a855f7" radius={[4, 4, 0, 0]} />
              <Bar dataKey="gate_peak" name="gate peak" fill="#ef4444" radius={[4, 4, 0, 0]} />
              <Bar dataKey="relative_drop" name="相对衰减" fill="#94a3b8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(320px, 0.95fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 各长度最优动态策略</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={lengthRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="best_score" name="最佳闭环分数" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="gain_vs_fixed" name="相对固定 tau=1.0 增益" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px', display: 'grid', gap: '10px' }}>
          <div>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 当前结论</div>
            <div style={{ color: '#e2e8f0', fontSize: '12px', lineHeight: '1.8' }}>
              当前最强动态策略是
              <span style={{ color: '#22c55e', fontWeight: 'bold' }}> 长度自适应温度 </span>
              。它把平均闭环从固定 `tau=1.0` 的 `0.5091` 提到 `0.5195`，同时把平均保留从 `0.3507` 提到 `0.3607`，但没有继续抬高最长任务末端分数。
            </div>
          </div>
          <div>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>5. 含义</div>
            <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.8' }}>
              这说明动态温度已经开始有效，但当前更像“把不同长度任务分开处理”带来的收益，还不是更强的上下文决策律。也就是 `tau_g(L)` 已经值得继续，而 `tau_g(h_t, uncertainty_t)` 还没跑赢固定基线。
            </div>
          </div>
          <div>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>6. 下一步</div>
            <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.8' }}>
              下一步应该把长度信号和任务阶段信号联动到同一门控器里，并且显式输入任务剩余步数、当前阶段和记忆竞争强度，而不是只靠局部隐藏态近似。
            </div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gap: '8px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px' }}>7. 策略说明</div>
        {ranking.map((row) => (
          <div key={row.system} style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
            <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}>
              {`${labelOfPolicy(row.policy)} | 平均闭环 ${fmt(row.mean_closure_score)} | 最长任务 ${fmt(row.max_length_score)}`}
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
