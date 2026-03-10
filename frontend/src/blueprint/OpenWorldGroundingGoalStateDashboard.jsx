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
import samplePayload from './data/open_world_grounding_action_loop_goal_state_scan_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.best_config && payload.baseline_direct_action && payload.baseline_stateful_best);
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

export default function OpenWorldGroundingGoalStateDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const bars = useMemo(() => {
    const direct = payload?.baseline_direct_action || {};
    const stateful = payload?.baseline_stateful_best || {};
    const goal = payload?.best_config || {};
    return [
      { item: '纠错后动作', direct: Number(direct.corrected_action_accuracy || 0), stateful: Number(stateful.corrected_action_accuracy || 0), goal: Number(goal.corrected_action_accuracy || 0) },
      { item: '旧概念保留', direct: Number(direct.old_concept_retention || 0), stateful: Number(stateful.old_concept_retention || 0), goal: Number(goal.old_concept_retention || 0) },
      { item: '动作闭环分数', direct: Number(direct.loop_score || 0), stateful: Number(stateful.loop_score || 0), goal: Number(goal.loop_score || 0) },
    ];
  }, [payload]);

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) throw new Error('缺少 best_config / baseline 字段');
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`开放世界 goal-state JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(34,197,94,0.14), transparent 28%), radial-gradient(circle at top right, rgba(16,185,129,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(34,197,94,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>开放世界长期目标 / 保留状态闭环</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            在最小动作回路之上加入旧概念保留目标和回放储备，测试长期状态能否把代理闭环从负边界拉回正区。
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
        <MetricCard label="loop gain vs direct" value={fmt(best.loop_score_gain_vs_direct)} />
        <MetricCard label="loop gain vs stateful" value={fmt(best.loop_score_gain_vs_stateful)} />
        <MetricCard label="retention gain vs direct" value={fmt(best.retention_gain_vs_direct)} />
        <MetricCard label="最佳 reserve / replay" value={`${best.reserve_target ?? '-'} / ${best.replay_count ?? '-'}`} />
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三阶段对比</div>
        <ResponsiveContainer width="100%" height={290}>
          <BarChart data={bars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip
              formatter={(value) => [fmt(value), 'score']}
              contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }}
            />
            <Legend />
            <Bar dataKey="direct" name="direct_action" fill="#64748b" radius={[4, 4, 0, 0]} />
            <Bar dataKey="stateful" name="stateful_trust" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            <Bar dataKey="goal" name="goal_state_replay" fill="#22c55e" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            这一轮说明代理闭环真正缺的不是单独一层动作信任，而是长期保留状态。只要旧概念保留目标和回放储备并入，闭环分数就能相对 direct 翻正。
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 下一步</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            下一步不能停在“旧概念保留”本身，而是要把真正的长期多步目标状态也并进同一控制面，
            直接看开放环境里更长时程的目标维持和自纠错是否还能稳定成立。
          </div>
        </div>
      </div>
    </div>
  );
}
