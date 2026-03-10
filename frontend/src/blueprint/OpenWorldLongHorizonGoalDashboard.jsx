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
import samplePayload from './data/open_world_long_horizon_goal_state_benchmark_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.best_config && payload.baseline_direct_action && payload.baseline_stateful_trust);
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

export default function OpenWorldLongHorizonGoalDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const bars = useMemo(() => {
    const direct = payload?.baseline_direct_action || {};
    const stateful = payload?.baseline_stateful_trust || {};
    const goal = payload?.best_config || {};
    return [
      {
        item: '阶段切换',
        direct: Number(direct.phase_switch_accuracy || 0),
        stateful: Number(stateful.phase_switch_accuracy || 0),
        goal: Number(goal.phase_switch_accuracy || 0),
      },
      {
        item: '旧概念保留',
        direct: Number(direct.old_concept_retention || 0),
        stateful: Number(stateful.old_concept_retention || 0),
        goal: Number(goal.old_concept_retention || 0),
      },
      {
        item: '长期闭环分数',
        direct: Number(direct.long_horizon_loop_score || 0),
        stateful: Number(stateful.long_horizon_loop_score || 0),
        goal: Number(goal.long_horizon_loop_score || 0),
      },
    ];
  }, [payload]);

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 best_config / baseline 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`长期多步目标 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(249,115,22,0.14), transparent 28%), radial-gradient(circle at top right, rgba(234,88,12,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(249,115,22,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>开放世界长期多步目标维持</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把旧概念保留目标扩成 repeated family switching 的长期目标链，直接比较不同系统在阶段切换、目标捕获和长期闭环分数上的差异。
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
        <MetricCard label="phase-switch gain vs direct" value={fmt(best.phase_switch_gain_vs_direct)} />
        <MetricCard label="retention gain vs direct" value={fmt(best.retention_gain_vs_direct)} />
        <MetricCard label="最佳 phase / reserve / replay" value={`${best.phase_span ?? '-'} / ${best.reserve_target ?? '-'} / ${best.replay_count ?? '-'} / ${best.replay_mode ?? '-'}`} />
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
            <Bar dataKey="stateful" name="stateful_trust" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            <Bar dataKey="goal" name="goal_state_replay" fill="#f97316" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            这一轮不再只看旧概念有没有保住，而是直接看 repeated switching 下目标能否维持。真正关键的指标是阶段切换准确率、
            目标捕获率和长期闭环分数能否一起抬起来。
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 下一步</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            如果长期多步目标还能保持正增益，下一步就该把 target family 进一步扩成真正的阶段性子目标程序，
            并显式加入失败回退和恢复逻辑。
          </div>
        </div>
      </div>
    </div>
  );
}
