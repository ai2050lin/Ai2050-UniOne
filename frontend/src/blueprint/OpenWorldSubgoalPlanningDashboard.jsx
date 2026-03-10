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
import samplePayload from './data/open_world_subgoal_planning_benchmark_sample.json';

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

export default function OpenWorldSubgoalPlanningDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const bars = useMemo(() => {
    const direct = payload?.baseline_direct_action || {};
    const stateful = payload?.baseline_stateful_trust || {};
    const best = payload?.best_config || {};
    return [
      {
        item: 'episode 成功率',
        direct: Number(direct.episode_success_rate || 0),
        stateful: Number(stateful.episode_success_rate || 0),
        best: Number(best.episode_success_rate || 0),
      },
      {
        item: '阶段过渡',
        direct: Number(direct.transition_accuracy || 0),
        stateful: Number(stateful.transition_accuracy || 0),
        best: Number(best.transition_accuracy || 0),
      },
      {
        item: '规划闭环分数',
        direct: Number(direct.planning_loop_score || 0),
        stateful: Number(stateful.planning_loop_score || 0),
        best: Number(best.planning_loop_score || 0),
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
      setError(`子目标规划 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(37,99,235,0.14), transparent 28%), radial-gradient(circle at top right, rgba(14,165,233,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(37,99,235,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>开放世界阶段性子目标程序</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把长期目标继续推进成多阶段子目标程序，直接比较 episode 成功率、阶段过渡和规划闭环分数，判断系统是不是开始进入真正的规划链。
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
        <MetricCard label="planning gain vs direct" value={fmt(best.planning_gain_vs_direct)} />
        <MetricCard label="episode success gain" value={fmt(best.episode_success_gain_vs_direct)} />
        <MetricCard label="transition gain" value={fmt(best.transition_gain_vs_direct)} />
        <MetricCard label="最佳 span / len / replay" value={`${best.subgoal_span ?? '-'} / ${best.program_len ?? '-'} / ${best.replay_mode ?? '-'}`} />
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三系统对比</div>
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
            <Bar dataKey="best" name="subgoal_planning" fill="#2563eb" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            这一步开始真正带出“规划味道”。关键已经不只是切换后还能不能识别，
            而是整段子目标程序能不能稳定完成，并在阶段切换时保持闭环不掉线。
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 下一步</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            如果阶段性子目标程序仍能保持正增益，下一步就该把固定 family 程序推进成可变长度、带失败回退的规划链，
            并把恢复率、污染控制和开放环境稳定性单独拉出来看。
          </div>
        </div>
      </div>
    </div>
  );
}
