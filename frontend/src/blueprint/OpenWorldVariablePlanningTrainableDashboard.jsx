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
import samplePayload from './data/open_world_variable_planning_trainable_benchmark_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && payload.gains && payload.headline_metrics);
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

export default function OpenWorldVariablePlanningTrainableDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const bars = useMemo(() => {
    const noRollback = payload?.systems?.stateful_no_rollback || {};
    const rollback = payload?.systems?.stateful_with_rollback || {};
    const trainable = payload?.systems?.trainable_planner || {};
    return [
      {
        item: 'episode_success',
        noRollback: Number(noRollback.episode_success_rate || 0),
        rollback: Number(rollback.episode_success_rate || 0),
        trainable: Number(trainable.episode_success_rate || 0),
      },
      {
        item: 'rollback_recovery',
        noRollback: Number(noRollback.rollback_recovery_rate || 0),
        rollback: Number(rollback.rollback_recovery_rate || 0),
        trainable: Number(trainable.rollback_recovery_rate || 0),
      },
      {
        item: 'contamination',
        noRollback: Number(noRollback.contamination_control || 0),
        rollback: Number(rollback.contamination_control || 0),
        trainable: Number(trainable.contamination_control || 0),
      },
      {
        item: 'stability',
        noRollback: Number(noRollback.open_environment_stability || 0),
        rollback: Number(rollback.open_environment_stability || 0),
        trainable: Number(trainable.open_environment_stability || 0),
      },
      {
        item: 'planning_score',
        noRollback: Number(noRollback.variable_planning_score || 0),
        rollback: Number(rollback.variable_planning_score || 0),
        trainable: Number(trainable.variable_planning_score || 0),
      },
    ];
  }, [payload]);

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 systems / gains / headline_metrics 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`可变长度规划链 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setPayload(samplePayload);
    setSource('内置样例');
    setError('');
  }

  const trainable = payload?.systems?.trainable_planner || {};

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(8,145,178,0.16), transparent 28%), radial-gradient(circle at top right, rgba(16,185,129,0.12), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(8,145,178,0.26)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>二点四十三、可变长度规划链与可学习闭环</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把固定 family 程序推进成可变长度规划链，并把失败回退、混合回放律、长期目标状态和动作策略并进同一可学习闭环。
            这一版优先外部化四个指标：episode 成功率、回退恢复率、旧概念污染控制和开放环境稳定性。
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
        <MetricCard label="trainable episode success" value={fmt(payload?.headline_metrics?.episode_success_rate)} />
        <MetricCard label="trainable rollback recovery" value={fmt(payload?.headline_metrics?.rollback_recovery_rate)} />
        <MetricCard label="trainable contamination control" value={fmt(payload?.headline_metrics?.contamination_control)} />
        <MetricCard label="trainable open-env stability" value={fmt(payload?.headline_metrics?.open_environment_stability)} />
        <MetricCard label="planning gain vs static rollback" value={fmt(payload?.gains?.trainable_variable_planning_gain_vs_static)} />
        <MetricCard label="learned reserve / replay / rollback" value={`${fmt(trainable.final_reserve_target, 3)} / ${fmt(trainable.final_replay_scale, 3)} / ${fmt(trainable.final_rollback_threshold, 3)}`} />
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三种系统对比</div>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={bars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip
              formatter={(value) => [fmt(value), 'score']}
              contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }}
            />
            <Legend />
            <Bar dataKey="noRollback" name="stateful_no_rollback" fill="#64748b" radius={[4, 4, 0, 0]} />
            <Bar dataKey="rollback" name="stateful_with_rollback" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
            <Bar dataKey="trainable" name="trainable_planner" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            固定规划程序已经不够用了。现在要看的不是单步能不能识别，而是失败后能不能回退、恢复，并在长链上继续维持目标。
            从这轮结果看，单纯加静态回退会抬高恢复率，但还不够；把回放律、目标状态和动作偏置做成可学习控制后，规划闭环分数继续上升。
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 下一步</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            下一步不是继续堆更多静态回放，而是把规划链、回退协议和长期状态再压成更统一的结构，然后把这个结构接回真实任务接口、
            脑侧候选约束和更严格的开放环境扰动。
          </div>
          <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
            可视化上建议继续增加“回退触发时间线”“恢复前后 trust / reserve / replay 参数曲线”和“旧概念被当前目标劫持的污染热图”。
          </div>
        </div>
      </div>
    </div>
  );
}
