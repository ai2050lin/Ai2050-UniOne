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
import samplePayload from './data/open_world_grounding_action_loop_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && payload.gains_vs_direct);
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

export default function OpenWorldGroundingActionLoopDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const bars = useMemo(
    () => [
      {
        item: '动作准确率',
        direct: Number(payload?.systems?.direct_action?.action_accuracy || 0),
        base: Number(payload?.systems?.shared_action_base?.action_accuracy || 0),
        tuned: Number(payload?.systems?.shared_action_tuned?.action_accuracy || 0),
      },
      {
        item: '纠错后准确率',
        direct: Number(payload?.systems?.direct_action?.corrected_action_accuracy || 0),
        base: Number(payload?.systems?.shared_action_base?.corrected_action_accuracy || 0),
        tuned: Number(payload?.systems?.shared_action_tuned?.corrected_action_accuracy || 0),
      },
      {
        item: '旧概念保留',
        direct: Number(payload?.systems?.direct_action?.old_concept_retention || 0),
        base: Number(payload?.systems?.shared_action_base?.old_concept_retention || 0),
        tuned: Number(payload?.systems?.shared_action_tuned?.old_concept_retention || 0),
      },
      {
        item: '动作闭环分数',
        direct: Number(payload?.systems?.direct_action?.loop_score || 0),
        base: Number(payload?.systems?.shared_action_base?.loop_score || 0),
        tuned: Number(payload?.systems?.shared_action_tuned?.loop_score || 0),
      },
    ],
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) throw new Error('缺少 systems / gains_vs_direct 字段');
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`开放世界动作回路 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(239,68,68,0.12), transparent 28%), radial-gradient(circle at top right, rgba(245,158,11,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(239,68,68,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>开放世界最小动作回路断点</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把连续流接地接上最小动作回路和自纠错环，直接看当前接地增益有没有真正传到代理闭环。这里的目标不是“挑好看分数”，
            而是明确断点究竟出在动作策略、纠错机制还是旧概念保留。
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
        <MetricCard label="base loop gain vs direct" value={fmt(payload?.gains_vs_direct?.base_loop_score_gain)} />
        <MetricCard label="tuned loop gain vs direct" value={fmt(payload?.gains_vs_direct?.tuned_loop_score_gain)} />
        <MetricCard label="tuned corrected action gain" value={fmt(payload?.gains_vs_direct?.tuned_corrected_action_gain)} />
        <MetricCard label="tuned old retention gain" value={fmt(payload?.gains_vs_direct?.tuned_old_concept_retention_gain)} />
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 动作回路四个核心指标</div>
        <ResponsiveContainer width="100%" height={300}>
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
            <Bar dataKey="base" name="shared_action_base" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            <Bar dataKey="tuned" name="shared_action_tuned" fill="#ef4444" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            这一步给出的结论很直接：感知层的接地增益不会自动传到动作层。即使连续流接地有改善，动作成功率、纠错能力和旧概念保留依然可能同时失真。
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 下一步</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            下一步不是只继续调接地更新率，而是把动作策略、长期状态和自纠错一起并入统一更新律，
            否则开放环境里的代理闭环仍然会在动作层断开。
          </div>
        </div>
      </div>
    </div>
  );
}
