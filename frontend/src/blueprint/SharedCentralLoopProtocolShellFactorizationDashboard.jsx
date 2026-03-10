import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/shared_central_loop_protocol_shell_factorization_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.factorized_protocol_shells);
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function yesNo(flag) {
  return flag ? '成立' : '未成立';
}

function MetricCard({ label, value }) {
  return (
    <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
      <div style={{ color: '#94a3b8', fontSize: '10px' }}>{label}</div>
      <div style={{ color: '#f8fafc', fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>{value}</div>
    </div>
  );
}

export default function SharedCentralLoopProtocolShellFactorizationDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const rows = useMemo(
    () => [
      { shell: 'family 协议壳', gap: Number(payload?.factorized_protocol_shells?.family_protocol_shell?.mean_held_out_gap || 0) },
      { shell: 'relation 协议壳', gap: Number(payload?.factorized_protocol_shells?.relation_protocol_shell?.mean_held_out_gap || 0) },
      { shell: 'action/planning 协议壳', gap: Number(payload?.factorized_protocol_shells?.action_planning_protocol_shell?.mean_held_out_gap || 0) },
    ],
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 factorized_protocol_shells 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`协议壳细分实验 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(244,114,182,0.14), transparent 28%), radial-gradient(circle at top right, rgba(56,189,248,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(244,114,182,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>共享中央回路协议壳细分实验</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把当前最强的协议型输出壳继续拆成 family、relation、action/planning 三类协议壳，定位多模态差异主要重写哪种协议。
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
        <MetricCard label="当前最优协议壳" value={payload?.winner || '-'} />
        <MetricCard label="H1 relation 优于 family" value={yesNo(payload?.hypotheses?.H1_relation_beats_family)} />
        <MetricCard label="H2 relation 优于 action" value={yesNo(payload?.hypotheses?.H2_relation_beats_action)} />
        <MetricCard label="H3 协议壳偏 relation" value={yesNo(payload?.hypotheses?.H3_protocol_shell_is_relation_heavy)} />
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三种协议壳的留一法误差</div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={rows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="shell" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
            <Bar dataKey="gap" name="留一法误差" fill="#f472b6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 当前判断</div>
        <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
          这一步的关键不是继续堆模块，而是继续收缩差异的位置。
          如果 family 协议壳最强，就说明多模态差异更像先落在语义家族和概念族群的粗粒度协议重写，再往下游展开更细关系和任务结构。
        </div>
        <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
          下一步问题：{payload?.project_readout?.next_question || '-'}
        </div>
      </div>
    </div>
  );
}
