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
import samplePayload from './data/shared_central_loop_output_shell_factorization_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.factorized_shells);
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

export default function SharedCentralLoopOutputShellFactorizationDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const rows = useMemo(
    () => [
      { shell: '校准壳', gap: Number(payload?.factorized_shells?.calibration_shell?.mean_held_out_gap || 0) },
      { shell: '协议壳', gap: Number(payload?.factorized_shells?.protocol_shell?.mean_held_out_gap || 0) },
      { shell: '任务读出壳', gap: Number(payload?.factorized_shells?.task_readout_shell?.mean_held_out_gap || 0) },
    ],
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 factorized_shells 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`输出壳细分实验 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(168,85,247,0.14), transparent 28%), radial-gradient(circle at top right, rgba(251,191,36,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(168,85,247,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>共享中央回路输出壳细分实验</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把当前最强的输出壳继续拆成校准壳、协议壳、任务读出壳，直接定位模态差异主要挂在哪种输出层。
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
        <MetricCard label="当前最优输出壳" value={payload?.winner || '-'} />
        <MetricCard label="H1 任务壳优于校准壳" value={yesNo(payload?.hypotheses?.H1_task_beats_calibration)} />
        <MetricCard label="H2 任务壳优于协议壳" value={yesNo(payload?.hypotheses?.H2_task_beats_protocol)} />
        <MetricCard label="H3 输出壳偏任务读出" value={yesNo(payload?.hypotheses?.H3_output_shell_is_readout_heavy)} />
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三种输出壳的留一法误差</div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={rows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="shell" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
            <Bar dataKey="gap" name="留一法误差" fill="#a855f7" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 当前判断</div>
        <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
          这一步的关键不是证明“统一回路已经成立”，而是继续收缩模态差异的位置。
          如果协议壳最强，就说明多模态差异更像落在关系/协议层的展开与重写，而不是浅层校准或最终任务读出。
        </div>
        <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
          下一步问题：{payload?.project_readout?.next_question || '-'}
        </div>
      </div>
    </div>
  );
}
