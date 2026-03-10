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
import samplePayload from './data/shared_central_loop_shell_hypothesis_sample.json';

function isValidPayload(payload) {
  return Boolean(
    payload &&
      payload.meta &&
      payload.fully_shared_law &&
      payload.parameterized_shared_law &&
      payload.shared_central_loop_law &&
      payload.shared_central_loop_shell_law
  );
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

export default function SharedCentralLoopShellDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const compareRows = useMemo(
    () => [
      { law: '完全共享', gap: Number(payload?.fully_shared_law?.mean_held_out_gap || 0) },
      { law: '参数化共享', gap: Number(payload?.parameterized_shared_law?.mean_held_out_gap || 0) },
      { law: '纯中央回路', gap: Number(payload?.shared_central_loop_law?.mean_held_out_gap || 0) },
      { law: '中央回路 + 外壳', gap: Number(payload?.shared_central_loop_shell_law?.mean_held_out_gap || 0) },
    ],
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少关键字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`中央回路外壳实验 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(34,197,94,0.14), transparent 28%), radial-gradient(circle at top right, rgba(6,182,212,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(34,197,94,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>共享中央回路 + 模态外壳实验</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            测试更现实的写法：统一回路负责公共处理，模态差异主要留在小型输入/输出壳层，而不是要求中央回路单独吃下全部模态差异。
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
        <MetricCard label="外壳版本留一法误差" value={fmt(payload?.shared_central_loop_shell_law?.mean_held_out_gap)} />
        <MetricCard label="H1 优于纯中央回路" value={yesNo(payload?.hypotheses?.H1_shell_improves_central_loop)} />
        <MetricCard label="H2 优于完全共享" value={yesNo(payload?.hypotheses?.H2_shell_beats_fully_shared)} />
        <MetricCard label="H3 接近参数化共享" value={yesNo(payload?.hypotheses?.H3_shell_close_to_parameterized)} />
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 四种写法留一法误差</div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={compareRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="law" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
            <Bar dataKey="gap" name="留一法误差" fill="#22c55e" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 当前判断</div>
        <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
          如果“中央回路 + 外壳”明显优于“纯中央回路”，就说明统一回路假设不必要求所有模态差异都被中央回路本体吞掉。
          更合理的结构可能是：统一回路处理公共关系与整合，模态外壳负责局部投影和残差补偿。
        </div>
      </div>
    </div>
  );
}
