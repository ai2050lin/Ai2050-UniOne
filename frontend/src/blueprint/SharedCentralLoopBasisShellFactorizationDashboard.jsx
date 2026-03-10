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
import samplePayload from './data/shared_central_loop_basis_shell_factorization_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.factorized_basis_shells);
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

export default function SharedCentralLoopBasisShellFactorizationDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const rows = useMemo(
    () => [
      { shell: '原型位置壳', gap: Number(payload?.factorized_basis_shells?.prototype_position_shell?.mean_held_out_gap || 0) },
      { shell: '边界宽度壳', gap: Number(payload?.factorized_basis_shells?.prototype_boundary_width_shell?.mean_held_out_gap || 0) },
      { shell: '原型间距壳', gap: Number(payload?.factorized_basis_shells?.prototype_spacing_shell?.mean_held_out_gap || 0) },
    ],
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 factorized_basis_shells 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`共享基底壳细分实验 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(59,130,246,0.14), transparent 28%), radial-gradient(circle at top right, rgba(16,185,129,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(59,130,246,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>共享中央回路共享基底壳细分实验</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把当前最强的共享基底壳继续拆成原型位置壳、原型边界宽度壳、原型间距壳，定位多模态差异更像先重写原型中心、边界还是家族原型之间的距离。
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
        <MetricCard label="当前最优基底壳" value={payload?.winner || '-'} />
        <MetricCard label="H1 位置优于边界" value={yesNo(payload?.hypotheses?.H1_position_beats_boundary)} />
        <MetricCard label="H2 位置优于间距" value={yesNo(payload?.hypotheses?.H2_position_beats_spacing)} />
        <MetricCard label="H3 基底壳偏原型位置" value={yesNo(payload?.hypotheses?.H3_basis_shell_is_position_heavy)} />
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三种共享基底壳的留一法误差</div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={rows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="shell" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip
              formatter={(value) => [fmt(value), '留一法误差']}
              contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }}
            />
            <Bar dataKey="gap" name="留一法误差" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 当前判断</div>
        <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
          这一层的关键不是再加新模块，而是继续压缩差异真正落在哪里。
          如果原型位置壳最强，就说明多模态差异更像先重写共享家族原型的中心位置，然后才进一步展开成边界宽度和原型间距差异。
        </div>
        <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
          下一步问题：{payload?.project_readout?.next_question || '-'}
        </div>
      </div>
    </div>
  );
}
