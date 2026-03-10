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
import samplePayload from './data/shared_central_loop_modality_hypothesis_sample.json';

function isValidPayload(payload) {
  return Boolean(
    payload &&
      payload.meta &&
      payload.fully_shared_law &&
      payload.parameterized_shared_law &&
      payload.shared_central_loop_law &&
      payload.modality_separate_oracle
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

export default function SharedCentralLoopModalityDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const compareRows = useMemo(
    () => [
      { law: '完全共享', held_out_gap: Number(payload?.fully_shared_law?.mean_held_out_gap || 0) },
      { law: '参数化共享', held_out_gap: Number(payload?.parameterized_shared_law?.mean_held_out_gap || 0) },
      { law: '共享中央回路', held_out_gap: Number(payload?.shared_central_loop_law?.mean_held_out_gap || 0) },
      { law: '模态独立上限', held_out_gap: Number(payload?.modality_separate_oracle?.mean_held_out_gap || 0) },
    ],
    [payload]
  );

  const modalityRows = useMemo(
    () =>
      (payload?.meta?.modalities || []).map((modality) => ({
        modality,
        fully_shared: Number(payload?.fully_shared_law?.modality_held_out_gap?.[modality] || 0),
        parameterized: Number(payload?.parameterized_shared_law?.modality_held_out_gap?.[modality] || 0),
        central_loop: Number(payload?.shared_central_loop_law?.modality_held_out_gap?.[modality] || 0),
      })),
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 fully_shared_law / parameterized_shared_law / shared_central_loop_law 等关键字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`共享中央回路实验 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(14,165,233,0.14), transparent 28%), radial-gradient(circle at top right, rgba(34,197,94,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(14,165,233,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>共享中央回路多模态实验</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            用“模态专属投影 + 共享中央回路 + 共享读出”的低秩写法，测试统一回路是否能处理视觉、触觉、语言等不同模态。
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
        <MetricCard label="中央回路 rank" value={fmt(payload?.shared_central_loop_law?.loop_rank, 0)} />
        <MetricCard label="中央回路留一法误差" value={fmt(payload?.shared_central_loop_law?.mean_held_out_gap)} />
        <MetricCard label="H1 优于完全共享" value={yesNo(payload?.hypotheses?.H1_central_loop_beats_fully_shared)} />
        <MetricCard label="H2 逼近参数化共享" value={yesNo(payload?.hypotheses?.H2_central_loop_close_to_parameterized)} />
        <MetricCard label="H3 统一回路假设" value={yesNo(payload?.hypotheses?.H3_shared_central_loop_supported)} />
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 四种写法留一法误差</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={compareRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="law" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="held_out_gap" name="留一法误差" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 各模态误差分布</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={modalityRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="modality" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="fully_shared" name="完全共享" fill="#94a3b8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="parameterized" name="参数化共享" fill="#10b981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="central_loop" name="共享中央回路" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 当前判断</div>
        <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
          如果共享中央回路写法稳定优于完全共享，同时逼近参数化共享，就支持“不同模态先进入参数化投影区，再交给统一回路处理”的猜想。
          当前最关键的问题不是模态是否完全独立，而是模态差异主要落在输入投影层、中央回路内部，还是最终读出标定层。
        </div>
        <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
          下一步问题：{payload?.project_readout?.next_question || '-'}
        </div>
      </div>
    </div>
  );
}
