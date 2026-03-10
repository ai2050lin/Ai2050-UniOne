import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/brain_d_real_cocalibrated_two_layer_unified_law_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.ranking_layer && payload.brain_d_real_cocalibrated_two_layer_law && payload.leave_one_out);
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function MetricCard({ label, value }) {
  return (
    <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
      <div style={{ color: '#94a3b8', fontSize: '10px' }}>{label}</div>
      <div style={{ color: '#f8fafc', fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>{value}</div>
    </div>
  );
}

export default function BrainDRealCocalibratedTwoLayerLawDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const metricRows = useMemo(
    () => [
      { item: '联合平均误差', value: Number(payload?.brain_d_real_cocalibrated_two_layer_law?.mean_absolute_gap || 0) },
      { item: '脑侧平均误差', value: Number(payload?.brain_d_real_cocalibrated_two_layer_law?.brain_mean_gap || 0) },
      { item: 'D 平均误差', value: Number(payload?.brain_d_real_cocalibrated_two_layer_law?.d_mean_gap || 0) },
      { item: '真实任务平均误差', value: Number(payload?.brain_d_real_cocalibrated_two_layer_law?.real_task_mean_gap || 0) },
      { item: '留一法平均误差', value: Number(payload?.brain_d_real_cocalibrated_two_layer_law?.held_out_mean_gap || 0) },
    ],
    [payload]
  );

  const brainRows = useMemo(
    () =>
      (payload?.brain_breakdown || []).map((row) => ({
        model_name: row.model_name,
        ranking_pressure: Number(row.ranking_pressure || 0),
        calibration_pressure: Number(row.calibration_pressure || 0),
        reference_score: Number(row.overall_bridge_score || 0),
        calibrated_score: Number(row.calibrated_score || 0),
      })),
    [payload]
  );

  const holdoutRows = useMemo(
    () =>
      (payload?.leave_one_out?.rows || []).map((row) => ({
        row_id: row.row_id,
        held_out_gap: Number(row.held_out_gap || 0),
      })),
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 ranking_layer、brain_d_real_cocalibrated_two_layer_law 或 leave_one_out 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`脑侧 + D + 真实任务共标定双层统一律 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(14,165,233,0.14), transparent 28%), radial-gradient(circle at top right, rgba(168,85,247,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(14,165,233,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>脑侧 + D + 真实任务共标定双层统一律</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把脑侧候选约束、D 闭环和真实任务收益放进同一套排序层与标定层，检查三域是否开始共享更小的统一结构。
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
        <MetricCard label="联合平均误差" value={fmt(payload?.brain_d_real_cocalibrated_two_layer_law?.mean_absolute_gap)} />
        <MetricCard label="联合相关系数" value={fmt(payload?.brain_d_real_cocalibrated_two_layer_law?.score_correlation)} />
        <MetricCard label="脑侧平均误差" value={fmt(payload?.brain_d_real_cocalibrated_two_layer_law?.brain_mean_gap)} />
        <MetricCard label="D 平均误差" value={fmt(payload?.brain_d_real_cocalibrated_two_layer_law?.d_mean_gap)} />
        <MetricCard label="真实任务平均误差" value={fmt(payload?.brain_d_real_cocalibrated_two_layer_law?.real_task_mean_gap)} />
        <MetricCard label="留一法平均误差" value={fmt(payload?.brain_d_real_cocalibrated_two_layer_law?.held_out_mean_gap)} />
        <MetricCard label="留一法相关系数" value={fmt(payload?.brain_d_real_cocalibrated_two_layer_law?.held_out_score_correlation)} />
        <MetricCard label="脑侧留一法误差" value={fmt(payload?.brain_d_real_cocalibrated_two_layer_law?.brain_held_out_gap)} />
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三域误差拆分</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={metricRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="value" name="误差" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 脑侧排序压力与标定压力</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={brainRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model_name" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="ranking_pressure" name="排序压力" fill="#a855f7" radius={[4, 4, 0, 0]} />
              <Bar dataKey="calibration_pressure" name="标定压力" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 脑侧参考分数与统一律分数</div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={brainRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model_name" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Line type="monotone" dataKey="reference_score" name="脑桥接参考分数" stroke="#38bdf8" strokeWidth={2} />
              <Line type="monotone" dataKey="calibrated_score" name="统一律分数" stroke="#f59e0b" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 留一法误差</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={holdoutRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="row_id" tick={{ fill: '#cbd5e1', fontSize: 10 }} angle={-18} textAnchor="end" height={58} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="held_out_gap" name="留一法误差" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>5. 当前判断</div>
        <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
          这一步检查的是：脑侧候选结构能否与 D 闭环、真实任务收益一起落到同一套双层统一律上。当前结果说明三域已经开始共标定，
          但脑侧域仍然是最弱的一环，说明脑侧约束现在更适合作为“外部限制条件”，还不是最强监督源。
        </div>
        <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
          下一步问题：{payload?.project_readout?.next_question || '-'}
        </div>
      </div>
    </div>
  );
}
