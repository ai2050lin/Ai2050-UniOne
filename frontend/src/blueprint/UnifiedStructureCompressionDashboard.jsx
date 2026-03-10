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
import samplePayload from './data/unified_structure_four_factor_compression_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.mapping && Array.isArray(payload.views) && payload.retention);
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

export default function UnifiedStructureCompressionDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const factorRows = useMemo(
    () =>
      Object.entries(payload?.factor_summary?.means || {}).map(([factor, score]) => ({
        factor,
        score: Number(score || 0),
      })),
    [payload]
  );

  const viewRows = useMemo(
    () =>
      (payload?.views || []).map((row) => ({
        view: row.view_id,
        reference_score: Number(row.reference_score || 0),
        compressed_score: Number(row.compressed_score || 0),
      })),
    [payload]
  );

  const mappingRows = useMemo(
    () =>
      Object.entries(payload?.mapping || {}).map(([factor, sources]) => ({
        factor,
        sources: Array.isArray(sources) ? sources.join(' / ') : String(sources || ''),
      })),
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) throw new Error('缺少 meta / mapping / views / retention 字段');
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`统一结构压缩 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(16,185,129,0.10), transparent 28%), radial-gradient(circle at top right, rgba(56,189,248,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(16,185,129,0.22)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>统一结构压缩</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把 `共享基底 / 个体偏移 / 关系协议 / 门控 / 拓扑 / 整合`
            压成 `base / adaptive_offset / routing / stabilization`，
            直接观察压缩后还保留了多少桥接解释力。
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
        <MetricCard label="压缩是否通过" value={payload?.retention?.compression_pass ? '通过' : '未通过'} />
        <MetricCard label="参考均值" value={fmt(payload?.retention?.reference_mean)} />
        <MetricCard label="压缩均值" value={fmt(payload?.retention?.compressed_mean)} />
        <MetricCard label="平均绝对误差" value={fmt(payload?.retention?.mean_absolute_gap)} />
        <MetricCard label="分数相关系数" value={fmt(payload?.retention?.score_correlation)} />
        <MetricCard label="最强因子" value={payload?.factor_summary?.strongest_factor || '-'} />
        <MetricCard label="最弱因子" value={payload?.factor_summary?.weakest_factor || '-'} />
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 四因子平均分</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={factorRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="factor" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="score" name="score" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 参考分数 vs 压缩分数</div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={viewRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="view" tick={{ fill: '#cbd5e1', fontSize: 10 }} angle={-18} textAnchor="end" height={58} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Line type="monotone" dataKey="reference_score" name="reference_score" stroke="#38bdf8" strokeWidth={2} />
              <Line type="monotone" dataKey="compressed_score" name="compressed_score" stroke="#f59e0b" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(280px, 0.9fr) minmax(0, 1.1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 六对象到四因子映射</div>
          <div style={{ display: 'grid', gap: '8px' }}>
            {mappingRows.map((row) => (
              <div key={row.factor} style={{ border: '1px solid rgba(148,163,184,0.16)', borderRadius: '10px', padding: '8px 10px' }}>
                <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}>{row.factor}</div>
                <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>{row.sources}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>{payload?.project_readout?.current_verdict || '-'}</div>
          <div style={{ marginTop: '10px', color: '#dbeafe', fontSize: '12px' }}>为什么先压成四因子</div>
          <div style={{ marginTop: '8px', display: 'grid', gap: '8px' }}>
            {(payload?.project_readout?.why_compress || []).map((item) => (
              <div key={item} style={{ border: '1px solid rgba(148,163,184,0.16)', borderRadius: '10px', padding: '8px 10px', color: '#cbd5e1', fontSize: '11px', lineHeight: '1.7' }}>
                {item}
              </div>
            ))}
          </div>
          <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
            下一步问题：{payload?.project_readout?.next_question || '-'}
          </div>
        </div>
      </div>
    </div>
  );
}
