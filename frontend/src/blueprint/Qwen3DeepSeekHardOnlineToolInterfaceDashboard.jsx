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
import samplePayload from './data/qwen3_deepseek7b_hard_online_tool_interface_sample.json';

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function isValidPayload(payload) {
  return Boolean(payload && payload.models && payload.headline_metrics && payload.hypotheses);
}

const MODEL_LABELS = {
  qwen3_4b: 'Qwen3-4B',
  deepseek_7b: 'DeepSeek-7B',
};

export default function Qwen3DeepSeekHardOnlineToolInterfaceDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const summaryRows = useMemo(() => {
    return Object.entries(payload?.models || {}).map(([key, row]) => ({
      model: MODEL_LABELS[key] || key,
      tool头成功率: Number(row?.tool_stage_head_online_tool_interface?.success_rate || 0),
      联合头成功率: Number(row?.relation_tool_joint_head_online_tool_interface?.success_rate || 0),
      tool头触发率: Number(row?.tool_stage_head_online_tool_interface?.rollback_trigger_rate || 0),
      联合头触发率: Number(row?.relation_tool_joint_head_online_tool_interface?.rollback_trigger_rate || 0),
    }));
  }, [payload]);

  const failureRows = useMemo(() => {
    return Object.entries(payload?.models || {}).map(([key, row]) => ({
      model: MODEL_LABELS[key] || key,
      tool头工具失败率: Number(row?.tool_stage_head_online_tool_interface?.tool_failure_rate || 0),
      联合头工具失败率: Number(row?.relation_tool_joint_head_online_tool_interface?.tool_failure_rate || 0),
      tool头恢复率: Number(row?.tool_stage_head_online_tool_interface?.rollback_recovery_rate || 0),
      联合头恢复率: Number(row?.relation_tool_joint_head_online_tool_interface?.rollback_recovery_rate || 0),
    }));
  }, [payload]);

  const breakdownRows = useMemo(() => {
    const rows = [];
    for (const [key, row] of Object.entries(payload?.models || {})) {
      const breakdown = row?.relation_tool_joint_head_online_tool_interface?.failure_breakdown || {};
      rows.push({ id: `${MODEL_LABELS[key] || key}:schema`, 比例: Number(breakdown.schema_mismatch || 0) });
      rows.push({ id: `${MODEL_LABELS[key] || key}:timeout`, 比例: Number(breakdown.timeout_pressure || 0) });
      rows.push({ id: `${MODEL_LABELS[key] || key}:state`, 比例: Number(breakdown.state_drift || 0) });
      rows.push({ id: `${MODEL_LABELS[key] || key}:verify`, 比例: Number(breakdown.verify_mismatch || 0) });
    }
    return rows;
  }, [payload]);

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 models / headline_metrics / hypotheses 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`更硬在线工具接口 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(239,68,68,0.14), transparent 28%), radial-gradient(circle at bottom right, rgba(34,197,94,0.12), transparent 26%), rgba(2,6,23,0.68)',
        border: '1px solid rgba(239,68,68,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#f8fafc', fontSize: '15px', fontWeight: 'bold' }}>五点五十七、更硬在线工具接口</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px', maxWidth: '920px' }}>
            这一轮不再只看软代理链，而是加入 `schema_mismatch / timeout_pressure / state_drift / verify_mismatch`
            四类失败，直接比较 `tool` 头和 `relation/tool` 联合头在更硬接口下的成功率、触发率和工具失败率。
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Qwen 成功率增益</div>
          <div style={{ color: '#22c55e', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.gains?.qwen_joint_minus_tool_head_success)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek 成功率增益</div>
          <div style={{ color: '#16a34a', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.gains?.deepseek_joint_minus_tool_head_success)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Qwen 触发率下降</div>
          <div style={{ color: '#38bdf8', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.gains?.qwen_tool_head_minus_joint_trigger_rate)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek 触发率下降</div>
          <div style={{ color: '#0ea5e9', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.gains?.deepseek_tool_head_minus_joint_trigger_rate)}</div>
        </div>
      </div>

      <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {Object.entries(payload?.hypotheses || {}).map(([key, value]) => (
          <div
            key={key}
            style={{
              borderRadius: '999px',
              padding: '6px 10px',
              fontSize: '11px',
              color: value ? '#dcfce7' : '#fee2e2',
              background: value ? 'rgba(34,197,94,0.16)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(34,197,94,0.30)' : 'rgba(248,113,113,0.30)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '未成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 成功率与触发率对照</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={summaryRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="tool头成功率" fill="#94a3b8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="联合头成功率" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="tool头触发率" fill="#ef4444" radius={[4, 4, 0, 0]} />
              <Bar dataKey="联合头触发率" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 工具失败率与恢复率</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={failureRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="tool头工具失败率" fill="#ef4444" radius={[4, 4, 0, 0]} />
              <Bar dataKey="联合头工具失败率" fill="#f97316" radius={[4, 4, 0, 0]} />
              <Bar dataKey="tool头恢复率" fill="#94a3b8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="联合头恢复率" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '12px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 联合头失败类型分布</div>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={breakdownRows} margin={{ top: 10, right: 12, left: 0, bottom: 20 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="id" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.35)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
            <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
            <Legend />
            <Bar dataKey="比例" fill="#a855f7" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
