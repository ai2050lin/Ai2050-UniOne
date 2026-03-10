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
import samplePayload from './data/qwen3_deepseek7b_real_model_recovery_proxy_atlas_sample.json';

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

export default function Qwen3DeepSeekRealModelRecoveryProxyAtlasDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const summaryRows = useMemo(() => {
    return Object.entries(payload?.models || {}).map(([key, row]) => ({
      model: MODEL_LABELS[key] || key,
      恢复代理分: Number(row?.global_summary?.recovery_proxy_score || 0),
      桥接侧收益: Number(row?.global_summary?.bridge_side_gain || 0),
      任务侧收益: Number(row?.global_summary?.task_side_gain || 0),
      取向惩罚: Number(row?.global_summary?.orientation_gap_abs || 0),
    }));
  }, [payload]);

  const relationRows = useMemo(() => {
    const rows = [];
    for (const [modelKey, row] of Object.entries(payload?.models || {})) {
      for (const relationRow of (row?.relation_recovery_rows || []).slice(0, 6)) {
        rows.push({
          id: `${MODEL_LABELS[modelKey] || modelKey}:${relationRow.relation}`,
          恢复代理: Number(relationRow.repair_proxy || 0),
          桥接收益: Number(relationRow.behavior_gain || 0),
          共享层命中: Number(relationRow.shared_layer_hit_ratio || 0),
        });
      }
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
      setError(`恢复代理 atlas JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(34,197,94,0.16), transparent 28%), radial-gradient(circle at bottom right, rgba(239,68,68,0.12), transparent 26%), rgba(2,6,23,0.68)',
        border: '1px solid rgba(34,197,94,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#f8fafc', fontSize: '15px', fontWeight: 'bold' }}>五点五十二、真实模型恢复代理 Atlas</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px', maxWidth: '920px' }}>
            这一步不宣称已经测到真实恢复链本体，而是把真实模型里与恢复最相关的桥接收益、结构收益、目标层带风险和取向惩罚压成一张恢复代理 atlas。重点是先把“恢复收益”和“恢复瓶颈”同时摆出来。
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Qwen 恢复代理分</div>
          <div style={{ color: '#22c55e', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.headline_metrics?.qwen_recovery_proxy_score)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek 恢复代理分</div>
          <div style={{ color: '#16a34a', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.headline_metrics?.deepseek_recovery_proxy_score)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Qwen 任务侧收益</div>
          <div style={{ color: '#38bdf8', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.headline_metrics?.qwen_task_side_gain)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek 取向惩罚</div>
          <div style={{ color: '#ef4444', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.headline_metrics?.deepseek_orientation_gap_abs)}</div>
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
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 模型级恢复代理总览</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={summaryRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="恢复代理分" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="桥接侧收益" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="任务侧收益" fill="#a855f7" radius={[4, 4, 0, 0]} />
              <Bar dataKey="取向惩罚" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 关系级恢复代理</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={relationRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="id" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="恢复代理" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="桥接收益" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="共享层命中" fill="#eab308" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
