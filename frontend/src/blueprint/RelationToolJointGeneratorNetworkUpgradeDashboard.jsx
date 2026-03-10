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
import samplePayload from './data/relation_tool_joint_generator_network_upgrade_sample.json';

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function isValidPayload(payload) {
  return Boolean(payload && payload.models && payload.generator_profiles && payload.headline_metrics && payload.hypotheses);
}

const MODEL_LABELS = {
  qwen3_4b: 'Qwen3-4B',
  deepseek_7b: 'DeepSeek-7B',
};

function stageUndercoverage(matchRow, stage) {
  const row = (matchRow?.rows || []).find((item) => item.stage === stage);
  return Number(row?.undercoverage || 0);
}

export default function RelationToolJointGeneratorNetworkUpgradeDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const summaryRows = useMemo(() => {
    return Object.entries(payload?.models || {}).map(([key, row]) => ({
      model: MODEL_LABELS[key] || key,
      tool头平均缺口: Number(row?.tool_stage_head_match?.mean_undercoverage || 0),
      联合头平均缺口: Number(row?.relation_tool_joint_head_match?.mean_undercoverage || 0),
      tool头relation缺口: stageUndercoverage(row?.tool_stage_head_match, 'relation'),
      联合头relation缺口: stageUndercoverage(row?.relation_tool_joint_head_match, 'relation'),
    }));
  }, [payload]);

  const stageRows = useMemo(() => {
    return Object.entries(payload?.models || {}).map(([key, row]) => ({
      model: MODEL_LABELS[key] || key,
      tool头relation缺口: stageUndercoverage(row?.tool_stage_head_match, 'relation'),
      联合头relation缺口: stageUndercoverage(row?.relation_tool_joint_head_match, 'relation'),
      tool头tool缺口: stageUndercoverage(row?.tool_stage_head_match, 'tool'),
      联合头tool缺口: stageUndercoverage(row?.relation_tool_joint_head_match, 'tool'),
    }));
  }, [payload]);

  const demandRows = useMemo(() => {
    const rows = [];
    const capacity = payload?.generator_profiles?.relation_tool_joint_head_generator_network?.stage_capacity || {};
    for (const [key, row] of Object.entries(payload?.models || {})) {
      const demand = row?.stage_demand || {};
      for (const stage of payload?.meta?.stages || []) {
        rows.push({
          id: `${MODEL_LABELS[key] || key}:${stage}`,
          阶段需求: Number(demand?.[stage] || 0),
          联合头容量: Number(capacity?.[stage] || 0),
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
        throw new Error('缺少 models / generator_profiles / headline_metrics / hypotheses 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`relation/tool 联合头升级 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setPayload(samplePayload);
    setSource('内置样例');
    setError('');
  }

  const upgradeSpec = payload?.generator_profiles?.relation_tool_joint_head_generator_network?.upgrade_spec || {};

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(34,197,94,0.15), transparent 28%), radial-gradient(circle at bottom right, rgba(14,165,233,0.10), transparent 26%), rgba(2,6,23,0.68)',
        border: '1px solid rgba(34,197,94,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#f8fafc', fontSize: '15px', fontWeight: 'bold' }}>五点五十六、relation/tool 联合头升级</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px', maxWidth: '920px' }}>
            在 `tool` 头已经修复主瓶颈之后，继续加一个 `relation/tool` 联合头。
            这一轮只回答一个问题：能不能把新暴露出的 relation 风险压下去，同时不把刚修好的 tool 段重新打坏。
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Qwen relation 缺口收缩</div>
          <div style={{ color: '#22c55e', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.headline_metrics?.qwen_relation_undercoverage_gain)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek relation 缺口收缩</div>
          <div style={{ color: '#16a34a', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.headline_metrics?.deepseek_relation_undercoverage_gain)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Qwen tool 回弹</div>
          <div style={{ color: '#38bdf8', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.headline_metrics?.qwen_tool_regression_after_joint_head)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek tool 回弹</div>
          <div style={{ color: '#0ea5e9', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.headline_metrics?.deepseek_tool_regression_after_joint_head)}</div>
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

      <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>relation 头增益</div>
          <div style={{ color: '#f8fafc', fontSize: '14px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(upgradeSpec.relation_head_gain, 2)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>tool 头增益</div>
          <div style={{ color: '#f8fafc', fontSize: '14px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(upgradeSpec.tool_head_gain, 2)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>cross gate</div>
          <div style={{ color: '#f8fafc', fontSize: '14px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(upgradeSpec.cross_gate_gain, 2)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>verify 护栏</div>
          <div style={{ color: '#f8fafc', fontSize: '14px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(upgradeSpec.verify_guard_gain, 2)}</div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 平均缺口与 relation 缺口</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={summaryRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="tool头平均缺口" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="联合头平均缺口" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="tool头relation缺口" fill="#a855f7" radius={[4, 4, 0, 0]} />
              <Bar dataKey="联合头relation缺口" fill="#f97316" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. relation 与 tool 缺口联动</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={stageRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="tool头relation缺口" fill="#a855f7" radius={[4, 4, 0, 0]} />
              <Bar dataKey="联合头relation缺口" fill="#f97316" radius={[4, 4, 0, 0]} />
              <Bar dataKey="tool头tool缺口" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="联合头tool缺口" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '12px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 联合头容量 vs 真实阶段需求</div>
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={demandRows} margin={{ top: 10, right: 12, left: 0, bottom: 20 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="id" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.35)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
            <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
            <Legend />
            <Bar dataKey="阶段需求" fill="#ef4444" radius={[4, 4, 0, 0]} />
            <Bar dataKey="联合头容量" fill="#22c55e" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
