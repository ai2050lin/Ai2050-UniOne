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
import samplePayload from './data/tool_stage_generator_network_upgrade_sample.json';

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

function toolUndercoverage(matchRow) {
  const row = (matchRow?.rows || []).find((item) => item.stage === 'tool');
  return Number(row?.undercoverage || 0);
}

export default function ToolStageGeneratorNetworkUpgradeDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const summaryRows = useMemo(() => {
    return Object.entries(payload?.models || {}).map(([key, row]) => ({
      model: MODEL_LABELS[key] || key,
      搜索版平均缺口: Number(row?.searched_generator_match?.mean_undercoverage || 0),
      端到端平均缺口: Number(row?.end_to_end_generator_match?.mean_undercoverage || 0),
      tool头平均缺口: Number(row?.tool_stage_head_match?.mean_undercoverage || 0),
    }));
  }, [payload]);

  const toolRows = useMemo(() => {
    return Object.entries(payload?.models || {}).map(([key, row]) => ({
      model: MODEL_LABELS[key] || key,
      搜索版tool缺口: toolUndercoverage(row?.searched_generator_match),
      端到端tool缺口: toolUndercoverage(row?.end_to_end_generator_match),
      tool头tool缺口: toolUndercoverage(row?.tool_stage_head_match),
    }));
  }, [payload]);

  const stageRows = useMemo(() => {
    const rows = [];
    const upgradedCapacity = payload?.generator_profiles?.tool_stage_head_generator_network?.stage_capacity || {};
    for (const [key, row] of Object.entries(payload?.models || {})) {
      const demand = row?.stage_demand || {};
      for (const stage of payload?.meta?.stages || []) {
        rows.push({
          id: `${MODEL_LABELS[key] || key}:${stage}`,
          阶段需求: Number(demand?.[stage] || 0),
          升级后容量: Number(upgradedCapacity?.[stage] || 0),
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
      setError(`tool 阶段生成网络升级 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setPayload(samplePayload);
    setSource('内置样例');
    setError('');
  }

  const upgradeSpec = payload?.generator_profiles?.tool_stage_head_generator_network?.upgrade_spec || {};

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(249,115,22,0.16), transparent 28%), radial-gradient(circle at bottom right, rgba(14,165,233,0.12), transparent 26%), rgba(2,6,23,0.68)',
        border: '1px solid rgba(249,115,22,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#f8fafc', fontSize: '15px', fontWeight: 'bold' }}>五点五十五、tool 阶段生成网络升级</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px', maxWidth: '920px' }}>
            这一步不再泛泛加容量，而是只针对真实在线链里最明显的瓶颈 `tool` 阶段加一个专门头，
            直接比较搜索版、端到端版和升级版在真实层带需求上的缺口变化，同时检查非 `tool` 阶段的副作用是否仍受控。
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Qwen tool 缺口收缩</div>
          <div style={{ color: '#22c55e', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.headline_metrics?.qwen_tool_undercoverage_gain)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek tool 缺口收缩</div>
          <div style={{ color: '#16a34a', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(payload?.headline_metrics?.deepseek_tool_undercoverage_gain)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Qwen 升级后最差阶段</div>
          <div style={{ color: '#f97316', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{payload?.headline_metrics?.qwen_worst_stage_after_upgrade || '-'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek 升级后最差阶段</div>
          <div style={{ color: '#fb7185', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{payload?.headline_metrics?.deepseek_worst_stage_after_upgrade || '-'}</div>
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>tool 头增益</div>
          <div style={{ color: '#f8fafc', fontSize: '14px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(upgradeSpec.tool_head_gain, 2)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>relation 辅助</div>
          <div style={{ color: '#f8fafc', fontSize: '14px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(upgradeSpec.relation_assist_gain, 2)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>verify 护栏</div>
          <div style={{ color: '#f8fafc', fontSize: '14px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(upgradeSpec.verify_guard_gain, 2)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>concept 护栏</div>
          <div style={{ color: '#f8fafc', fontSize: '14px', fontWeight: 'bold', marginTop: '4px' }}>{fmt(upgradeSpec.concept_guard_gain, 2)}</div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 模型级平均缺口对照</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={summaryRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="搜索版平均缺口" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="端到端平均缺口" fill="#a855f7" radius={[4, 4, 0, 0]} />
              <Bar dataKey="tool头平均缺口" fill="#f97316" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. tool 阶段缺口对照</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={toolRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="搜索版tool缺口" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="端到端tool缺口" fill="#a855f7" radius={[4, 4, 0, 0]} />
              <Bar dataKey="tool头tool缺口" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '12px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 升级后容量 vs 真实阶段需求</div>
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={stageRows} margin={{ top: 10, right: 12, left: 0, bottom: 20 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="id" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.35)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
            <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
            <Legend />
            <Bar dataKey="阶段需求" fill="#ef4444" radius={[4, 4, 0, 0]} />
            <Bar dataKey="升级后容量" fill="#22c55e" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
