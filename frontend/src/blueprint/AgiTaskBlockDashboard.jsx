import React, { useMemo, useState } from 'react';
import samplePayload from './data/agi_task_block_summary_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.blocks && payload.next_plan);
}

function fmt(v, digits = 4) {
  return Number(v || 0).toFixed(digits);
}

function statusMeta(status) {
  if (status === 'completed') return { label: '已完成', color: '#22c55e', glow: 'rgba(34,197,94,0.18)' };
  if (status === 'current') return { label: '当前进行中', color: '#38bdf8', glow: 'rgba(56,189,248,0.18)' };
  return { label: '部分完成', color: '#f59e0b', glow: 'rgba(245,158,11,0.18)' };
}

export default function AgiTaskBlockDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const blockRows = useMemo(
    () =>
      Object.entries(payload?.blocks || {}).map(([key, value]) => ({
        id: key,
        ...value,
      })),
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) throw new Error('缺少 blocks / next_plan 字段');
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`任务块总览 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(14,165,233,0.12), transparent 28%), radial-gradient(circle at top right, rgba(34,197,94,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(14,165,233,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>A / B / C / D 任务块总览</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把四个大任务块的当前状态、主分数、关键卡点和下一步计划放到一张面板里，
            直接看哪一块已经形成稳定证据，哪一块仍然是主战场。
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

      <div style={{ marginTop: '14px', display: 'grid', gap: '10px' }}>
        {blockRows.map((block) => {
          const meta = statusMeta(block.status);
          return (
            <div
              key={block.id}
              style={{
                border: `1px solid ${meta.glow}`,
                borderRadius: '14px',
                padding: '12px',
                background: 'rgba(15,23,42,0.42)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', flexWrap: 'wrap', alignItems: 'center' }}>
                <div>
                  <div style={{ color: '#f8fafc', fontSize: '14px', fontWeight: 'bold' }}>{`${block.id} ${block.title}`}</div>
                  <div style={{ color: meta.color, fontSize: '11px', marginTop: '3px' }}>{meta.label}</div>
                </div>
                <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
                  <div>
                    <div style={{ color: '#94a3b8', fontSize: '10px' }}>主分数</div>
                    <div style={{ color: '#7dd3fc', fontSize: '16px', fontWeight: 'bold' }}>{fmt(block.headline_score)}</div>
                  </div>
                  <div>
                    <div style={{ color: '#94a3b8', fontSize: '10px' }}>副分数</div>
                    <div style={{ color: '#fcd34d', fontSize: '16px', fontWeight: 'bold' }}>{fmt(block.sub_score)}</div>
                  </div>
                </div>
              </div>

              <div style={{ marginTop: '8px', color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>{block.statement}</div>

              <div style={{ marginTop: '10px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '8px' }}>
                {Object.entries(block.metrics || {}).map(([name, value]) => (
                  <div
                    key={name}
                    style={{
                      border: '1px solid rgba(148,163,184,0.18)',
                      borderRadius: '10px',
                      padding: '8px 10px',
                      color: '#cbd5e1',
                      fontSize: '11px',
                    }}
                  >
                    <div style={{ color: '#94a3b8', marginBottom: '4px' }}>{name}</div>
                    <div style={{ color: '#f8fafc', fontWeight: 'bold' }}>{fmt(value)}</div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>下一步计划</div>
        <div style={{ display: 'grid', gap: '8px' }}>
          {(payload?.next_plan || []).map((item) => (
            <div
              key={item}
              style={{
                color: '#cbd5e1',
                fontSize: '11px',
                lineHeight: '1.7',
                border: '1px solid rgba(148,163,184,0.16)',
                borderRadius: '10px',
                padding: '8px 10px',
              }}
            >
              {item}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
