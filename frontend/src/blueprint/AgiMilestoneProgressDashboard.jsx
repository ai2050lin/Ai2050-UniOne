import React, { useMemo, useState } from 'react';
import samplePayload from './data/agi_milestone_progress_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && Array.isArray(payload.milestones) && Array.isArray(payload.next_plan));
}

function pct(v) {
  return `${(Number(v || 0) * 100).toFixed(0)}%`;
}

function statusMeta(status) {
  if (status === 'completed') return { label: '已完成', color: '#22c55e', glow: 'rgba(34,197,94,0.18)' };
  if (status === 'current') return { label: '当前节点', color: '#38bdf8', glow: 'rgba(56,189,248,0.18)' };
  return { label: '待推进', color: '#f59e0b', glow: 'rgba(245,158,11,0.18)' };
}

export default function AgiMilestoneProgressDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const currentStage = useMemo(
    () => payload?.milestones?.find((item) => item.id === payload?.meta?.current_stage_id),
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 meta / milestones / next_plan 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`里程碑进度 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(34,197,94,0.12), transparent 28%), radial-gradient(circle at top right, rgba(56,189,248,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(34,197,94,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>AGI 里程碑进度总览</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            用统一时间轴显示整体研究进度、当前所处节点、每个里程碑的证据状态，以及接下来最值得推进的工作计划。
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>整体进度</div>
          <div style={{ color: '#86efac', fontSize: '22px', fontWeight: 'bold' }}>{pct(payload?.meta?.overall_progress)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>当前节点</div>
          <div style={{ color: '#7dd3fc', fontSize: '18px', fontWeight: 'bold' }}>{currentStage?.id || '-'}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{currentStage?.title || '-'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>已完成节点</div>
          <div style={{ color: '#fcd34d', fontSize: '22px', fontWeight: 'bold' }}>
            {payload?.summary?.completed_milestones ?? '-'} / {payload?.summary?.total_milestones ?? '-'}
          </div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>当前焦点</div>
          <div style={{ color: '#f8fafc', fontSize: '14px', fontWeight: 'bold', lineHeight: '1.6' }}>{payload?.summary?.current_focus || '-'}</div>
        </div>
      </div>

      <div style={{ marginTop: '14px', color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8', border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
        {payload?.summary?.statement || '-'}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1.1fr) minmax(320px, 0.9fr)', gap: '12px' }}>
        <div style={{ display: 'grid', gap: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px' }}>1. 里程碑时间轴</div>
          {(payload?.milestones || []).map((item, index) => {
            const meta = statusMeta(item.status);
            return (
              <div
                key={item.id}
                style={{
                  border: `1px solid ${meta.glow}`,
                  borderRadius: '14px',
                  padding: '12px',
                  background: 'rgba(15,23,42,0.42)',
                  boxShadow: item.status === 'current' ? `0 0 0 1px ${meta.glow} inset` : 'none',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
                  <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    <div
                      style={{
                        width: '26px',
                        height: '26px',
                        borderRadius: '999px',
                        background: meta.glow,
                        border: `1px solid ${meta.color}`,
                        color: meta.color,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '11px',
                        fontWeight: 'bold',
                      }}
                    >
                      {index + 1}
                    </div>
                    <div>
                      <div style={{ color: '#f8fafc', fontSize: '13px', fontWeight: 'bold' }}>{`${item.id} ${item.title}`}</div>
                      <div style={{ color: meta.color, fontSize: '11px', marginTop: '2px' }}>{meta.label}</div>
                    </div>
                  </div>
                  <div style={{ color: '#e2e8f0', fontSize: '12px', fontWeight: 'bold' }}>{pct(item.progress)}</div>
                </div>

                <div style={{ marginTop: '10px', height: '8px', borderRadius: '999px', background: 'rgba(30,41,59,0.9)', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: pct(item.progress),
                      height: '100%',
                      background: `linear-gradient(90deg, ${meta.color}, rgba(255,255,255,0.7))`,
                    }}
                  />
                </div>

                <div style={{ marginTop: '10px', color: '#cbd5e1', fontSize: '11px', lineHeight: '1.8' }}>{item.goal}</div>
                <div style={{ marginTop: '8px', display: 'grid', gap: '6px' }}>
                  {(item.evidence || []).map((evidence) => (
                    <div
                      key={evidence}
                      style={{
                        color: '#cbd5e1',
                        fontSize: '11px',
                        lineHeight: '1.7',
                        border: '1px solid rgba(148,163,184,0.16)',
                        borderRadius: '10px',
                        padding: '7px 9px',
                      }}
                    >
                      {evidence}
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: '8px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>{`过关条件: ${item.next_gate}`}</div>
              </div>
            );
          })}
        </div>

        <div style={{ display: 'grid', gap: '12px' }}>
          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 当前节点解读</div>
            <div style={{ color: '#f8fafc', fontSize: '14px', fontWeight: 'bold' }}>{`${currentStage?.id || '-'} ${currentStage?.title || ''}`}</div>
            <div style={{ marginTop: '8px', color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>{currentStage?.goal || '-'}</div>
            <div style={{ marginTop: '10px', display: 'grid', gap: '8px' }}>
              {(currentStage?.evidence || []).map((item) => (
                <div key={item} style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.7', border: '1px solid rgba(148,163,184,0.18)', borderRadius: '10px', padding: '8px 10px' }}>
                  {item}
                </div>
              ))}
            </div>
          </div>

          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 接下来工作计划</div>
            <div style={{ display: 'grid', gap: '8px' }}>
              {(payload?.next_plan || []).map((item) => (
                <div key={`${item.priority}-${item.title}`} style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '10px', padding: '9px 10px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', alignItems: 'center' }}>
                    <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}>{item.title}</div>
                    <div style={{ color: item.priority === 'P0' ? '#f97316' : '#38bdf8', fontSize: '11px', fontWeight: 'bold' }}>{item.priority}</div>
                  </div>
                  <div style={{ marginTop: '6px', color: '#cbd5e1', fontSize: '11px', lineHeight: '1.7' }}>{item.description}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
