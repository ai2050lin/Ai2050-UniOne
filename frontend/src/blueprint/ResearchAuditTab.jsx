import { AlertTriangle, CheckCircle2, GitBranch, ShieldAlert } from 'lucide-react';
import { useState } from 'react';
import { dispatchAudit3DFocus, getAudit3DFocus } from './audit3dBridge';

const clamp01 = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
};

const toPct = (value) => `${Math.round(clamp01(value) * 100)}%`;

const toneMap = {
  theory_evidence_hardened: {
    color: '#22c55e',
    softBorder: 'rgba(34, 197, 94, 0.22)',
    softBg: 'rgba(34, 197, 94, 0.06)',
  },
  theory_evidence_transition: {
    color: '#f59e0b',
    softBorder: 'rgba(245, 158, 11, 0.22)',
    softBg: 'rgba(245, 158, 11, 0.06)',
  },
  unproven_explanatory_framework: {
    color: '#f87171',
    softBorder: 'rgba(248, 113, 113, 0.24)',
    softBg: 'rgba(248, 113, 113, 0.06)',
  },
};

const panelStyle = (borderColor, background) => ({
  padding: '28px',
  borderRadius: '22px',
  border: `1px solid ${borderColor}`,
  background,
});

const infoCardStyle = {
  padding: '14px 16px',
  borderRadius: '14px',
  background: 'rgba(0,0,0,0.24)',
  border: '1px solid rgba(255,255,255,0.06)',
};

const infoTitleStyle = {
  fontSize: '11px',
  color: '#64748b',
  marginBottom: '6px',
};

const routeSnapshotStyle = {
  padding: '12px 14px',
  borderRadius: '12px',
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.06)',
};

const stageTitleMap = {
  stage71: '阶段71 统一主核',
  stage73: '阶段73 判伪边界',
  stage80: '阶段80 失效图谱',
  stage81: '阶段81 前后向统一',
  stage82: '阶段82 新颖泛化修复',
  stage83: '阶段83 定理主核',
  stage84: '阶段84 可判伪计算核',
};

const checkToneMap = {
  high: {
    color: '#f87171',
    background: 'rgba(127,29,29,0.20)',
    border: 'rgba(248,113,113,0.24)',
  },
  medium: {
    color: '#fbbf24',
    background: 'rgba(120,53,15,0.18)',
    border: 'rgba(251,191,36,0.22)',
  },
  low: {
    color: '#4ade80',
    background: 'rgba(20,83,45,0.18)',
    border: 'rgba(74,222,128,0.22)',
  },
};

const sortStageEntries = (entries) =>
  [...entries].sort((a, b) => {
    const aMatch = String(a[0]).match(/stage(\d+)/);
    const bMatch = String(b[0]).match(/stage(\d+)/);
    return Number(aMatch?.[1] || 0) - Number(bMatch?.[1] || 0);
  });

const formatStageLabel = (value) => {
  if (!value) return '未知阶段';
  if (stageTitleMap[value]) return stageTitleMap[value];
  const match = String(value).match(/^stage(\d+)(?:_(.+))?$/);
  if (!match) return String(value);
  return match[2] ? `阶段${match[1]} / ${match[2]}` : `阶段${match[1]}`;
};

const focusButtonStyle = (active = false) => ({
  borderRadius: '999px',
  border: active ? '1px solid rgba(103,232,249,0.55)' : '1px solid rgba(255,255,255,0.08)',
  background: active ? 'rgba(8,145,178,0.18)' : 'rgba(255,255,255,0.03)',
  color: active ? '#cffafe' : '#cbd5e1',
  padding: '6px 10px',
  fontSize: '11px',
  cursor: 'pointer',
});

export const ResearchAuditTab = ({
  auditData,
  auditLoading,
  auditError,
  selectedRoute,
  timelineRoutes,
}) => {
  const [lastFocusedStage, setLastFocusedStage] = useState(null);

  if (auditLoading) {
    return (
      <div style={{ animation: 'roadmapFade 0.5s ease-out' }}>
        <div style={{ ...panelStyle('rgba(255,255,255,0.12)', 'rgba(255,255,255,0.025)'), color: '#94a3b8' }}>
          正在加载严格审查结果...
        </div>
      </div>
    );
  }

  if (auditError) {
    return (
      <div style={{ animation: 'roadmapFade 0.5s ease-out' }}>
        <div style={{ ...panelStyle('rgba(248,113,113,0.26)', 'rgba(127,29,29,0.18)'), color: '#fecaca' }}>
          严格审查结果加载失败：{auditError}
        </div>
      </div>
    );
  }

  const hm = auditData?.headline_metrics || {};
  const status = auditData?.status || {};
  const projectReadout = auditData?.project_readout || {};
  const findings = Array.isArray(auditData?.audit_findings) ? auditData.audit_findings : [];
  const lawRank = Array.isArray(auditData?.law_rank) ? auditData.law_rank : [];
  const dependencyEntries = sortStageEntries(Object.entries(auditData?.dependency_graph || {}));
  const auditChecks = Array.isArray(auditData?.audit_checks) ? auditData.audit_checks : [];
  const backfeedPaths = Array.isArray(auditData?.backfeed_paths) ? auditData.backfeed_paths : [];
  const tone = toneMap[status.status_short] || toneMap.unproven_explanatory_framework;
  const routeSnapshots = Array.isArray(timelineRoutes) ? timelineRoutes.slice(0, 4) : [];

  const metricCards = [
    { label: '理论可信度', value: toPct(hm.theory_correctness_confidence), hint: '当前理论被强证据锁定的程度' },
    { label: '证据独立性', value: toPct(hm.evidence_independence_score), hint: '高层结论与下层摘要的解耦程度' },
    { label: '测试强度', value: toPct(hm.test_strength_score), hint: '测试更接近理论判伪还是脚本回归' },
    {
      label: '最优律领先幅度',
      value: Number.isFinite(Number(hm.stage82_best_law_margin)) ? Number(hm.stage82_best_law_margin).toFixed(4) : '-',
      hint: `当前最优候选：${hm.stage82_best_law_name || '未给出'}`,
    },
  ];

  const riskFlags = [
    {
      active: !!hm.derived_falsification_flag,
      label: hm.derived_falsification_flag ? '存在脚本内构造判伪' : '未发现脚本内构造判伪',
    },
    {
      active: !!hm.best_law_fragility_flag,
      label: hm.best_law_fragility_flag ? '最优律优势仍处脆弱区' : '最优律优势暂时较稳',
    },
    {
      active: !!hm.status_label_mismatch_flag,
      label: hm.status_label_mismatch_flag ? '代码口径与文档口径未完全收敛' : '代码口径与文档口径基本一致',
    },
  ];

  const stageFocusEntries = dependencyEntries.slice(0, 5).map(([stageId]) => ({
    stageId,
    ...getAudit3DFocus(stageId),
  }));

  const handleSendFocusTo3D = (stageId) => {
    const focus = dispatchAudit3DFocus(stageId, {
      routeTitle: selectedRoute?.title || '',
      source: 'research_audit_tab',
    });
    if (focus) {
      setLastFocusedStage(stageId);
    }
  };

  return (
    <div style={{ animation: 'roadmapFade 0.5s ease-out', maxWidth: '1220px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '30px', gap: '20px' }}>
        <div>
          <h2 style={{ fontSize: '34px', fontWeight: '900', color: '#fff', margin: '0 0 8px 0' }}>严格审查中心</h2>
          <div style={{ color: '#94a3b8', fontSize: '14px', lineHeight: '1.8', maxWidth: '780px' }}>
            这里不再只展示项目走到了哪里，而是直接回答当前理论为什么还不能轻易下“已经正确”的结论。
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '12px', color: tone.color, fontWeight: 'bold', letterSpacing: '2px', marginBottom: '6px' }}>
            审查状态
          </div>
          <div style={{ fontSize: '16px', color: '#fff', fontWeight: 'bold', maxWidth: '320px', lineHeight: '1.6' }}>
            {status.status_label || '当前没有可用的审查结论。'}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '22px', marginBottom: '22px' }}>
        <div style={panelStyle(tone.softBorder, tone.softBg)}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
            <ShieldAlert size={18} color={tone.color} />
            <div style={{ fontSize: '12px', color: tone.color, fontWeight: 'bold', letterSpacing: '2px' }}>理论状态判断</div>
          </div>
          <div style={{ fontSize: '24px', color: '#fff', fontWeight: 'bold', marginBottom: '12px' }}>
            {status.status_short || 'audit_unknown'}
          </div>
          <div style={{ fontSize: '14px', color: '#d1d5db', lineHeight: '1.75', marginBottom: '14px' }}>
            {projectReadout.summary || '当前没有可用的项目审查摘要。'}
          </div>
          <div style={{ height: '8px', borderRadius: '999px', background: 'rgba(0,0,0,0.28)', overflow: 'hidden', marginBottom: '10px' }}>
            <div
              style={{
                width: toPct(hm.theory_correctness_confidence),
                height: '100%',
                borderRadius: '999px',
                background: tone.color,
              }}
            />
          </div>
          <div style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: '1.7' }}>
            下一步关键问题：{projectReadout.next_question || '需要继续增加外部反例和稳健性测试。'}
          </div>
        </div>

        <div style={panelStyle('rgba(255,255,255,0.08)', 'rgba(255,255,255,0.025)')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
            <GitBranch size={18} color="#60a5fa" />
            <div style={{ fontSize: '12px', color: '#60a5fa', fontWeight: 'bold', letterSpacing: '2px' }}>当前路线上下文</div>
          </div>
          <div style={{ marginBottom: '12px', color: '#fff', fontSize: '20px', fontWeight: 'bold' }}>
            {selectedRoute?.title || '未选中路线'}
          </div>
          <div style={{ color: '#94a3b8', fontSize: '13px', lineHeight: '1.7', marginBottom: '14px' }}>
            {selectedRoute?.routeDescription || selectedRoute?.theorySummary || '当前没有可用的路线描述。'}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div style={infoCardStyle}>
              <div style={infoTitleStyle}>路线准备度</div>
              <div style={{ color: '#67e8f9', fontSize: '24px', fontWeight: 'bold', fontFamily: 'monospace' }}>
                {selectedRoute?.stats?.routeProgress ?? '-'}%
              </div>
            </div>
            <div style={infoCardStyle}>
              <div style={infoTitleStyle}>已接入实验路线</div>
              <div style={{ color: '#e5e7eb', fontSize: '24px', fontWeight: 'bold', fontFamily: 'monospace' }}>
                {Array.isArray(timelineRoutes) ? timelineRoutes.length : 0}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px', marginBottom: '22px' }}>
        {metricCards.map((item) => (
          <div key={item.label} style={infoCardStyle}>
            <div style={infoTitleStyle}>{item.label}</div>
            <div style={{ color: '#fff', fontSize: '26px', fontWeight: 'bold', fontFamily: 'monospace', marginBottom: '6px' }}>
              {item.value}
            </div>
            <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.6' }}>{item.hint}</div>
          </div>
        ))}
      </div>

      <div style={{ ...panelStyle('rgba(34,197,94,0.22)', 'rgba(34,197,94,0.05)'), marginBottom: '22px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', marginBottom: '12px' }}>
          <div>
            <div style={{ fontSize: '12px', color: '#22c55e', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '6px' }}>3D 机制联动</div>
            <div style={{ color: '#d1d5db', fontSize: '13px', lineHeight: '1.7' }}>
              从严格审查直接把阶段焦点发送到 3D 工作台，让机制层自动切换到对应理论对象、动作模式和动画脚本。
            </div>
          </div>
          <div style={{ color: '#86efac', fontSize: '12px', textAlign: 'right', lineHeight: '1.7' }}>
            {lastFocusedStage ? `最近已发送：${formatStageLabel(lastFocusedStage)}` : '尚未发送 3D 聚焦方案'}
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '12px' }}>
          {stageFocusEntries.map((item) => (
            <div key={item.stageId} style={routeSnapshotStyle}>
              <div style={{ color: '#fff', fontSize: '13px', fontWeight: 'bold', marginBottom: '6px' }}>{item.stageLabel}</div>
              <div style={{ color: '#94a3b8', fontSize: '12px', lineHeight: '1.65', minHeight: '58px', marginBottom: '10px' }}>
                {item.summary}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '10px' }}>
                <span style={focusButtonStyle(false)}>{`对象 ${item.theoryObject}`}</span>
                <span style={focusButtonStyle(false)}>{`动作 ${item.analysisMode}`}</span>
              </div>
              <button type="button" onClick={() => handleSendFocusTo3D(item.stageId)} style={focusButtonStyle(lastFocusedStage === item.stageId)}>
                发送到 3D 工作台
              </button>
            </div>
          ))}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: '22px', marginBottom: '22px' }}>
        <div style={panelStyle('rgba(56,189,248,0.22)', 'rgba(56,189,248,0.05)')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
            <GitBranch size={18} color="#38bdf8" />
            <div style={{ fontSize: '12px', color: '#38bdf8', fontWeight: 'bold', letterSpacing: '2px' }}>证据依赖拓扑</div>
          </div>
          {dependencyEntries.length === 0 ? (
            <div style={{ ...routeSnapshotStyle, color: '#94a3b8', fontSize: '13px' }}>当前没有可用的依赖拓扑数据。</div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
              {dependencyEntries.map(([nodeName, deps]) => (
                <div key={nodeName} style={routeSnapshotStyle}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', gap: '10px' }}>
                    <div style={{ color: '#fff', fontSize: '14px', fontWeight: 'bold' }}>{formatStageLabel(nodeName)}</div>
                    <div style={{ color: '#7dd3fc', fontSize: '12px', fontFamily: 'monospace' }}>扇入 {deps.length}</div>
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {deps.map((dep) => (
                      <span
                        key={dep}
                        style={{
                          padding: '5px 8px',
                          borderRadius: '999px',
                          background: dep.startsWith('stage7') || dep.startsWith('stage8') ? 'rgba(56,189,248,0.14)' : 'rgba(255,255,255,0.04)',
                          border: dep.startsWith('stage7') || dep.startsWith('stage8') ? '1px solid rgba(56,189,248,0.22)' : '1px solid rgba(255,255,255,0.06)',
                          color: dep.startsWith('stage7') || dep.startsWith('stage8') ? '#bae6fd' : '#94a3b8',
                          fontSize: '11px',
                          lineHeight: '1.4',
                        }}
                      >
                        {formatStageLabel(dep)}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={panelStyle('rgba(14,165,233,0.18)', 'rgba(255,255,255,0.025)')}>
          <div style={{ fontSize: '12px', color: '#67e8f9', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '14px' }}>
            审查检查项
          </div>
          {auditChecks.length === 0 ? (
            <div style={{ ...routeSnapshotStyle, color: '#94a3b8', fontSize: '13px' }}>当前没有可用的审查检查项。</div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px', marginBottom: backfeedPaths.length > 0 ? '12px' : 0 }}>
              {auditChecks.map((item) => {
                const toneInfo = checkToneMap[item.risk_level] || checkToneMap.medium;
                return (
                  <div
                    key={item.name}
                    style={{
                      padding: '12px 14px',
                      borderRadius: '12px',
                      background: toneInfo.background,
                      border: `1px solid ${toneInfo.border}`,
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', marginBottom: '6px' }}>
                      <div style={{ color: '#fff', fontSize: '13px', fontWeight: 'bold' }}>{item.name}</div>
                      <div style={{ color: toneInfo.color, fontSize: '11px', fontWeight: 'bold' }}>
                        {item.passed ? '已通过' : '未通过'} / {item.risk_level}
                      </div>
                    </div>
                    <div style={{ color: '#d1d5db', fontSize: '12px', lineHeight: '1.65' }}>{item.detail}</div>
                  </div>
                );
              })}
            </div>
          )}
          {backfeedPaths.length > 0 ? (
            <div style={{ ...routeSnapshotStyle, marginTop: '12px' }}>
              <div style={{ color: '#67e8f9', fontSize: '12px', fontWeight: 'bold', marginBottom: '8px' }}>检测到的回灌路径</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
                {backfeedPaths.map((item) => (
                  <div key={item} style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.6' }}>
                    {item}
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '0.95fr 1.05fr', gap: '22px', marginBottom: '22px' }}>
        <div style={panelStyle('rgba(248,113,113,0.22)', 'rgba(248,113,113,0.05)')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
            <AlertTriangle size={18} color="#f87171" />
            <div style={{ fontSize: '12px', color: '#f87171', fontWeight: 'bold', letterSpacing: '2px' }}>风险标记</div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {riskFlags.map((item) => (
              <div
                key={item.label}
                style={{
                  padding: '12px 14px',
                  borderRadius: '12px',
                  background: item.active ? 'rgba(127,29,29,0.22)' : 'rgba(255,255,255,0.03)',
                  border: item.active ? '1px solid rgba(248,113,113,0.28)' : '1px solid rgba(255,255,255,0.06)',
                  color: item.active ? '#fecaca' : '#cbd5e1',
                  fontSize: '13px',
                  lineHeight: '1.65',
                }}
              >
                {item.label}
              </div>
            ))}
          </div>
        </div>

        <div style={panelStyle('rgba(255,255,255,0.08)', 'rgba(255,255,255,0.025)')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
            <CheckCircle2 size={18} color="#67e8f9" />
            <div style={{ fontSize: '12px', color: '#67e8f9', fontWeight: 'bold', letterSpacing: '2px' }}>审查发现</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
            {findings.length === 0 ? (
              <div style={{ ...routeSnapshotStyle, color: '#94a3b8', fontSize: '13px' }}>暂无审查发现。</div>
            ) : (
              findings.map((item, idx) => (
                <div key={`${idx}-${item}`} style={{ ...routeSnapshotStyle, color: '#dbe4ee', fontSize: '13px', lineHeight: '1.7' }}>
                  {idx + 1}. {item}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '22px' }}>
        <div style={panelStyle('rgba(99,102,241,0.24)', 'rgba(99,102,241,0.05)')}>
          <div style={{ fontSize: '12px', color: '#818cf8', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '14px' }}>
            候选更新律排序
          </div>
          {lawRank.length === 0 ? (
            <div style={{ ...routeSnapshotStyle, color: '#94a3b8', fontSize: '13px' }}>当前摘要里没有可用的更新律排序。</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {lawRank.map((item, idx) => (
                <div key={`${item.law_name}-${idx}`} style={routeSnapshotStyle}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <div style={{ color: idx === 0 ? '#c4b5fd' : '#e5e7eb', fontSize: '14px', fontWeight: 'bold' }}>
                      {idx + 1}. {item.law_name}
                    </div>
                    <div style={{ color: '#93c5fd', fontSize: '12px', fontFamily: 'monospace' }}>
                      novelty {Number(item.repaired_novelty_score || 0).toFixed(4)}
                    </div>
                  </div>
                  <div style={{ color: '#94a3b8', fontSize: '12px' }}>
                    修复后失败量：{Number(item.failure_after || 0).toFixed(4)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={panelStyle('rgba(34,197,94,0.22)', 'rgba(34,197,94,0.05)')}>
          <div style={{ fontSize: '12px', color: '#22c55e', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '14px' }}>
            实验路线快照
          </div>
          {routeSnapshots.length === 0 ? (
            <div style={{ ...routeSnapshotStyle, color: '#94a3b8', fontSize: '13px' }}>当前没有可用的时间线路线快照。</div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
              {routeSnapshots.map((item) => (
                <div key={item.route_id || item.title} style={routeSnapshotStyle}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <div style={{ color: '#fff', fontSize: '14px', fontWeight: 'bold' }}>{item.title}</div>
                    <div style={{ color: '#86efac', fontSize: '12px', fontFamily: 'monospace' }}>
                      {Math.round(Number(item?.stats?.routeProgress || 0))}%
                    </div>
                  </div>
                  <div style={{ color: '#94a3b8', fontSize: '12px', lineHeight: '1.6' }}>
                    run {item?.stats?.totalRuns || 0} | success {item?.stats?.completedRuns || 0} | avg{' '}
                    {Number(item?.stats?.avgScore || 0).toFixed(3)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
