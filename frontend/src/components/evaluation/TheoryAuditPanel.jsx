import { AlertTriangle, CheckCircle2, ShieldAlert } from 'lucide-react';

function clamp01(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function toPct(value) {
  return Math.round(clamp01(value) * 100);
}

function getStatusTone(statusShort) {
  if (statusShort === 'theory_evidence_hardened') {
    return {
      badge: 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/20',
      bar: 'bg-emerald-500/80',
    };
  }
  if (statusShort === 'theory_evidence_transition') {
    return {
      badge: 'bg-amber-500/15 text-amber-300 border border-amber-500/20',
      bar: 'bg-amber-500/80',
    };
  }
  return {
    badge: 'bg-rose-500/15 text-rose-300 border border-rose-500/20',
    bar: 'bg-rose-500/80',
  };
}

function flagLabel(flag, trueLabel, falseLabel) {
  return flag ? trueLabel : falseLabel;
}

export default function TheoryAuditPanel({ auditData, loading, error }) {
  if (loading) {
    return (
      <div className="rounded-xl border border-white/10 bg-zinc-900/40 p-4 text-sm text-zinc-400">
        正在加载严格审查结果...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-500/20 bg-red-900/10 p-4 text-sm text-red-300">
        {error}
      </div>
    );
  }

  const hm = auditData?.headline_metrics || {};
  const status = auditData?.status || {};
  const findings = Array.isArray(auditData?.audit_findings) ? auditData.audit_findings : [];
  const tone = getStatusTone(status.status_short);

  const metricRows = [
    { label: '理论可信度', value: `${toPct(hm.theory_correctness_confidence)}%` },
    { label: '证据独立性', value: `${toPct(hm.evidence_independence_score)}%` },
    { label: '测试强度', value: `${toPct(hm.test_strength_score)}%` },
    { label: '最优律领先', value: hm.stage82_best_law_margin ? hm.stage82_best_law_margin.toFixed(4) : '-' },
  ];

  const flagRows = [
    flagLabel(hm.derived_falsification_flag, '存在脚本内构造判伪', '未发现脚本内构造判伪'),
    flagLabel(hm.best_law_fragility_flag, '最优律结论处于脆弱区', '最优律结论暂不脆弱'),
    flagLabel(hm.status_label_mismatch_flag, '文档口径与代码口径不完全一致', '文档口径与代码口径基本一致'),
  ];

  return (
    <div className="rounded-xl border border-white/10 bg-zinc-900/40 p-4">
      <div className="flex items-start justify-between gap-3 mb-4">
        <div>
          <div className="text-xs text-zinc-500 mb-1">严格审查</div>
          <div className="text-sm font-semibold text-zinc-100">理论证据状态</div>
        </div>
        <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold ${tone.badge}`}>
          {status.status_short || 'audit_unknown'}
        </span>
      </div>

      <div className="rounded-lg border border-white/10 bg-black/20 p-3 mb-3">
        <div className="flex items-center gap-2 text-zinc-200 mb-2">
          <ShieldAlert size={14} className="text-amber-300" />
          <span className="text-sm font-medium">当前判断</span>
        </div>
        <div className="text-xs text-zinc-400 leading-6">
          {status.status_label || '当前没有可用的严格审查标签。'}
        </div>
        <div className="mt-3 h-2 rounded bg-black/30 overflow-hidden">
          <div className={`h-full ${tone.bar}`} style={{ width: `${toPct(hm.theory_correctness_confidence)}%` }} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-3">
        {metricRows.map((item) => (
          <div key={item.label} className="rounded-lg border border-white/10 bg-black/20 p-3">
            <div className="text-[11px] text-zinc-500 mb-1">{item.label}</div>
            <div className="text-lg font-semibold text-zinc-100">{item.value}</div>
          </div>
        ))}
      </div>

      <div className="rounded-lg border border-white/10 bg-black/20 p-3 mb-3">
        <div className="flex items-center gap-2 text-zinc-200 mb-2">
          <AlertTriangle size={14} className="text-rose-300" />
          <span className="text-sm font-medium">风险标记</span>
        </div>
        <div className="space-y-2">
          {flagRows.map((item) => (
            <div key={item} className="text-xs text-zinc-400 flex items-start gap-2">
              <span className="text-rose-300 mt-0.5">•</span>
              <span>{item}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-white/10 bg-black/20 p-3">
        <div className="flex items-center gap-2 text-zinc-200 mb-2">
          <CheckCircle2 size={14} className="text-cyan-300" />
          <span className="text-sm font-medium">审查发现</span>
        </div>
        <div className="space-y-2">
          {findings.length === 0 ? (
            <div className="text-xs text-zinc-500">暂无审查发现。</div>
          ) : (
            findings.slice(0, 4).map((item) => (
              <div key={item} className="text-xs text-zinc-400 leading-6">
                {item}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
