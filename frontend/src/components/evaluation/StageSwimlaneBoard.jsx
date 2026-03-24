import { AlertTriangle, BookOpen, CheckCircle2, Clock3 } from 'lucide-react';
import { useMemo } from 'react';

function toTimestamp(value) {
  const ts = Date.parse(value || '');
  return Number.isFinite(ts) ? ts : 0;
}

function formatTime(value) {
  if (!value) return '-';
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

function normalizeStatus(value) {
  return String(value || '').toLowerCase();
}

function getStatusTone(status) {
  if (status.includes('done') || status.includes('complete') || status.includes('成功')) {
    return 'text-emerald-300 border-emerald-500/20 bg-emerald-500/10';
  }
  if (status.includes('progress') || status.includes('active') || status.includes('运行')) {
    return 'text-blue-300 border-blue-500/20 bg-blue-500/10';
  }
  if (status.includes('failed') || status.includes('error') || status.includes('风险')) {
    return 'text-rose-300 border-rose-500/20 bg-rose-500/10';
  }
  return 'text-zinc-400 border-white/10 bg-black/20';
}

function buildRecentTests(timeline) {
  const routes = Array.isArray(timeline?.routes) ? timeline.routes : [];
  return routes
    .flatMap((routeItem) => {
      const tests = Array.isArray(routeItem?.tests) ? routeItem.tests : [];
      return tests.map((test) => ({
        id: test?.test_id || `${routeItem.route}-${test?.timestamp}`,
        title: `${routeItem.route} / ${test?.analysis_type || 'unknown'}`,
        subtitle: test?.evaluation?.summary || `状态 ${test?.status || 'unknown'}`,
        status: test?.status || 'unknown',
        timestamp: test?.timestamp || '',
        meta: formatTime(test?.timestamp),
      }));
    })
    .sort((a, b) => toTimestamp(b.timestamp) - toTimestamp(a.timestamp))
    .slice(0, 6);
}

export default function StageSwimlaneBoard({ phases, timeline, auditData, logs }) {
  const recentTests = useMemo(() => buildRecentTests(timeline), [timeline]);
  const auditFindings = Array.isArray(auditData?.audit_findings) ? auditData.audit_findings : [];
  const auditMetrics = auditData?.headline_metrics || {};

  const lanes = [
    {
      key: 'phases',
      title: '阶段',
      icon: CheckCircle2,
      items: (Array.isArray(phases) ? phases : []).slice(0, 6).map((phase, idx) => ({
        id: `${phase?.title || 'phase'}-${idx}`,
        title: phase?.title || `Phase ${idx + 1}`,
        subtitle: phase?.summary || '暂无阶段摘要',
        status: phase?.status || 'unknown',
        meta: `阶段 ${idx + 1}`,
      })),
    },
    {
      key: 'tests',
      title: '测试',
      icon: Clock3,
      items: recentTests,
    },
    {
      key: 'audit',
      title: '审查',
      icon: AlertTriangle,
      items: [
        {
          id: 'audit-status',
          title: auditData?.status?.status_short || 'audit_unknown',
          subtitle: auditData?.status?.status_label || '暂无审查状态',
          status: auditData?.status?.status_short || 'unknown',
          meta: `可信度 ${Math.round(Number(auditMetrics.theory_correctness_confidence || 0) * 100)}%`,
        },
        ...auditFindings.slice(0, 3).map((item, idx) => ({
          id: `audit-finding-${idx}`,
          title: `发现 ${idx + 1}`,
          subtitle: item,
          status: 'risk',
          meta: '严格审查',
        })),
      ],
    },
    {
      key: 'logs',
      title: '文档',
      icon: BookOpen,
      items: (Array.isArray(logs) ? logs : []).slice(0, 4).map((log, idx) => ({
        id: `log-${idx}`,
        title: `研究记录 ${idx + 1}`,
        subtitle: log,
        status: 'memo',
        meta: 'MEMO / LOG',
      })),
    },
  ];

  return (
    <div className="rounded-xl border border-white/10 bg-zinc-900/40 p-4">
      <div className="flex items-center justify-between gap-3 mb-4">
        <div>
          <div className="text-xs text-zinc-500 mb-1">研究泳道</div>
          <div className="text-sm font-semibold text-zinc-100">阶段 / 测试 / 审查 / 文档</div>
        </div>
        <div className="text-[10px] text-zinc-500">
          目标：把理论推进、实验推进和审查推进放到同一时间视图
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
        {lanes.map((lane) => {
          const Icon = lane.icon;
          return (
            <div key={lane.key} className="rounded-lg border border-white/10 bg-black/20 p-3">
              <div className="flex items-center gap-2 mb-3 text-zinc-200">
                <Icon size={14} className="text-cyan-300" />
                <span className="text-sm font-medium">{lane.title}</span>
              </div>

              <div className="space-y-2">
                {lane.items.length === 0 ? (
                  <div className="text-xs text-zinc-500">暂无数据</div>
                ) : (
                  lane.items.map((item) => (
                    <div key={item.id} className="rounded-lg border border-white/10 bg-zinc-950/60 p-3">
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <div className="text-xs font-semibold text-zinc-100">{item.title}</div>
                        <span className={`px-2 py-0.5 rounded-full text-[10px] border ${getStatusTone(normalizeStatus(item.status))}`}>
                          {item.meta}
                        </span>
                      </div>
                      <div className="text-xs text-zinc-400 leading-6 line-clamp-4">
                        {item.subtitle}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
