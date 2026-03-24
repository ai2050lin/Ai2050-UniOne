import { AlertTriangle, CheckCircle2, Clock3, TrendingUp } from 'lucide-react';
import { useMemo } from 'react';

function clamp01(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function toPct(value) {
  return Math.round(clamp01(value) * 100);
}

function countTimelineTests(timeline) {
  const routes = Array.isArray(timeline?.routes) ? timeline.routes : [];
  return routes.reduce((sum, routeItem) => {
    const tests = Array.isArray(routeItem?.tests) ? routeItem.tests : [];
    return sum + tests.length;
  }, 0);
}

export default function ProgressRiskDualAxis({ progressData, timeline, auditData }) {
  const overview = progressData?.systemic || {};
  const roadmap = Array.isArray(overview?.roadmap) ? overview.roadmap : [];
  const auditMetrics = auditData?.headline_metrics || {};

  const progressRatio = clamp01(
    overview?.convergence_index
      ?? (roadmap.length > 0
        ? roadmap.filter((item) => String(item?.status || '').toLowerCase() === 'done').length / roadmap.length
        : 0)
  );
  const confidenceRatio = clamp01(auditMetrics.theory_correctness_confidence);
  const evidenceRatio = clamp01(auditMetrics.evidence_independence_score);
  const testStrengthRatio = clamp01(auditMetrics.test_strength_score);

  const routeCount = Array.isArray(timeline?.routes) ? timeline.routes.length : 0;
  const totalTests = countTimelineTests(timeline);
  const riskFlags = [
    Boolean(auditMetrics.derived_falsification_flag),
    Boolean(auditMetrics.best_law_fragility_flag),
    Boolean(auditMetrics.status_label_mismatch_flag),
  ].filter(Boolean).length;

  const headline = useMemo(() => {
    if (progressRatio >= 0.6 && confidenceRatio < 0.3) {
      return '研究推进明显快于证据加固，当前更适合当作“进展中理论”，不适合当作“已成立理论”。';
    }
    if (confidenceRatio >= 0.6) {
      return '当前研究推进与证据强度相对接近，可以逐步进入更强的外部判伪验证。';
    }
    return '当前项目更像强解释框架，客户端应持续把风险与证据缺口放在主视觉中心。';
  }, [confidenceRatio, progressRatio]);

  const cards = [
    {
      label: '研究推进',
      value: `${toPct(progressRatio)}%`,
      hint: '来自当前总路线与系统收敛口径',
      color: 'text-blue-300',
      icon: TrendingUp,
    },
    {
      label: '严格可信度',
      value: `${toPct(confidenceRatio)}%`,
      hint: '来自 stage83 理论证据审查',
      color: confidenceRatio >= 0.6 ? 'text-emerald-300' : 'text-amber-300',
      icon: CheckCircle2,
    },
    {
      label: '时间线覆盖',
      value: `${routeCount} / ${totalTests}`,
      hint: '路线数 / 测试记录数',
      color: 'text-cyan-300',
      icon: Clock3,
    },
    {
      label: '风险标记',
      value: String(riskFlags),
      hint: '自构造判伪 / 脆弱最优律 / 口径不一致',
      color: riskFlags > 0 ? 'text-rose-300' : 'text-emerald-300',
      icon: AlertTriangle,
    },
  ];

  return (
    <div className="rounded-xl border border-white/10 bg-zinc-900/40 p-4">
      <div className="flex items-center justify-between gap-3 mb-4">
        <div>
          <div className="text-xs text-zinc-500 mb-1">研究驾驶舱总览</div>
          <div className="text-sm font-semibold text-zinc-100">进度 / 可信度双轴</div>
        </div>
        <div className="text-[10px] text-zinc-500">
          目标：把“推进了什么”和“站住了什么”同时显示
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3 mb-4">
        {cards.map((item) => {
          const Icon = item.icon;
          return (
            <div key={item.label} className="rounded-lg border border-white/10 bg-black/20 p-3">
              <div className="flex items-center justify-between mb-2">
                <div className="text-xs text-zinc-400">{item.label}</div>
                <Icon size={13} className={item.color} />
              </div>
              <div className={`text-xl font-bold ${item.color}`}>{item.value}</div>
              <div className="text-[11px] text-zinc-500 mt-1">{item.hint}</div>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="rounded-lg border border-white/10 bg-black/20 p-3">
          <div className="flex items-center justify-between mb-2 text-xs">
            <span className="text-zinc-400">研究推进轴</span>
            <span className="text-blue-300">{toPct(progressRatio)}%</span>
          </div>
          <div className="h-2 rounded bg-black/30 overflow-hidden">
            <div className="h-full bg-blue-500/80" style={{ width: `${toPct(progressRatio)}%` }} />
          </div>
          <div className="text-[11px] text-zinc-500 mt-2">
            当前更偏“研究推进状态”，适合持续显示路线、阶段和结果演化。
          </div>
        </div>

        <div className="rounded-lg border border-white/10 bg-black/20 p-3">
          <div className="flex items-center justify-between mb-2 text-xs">
            <span className="text-zinc-400">证据加固轴</span>
            <span className={confidenceRatio >= 0.6 ? 'text-emerald-300' : 'text-amber-300'}>
              {toPct(confidenceRatio)}%
            </span>
          </div>
          <div className="h-2 rounded bg-black/30 overflow-hidden">
            <div
              className={confidenceRatio >= 0.6 ? 'h-full bg-emerald-500/80' : 'h-full bg-amber-500/80'}
              style={{ width: `${toPct(confidenceRatio)}%` }}
            />
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-zinc-500">
            <div>证据独立性：{toPct(evidenceRatio)}%</div>
            <div>测试强度：{toPct(testStrengthRatio)}%</div>
          </div>
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-white/10 bg-black/20 p-3 text-xs text-zinc-300 leading-6">
        {headline}
      </div>
    </div>
  );
}
