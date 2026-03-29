import { Brain, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { pollRuntimeWithFallback } from './utils/runtimeClient';
import { ProjectRoadmapTab } from './blueprint/ProjectRoadmapTab';
import { LanguageAnalysisTab } from './blueprint/LanguageAnalysisTab';
import { DeepAnalysisTab } from './blueprint/DeepAnalysisTab';
import { ResearchProgressTab } from './blueprint/ResearchProgressTab';
import { SystemStatusTab } from './blueprint/SystemStatusTab';
import {
  PHASES,
  IMPROVEMENTS,
  DNN_ANALYSIS_PLAN,
  EVIDENCE_DRIVEN_PLAN,
  EXECUTION_PLAYBOOK,
  MATH_ROUTE_SYSTEM_PLAN,
} from './blueprint/blueprintConfig';
import { API_BASE, mapLegacyConsciousField, mapRuntimeConsciousField } from './blueprint/blueprintRuntimeUtils';

const BLUEPRINT_TABS = new Set(['roadmap', 'language', 'analysis', 'progress', 'system']);

export const HLAIBlueprint = ({ onClose, initialTab = 'roadmap' }) => {
  const [activeTab, setActiveTab] = useState(initialTab); // roadmap, progress, system
  const [selectedRouteId, setSelectedRouteId] = useState('fiber_bundle');
  const [timelineRoutes, setTimelineRoutes] = useState([]);
  const [expandedFormulaIdx, setExpandedFormulaIdx] = useState(null);
  const [expandedParam, setExpandedParam] = useState(null);
  const [expandedEngPhase, setExpandedEngPhase] = useState(null);
  const [expandedImprovementPhase, setExpandedImprovementPhase] = useState(IMPROVEMENTS[0]?.id || null);
  const [expandedImprovementTest, setExpandedImprovementTest] = useState(null);
  const [consciousField, setConsciousField] = useState(null);
  const [multimodalSummary, setMultimodalSummary] = useState(null);
  const [multimodalView, setMultimodalView] = useState('multimodal_connector');
  const [multimodalError, setMultimodalError] = useState(null);
  const [runtimeStatusSummary, setRuntimeStatusSummary] = useState(null);
  const runtimeStepRef = useRef(0);

  useEffect(() => {
    setActiveTab(BLUEPRINT_TABS.has(initialTab) ? initialTab : 'roadmap');
  }, [initialTab]);

  // Real-time Consciousness Polling
  useEffect(() => {
    let mounted = true;

    const fetchLegacyConsciousField = async () => {
      const res = await fetch(`${API_BASE}/nfb_ra/unified_conscious_field`);
      if (!res.ok) throw new Error(`legacy conscious field failed: ${res.status}`);
      const data = await res.json();
      if (data?.status !== 'success') throw new Error('legacy conscious field unavailable');
      return mapLegacyConsciousField(data);
    };

    const pollConsciousField = async () => {
      const stepId = runtimeStepRef.current++;
      try {
        const result = await pollRuntimeWithFallback({
          apiBase: API_BASE,
          runRequest: {
            route: 'fiber_bundle',
            analysis_type: 'unified_conscious_field',
            params: { step_id: stepId, noise_scale: 0.4 },
            input_payload: {},
          },
          mapRuntimeEvents: mapRuntimeConsciousField,
          fetchLegacy: fetchLegacyConsciousField,
          eventLimit: 20,
        });
        if (!mounted) return;
        setConsciousField({ ...result.data, source: result.source });
      } catch (err) {
        if (!mounted) return;
        setConsciousField(null);
        console.warn('Unified Conscious Field unreachable.', err);
      }
    };

    pollConsciousField();
    const interval = setInterval(pollConsciousField, 2000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let mounted = true;

    const fetchRuntimeStatusSummary = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/system_status/runtime_summary`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = await res.json();
        if (!mounted || payload?.status !== 'success') return;
        setRuntimeStatusSummary(payload);
      } catch {
        if (!mounted) return;
        setRuntimeStatusSummary(null);
      }
    };

    fetchRuntimeStatusSummary();
    const interval = setInterval(fetchRuntimeStatusSummary, 10000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    const fetchTimelineRoutes = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/experiments/timeline?limit=120`);
        if (!res.ok) return;
        const payload = await res.json();
        if (!mounted || payload?.status !== 'success') return;
        const routes = Array.isArray(payload?.timeline?.routes) ? payload.timeline.routes : [];
        setTimelineRoutes(routes);
      } catch {
        // Keep local defaults when runtime API is unavailable.
      }
    };
    fetchTimelineRoutes();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    const fetchMultimodalSummary = async () => {
      try {
        const res = await fetch(`${API_BASE}/nfb/multimodal/summary`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = await res.json();
        if (!mounted) return;
        if (payload?.status !== 'success') throw new Error('invalid payload');
        setMultimodalSummary(payload);
        setMultimodalError(null);
      } catch (err) {
        if (!mounted) return;
        setMultimodalError(err?.message || 'multimodal summary unavailable');
      }
    };
    fetchMultimodalSummary();
    const interval = setInterval(fetchMultimodalSummary, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);


  useEffect(() => {
    const available = Array.isArray(multimodalSummary?.available_views)
      ? multimodalSummary.available_views
      : [];
    if (available.length > 0 && !available.includes(multimodalView)) {
      setMultimodalView(available[0]);
    }
  }, [multimodalSummary, multimodalView]);

  const baseStatusData = PHASES.find(p => p.id === 'agi_status');
  const statusData = useMemo(() => {
    if (!baseStatusData) return baseStatusData;
    const runtimeModelSummary = runtimeStatusSummary?.model_summary || {};
    const runtimeLanguage = runtimeStatusSummary?.runtime_language || {};
    const phaseaRuntime = runtimeStatusSummary?.phasea_runtime || {};
    const researchOverview = runtimeStatusSummary?.research_overview || {};
    return {
      ...baseStatusData,
      model_summary: {
        ...(baseStatusData.model_summary || {}),
        ...runtimeModelSummary,
      },
      runtime_language: runtimeLanguage,
      phasea_runtime: {
        ...(baseStatusData.phasea_runtime || {}),
        ...phaseaRuntime,
      },
      research_overview: {
        ...(baseStatusData.research_overview || {}),
        ...researchOverview,
      },
    };
  }, [baseStatusData, runtimeStatusSummary]);
  const roadmapData = PHASES.find(p => p.id === 'roadmap');
  const theoryPhase = PHASES.find(p => p.id === 'theory');
  const analysisPhase = PHASES.find(p => p.id === 'analysis');
  const engineeringPhase = PHASES.find(p => p.id === 'engineering');
  const milestonePhase = PHASES.find(p => p.id === 'agi_goal');

  const routeBlueprints = useMemo(
    () => ({
      fiber_bundle: {
        id: 'fiber_bundle',
        title: 'Fiber Bundle',
        subtitle: '几何原生智能路线',
        routeDescription: '以神经纤维丛与几何推理为核心，验证结构化智能的可行性。',
        engineeringProcessDescription:
          '计算流程：输入先映射到底流形进行逻辑定位，再进入纤维记忆检索候选语义；通过联络传输层完成跨束对齐，最后由全局工作空间执行 Top-K 裁决并输出结果。',
        theoryTitle: theoryPhase?.definition?.headline || 'Intelligence = Geometry + Physics',
        theorySummary: theoryPhase?.definition?.summary || '',
        theoryBullets: (theoryPhase?.theory_content || []).slice(0, 4).map((item) => item.title),
        theoryFormulas: [
          {
            title: '神经纤维丛原理 (NFB Principle)',
            formula: 'φ(x) = M ⊗ F',
            detail:
              '把智能状态拆成“逻辑骨架 (底流形)”和“知识内容 (纤维)”的张量积，逻辑稳定、内容可扩展。',
          },
          {
            title: '全局工作空间 (Global Workspace)',
            formula: 'W_G = ∫ (w_i · P_i) dμ',
            detail:
              '将多模块竞争后的有效信息做全局聚合，形成当前时刻的统一意识场与决策上下文。',
          },
          {
            title: '高维全息编码 (SHDC Encoding)',
            formula: '⟨v_i, v_j⟩ ≈ δ_ij',
            detail:
              '利用高维近似正交，让特征编码尽量互不干扰，从而支持高容量、低串扰的知识表示。',
          },
          {
            title: '联络与推理 (Connection Equation)',
            formula: '∇_X s = 0',
            detail:
              '将推理视为语义流形上的平行移动，约束语义在传输中保持一致，减少无关漂移。',
          },
        ],
        engineeringItems: [
          {
            name: 'Base Manifold Controller',
            status: 'done',
            focus: '底流形调度与全局约束',
            detail: '维护逻辑骨架状态，统一管理各子模块输入输出与全局稳定性边界。',
          },
          {
            name: 'Fiber Memory Bank',
            status: 'done',
            focus: '知识纤维写入与检。',
            detail: '负责高维语义纤维存储、索引与按联络条件的快速检索。',
          },
          {
            name: 'Connection Transport Layer',
            status: 'in_progress',
            focus: '跨束信息传输',
            detail: '执行底流形与纤维空间间的并行传输与语义一致性对齐。',
          },
          {
            name: 'Ricci Flow Optimizer',
            status: 'in_progress',
            focus: '流形平滑与冲突修。',
            detail: '在离。在线周期中优化曲率分布，减少推理路径扭曲与幻觉风险。',
          },
          {
            name: 'Global Workspace Arbiter',
            status: 'in_progress',
            focus: '全局工作空间竞争裁决',
            detail: '对多模块候选表征执。Top-K 选择，形成当前时刻统一决策上下文。',
          },
          {
            name: 'Alignment & Surgery Interface',
            status: 'done',
            focus: '可交互价值对。',
            detail: '通过流形手术接口对语义方向进行可控干预，支持偏差修复与对齐验证。',
          },
        ],
        nfbtProcessSteps: [
          { step: '1. 邻域图', input: 'X[N,D]', output: '邻居索引', complexity: 'O(N^2D)', op: '距离计算 / 近邻搜索' },
          { step: '2. 局部坐标', input: '邻居点', output: 'basis[d,D]', complexity: 'O(kDd)', op: '局部SVD / 随机SVD' },
          { step: '3. 度量张量', input: 'coords', output: 'g[d,d]', complexity: 'O(kd^2)', op: '局部协方差与正则化' },
          { step: '4. 联络', input: 'g, dg', output: 'Γ[d,d,d]', complexity: 'O(d^3)', op: '偏导组合与指标变换' },
          { step: '5. 曲率', input: 'Γ', output: 'R[d,d,d,d]', complexity: 'O(d^4)', op: '张量收缩与对称化' },
          { step: '6. 平行移动', input: 'Γ, v, dx', output: 'v_new', complexity: 'O(d^2)', op: '联络驱动向量更新' },
          { step: '7. Ricci Flow', input: 'R, X', output: 'X_new', complexity: 'O(T*n*d^4)', op: '离散演化迭代' },
        ],
        nfbtOptimization:
          '关键优化：d << D（如 d=4, D=128），将核心几何计算从 O(D^4) 降到 O(d^4)，并结合近似kNN、截断SVD与曲率张量对称约简降低总成本。',
        milestoneTitle: '里程碑（原 AGI 终点）',
        milestoneGoals: milestonePhase?.goals || [],
        milestoneMetrics: milestonePhase?.metrics || {},
        milestoneStages: [
          {
            id: 'prototype',
            name: '原型阶段',
            status: 'done',
            featurePoints: [
              '完成 FiberNet 核心逻辑层与底流形建。',
              '打。NFB 几何编码与基础推理链路',
              '建立最小可用结构分析工具链（Logit Lens/TDA。',
            ],
            tests: [
              {
                name: 'Z113 閫昏緫闂寘楠岃瘉',
                params: 'layers=12, d=4, D=128, optimizer=adamw, lr=1e-3',
                dataset: 'Z113 模运算合成数据集',
                result: '准确。99.4%，可恢复稳定环面结构',
                summary: '证明原型具备几何逻辑骨架，不是纯统计拟合。',
              },
              {
                name: '基础拓扑可解释性测。',
                params: 'topk_heads=8, tda_threshold=0.1',
                dataset: '内部语义提示词基准集 v1',
                result: '关键层拓扑特征可稳定复现',
                summary: '原型阶段已具备可观测、可解释的结构分析能力。',
              },
            ],
          },
          {
            id: 'scale',
            name: '规模化阶段',
            status: 'in_progress',
            featurePoints: [
              '完成参数规模 × 数据规模的系统化训练矩阵验证（full preset）',
              '完成 8.5M 大模型专项调参（warmup + grad accumulation）并恢复收敛',
              '完成 5-seed 大规模稳定性复现实验，沉淀统计报告与基准文件',
              '完成 d_100k 低资源长程训练对照（36 epochs），确认当前瓶颈主要来自数据规模而非训练轮次',
            ],
            tests: [
              {
                name: 'Full Matrix 基线测试（16 runs）',
                params: 'preset=full, epochs=12, batch=256, eval_batch=2048, device=cuda',
                dataset: 'Modular Addition 合成集：d_100k/d_300k/d_700k/d_1200k',
                result: '总耗时 18.15 分钟；m_0.4m/m_1.4m/m_3.2m 在中大数据规模可收敛；m_8.5m 在默认超参下失稳（~0.009）。',
                summary: '验证了“参数放大后训练策略敏感性显著增加”，大模型不可直接复用小模型超参。',
              },
              {
                name: 'm_8.5m 专项调参测试（4 runs）',
                params: 'epochs=24, lr=2e-4, weight_decay=0.01, warmup=0.1, min_lr_scale=0.1, grad_accum=2, grad_clip=0.5, dropout=0.0',
                dataset: '同 full 数据规模四档：d_100k/d_300k/d_700k/d_1200k',
                result: 'best_val_acc：0.7984 / 0.9905 / 0.9999 / 1.0000（由默认配置的 ~0.009 全面恢复）。',
                summary: '调参后 8.5M 已具备稳定收敛能力，且随数据规模增加表现持续提升。',
              },
              {
                name: 'm_8.5m 多随机种子稳定性（5 seeds, 20 runs）',
                params: '固定 tuned 配置，seed 组：42 / 314 / 2026 / 4096 / 8192',
                dataset: 'd_100k/d_300k/d_700k/d_1200k',
                result: '均值(best)：0.793640 / 0.990581 / 0.999949 / 1.000000；std：0.004160 / 0.000616 / 0.000042 / 0.000000',
                summary: '300k+ 数据规模下结果稳定且高分，100k 档位仍存在数据瓶颈（约 0.79 上限）。',
              },
              {
                name: 'm_8.5m 低资源长程训练（3 seeds, d_100k, epochs=36）',
                params: '同 tuned 配置，epochs 从 24 提升到 36；seed：10001 / 20002 / 30003',
                dataset: 'd_100k',
                result: 'best_val_acc：0.794689 / 0.788311 / 0.791244；mean=0.791415，std=0.002607',
                summary: '与 epochs=24 的 d_100k 结果相比无显著提升，说明低资源场景应优先补充数据或引入更强正则与数据增强策略。',
              },
              {
                name: 'WikiText 几何涌现 (Phase 3)',
                params: '20M Params, Split Stream',
                dataset: 'WikiText-2 (10M Tokens)',
                result: 'Loss: 0.529, ID Peak: 31.5',
                summary: '时间: 2026-02-19 | 数据: Ep 1-47 | 分析: 观测到完整的 ID 压缩(10.5)->膨胀(31.5)->微调(29.3) 呼吸周期。 | 结论: 验证了 SHMC 理论中流形动态重组的物理机制。',
              },
            ],
          },
          {
            id: 'agi',
            name: 'AGI阶段',
            status: 'planned',
            featurePoints: [
              '构建统一意识裁决中心（多路线仲裁。',
              '实现具身控制闭环与安全对齐机。',
              '完成跨模型迁移与长期自治学习框架',
            ],
            tests: [
              {
                name: '全局工作空间端到端压。',
                params: 'modules>=7, arbitration=Top-K, latency<200ms',
                dataset: 'Multi-Agent Conflict Suite',
                result: '待执行',
                summary: '用于验证复杂冲突场景下的稳定裁决能力。',
              },
              {
                name: '具身控制闭环测试',
                params: 'control_horizon=128, safety_guard=on',
                dataset: 'Embodied Interaction Set',
                result: '待执行',
                summary: '用于验证感知-决策-行动闭环的一致性与安全边界。',
              },
            ],
          },
        ],
        milestonePlanEvaluation: {
          assessment:
            '里程碑已从“功能演示”升级为“规模化证据链”：完成 full 矩阵、专项调参与多 seed 复现，证明大模型可训练性与稳定性。',
          suggestions: [

            '将每阶段验收门槛量化（准确率、稳定性、时延、成本）。',
            '规模化阶段增加故障注入与恢复时间指标（MTTR）。',
            'Phase 4: TinyStories 规模化结晶实验 (100M Params) 正在运行 (Batch 500+, ID~12.0)。',

            '将规模化阶段验收门槛固定为：mean/std、训练耗时、吞吐、失败率四项硬指标。',
            '补充 OOD 与噪声扰动测试，验证高分是否可迁移而非数据内记忆。',
            '针对 d_100k 低资源场景继续优化（更长训练、正则与学习率策略），形成小数据稳态方案。',
          ],
        },
      },
      transformer_baseline: {
        id: 'transformer_baseline',
        title: 'Transformer Baseline',
        subtitle: '标准深度网络路线',
        routeDescription: '。Transformer 标准范式建立可复现基线，沉淀稳定分析工具链。',
        engineeringProcessDescription:
          '计算流程：token 嵌入后经过多。Attention+MLP 堆叠，利用残差流聚合上下文，再由输出层完成概率分布与结果生成。',
        theoryTitle: 'Statistical Scaling + Circuit Discovery',
        theorySummary:
          '。Transformer 规模定律和可解释性工具链为核心，优先验证“结构分析能力”与“可复现实验结论”。',
        theoryBullets: [
          '利用 Logit Lens / TDA 形成层级证据。',
          '在相同任务上。Fiber 路线。A/B 评测',
          '优先提升稳定性与工程可维护。',
        ],
        theoryFormulas: [],
        engineeringItems: [
          { name: '数据与任务基线', status: 'done', focus: '统一评测集与日志规范' },
          { name: '可解释性探针', status: 'in_progress', focus: '激活、注意力、回路追踪' },
          { name: '可行性评估闭环', status: 'in_progress', focus: '周报 + 时间线 + 失败归因' },
        ],
        nfbtProcessSteps: [],
        nfbtOptimization: '',
        milestoneTitle: '里程碑（原 AGI 终点）',
        milestoneGoals: [
          '形成可持续复现的深度网络分析基线',
          '对关键任务达到稳定可解释的效果上。',
          '沉淀可迁移到其他路线的通用工具。',
        ],
        milestoneMetrics: { Priority: '可复现性', Horizon: '近中期' },
        milestoneStages: [
          {
            id: 'prototype',
            name: '原型阶段',
            status: 'done',
            featurePoints: [
              '建立 Transformer 统一实验入口与推理日志格。',
              '完成基础 attention/mlp/topology 分析联。',
              '搭建可复现基准任务与版本化数据管。',
            ],
            tests: [
              {
                name: '基础推理链路连通测。',
                params: 'model=gpt2-small, max_len=128, seed=42',
                dataset: 'Prompt Regression Set v1',
                result: '核心接口通过，输出稳。',
                summary: '确认基线系统可持续复现实验流程。',
              },
              {
                name: '注意力结构可解释性测。',
                params: 'heads=all, layer_range=0-11',
                dataset: 'Interpretability Probe Set',
                result: '关键头部激活可视化可复。',
                summary: '为后续跨路线对照提供统一解释基线。',
              },
            ],
          },
          {
            id: 'scale',
            name: '规模化阶。',
            status: 'in_progress',
            featurePoints: [
              '扩展多任务评测矩阵与失败归因统计',
              '接入时间。周报治理与自动导。',
              '优化长序列时延和显存占用',
            ],
            tests: [
              {
                name: '多任务回归压。',
                params: 'tasks=12, batch=16, seq_len=512',
                dataset: 'Mixed Reasoning Benchmark',
                result: '任务间性能有波动，整体可控',
                summary: '规模化可行，但需继续收敛稳定性指标。',
              },
              {
                name: '失败模式聚合测试',
                params: 'window=30d, top_failures=8',
                dataset: 'Experiment Timeline JSON',
                result: '可稳定提取高频失败原。',
                summary: '治理闭环有效，支持后续针对性修复。',
              },
            ],
          },
          {
            id: 'agi',
            name: 'AGI阶段',
            status: 'planned',
            featurePoints: [
              '与几何路线形成长期协同基。',
              '支持跨模型迁移评测与策略切换',
              '完善安全对齐和回退控制策略',
            ],
            tests: [
              {
                name: '跨模型迁移验。',
                params: 'source=gpt2, target=qwen, adapter=on',
                dataset: 'Cross-Model Transfer Set',
                result: '待执行',
                summary: '用于评估基线能力的可迁移上限。',
              },
              {
                name: '安全约束回退测试',
                params: 'safety_guard=strict, rollback=enabled',
                dataset: 'Safety Red Team Set',
                result: '待执行',
                summary: '用于验证异常场景下的可控与可恢复性。',
              },
            ],
          },
        ],
        milestonePlanEvaluation: {
          assessment:
            '路线定位清晰，适合作为可复现实验基线与对照组。',
          suggestions: [
            '增加统一时延/成本指标，避免仅看准确率。',
            '对失败原因设置优先级并建立修复SLA。',
            '提前定义跨模型迁移的验收阈值。',
          ],
        },
      },
      hybrid_workspace: {
        id: 'hybrid_workspace',
        title: 'Hybrid Workspace',
        subtitle: '全局工作空间混合路线',
        routeDescription: '以全局工作空间整合多路线输出，提升跨模块协同与鲁棒性。',
        engineeringProcessDescription:
          '计算流程：不同路线并行产出候选表征，统一映射到共享工作空间后执行冲突消解与优先级仲裁，最终输出融合结果并回写路线状态。',
        theoryTitle: 'Global Workspace + Route Ensemble',
        theorySummary:
          '将多路线输出映射到统一工作空间，通过竞争与仲裁机制在稳定性和泛化能力之间寻找最优平衡。',
        theoryBullets: [
          '跨路线共享状态空间与指标协议',
          'Top-K 竞争裁决不同模块候选输。',
          '结合失败模式实现动态路由调。',
        ],
        theoryFormulas: [],
        engineeringItems: [
          { name: '统一路由协议', status: 'done', focus: 'route + analysis_type + summary' },
          { name: '全局仲裁器', status: 'in_progress', focus: '冲突管理与优先级决策' },
          { name: '在线自适应调度', status: 'pending', focus: '基于历史可行性自动选路' },
        ],
        nfbtProcessSteps: [],
        nfbtOptimization: '',
        milestoneTitle: '里程碑（原 AGI 终点）',
        milestoneGoals: [
          '实现多路线协同下的稳定推理输。',
          '将失败恢复时间降低到分钟。',
          '完成面向长期 AGI 研究的统一实验操作系统',
        ],
        milestoneMetrics: { Priority: '协同能力', Horizon: '中长期' },
        milestoneStages: [
          {
            id: 'prototype',
            name: '原型阶段',
            status: 'done',
            featurePoints: [
              '完成多路线统一协议与事件流标准。',
              '实现基础仲裁器与结果融合接口',
              '建立路线状态与评分映射机制',
            ],
            tests: [
              {
                name: '双路线仲裁连通测。',
                params: 'routes=2, arbitration=topk(1), timeout=2s',
                dataset: 'Route Arbitration Smoke Set',
                result: '仲裁链路可稳定执。',
                summary: '确认混合路由原型可用。',
              },
              {
                name: '融合输出一致性测。',
                params: 'fusion=weighted, weights=static',
                dataset: 'Consistency Benchmark v1',
                result: '一致性优于单路线平均。',
                summary: '融合策略有效，但动态权重仍需优化。',
              },
            ],
          },
          {
            id: 'scale',
            name: '规模化阶。',
            status: 'in_progress',
            featurePoints: [
              '扩展到多路线并行调度',
              '引入失败快速恢复与自动降级策略',
              '建立跨路线趋势分析与周报治理',
            ],
            tests: [
              {
                name: '多路线并发压。',
                params: 'routes=5, concurrent_runs=20',
                dataset: 'Concurrent Routing Stress Set',
                result: '高并发下存在尾延。',
                summary: '需优化队列调度与资源隔离。',
              },
              {
                name: '故障注入恢复测试',
                params: 'failure_rate=0.2, fallback=enabled',
                dataset: 'Failure Injection Set',
                result: '大部分场景可自动恢复',
                summary: '恢复机制有效，需进一步缩。MTTR。',
              },
            ],
          },
          {
            id: 'agi',
            name: 'AGI阶段',
            status: 'planned',
            featurePoints: [
              '形成具备自适应选路能力的统一工作空间',
              '支持长期任务下的策略演化与记忆整。',
              '建立可审计的安全治理与人工接管机。',
            ],
            tests: [
              {
                name: '自适应选路策略验证',
                params: 'policy=bandit, reward=feasibility_score',
                dataset: 'Long-Horizon Route Selection Set',
                result: '待执行',
                summary: '用于验证长期任务中的选路收敛能力。',
              },
              {
                name: '全链路安全审计测。',
                params: 'audit=full, intervention=manual+auto',
                dataset: 'Governance Compliance Set',
                result: '待执行',
                summary: '用于评估可审计性与可控性是否达标。',
              },
            ],
          },
        ],
        milestonePlanEvaluation: {
          assessment:
            '具备成为“路线操作系统”的潜力，关键在于稳定调度与治理可控性。',
          suggestions: [
            '优先优化并发场景下尾延迟与资源竞争问题。',
            '将MTTR纳入核心KPI并周度跟踪。',
            '提前定义人工接管触发条件与回退策略。',
          ],
        },
      },
    }),
    [engineeringPhase?.sub_phases, milestonePhase?.goals, milestonePhase?.metrics, theoryPhase?.definition?.headline, theoryPhase?.definition?.summary, theoryPhase?.theory_content]
  );

  const routeList = useMemo(() => {
    const runtimeIds = timelineRoutes
      .map((item) => item?.route)
      .filter((id) => typeof id === 'string' && id.length > 0);
    const baseIds = Object.keys(routeBlueprints);
    const allIds = Array.from(new Set([...baseIds, ...runtimeIds]));

    return allIds.map((id) => {
      const base = routeBlueprints[id] || {
        id,
        title: id,
        subtitle: '实验路线',
        routeDescription: '该路线正在构建中，描述信息待补充。',
        engineeringProcessDescription: '计算流程说明待补充。',
        theoryTitle: '寰呰ˉ鍏呯悊璁?',
        theorySummary: '该路线尚未配置详细理论描述。',
        theoryBullets: [],
        theoryFormulas: [],
        engineeringItems: [],
        nfbtProcessSteps: [],
        nfbtOptimization: '',
        milestoneTitle: '里程碑目标（。AGI 终点。',
        milestoneGoals: [],
        milestoneMetrics: {},
        milestoneStages: [],
        milestonePlanEvaluation: null,
      };
      const runtime = timelineRoutes.find((item) => item?.route === id);
      const stats = runtime?.stats || {};
      const totalRuns = Number(stats.total_runs || 0);
      const completedRuns = Number(stats.completed_runs || 0);
      const avgScore = Number(stats.avg_score || 0);
      const routeProgress =
        totalRuns > 0
          ? Math.max(
            0,
            Math.min(100, Math.round((completedRuns / Math.max(1, totalRuns)) * 60 + avgScore * 40))
          )
          : 0;
      return {
        ...base,
        stats: {
          totalRuns,
          completedRuns,
          failedRuns: Number(stats.failed_runs || 0),
          avgScore,
          routeProgress,
        },
      };
    });
  }, [routeBlueprints, timelineRoutes]);

  useEffect(() => {
    if (routeList.length === 0) return;
    if (!routeList.some((item) => item.id === selectedRouteId)) {
      setSelectedRouteId(routeList[0].id);
    }
  }, [routeList, selectedRouteId]);

  useEffect(() => {
    setExpandedFormulaIdx(null);
    setExpandedEngPhase(null);
    setExpandedParam(null);
  }, [selectedRouteId]);

  const selectedRoute = routeList.find((item) => item.id === selectedRouteId) || routeList[0];
  const systemRouteOptions = routeList.filter((item) =>
    ['fiber_bundle', 'transformer_baseline', 'hybrid_workspace'].includes(item.id)
  );
  const selectedMultimodalData = multimodalSummary?.views?.[multimodalView] || null;
  const selectedMultimodalReport = selectedMultimodalData?.report || null;
  const selectedMultimodalBest = selectedMultimodalReport?.summary?.best || null;
  const selectedMultimodalLatest = selectedMultimodalData?.latest_test || null;

  const multimodalMetricRows = useMemo(() => {
    if (!selectedMultimodalBest) return [];
    if (multimodalView === 'vision_alignment') {
      return [
        { label: '最佳轮次', value: selectedMultimodalBest.epoch },
        { label: 'Val Accuracy', value: Number(selectedMultimodalBest.val_acc || 0).toFixed(4) },
        { label: 'Anchor Cos', value: Number(selectedMultimodalBest.val_anchor_cos || 0).toFixed(4) },
        { label: 'Val Loss', value: Number(selectedMultimodalBest.val_loss || 0).toFixed(4) },
      ];
    }
    return [
      { label: '最佳轮次', value: selectedMultimodalBest.epoch },
      { label: 'Val Fused Acc', value: Number(selectedMultimodalBest.val_fused_acc || 0).toFixed(4) },
      { label: 'Retrieval@1', value: Number(selectedMultimodalBest.val_retrieval_top1 || 0).toFixed(4) },
      { label: 'Align Cos', value: Number(selectedMultimodalBest.val_alignment_cos || 0).toFixed(4) },
    ];
  }, [selectedMultimodalBest, multimodalView]);

  const getRouteImpl = (capability) => {
    const map = capability?.implementation_by_route || {};
    return (
      map[selectedRouteId] ||
      map[selectedRoute?.id] ||
      capability?.desc ||
      '该路线实现描述待补充。'
    );
  };

  const systemProfiles = useMemo(
    () => ({
      fiber_bundle: {
        metricCards: [
          {
            label: '内稳态调。',
            brain_ability: '稳态维持与资源分配',
            value: consciousField ? `${((consciousField.stability || 0) * 100).toFixed(1)}%` : '92.0%',
            color: '#10b981',
          },
          {
            label: '工作记忆负载',
            brain_ability: '短时记忆与上下文保持',
            value: consciousField ? `${consciousField.memory_load || 0}%` : '68%',
            color: '#00d2ff',
          },
          {
            label: '跨域共振',
            brain_ability: '跨模态联想整。',
            value: consciousField ? (consciousField.resonance || 0).toFixed(3) : '0.742',
            color: '#ffaa00',
          },
          {
            label: '意识竞争强度',
            brain_ability: '注意焦点竞争与广。',
            value: consciousField ? (consciousField.gws_intensity || 0).toFixed(2) : '0.81',
            color: '#a855f7',
          },
        ],
        parameterCards: [
          {
            name: '几何潜空间配。',
            brain_ability: '抽象结构建模',
            route_param: 'd=4, D=128, manifold=riemannian',
            detail: '低维几何。+ 高维语义外壳',
            desc: '通过 d<<D 降低核心几何计算复杂度，同时保持语义表达容量。',
            value_meaning: '兼顾可解释性、稳定性与计算成本。',
            why_important: '决定几何推理是否可持续扩展。',
          },
          {
            name: '联络与平行移。',
            brain_ability: '推理路径保持',
            route_param: 'transport=connection_based, step=adaptive',
            detail: 'Γ 驱动语义平移',
            desc: '沿流形执行平行移动，减少语义漂移。',
            value_meaning: '推理链更稳定，抗扰动能力更强。',
            why_important: '是从“拟合”走向“结构推理”的关键。'
          },
          {
            name: 'Ricci Flow 婕斿寲',
            brain_ability: '睡眠重整与冲突修。',
            route_param: 'iterations=100, reg=1e-3',
            detail: '离线曲率平滑',
            desc: '通过流形平滑降低逻辑尖峰与幻觉风险。',
            value_meaning: '提升长期稳定性与一致性。',
            why_important: '支持系统持续自我修复。'
          },
          {
            name: '全局工作空间',
            brain_ability: '意识竞争裁决',
            route_param: 'top_k=8, arbitration=winner_take_all',
            detail: '多模块竞争广。',
            desc: '在冲突候选中选取最优表示并广播。',
            value_meaning: '保证实时决策聚焦有效信息。',
            why_important: '直接影响系统响应质量与时延。'
          },
        ],
        validationRecords: (statusData?.passed_tests || []).map((t) => ({
          ...t,
          brain_ability: t.brain_ability || '结构推理稳定与记忆重。',
          route_param_focus: t.route_param_focus || 'manifold_dim=4, top_k=8, ricci_iterations=100',
        })),
      },
      transformer_baseline: {
        metricCards: [
          { label: '上下文保持', brain_ability: '工作记忆', value: '74%', color: '#10b981' },
          { label: '模式泛化', brain_ability: '经验迁移', value: '0.69', color: '#00d2ff' },
          { label: '可解释覆盖', brain_ability: '自我监控', value: '83%', color: '#ffaa00' },
          { label: '推理稳定性', brain_ability: '执行控制', value: '0.71', color: '#a855f7' },
        ],
        parameterCards: [
          {
            name: '模型与序列配。',
            brain_ability: '语言工作记忆',
            route_param: 'model=gpt2-small, seq_len=512, batch=16',
            detail: '标准 Transformer 主干',
            desc: '通过标准注意力机制建模上下文依赖。',
            value_meaning: '易复现、生态成熟。',
            why_important: '是对照路线的基础能力锚点。',
          },
          {
            name: '可解释性探。',
            brain_ability: '内省与自检',
            route_param: 'logit_lens=on, tda=on, head_probe=full',
            detail: '多探针并。',
            desc: '输出层到中间层的解释链路可追踪。',
            value_meaning: '便于定位错误来源。',
            why_important: '支撑实验可解释与回归分析。',
          },
          {
            name: '训练稳定性策。',
            brain_ability: '执行控制与抑。',
            route_param: 'lr=1e-4, warmup=2k, clip=1.0',
            detail: '标准优化管线',
            desc: '控制梯度震荡与训练发散风险。',
            value_meaning: '提升训练一致性。',
            why_important: '影响规模化阶段可持续性。',
          },
          {
            name: 'RAG 鎵╁睍',
            brain_ability: '长期知识提取',
            route_param: 'retriever=faiss, topk=5, rerank=on',
            detail: '外部知识增强',
            desc: '通过检索补偿参数内知识不足。',
            value_meaning: '提高事实一致性。',
            why_important: '降低知识时效衰减。'
          },
        ],
        validationRecords: [
          {
            name: '多任务基线回。',
            date: '2026-02-16',
            result: 'PASS',
            brain_ability: '工作记忆稳定与任务切换',
            route_param_focus: 'seq_len=512, batch=16, clip=1.0',
            target: '验证标准路线在多任务上的稳定性能。',
            process: '统一任务集批量回。+ 误差曲线对齐。',
            significance: '确认基线可作为长期对照参照系。'
          },
          {
            name: '解释链完整性测。',
            date: '2026-02-16',
            result: 'PASS',
            brain_ability: '内省监控与因果追踪',
            route_param_focus: 'logit_lens=on, tda=on, head_probe=full',
            target: '验证从激活到输出的可追溯性。',
            process: 'Logit Lens + 头部归因联合验证。',
            significance: '确保分析结论具备可复检性。'
          },
        ],
      },
      hybrid_workspace: {
        metricCards: [
          { label: '跨路线一致性', brain_ability: '跨脑区整合', value: '0.76', color: '#10b981' },
          { label: '仲裁收敛速度', brain_ability: '注意竞争调度', value: '148ms', color: '#00d2ff' },
          { label: '故障恢复能力', brain_ability: '稳态恢复', value: 'MTTR 2.6m', color: '#ffaa00' },
          { label: '融合收益', brain_ability: '多通道协同', value: '+8.9%', color: '#a855f7' },
        ],
        parameterCards: [
          {
            name: '仲裁器参。',
            brain_ability: '鍐茬獊鍐崇瓥',
            route_param: 'routes=5, top_k=2, timeout=200ms',
            detail: '多路线竞争框。',
            desc: '基于评分和稳定性进行候选筛选。',
            value_meaning: '平衡质量与时延。',
            why_important: '决定融合输出可用性。',
          },
          {
            name: '融合策略参数',
            brain_ability: '多源信息整合',
            route_param: 'fusion=weighted, dynamic_weight=on',
            detail: '动态权重融。',
            desc: '根据路线可靠度动态调节贡献。',
            value_meaning: '减少单路线失效影响。',
            why_important: '提高鲁棒性与连续性。',
          },
          {
            name: '鎭㈠涓庨檷绾?',
            brain_ability: '异常处理',
            route_param: 'fallback=enabled, retry=2, degrade=graceful',
            detail: '故障容错机制',
            desc: '出现异常时自动切换到安全路径。',
            value_meaning: '避免全链路中断。',
            why_important: '保障系统稳定运行。'
          },
          {
            name: '治理追踪参数',
            brain_ability: '长期自我监督',
            route_param: 'timeline=on, weekly_report=on',
            detail: '全链路可审计',
            desc: '持续记录决策证据与失败归因。',
            value_meaning: '支持迭代改进闭环。',
            why_important: '提升工程治理效率。'
          },
        ],
        validationRecords: [
          {
            name: '多路线并发仲裁测。',
            date: '2026-02-17',
            result: 'PASS',
            brain_ability: '冲突决策与全局调度',
            route_param_focus: 'routes=5, top_k=2, timeout=200ms',
            target: '验证多路线并发下仲裁稳定性。',
            process: '构造冲突任务，统计收敛时间与一致性。',
            significance: '确认仲裁机制在复杂场景可用。'
          },
          {
            name: '故障注入恢复测试',
            date: '2026-02-17',
            result: 'PASS',
            brain_ability: '稳态恢复与容错控制',
            route_param_focus: 'fallback=enabled, retry=2, degrade=graceful',
            target: '验证路由失败时自动降级能力。',
            process: '注入随机失败，观。MTTR 与回退质量。',
            significance: '证明混合路线具备工程级容错能力。'
          },
        ],
      },
    }),
    [consciousField, selectedRouteId, statusData?.passed_tests]
  );

  const activeSystemProfile = systemProfiles[selectedRouteId] || systemProfiles.fiber_bundle;
  const mergedMilestoneStages = useMemo(() => {
    const baseStages = selectedRoute?.milestoneStages || [];
    const routeTests = activeSystemProfile?.validationRecords || [];
    if (!routeTests.length) return baseStages;

    const routeValidationStage = {
      id: 'route_validation',
      name: '路线测试记录',
      status: routeTests.every((t) => String(t?.result || '').toUpperCase().includes('PASS')) ? 'done' : 'in_progress',
      featurePoints: [
        `来源：系统状态 / ${selectedRoute?.title || selectedRouteId}`,
        `测试数量：${routeTests.length}`,
        '作为里程碑验收证据沉淀到研发进展',
      ],
      tests: routeTests.map((t) => ({
        name: t.name || '未命名测试',
        params: t.route_param_focus || t.params || '-',
        dataset: t.dataset || (t.date ? `验证日期: ${t.date}` : '-'),
        result: t.result || '-',
        summary: t.significance || t.summary || '-',
      })),
    };

    return [...baseStages, routeValidationStage];
  }, [selectedRoute, selectedRouteId, activeSystemProfile]);
  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh',
      backgroundColor: 'rgba(5, 5, 10, 0.98)', backdropFilter: 'blur(30px)', zIndex: 2000,
      display: 'flex', flexDirection: 'column', color: '#fff',
      fontFamily: '"SF Mono", "Roboto Mono", monospace', overflow: 'hidden'
    }}>
      {/* Custom Keyframes */}
      <style>{`
        @keyframes roadmapFade { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes brainPulse { from { scale: 0.95; opacity: 0.8; } to { scale: 1.05; opacity: 1; } }
        @keyframes brainRotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes brainRotateReverse { from { transform: rotate(0deg); } to { transform: rotate(-360deg); } }
        @keyframes synapsePulse { 0%, 100% { opacity: 0.3; scale: 0.8; } 50% { opacity: 1; scale: 1.2; } }
      `}</style>

      {/* Top Header / Navigation */}
      <div style={{
        padding: '0 40px', height: '80px', display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.1)', background: 'rgba(0,0,0,0.3)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '50px', height: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Brain size={28} color="#00d2ff" />
            <span style={{ fontSize: '18px', fontWeight: 'bold', letterSpacing: '2px' }}>理论分析</span>
          </div>

          <nav style={{ display: 'flex', gap: '10px', height: '100%', overflowX: 'auto', scrollbarWidth: 'thin' }}>
            {[
              { id: 'roadmap', label: '项目大纲' },
              { id: 'language', label: '语言分析' },
              { id: 'analysis', label: '智能理论' },
              { id: 'progress', label: '模型研发' },
              { id: 'system', label: '系统状态' },
            ].map(t => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                style={{
                  background: 'transparent', border: 'none', color: activeTab === t.id ? '#00d2ff' : '#666',
                  fontSize: '15px', fontWeight: 'bold', cursor: 'pointer', padding: '0 25px',
                  borderBottom: activeTab === t.id ? '3px solid #00d2ff' : '3px solid transparent',
                  transition: 'all 0.3s', height: '100%'
                }}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button onClick={onClose} style={{
            background: 'rgba(255,255,255,0.05)', border: 'none', color: '#fff', cursor: 'pointer',
            width: '40px', height: '40px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center'
          }} onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,100,100,0.2)'} onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}>
            <X size={22} />
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* Sub-Sidebar for Research Progress */}
        {activeTab === 'progress' && (
          <div style={{
            width: '280px', borderRight: '1px solid rgba(255,255,255,0.1)',
            padding: '30px 20px', background: 'rgba(0,0,0,0.2)', overflowY: 'auto',
            position: 'relative'
          }}>
            <div style={{ fontSize: '10px', color: '#444', textTransform: 'uppercase', marginBottom: '30px', letterSpacing: '2px', fontWeight: 'bold' }}>Research Routes</div>

            {/* Vertical Timeline Line */}
            <div style={{
              position: 'absolute', left: '38px', top: '80px', bottom: '40px',
              width: '1px', background: 'linear-gradient(to bottom, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%)',
              zIndex: 0
            }} />

            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', position: 'relative', zIndex: 1 }}>
              {routeList.map((routeItem) => (
                <div key={routeItem.id} style={{ position: 'relative' }}>
                  {/* Timeline Dot */}
                  <div style={{
                    position: 'absolute', left: '15px', top: '22px',
                    width: '6px', height: '6px', borderRadius: '50%',
                    background: selectedRoute?.id === routeItem.id ? '#00d2ff' : '#222',
                    border: `2px solid ${selectedRoute?.id === routeItem.id ? '#000' : 'rgba(255,255,255,0.1)'}`,
                    boxShadow: selectedRoute?.id === routeItem.id ? '0 0 10px #00d2ff' : 'none',
                    transition: 'all 0.3s'
                  }} />

                  <button
                    onClick={() => setSelectedRouteId(routeItem.id)}
                    style={{
                      width: '100%', padding: '12px 12px 12px 45px', borderRadius: '14px',
                      textAlign: 'left', cursor: 'pointer',
                      background: selectedRoute?.id === routeItem.id ? 'rgba(255,255,255,0.03)' : 'transparent',
                      border: 'none',
                      color: selectedRoute?.id === routeItem.id ? '#fff' : '#666', transition: 'all 0.3s',
                      display: 'flex', flexDirection: 'column', gap: '4px'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                      <span style={{ fontSize: '14px', fontWeight: 'bold', color: selectedRoute?.id === routeItem.id ? '#fff' : '#888' }}>
                        {routeItem.title}
                      </span>
                      <span style={{
                        fontSize: '11px', fontFamily: 'monospace', fontWeight: 'bold',
                        color: selectedRoute?.id === routeItem.id ? '#00d2ff' : '#444'
                      }}>
                        {routeItem.stats.routeProgress}%
                      </span>
                    </div>
                    <div style={{ fontSize: '10px', color: '#444' }}>
                      run {routeItem.stats.totalRuns} | success {routeItem.stats.completedRuns} | avg {(routeItem.stats.avgScore * 100).toFixed(1)}%
                    </div>
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Content Details */}
        <div style={{
          flex: 1, padding: '50px 80px', overflowY: 'auto',
          background: 'radial-gradient(circle at 50% 10%, rgba(0, 100, 200, 0.05) 0%, transparent 70%)'
        }}>

          {/* TAB: Project Roadmap */}
          {activeTab === 'roadmap' && (
            <ProjectRoadmapTab
              roadmapData={roadmapData}
              analysisPhase={analysisPhase}
              evidenceDrivenPlan={EVIDENCE_DRIVEN_PLAN}
              executionPlaybook={EXECUTION_PLAYBOOK}
              mathRouteSystemPlan={MATH_ROUTE_SYSTEM_PLAN}
              improvements={IMPROVEMENTS}
              expandedImprovementPhase={expandedImprovementPhase}
              setExpandedImprovementPhase={setExpandedImprovementPhase}
              expandedImprovementTest={expandedImprovementTest}
              setExpandedImprovementTest={setExpandedImprovementTest}
            />
          )}

          {/* TAB: Language Analysis */}
          {activeTab === 'language' && (
            <LanguageAnalysisTab />
          )}

          {/* TAB: Deep Analysis / Model Comparison */}
          {activeTab === 'analysis' && (
            <DeepAnalysisTab
              evidenceDrivenPlan={EVIDENCE_DRIVEN_PLAN}
              improvements={IMPROVEMENTS}
              expandedImprovementPhase={expandedImprovementPhase}
              setExpandedImprovementPhase={setExpandedImprovementPhase}
              expandedImprovementTest={expandedImprovementTest}
              setExpandedImprovementTest={setExpandedImprovementTest}
            />
          )}

          {/* TAB: Research Progress (Route-Centric Command) */}
          {activeTab === 'progress' && selectedRoute && (
            <ResearchProgressTab
              selectedRoute={selectedRoute}
              expandedFormulaIdx={expandedFormulaIdx}
              setExpandedFormulaIdx={setExpandedFormulaIdx}
              dnnAnalysisPlan={DNN_ANALYSIS_PLAN}
              expandedEngPhase={expandedEngPhase}
              setExpandedEngPhase={setExpandedEngPhase}
              mergedMilestoneStages={mergedMilestoneStages}
              multimodalView={multimodalView}
              setMultimodalView={setMultimodalView}
              multimodalError={multimodalError}
              selectedMultimodalReport={selectedMultimodalReport}
              selectedMultimodalData={selectedMultimodalData}
              selectedMultimodalLatest={selectedMultimodalLatest}
              multimodalMetricRows={multimodalMetricRows}
            />
          )}

          {/* TAB: AGI System Status */}
          {activeTab === 'system' && (
            <SystemStatusTab
              consciousField={consciousField}
              systemRouteOptions={systemRouteOptions}
              routeList={routeList}
              setSelectedRouteId={setSelectedRouteId}
              selectedRouteId={selectedRouteId}
              activeSystemProfile={activeSystemProfile}
              statusData={statusData}
              selectedRoute={selectedRoute}
              getRouteImpl={getRouteImpl}
              expandedParam={expandedParam}
              setExpandedParam={setExpandedParam}
            />
          )}

        </div>
      </div>
    </div>
  );
};






