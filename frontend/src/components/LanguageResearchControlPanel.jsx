import { AlertTriangle, Boxes, Brain, ChevronDown, GitBranch, Layers3, Route, ShieldAlert, Sparkles } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

const RESEARCH_LAYERS = [
  {
    id: 'static_encoding',
    label: '静态编码层',
    desc: '看共享基底、薄差分和对象群聚。',
    theoryObject: 'family_patch',
    analysisMode: 'static',
    animationMode: 'family_patch_formation',
  },
  {
    id: 'dynamic_route',
    label: '动态路径层',
    desc: '看任务词、方向词和路径分流。',
    theoryObject: 'relation_context_fiber',
    analysisMode: 'dynamic_prediction',
    animationMode: 'successor_transport',
  },
  {
    id: 'result_recovery',
    label: '结果回收层',
    desc: '看原生闭合、修复闭合和读出桥。',
    theoryObject: 'protocol_bridge',
    analysisMode: 'minimal_circuit',
    animationMode: 'protocol_bridge',
  },
  {
    id: 'propagation_encoding',
    label: '传播编码层',
    desc: '看来源链、跨层接力和保真断裂点。',
    theoryObject: 'stage_conditioned_transport',
    analysisMode: 'cross_layer_transport',
    animationMode: 'cross_layer_relay',
  },
  {
    id: 'semantic_roles',
    label: '语义角色层',
    desc: '看对象、属性、位置、操作、约束、结果的角色组合。',
    theoryObject: 'attribute_fiber',
    analysisMode: 'compositionality',
    animationMode: 'attribute_fiber',
  },
];

const OBJECT_GROUPS = [
  { id: 'fruit', label: '水果' },
  { id: 'animal', label: '动物' },
  { id: 'tool', label: '工具' },
  { id: 'brand', label: '品牌' },
  { id: 'code_file', label: '文件/代码' },
];

const TASK_GROUPS = [
  { id: 'translation', label: '翻译', analysisMode: 'cross_layer_transport' },
  { id: 'modify', label: '修改', analysisMode: 'causal_intervention' },
  { id: 'refactor', label: '重构', analysisMode: 'minimal_circuit' },
  { id: 'long_chain', label: '长链', analysisMode: 'robustness' },
  { id: 'competition', label: '高竞争', analysisMode: 'counterfactual' },
];

const ROLE_GROUPS = [
  { id: 'object', label: '对象' },
  { id: 'attribute', label: '属性' },
  { id: 'position', label: '位置' },
  { id: 'operation', label: '操作' },
  { id: 'constraint', label: '约束' },
  { id: 'result', label: '结果' },
];

const STRUCTURE_OVERLAYS = [
  { id: 'shared_base', label: '共享基底' },
  { id: 'local_delta', label: '局部差分' },
  { id: 'path_amplification', label: '路径放大' },
  { id: 'semantic_roles', label: '语义角色' },
  { id: 'fidelity', label: '来源保真' },
];

const MODEL_OPTIONS = [
  { id: 'gpt2', label: 'GPT-2' },
  { id: 'qwen', label: 'Qwen' },
  { id: 'deepseek', label: 'DeepSeek' },
];

const STAGE_OPTIONS = [
  { id: 'stage257', label: 'Stage257' },
  { id: 'stage258', label: 'Stage258' },
  { id: 'stage260', label: 'Stage260' },
  { id: 'stage262', label: 'Stage262' },
  { id: 'stage265', label: 'Stage265' },
];

const COMPARE_OPTIONS = [
  { id: 'single_model', label: '单模型' },
  { id: 'cross_model', label: '跨模型' },
  { id: 'single_stage', label: '单阶段' },
  { id: 'stage_replay', label: '阶段回放' },
];

const HARD_PROBLEMS = [
  { id: 'fidelity', title: '天然来源保真低', tone: '#ff7a7a' },
  { id: 'competition', title: '同类高竞争边界脆弱', tone: '#ffb86b' },
  { id: 'closure', title: '修复强于原生闭合', tone: '#ffd36b' },
  { id: 'brand', title: '品牌义边界弱', tone: '#ff9fd8' },
  { id: 'cross_model', title: '跨模型硬主核仍少', tone: '#8acbff' },
];

const cardStyle = {
  padding: '12px',
  borderRadius: '10px',
  background: 'rgba(255,255,255,0.04)',
  border: '1px solid rgba(255,255,255,0.08)',
};

const sectionTitleStyle = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  marginBottom: '10px',
  color: '#dfe8ff',
  fontSize: '13px',
  fontWeight: 700,
};

function ChipButton({ active, label, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '7px 10px',
        borderRadius: '999px',
        border: active ? '1px solid rgba(79, 172, 254, 0.65)' : '1px solid rgba(255,255,255,0.1)',
        background: active ? 'rgba(79, 172, 254, 0.16)' : 'rgba(255,255,255,0.02)',
        color: active ? '#dff0ff' : '#9fb0c8',
        cursor: 'pointer',
        fontSize: '11px',
        fontWeight: 600,
        transition: 'all 0.2s ease',
      }}
    >
      {label}
    </button>
  );
}

export default function LanguageResearchControlPanel({ workspace, legacyControls = null }) {
  const [legacyOpen, setLegacyOpen] = useState(false);
  const [fallbackFocus, setFallbackFocus] = useState({
    researchLayer: 'static_encoding',
    objectGroup: 'fruit',
    taskGroup: 'translation',
    roleGroup: 'object',
    modelKey: 'gpt2',
    stageKey: 'stage260',
    compareMode: 'single_model',
    riskFocus: 'fidelity',
    structureOverlays: ['shared_base', 'local_delta', 'path_amplification'],
  });

  const languageFocus = workspace?.languageFocus || fallbackFocus;
  const updateLanguageFocus = (patch) => {
    if (workspace?.setLanguageFocus) {
      workspace.setLanguageFocus((prev) => ({ ...prev, ...patch }));
      return;
    }
    setFallbackFocus((prev) => ({ ...prev, ...patch }));
  };

  const researchLayer = languageFocus.researchLayer;
  const objectGroup = languageFocus.objectGroup;
  const taskGroup = languageFocus.taskGroup;
  const roleGroup = languageFocus.roleGroup;
  const modelKey = languageFocus.modelKey;
  const stageKey = languageFocus.stageKey;
  const compareMode = languageFocus.compareMode;
  const riskFocus = languageFocus.riskFocus;
  const overlays = Array.isArray(languageFocus.structureOverlays) ? languageFocus.structureOverlays : [];

  const currentLayerMeta = useMemo(
    () => RESEARCH_LAYERS.find((item) => item.id === researchLayer) || RESEARCH_LAYERS[0],
    [researchLayer]
  );

  useEffect(() => {
    if (!workspace) return;

    if (currentLayerMeta.theoryObject && workspace.setTheoryObject) {
      workspace.setTheoryObject(currentLayerMeta.theoryObject);
    }

    if (currentLayerMeta.analysisMode && workspace.setAnalysisMode) {
      workspace.setAnalysisMode(currentLayerMeta.analysisMode);
    }

    if (currentLayerMeta.animationMode && workspace.setAnimationMode) {
      workspace.setAnimationMode(currentLayerMeta.animationMode);
    }
  }, [currentLayerMeta, workspace]);

  useEffect(() => {
    if (!workspace?.setQueryCategoryInput) return;
    workspace.setQueryCategoryInput(objectGroup);
  }, [objectGroup, workspace]);

  useEffect(() => {
    const taskMeta = TASK_GROUPS.find((item) => item.id === taskGroup);
    if (!taskMeta?.analysisMode || !workspace?.setAnalysisMode) return;
    workspace.setAnalysisMode(taskMeta.analysisMode);
  }, [taskGroup, workspace]);

  useEffect(() => {
    const nextOverlays = ['shared_base', 'local_delta', 'path_amplification'];
    if (researchLayer === 'semantic_roles') nextOverlays.push('semantic_roles');
    if (researchLayer === 'propagation_encoding' || researchLayer === 'result_recovery') nextOverlays.push('fidelity');
    updateLanguageFocus({ structureOverlays: Array.from(new Set(nextOverlays)) });
  }, [researchLayer]);

  const toggleOverlay = (overlayId) => {
    const nextOverlays = overlays.includes(overlayId)
      ? overlays.filter((item) => item !== overlayId)
      : [...overlays, overlayId];
    updateLanguageFocus({ structureOverlays: nextOverlays });
  };

  const currentTheoryLabel = workspace?.currentTheoryObject?.labelZh || '-';
  const currentModeLabel = workspace?.analysisModes?.find((item) => item.id === workspace?.analysisMode)?.label || workspace?.analysisMode || '-';
  const selectedNodeLabel = workspace?.selected?.label || workspace?.selected?.concept || '未选中';

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div style={{
        padding: '14px',
        borderRadius: '12px',
        background: 'linear-gradient(160deg, rgba(58,123,213,0.26), rgba(16,24,40,0.92))',
        border: '1px solid rgba(125, 211, 252, 0.22)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <Brain size={16} color="#8fd4ff" />
          <div style={{ color: '#eff6ff', fontSize: '14px', fontWeight: 800 }}>语言主线控制台</div>
        </div>
        <div style={{ color: '#bdd6f7', fontSize: '11px', lineHeight: 1.6 }}>
          左侧面板已切到“研究拼图”模式。先选研究层，再选对象、任务和结构图层，主 3D 空间会跟着切换。
        </div>
      </div>

      <div style={cardStyle}>
        <div style={sectionTitleStyle}>
          <Layers3 size={15} color="#8fd4ff" />
          <span>五层测试体系</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
          {RESEARCH_LAYERS.map((item) => (
            <button
              key={item.id}
              onClick={() => updateLanguageFocus({ researchLayer: item.id })}
              style={{
                textAlign: 'left',
                padding: '10px 12px',
                borderRadius: '10px',
                border: researchLayer === item.id ? '1px solid rgba(143, 212, 255, 0.55)' : '1px solid rgba(255,255,255,0.08)',
                background: researchLayer === item.id ? 'rgba(143, 212, 255, 0.12)' : 'rgba(255,255,255,0.02)',
                color: researchLayer === item.id ? '#eef7ff' : '#a8b7cc',
                cursor: 'pointer',
              }}
            >
              <div style={{ fontSize: '12px', fontWeight: 700, marginBottom: '3px' }}>{item.label}</div>
              <div style={{ fontSize: '10px', lineHeight: 1.5, color: researchLayer === item.id ? '#c8e6ff' : '#7e91ab' }}>{item.desc}</div>
            </button>
          ))}
        </div>
      </div>

      <div style={cardStyle}>
        <div style={sectionTitleStyle}>
          <Boxes size={15} color="#7dd3fc" />
          <span>对象 / 任务 / 角色</span>
        </div>

        <div style={{ marginBottom: '10px' }}>
          <div style={{ fontSize: '10px', color: '#71839f', marginBottom: '6px' }}>对象组</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {OBJECT_GROUPS.map((item) => (
              <ChipButton key={item.id} active={objectGroup === item.id} label={item.label} onClick={() => updateLanguageFocus({ objectGroup: item.id })} />
            ))}
          </div>
        </div>

        <div style={{ marginBottom: '10px' }}>
          <div style={{ fontSize: '10px', color: '#71839f', marginBottom: '6px' }}>任务组</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {TASK_GROUPS.map((item) => (
              <ChipButton key={item.id} active={taskGroup === item.id} label={item.label} onClick={() => updateLanguageFocus({ taskGroup: item.id })} />
            ))}
          </div>
        </div>

        <div>
          <div style={{ fontSize: '10px', color: '#71839f', marginBottom: '6px' }}>角色组</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {ROLE_GROUPS.map((item) => (
              <ChipButton key={item.id} active={roleGroup === item.id} label={item.label} onClick={() => updateLanguageFocus({ roleGroup: item.id })} />
            ))}
          </div>
        </div>
      </div>

      <div style={cardStyle}>
        <div style={sectionTitleStyle}>
          <Route size={15} color="#4ade80" />
          <span>结构图层</span>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
          {STRUCTURE_OVERLAYS.map((item) => (
            <ChipButton
              key={item.id}
              active={overlays.includes(item.id)}
              label={item.label}
              onClick={() => toggleOverlay(item.id)}
            />
          ))}
        </div>
        <div style={{ marginTop: '10px', fontSize: '10px', color: '#7e91ab', lineHeight: 1.6 }}>
          已选图层：{overlays.map((item) => STRUCTURE_OVERLAYS.find((meta) => meta.id === item)?.label || item).join('、')}
        </div>
      </div>

      <div style={cardStyle}>
        <div style={sectionTitleStyle}>
          <GitBranch size={15} color="#f5c87a" />
          <span>模型与阶段口径</span>
        </div>

        <div style={{ marginBottom: '10px' }}>
          <div style={{ fontSize: '10px', color: '#71839f', marginBottom: '6px' }}>模型</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {MODEL_OPTIONS.map((item) => (
              <ChipButton key={item.id} active={modelKey === item.id} label={item.label} onClick={() => updateLanguageFocus({ modelKey: item.id })} />
            ))}
          </div>
        </div>

        <div style={{ marginBottom: '10px' }}>
          <div style={{ fontSize: '10px', color: '#71839f', marginBottom: '6px' }}>阶段</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {STAGE_OPTIONS.map((item) => (
              <ChipButton key={item.id} active={stageKey === item.id} label={item.label} onClick={() => updateLanguageFocus({ stageKey: item.id })} />
            ))}
          </div>
        </div>

        <div>
          <div style={{ fontSize: '10px', color: '#71839f', marginBottom: '6px' }}>对照模式</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {COMPARE_OPTIONS.map((item) => (
              <ChipButton key={item.id} active={compareMode === item.id} label={item.label} onClick={() => updateLanguageFocus({ compareMode: item.id })} />
            ))}
          </div>
        </div>
      </div>

      <div style={cardStyle}>
        <div style={sectionTitleStyle}>
          <Sparkles size={15} color="#93c5fd" />
          <span>当前联动摘要</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '92px 1fr', gap: '8px', fontSize: '11px', color: '#a9bdd8', lineHeight: 1.6 }}>
          <div>研究层</div><div>{currentLayerMeta.label}</div>
          <div>理论对象</div><div>{currentTheoryLabel}</div>
          <div>当前动作</div><div>{currentModeLabel}</div>
          <div>对象组</div><div>{OBJECT_GROUPS.find((item) => item.id === objectGroup)?.label}</div>
          <div>任务组</div><div>{TASK_GROUPS.find((item) => item.id === taskGroup)?.label}</div>
          <div>角色组</div><div>{ROLE_GROUPS.find((item) => item.id === roleGroup)?.label}</div>
          <div>节点焦点</div><div>{selectedNodeLabel}</div>
          <div>阶段口径</div><div>{stageKey} / {MODEL_OPTIONS.find((item) => item.id === modelKey)?.label}</div>
        </div>
      </div>

      <div style={{ ...cardStyle, borderColor: 'rgba(248, 113, 113, 0.18)' }}>
        <div style={sectionTitleStyle}>
          <ShieldAlert size={15} color="#ff8a80" />
          <span>当前硬伤</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {HARD_PROBLEMS.map((item) => {
            const active = riskFocus === item.id;
            return (
              <button
                key={item.id}
                onClick={() => updateLanguageFocus({ riskFocus: item.id })}
                style={{
                  textAlign: 'left',
                  padding: '10px 12px',
                  borderRadius: '10px',
                  border: active ? `1px solid ${item.tone}` : '1px solid rgba(255,255,255,0.08)',
                  background: active ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.02)',
                  color: active ? '#fff2f2' : '#adb7c7',
                  cursor: 'pointer',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '3px' }}>
                  <AlertTriangle size={12} color={item.tone} />
                  <span style={{ fontSize: '12px', fontWeight: 700 }}>{item.title}</span>
                </div>
                <div style={{ fontSize: '10px', color: active ? '#ffd7d7' : '#7f91a9' }}>
                  {item.id === 'fidelity' && '后段天然闭合仍弱于前段分流。'}
                  {item.id === 'competition' && '水果内部这类高共享对象最容易失稳。'}
                  {item.id === 'closure' && '系统更像补救型闭合，不是天然稳态闭合。'}
                  {item.id === 'brand' && '品牌触发已经出现，但边界还不够厚。'}
                  {item.id === 'cross_model' && '单模型亮点不少，但跨模型硬主核仍偏少。'}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <div style={cardStyle}>
        <button
          onClick={() => setLegacyOpen((prev) => !prev)}
          style={{
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'transparent',
            border: 'none',
            color: '#d6e4ff',
            cursor: 'pointer',
            padding: 0,
            fontSize: '13px',
            fontWeight: 700,
          }}
        >
          <span>高级分析工具</span>
          <ChevronDown
            size={16}
            style={{
              transform: legacyOpen ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s ease',
            }}
          />
        </button>
        <div style={{ marginTop: legacyOpen ? '12px' : 0, display: legacyOpen ? 'block' : 'none' }}>
          {legacyControls}
        </div>
      </div>
    </div>
  );
}
