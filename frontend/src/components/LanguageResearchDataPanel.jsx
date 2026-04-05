import { Boxes, Database, FileSearch, Focus, Route } from 'lucide-react';
import { useMemo } from 'react';
import {
  AppleNeuronCategoryComparePanel,
  AppleNeuronEncodingInfoPanels,
  AppleNeuronResearchAssetInfoPanel,
  AppleNeuronSelectedLegendPanels,
} from '../blueprint/appleNeuronInfoPanelsBridge';
import { PERSISTED_PUZZLE_RECORDS_V1 } from '../blueprint/data/persisted_puzzle_records_v1';
import { PERSISTED_REPAIR_REPLAY_SAMPLE_SLOTS_V1 } from '../blueprint/data/persisted_repair_replay_sample_slots_v1';

const PANEL_TABS = [
  { key: 'focus', label: '当前焦点', icon: Focus },
  { key: 'layers', label: '层数据', icon: Boxes },
  { key: 'replay', label: '样本回放', icon: Route },
  { key: 'assets', label: '资产与证据', icon: FileSearch },
];

const cardStyle = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: '8px',
  padding: '10px',
};

const compactGridStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
  gap: '8px',
};

function safePercent(value, fallback = 0) {
  const numeric = Number(value);
  const resolved = Number.isFinite(numeric) ? numeric : fallback;
  return `${Math.round(resolved * 100)}%`;
}

function formatDisplayLevels(displayLevels = {}) {
  const labelMap = {
    foundation: '基础层',
    parameter_state: '参数位',
    dynamic_route: '动态路径',
    result_recovery: '结果回收',
    propagation_encoding: '传播编码',
    semantic_roles: '语义角色',
    object_family: '对象族',
  };
  return Object.entries(labelMap)
    .filter(([key]) => displayLevels?.[key] !== false)
    .map(([, label]) => label)
    .join(' / ') || '暂无';
}

function renderHoverDetail(info) {
  if (!info) {
    return '当前没有选中对象，悬停或点击 3D 对象后，这里会显示精确数据。';
  }
  if (info.type === 'feature') {
    return `特征 #${info.featureId}，激活值 ${info.activation?.toFixed?.(4) || '-'}`;
  }
  if (info.type === 'manifold') {
    return `流形点 ${info.index}，PC1/2/3 = ${info.pc1?.toFixed?.(2) || '-'}, ${info.pc2?.toFixed?.(2) || '-'}, ${info.pc3?.toFixed?.(2) || '-'}`;
  }
  if (info.detailType === 'apple_switch_unit') {
    return [
      info.unitId ? `单元 ${info.unitId}` : null,
      info.modelName ? `模型 ${info.modelName}` : null,
      info.roleLabel ? `角色 ${info.roleLabel}` : null,
      info.unitTypeLabel ? `类型 ${info.unitTypeLabel}` : null,
      typeof info.actualLayer === 'number' ? `真实层 L${info.actualLayer}` : null,
      typeof info.effectiveScore === 'number' ? `有效分数 ${info.effectiveScore.toFixed(4)}` : null,
      info.directionLabel ? `方向 ${info.directionLabel}` : null,
    ].filter(Boolean).join(' | ');
  }
  if (info.type?.startsWith?.('encoding3d_') || info.type?.startsWith?.('layerfirst_')) {
    return [
      info.label ? `名称 ${info.label}` : null,
      info.role ? `角色 ${info.role}` : null,
      info.nodeKind ? `节点类型 ${info.nodeKind}` : null,
      info.layerLabel ? `层 ${info.layerLabel}` : null,
      typeof info.score === 'number' ? `分数 ${info.score.toFixed(4)}` : null,
      typeof info.gain === 'number' ? `增益 ${info.gain.toFixed(4)}` : null,
      typeof info.memberCount === 'number' ? `成员数 ${info.memberCount}` : null,
      Array.isArray(info.members) ? `参数位 ${info.members.join(', ')}` : null,
      info.detailText ? `说明 ${info.detailText}` : null,
    ].filter(Boolean).join(' | ');
  }
  if (info.label || typeof info.probability === 'number') {
    return [
      info.label ? `词元 "${info.label}"` : null,
      typeof info.probability === 'number' ? `概率 ${(info.probability * 100).toFixed(1)}%` : null,
      info.actual ? `实际 "${info.actual}"` : null,
    ].filter(Boolean).join(' | ');
  }
  return '当前对象存在，但还没有结构化细节。';
}

function SummaryCard({ label, value }) {
  return (
    <div style={cardStyle}>
      <div style={{ fontSize: '10px', color: '#8ea5c5', marginBottom: '4px' }}>{label}</div>
      <div style={{ fontSize: '13px', fontWeight: 700, color: '#eef7ff', lineHeight: 1.5 }}>{value}</div>
    </div>
  );
}

export default function LanguageResearchDataPanel({
  workspace,
  hoveredInfo = null,
  displayInfo = null,
  infoPanelTab = 'focus',
  setInfoPanelTab = null,
  showAppleCategoryCompare = true,
  showAppleEncodingInfo = true,
  showAppleResearchAsset = true,
  showAppleLegend = true,
  currentAlgorithmName = '当前算法',
  currentAlgorithmFocus = '',
  structureTab = 'circuit',
}) {
  const languageFocus = workspace?.languageFocus || {};
  const activePuzzle = useMemo(
    () => PERSISTED_PUZZLE_RECORDS_V1.find((item) => item.id === languageFocus?.activePuzzleId) || null,
    [languageFocus?.activePuzzleId]
  );
  const selectedReplaySlot = useMemo(
    () => PERSISTED_REPAIR_REPLAY_SAMPLE_SLOTS_V1.find((item) => item.slot_id === languageFocus?.selectedRepairReplaySlotId) || null,
    [languageFocus?.selectedRepairReplaySlotId]
  );
  const focusInfo = hoveredInfo || displayInfo || workspace?.selected || null;
  const conceptAssociationState = workspace?.conceptAssociationState || null;
  const visibleNodeCount = Array.isArray(workspace?.nodes) ? workspace.nodes.length : 0;
  const visibleLinkCount = Array.isArray(workspace?.links) ? workspace.links.length : 0;
  const displayLevelSummary = formatDisplayLevels(workspace?.displayLevels);
  const currentPhase = languageFocus?.selectedRepairReplayPhase || 'bridge';
  const scanPreviewData = workspace?.scanPreviewData || null;
  const scanPreviewSummary = scanPreviewData?.summary || '当前没有载入新的资产摘要。';
  const scanAssetStatus = workspace?.scanPreviewLoading ? '加载中' : (workspace?.scanPreviewError ? '读取失败' : '可读取');
  const replayFocusLabel = selectedReplaySlot
    ? `${selectedReplaySlot.label} / ${selectedReplaySlot.matchTypeLabel || '待定类型'}`
    : '当前未选中回放槽位';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <div style={{ display: 'flex', gap: '6px' }}>
        {PANEL_TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = infoPanelTab === tab.key;
          return (
            <button
              key={tab.key}
              type="button"
              onClick={() => setInfoPanelTab?.(tab.key)}
              style={{
                flex: 1,
                border: '1px solid rgba(255,255,255,0.14)',
                borderRadius: '6px',
                background: isActive ? 'rgba(0,210,255,0.18)' : 'rgba(255,255,255,0.02)',
                color: isActive ? '#e9f9ff' : '#9aa4b4',
                fontSize: '11px',
                padding: '6px 8px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '5px',
              }}
            >
              <Icon size={12} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {infoPanelTab === 'focus' ? (
        <div style={{ display: 'grid', gap: '8px' }}>
          <div style={cardStyle}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
              <Database size={13} color="#7dd3fc" />
              <div style={{ color: '#eef7ff', fontSize: '12px', fontWeight: 700 }}>当前焦点摘要</div>
            </div>
            <div style={{ color: '#9ea7b7', fontSize: '11px', lineHeight: 1.6 }}>
              {currentAlgorithmName} / {structureTab}
              {currentAlgorithmFocus ? `，${currentAlgorithmFocus}` : ''}
            </div>
          </div>

          <div style={compactGridStyle}>
            <SummaryCard label="研究层" value={languageFocus?.researchLayer || 'static_encoding'} />
            <SummaryCard label="当前拼图" value={activePuzzle?.title || '未选中'} />
            <SummaryCard label="当前概念" value={conceptAssociationState?.conceptLabel || workspace?.selected?.label || '未聚焦'} />
            <SummaryCard label="当前场景焦点" value={selectedReplaySlot ? selectedReplaySlot.label : (workspace?.selected?.label || '未选中')} />
          </div>

          <div style={cardStyle}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: '#dbeafe', marginBottom: '6px' }}>当前对象</div>
            <div style={{ fontSize: '11px', color: '#c8d1df', lineHeight: 1.7 }}>
              {renderHoverDetail(focusInfo)}
            </div>
          </div>

          {conceptAssociationState ? (
            <div style={compactGridStyle}>
              <SummaryCard label="概念类别" value={conceptAssociationState.categoryLabel || '未分类'} />
              <SummaryCard label="覆盖层数" value={String(conceptAssociationState.layers?.length || 0)} />
              <SummaryCard label="关联节点" value={String(conceptAssociationState.totalLinkedNodes || 0)} />
              <SummaryCard label="层间强度" value={safePercent(conceptAssociationState.totalRelationStrength, 0)} />
            </div>
          ) : null}
        </div>
      ) : null}

      {infoPanelTab === 'layers' ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={compactGridStyle}>
            <SummaryCard label="可见节点" value={String(visibleNodeCount)} />
            <SummaryCard label="可见链路" value={String(visibleLinkCount)} />
            <SummaryCard label="显示层级" value={displayLevelSummary} />
            <SummaryCard label="当前研究层" value={languageFocus?.researchLayer || 'static_encoding'} />
          </div>

          {showAppleCategoryCompare && <AppleNeuronCategoryComparePanel workspace={workspace} compact />}
          {showAppleEncodingInfo && <AppleNeuronEncodingInfoPanels workspace={workspace} compact />}
          {showAppleLegend && <AppleNeuronSelectedLegendPanels workspace={workspace} compact />}

          {!showAppleCategoryCompare && !showAppleEncodingInfo && !showAppleLegend ? (
            <div style={cardStyle}>
              <div style={{ fontSize: '11px', color: '#9ea7b7', lineHeight: 1.6 }}>
                当前层数据面板没有额外卡片，后续这里会继续承载右侧独立的数据查看结构。
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      {infoPanelTab === 'replay' ? (
        <div style={{ display: 'grid', gap: '8px' }}>
          <div style={compactGridStyle}>
            <SummaryCard label="当前槽位" value={replayFocusLabel} />
            <SummaryCard label="当前阶段" value={currentPhase} />
            <SummaryCard label="当前验证状态" value={selectedReplaySlot?.status || 'planned'} />
            <SummaryCard label="预测状态" value={workspace?.prediction?.statusText || '当前没有动态回放'} />
          </div>

          <div style={cardStyle}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: '#dbeafe', marginBottom: '6px' }}>样本回放详情</div>
            <div style={{ fontSize: '11px', color: '#c8d1df', lineHeight: 1.7 }}>
              {selectedReplaySlot ? (
                <>
                  <div>{`样本：${selectedReplaySlot.sample_label}`}</div>
                  <div>{`锚定变量：${selectedReplaySlot.anchor_variable || '待定'}`}</div>
                  <div>{`共享候选链：${selectedReplaySlot.shared_subcircuit_hint || '待补'}`}</div>
                  <div>{`就绪度：${safePercent(selectedReplaySlot.readiness, 0)}`}</div>
                  <div>{`目标：${selectedReplaySlot.validation_goal}`}</div>
                </>
              ) : (
                <div>当前没有选中的真实样本回放槽位，后续这里会成为样本级验证的主入口。</div>
              )}
            </div>
          </div>
        </div>
      ) : null}

      {infoPanelTab === 'assets' ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={compactGridStyle}>
            <SummaryCard label="资产读取状态" value={scanAssetStatus} />
            <SummaryCard label="当前资产路径" value={workspace?.selectedScanPath || '未选中'} />
            <SummaryCard label="当前拼图工件" value={activePuzzle?.artifactPath || '未挂接'} />
            <SummaryCard label="概念数据状态" value={conceptAssociationState ? '已建立关联' : '待补概念映射'} />
          </div>

          <div style={cardStyle}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: '#dbeafe', marginBottom: '6px' }}>资产摘要</div>
            <div style={{ fontSize: '11px', color: '#c8d1df', lineHeight: 1.7 }}>
              {scanPreviewSummary}
            </div>
          </div>

          {showAppleResearchAsset ? <AppleNeuronResearchAssetInfoPanel workspace={workspace} compact /> : null}
        </div>
      ) : null}
    </div>
  );
}
