from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class VisualizationWidget:
    id: str
    title: str
    chart_type: str
    target_data: List[str]
    target_math: List[str]
    visual_meaning: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "title": self.title,
            "chart_type": self.chart_type,
            "target_data": self.target_data,
            "target_math": self.target_math,
            "visual_meaning": self.visual_meaning,
        }


@dataclass
class VisualizationScreen:
    id: str
    title: str
    purpose: str
    integration_zone: str
    widgets: List[VisualizationWidget]
    interactions: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "title": self.title,
            "purpose": self.purpose,
            "integration_zone": self.integration_zone,
            "widgets": {widget.id: widget.to_dict() for widget in self.widgets},
            "interactions": self.interactions,
        }


@dataclass
class DnnExtractionVisualizationBlueprint:
    screens: List[VisualizationScreen]
    sources: Dict[str, str]

    @classmethod
    def from_repo(cls, root: Path) -> "DnnExtractionVisualizationBlueprint":
        temp = root / "tests" / "codex_temp"
        sources = {
            "systematic_mass_extraction": "tests/codex_temp/dnn_systematic_mass_extraction_block_20260315.json",
            "dense_real_unit_corpus": "tests/codex_temp/dnn_dense_real_unit_corpus_block_20260315.json",
            "activation_signature": "tests/codex_temp/dnn_activation_signature_mining_block_20260315.json",
            "math_restoration": "tests/codex_temp/dnn_math_restoration_status_block_20260315.json",
            "micro_meso_macro": "tests/codex_temp/qwen_deepseek_micro_meso_macro_encoding_map_20260315.json",
            "family_offset_math": "tests/codex_temp/qwen3_deepseek_family_patch_offset_math_mechanism_20260315.json",
            "successor_real_corpus": "tests/codex_temp/dnn_successor_real_corpus_block_20260315.json",
            "successor_stage_rows": "tests/codex_temp/dnn_successor_stage_row_corpus_block_20260315.json",
            "online_recovery_episode_export": "tests/codex_temp/dnn_successor_online_recovery_episode_export_block_20260315.json",
            "successor_math_restoration": "tests/codex_temp/dnn_successor_math_restoration_block_20260315.json",
        }

        screens = [
            VisualizationScreen(
                id="overview_command_wall",
                title="总览指挥墙",
                purpose="一屏看清当前 DNN 数据提取规模、真实度、恢复总分以及主瓶颈。",
                integration_zone="DNN 主工作台 / 顶层总览页",
                widgets=[
                    VisualizationWidget(
                        id="corpus_kpi_strip",
                        title="语料库 KPI 条",
                        chart_type="metric_strip",
                        target_data=["systematic_mass_extraction", "dense_real_unit_corpus", "activation_signature"],
                        target_math=["full_restoration", "evidence_scale"],
                        visual_meaning="显示 total_standardized_units、exact_real_fraction、signature_rows、full_restoration_score，让用户先看全局状态。",
                    ),
                    VisualizationWidget(
                        id="progress_radar",
                        title="恢复进度雷达",
                        chart_type="radar",
                        target_data=["math_restoration"],
                        target_math=["family_basis", "concept_offset", "protocol_field", "topology", "successor"],
                        visual_meaning="把五大数学恢复项放到同一张雷达图，直接看哪一项拖后腿。",
                    ),
                    VisualizationWidget(
                        id="hard_gap_bar",
                        title="硬伤排序条",
                        chart_type="ranked_bar",
                        target_data=["math_restoration", "successor_real_corpus"],
                        target_math=["successor_bottleneck", "dense_exact_gap"],
                        visual_meaning="把 exact_real_fraction、successor_exactness_fraction、successor_parametric_score 直接排序，显示当前最硬瓶颈。",
                    ),
                ],
                interactions=[
                    "点击任意恢复项，跳转到对应专题页。",
                    "悬停指标时显示来源文件和计算公式。",
                    "允许按 qwen / deepseek / codebook 筛选来源。",
                ],
            ),
            VisualizationScreen(
                id="corpus_atlas_view",
                title="提取语料库总图",
                purpose="完整展示现在到底提取了哪些数据对象、各自有多少、来自哪里。",
                integration_zone="DNN 主工作台 / 数据总图页",
                widgets=[
                    VisualizationWidget(
                        id="source_sankey",
                        title="来源流图",
                        chart_type="sankey",
                        target_data=["systematic_mass_extraction", "dense_real_unit_corpus"],
                        target_math=["evidence_flow"],
                        visual_meaning="从 qwen/deepseek/codebook 流向 layer_row、relation_topology_concept、structure_task_row 等真实单位类型，展示数据来源和汇聚。",
                    ),
                    VisualizationWidget(
                        id="unit_type_treemap",
                        title="单位类型树图",
                        chart_type="treemap",
                        target_data=["dense_real_unit_corpus"],
                        target_math=["evidence_composition"],
                        visual_meaning="把 layer_row、protocol_field_concept、relation_topology_concept、structure_bridge_task 等类型直接按体量可视化。",
                    ),
                    VisualizationWidget(
                        id="real_vs_proxy_stacked",
                        title="真实度堆叠图",
                        chart_type="stacked_bar",
                        target_data=["systematic_mass_extraction", "successor_real_corpus"],
                        target_math=["exactness_structure"],
                        visual_meaning="展示 exact_real_units、synthetic_units、inventory_mass_units、proxy_units 的比例，告诉用户哪些地方还在 proxy 层。",
                    ),
                ],
                interactions=[
                    "点击树图节点后，在右侧显示该类型对应的原始 artifact 路径。",
                    "支持按 micro / meso / macro、real / proxy、source 过滤。",
                ],
            ),
            VisualizationScreen(
                id="concept_family_offset_lab",
                title="概念与 Family/Offset 实验室",
                purpose="具体展示 concept 是怎么挂在 family patch 上，以及苹果、香蕉、猫、狗这类例子如何分布。",
                integration_zone="DNN 主工作台 / ICSPB 对象层 family_patch 与 concept_section",
                widgets=[
                    VisualizationWidget(
                        id="family_patch_3d_scatter",
                        title="Family Patch 3D 散点",
                        chart_type="3d_scatter",
                        target_data=["family_offset_math", "micro_meso_macro"],
                        target_math=["family_basis", "concept_offset"],
                        visual_meaning="以 family 为颜色、以 concept 为点，展示 apple/banana/pear、cat/dog 等在 family patch 上的分布和 offset 偏移。",
                    ),
                    VisualizationWidget(
                        id="concept_offset_vector_field",
                        title="Offset 向量场",
                        chart_type="vector_field",
                        target_data=["family_offset_math"],
                        target_math=["concept_offset"],
                        visual_meaning="从 family center 指向各 concept，直接把 Delta_apple、Delta_banana 画成偏移箭头。",
                    ),
                    VisualizationWidget(
                        id="pair_distance_matrix",
                        title="概念距离矩阵",
                        chart_type="heatmap",
                        target_data=["micro_meso_macro"],
                        target_math=["family_local_geometry"],
                        visual_meaning="显示 apple-banana、apple-pear、cat-dog 等 pairwise 距离，验证 same-family 更近、cross-family 更远。",
                    ),
                ],
                interactions=[
                    "选择一个 concept 后，右侧直接显示候选公式和关键数值。",
                    "支持从 fruit 切换到 animal / object / abstract。",
                ],
            ),
            VisualizationScreen(
                id="micro_protocol_topology_lab",
                title="属性/协议/拓扑实验室",
                purpose="把 attribute axis、protocol field、relation/attention topology 放进一套可以联动的视图里。",
                integration_zone="DNN 主工作台 / ICSPB 对象层 attribute_fiber、relation_context_fiber、protocol_bridge",
                widgets=[
                    VisualizationWidget(
                        id="attribute_axis_parallel",
                        title="属性轴平行坐标",
                        chart_type="parallel_coordinates",
                        target_data=["micro_meso_macro", "activation_signature"],
                        target_math=["micro_attribute_fibers"],
                        visual_meaning="展示颜色、甜度、圆润度等 attribute axis 在不同 concept 上的投影强弱。",
                    ),
                    VisualizationWidget(
                        id="protocol_field_head_map",
                        title="协议字段头图",
                        chart_type="layer_head_heatmap",
                        target_data=["dense_real_unit_corpus", "activation_signature"],
                        target_math=["protocol_field"],
                        visual_meaning="把 protocol field 在 layer-head 上的使用强度可视化，显示 concept 进入协议区时调用了哪些头带。",
                    ),
                    VisualizationWidget(
                        id="relation_attention_topology_graph",
                        title="关系/注意力拓扑图",
                        chart_type="graph_3d",
                        target_data=["dense_real_unit_corpus", "activation_signature"],
                        target_math=["topology", "relation_context_transport"],
                        visual_meaning="把 relation topology 与 attention topology 叠成同一图，展示 family 内 transport 和 cross-family 干扰。",
                    ),
                ],
                interactions=[
                    "属性轴、protocol field、topology 图三者联动高亮同一个 concept。",
                    "点击 layer/head 节点时回显对应 artifact 与 margin 指标。",
                ],
            ),
            VisualizationScreen(
                id="successor_chain_lab",
                title="Successor 链实验室",
                purpose="完整看到 successor 目前有哪些 direct/proxy 数据、哪些已经 dense、哪些还停在 summary/inventory。",
                integration_zone="DNN 主工作台 / ICSPB 对象层 successor_aligned_transport",
                widgets=[
                    VisualizationWidget(
                        id="successor_exactness_stack",
                        title="Successor Exactness 堆叠图",
                        chart_type="stacked_bar",
                        target_data=["successor_real_corpus"],
                        target_math=["successor_exactness"],
                        visual_meaning="把 direct_dense、summary_proxy、inventory_proxy、structured_math、export_contract 五块堆起来，直接看 successor 证据结构。",
                    ),
                    VisualizationWidget(
                        id="stage_row_ribbon",
                        title="Stage Row 带状图",
                        chart_type="ribbon_timeline",
                        target_data=["successor_stage_rows", "online_recovery_episode_export"],
                        target_math=["successor_stage_transport"],
                        visual_meaning="按 concept/relation/tool/verify 四个 stage 展示 trigger 和 recovery 分布，让 successor 不再是抽象分数。",
                    ),
                    VisualizationWidget(
                        id="replacement_priority_board",
                        title="替换优先级板",
                        chart_type="dual_axis_bar",
                        target_data=["successor_stage_rows"],
                        target_math=["execution_first_vs_gain_first"],
                        visual_meaning="一根轴显示 execution priority，一根轴显示 gain priority，明确 online_recovery 该先做，inventory 该后做但增益更大。",
                    ),
                    VisualizationWidget(
                        id="successor_restoration_gauge",
                        title="Successor 恢复仪表盘",
                        chart_type="gauge_cluster",
                        target_data=["successor_math_restoration", "math_restoration"],
                        target_math=["successor_structure", "successor_transport", "successor_exactness", "successor_restoration"],
                        visual_meaning="把 successor_structure_score、transport_score、exactness_score、restoration_score 分别显示，明确哪一项在拖后腿。",
                    ),
                ],
                interactions=[
                    "点击 direct/proxy 区块后，右侧显示缺的轴和目标 tensor layout。",
                    "可以切换 execution-first / gain-first 两种替换策略口径。",
                ],
            ),
            VisualizationScreen(
                id="math_theory_console",
                title="数学还原控制台",
                purpose="把每条数学式、分数、数据来源和典型概念案例放在同一控制台里。",
                integration_zone="DNN 主工作台 / 数学理论页",
                widgets=[
                    VisualizationWidget(
                        id="equation_card_stack",
                        title="公式卡片堆栈",
                        chart_type="equation_cards",
                        target_data=["family_offset_math", "micro_meso_macro", "math_restoration"],
                        target_math=["family_basis", "concept_offset", "protocol_field", "topology", "successor"],
                        visual_meaning="每张卡片显示一条候选方程、对应分数、对应数据来源和一个具体例子。",
                    ),
                    VisualizationWidget(
                        id="data_to_math_bipartite",
                        title="数据到数学二分图",
                        chart_type="bipartite_graph",
                        target_data=["systematic_mass_extraction", "dense_real_unit_corpus", "activation_signature", "successor_real_corpus"],
                        target_math=["family_basis", "concept_offset", "protocol_field", "topology", "successor"],
                        visual_meaning="左侧是数据块，右侧是数学项，中间连线表示哪类数据支撑了哪条数学恢复。",
                    ),
                    VisualizationWidget(
                        id="closure_ladder",
                        title="闭合阶梯图",
                        chart_type="step_ladder",
                        target_data=["math_restoration", "successor_math_restoration"],
                        target_math=["candidate_to_theorem_closure"],
                        visual_meaning="把当前理论处在 candidate、strong candidate、near closure、final theorem 哪一级可视化。",
                    ),
                ],
                interactions=[
                    "点击公式卡片后，底部弹出概念例子，比如 apple/bbanana/cat/dog 的具体数值。",
                    "支持切换仅看 exact-real 支撑或包含 proxy 支撑。",
                ],
            ),
            VisualizationScreen(
                id="provenance_trace_view",
                title="证据溯源视图",
                purpose="让每一个可视化元素都能追溯到具体 artifact、脚本和原始字段，避免图像脱离证据。",
                integration_zone="全局右侧详情抽屉",
                widgets=[
                    VisualizationWidget(
                        id="artifact_trace_table",
                        title="Artifact 溯源表",
                        chart_type="table",
                        target_data=list(sources.keys()),
                        target_math=["evidence_traceability"],
                        visual_meaning="显示每个视图元素对应的 JSON 文件、字段名、脚本来源。",
                    ),
                    VisualizationWidget(
                        id="field_path_tree",
                        title="字段路径树",
                        chart_type="tree",
                        target_data=list(sources.keys()),
                        target_math=["field_traceability"],
                        visual_meaning="把 JSON 字段路径树形展开，方便从图表反查到原始字段。",
                    ),
                ],
                interactions=[
                    "所有图表元素右键可打开对应 artifact 路径和字段路径。",
                    "支持复制公式、字段路径、数据来源。",
                ],
            ),
        ]

        return cls(screens=screens, sources=sources)

    def summary(self) -> Dict[str, object]:
        total_widgets = sum(len(screen.widgets) for screen in self.screens)
        three_d_widgets = sum(
            1 for screen in self.screens for widget in screen.widgets if "3d" in widget.chart_type.lower()
        )
        successor_widgets = sum(
            1
            for screen in self.screens
            for widget in screen.widgets
            if "successor" in " ".join(widget.target_math).lower()
        )
        return {
            "screen_count": len(self.screens),
            "widget_count": total_widgets,
            "three_d_widget_count": three_d_widgets,
            "successor_widget_count": successor_widgets,
            "source_count": len(self.sources),
            "screens": {screen.id: screen.to_dict() for screen in self.screens},
            "sources": self.sources,
        }
