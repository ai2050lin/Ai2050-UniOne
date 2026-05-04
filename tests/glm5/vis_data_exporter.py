"""
标准可视化数据导出模块

将实验结果转换为3D可视化客户端可读取的标准JSON格式。
支持 Schema v1.0 和 v2.0 规范。

v2.0 新增类型:
    - subspace_decomposition: W_U/W_U⊥子空间分解
    - force_line: 语义力线指数增长
    - grammar_role_matrix: 语法角色余弦矩阵
    - causal_chain: 因果链追踪
    - dark_matter_flow: 暗物质非线性转导

使用方法:
    from vis_data_exporter import export_trajectory, export_subspace_decomposition, save_vis_file
    
    # 在实验脚本中收集数据后调用
    vis = export_trajectory(phase="CCLXIV", model="qwen3", ...)
    subspace = export_subspace_decomposition(phase="CCL-M", model="qwen3", ...)
    save_vis_file("CCL-M", "qwen3", "grammar_decouple", [vis, subspace], model_info)
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.decomposition import PCA

VIS_DATA_DIR = Path("results/vis_data")

# ==================== 颜色方案 ====================

CATEGORY_COLORS = {
    "fruit": "#ff6b6b",
    "animal": "#4ecdc4",
    "vehicle": "#ffe66d",
    "tool": "#a855f7",
    "nature": "#34d399",
    "food": "#f97316",
    "person": "#ec4899",
    "abstract": "#6366f1",
}

LAYER_FUNCTION_COLORS = {
    "lexical": "#ff6b6b",    # 词法 L0-L9
    "semantic": "#4ecdc4",   # 语义 L6-L18
    "syntactic": "#ffe66d",  # 语法 L14-L24
    "decision": "#a855f7",   # 决策 L24+
    "correction": "#fbbf24", # 纠正层
}

TRAJECTORY_COLORS = [
    "#ff6b6b", "#4ecdc4", "#ffe66d", "#a855f7",
    "#f97316", "#34d399", "#ec4899", "#6366f1",
    "#ef4444", "#14b8a6", "#eab308", "#8b5cf6",
]

# 语法角色颜色
GRAMMAR_ROLE_COLORS = {
    "nsubj": "#ff6b6b",
    "dobj": "#4ecdc4",
    "amod": "#ffe66d",
    "aux": "#a855f7",
    "iobj": "#34d399",
    "ccomp": "#f97316",
    "xcomp": "#ec4899",
    "mark": "#6366f1",
}

# 子空间颜色
SUBSPACE_COLORS = {
    "w_u": "#4ecdc4",         # W_U可见 - 青绿
    "w_u_perp": "#ff6b6b",    # W_U⊥ - 红色
    "grammar": "#ffe66d",      # 语法 - 黄色
    "semantic": "#4ecdc4",     # 语义 - 青绿
    "logic": "#a855f7",        # 逻辑 - 紫色
    "dark_matter": "#f97316",  # 暗物质 - 橙色
}


def _get_layer_function(layer, n_layers):
    """根据层号判断功能类别"""
    ratio = layer / n_layers if n_layers > 0 else 0
    if ratio < 0.25:
        return "lexical"
    elif ratio < 0.5:
        return "semantic"
    elif ratio < 0.7:
        return "syntactic"
    else:
        return "decision"


def _delta_cos_to_color(delta_cos):
    """delta_cos → 颜色映射: 1.0=红, 0.5=橙, 0.0=蓝"""
    r = max(0, min(1, delta_cos))
    if r > 0.5:
        # 红→橙
        t = (r - 0.5) * 2
        red = int(239 * t + 245 * (1 - t))
        green = int(68 * t + 158 * (1 - t))
        blue = int(68 * t + 11 * (1 - t))
    else:
        # 橙→蓝
        t = r * 2
        red = int(245 * t + 59 * (1 - t))
        green = int(158 * t + 130 * (1 - t))
        blue = int(11 * t + 246 * (1 - t))
    return f"#{red:02x}{green:02x}{blue:02x}"


# ==================== 导出函数 ====================

def export_trajectory(phase, model, experiment_id, token, source_token,
                      template, per_layer_data, correction_layers=None,
                      pca_coords=None, color=None):
    """导出trajectory类型数据
    
    Args:
        phase: Phase编号, 如 "CCLXIV"
        model: 模型名, 如 "qwen3"
        experiment_id: 实验ID
        token: 目标token
        source_token: 源token
        template: 模板, 如 "The {} is"
        per_layer_data: list of dict, 每项包含:
            - layer: int, 层号
            - norm: float, 残差流范数
            - cos_with_target: float, 与原始target的cos
            - cos_with_source: float, 与原始source的cos
            - delta_cos: float, 差分方向保持度
        pca_coords: 可选, list of {x, y, z}, PCA降维坐标
        color: 可选, 轨迹颜色
    
    Returns:
        dict: 标准trajectory可视化对象
    """
    if pca_coords is None:
        # 默认3D坐标: X=层号(均匀展开), Y=delta_cos, Z=cos_with_target
        n = len(per_layer_data)
        pca_coords = []
        for i, d in enumerate(per_layer_data):
            # 用螺旋展开使轨迹不重叠
            angle = i * 0.3  # 每层旋转0.3弧度
            radius = 2 + i * 0.15
            x = radius * np.cos(angle)
            y = d.get("delta_cos", 0) * 10
            z = radius * np.sin(angle)
            pca_coords.append({"x": round(x, 4), "y": round(y, 4), "z": round(z, 4)})
    
    points = []
    for i, d in enumerate(per_layer_data):
        coord = pca_coords[i] if i < len(pca_coords) else {"x": 0, "y": 0, "z": 0}
        points.append({
            "layer": d["layer"],
            "x": round(coord["x"], 4),
            "y": round(coord["y"], 4),
            "z": round(coord["z"], 4),
            "norm": round(d.get("norm", 0), 2),
            "cos_with_target": round(d.get("cos_with_target", 0), 4),
            "cos_with_source": round(d.get("cos_with_source", 0), 4),
            "delta_cos": round(d.get("delta_cos", 0), 4),
            "cos_with_wu": round(d.get("cos_with_wu", 0), 4) if "cos_with_wu" in d else None,
        })
    
    return {
        "type": "trajectory",
        "id": f"{source_token}_to_{token}_delta",
        "label": f"{source_token}→{token} 差分注入",
        "token": token,
        "source_token": source_token,
        "template": template,
        "points": points,
        "color": color or TRAJECTORY_COLORS[hash(experiment_id) % len(TRAJECTORY_COLORS)],
        "correction_layers": correction_layers or [],
    }


def export_trajectory_from_pca(phase, model, experiment_id, token, source_token,
                                template, per_layer_vectors, per_layer_metrics,
                                correction_layers=None, n_pca_components=3):
    """从原始高维向量导出trajectory (自动PCA降维)
    
    Args:
        per_layer_vectors: dict {layer: numpy_array(d_model,)}, 逐层残差流向量
        per_layer_metrics: dict {layer: {norm, cos_with_target, cos_with_source, delta_cos}}
        n_pca_components: PCA降维维度(默认3)
    
    Returns:
        dict: 标准trajectory可视化对象
    """
    layers = sorted(per_layer_vectors.keys())
    vectors = np.stack([per_layer_vectors[l] for l in layers])
    
    # PCA降维到3D
    n_components = min(n_pca_components, vectors.shape[0], vectors.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(vectors)
    
    # 归一化到合理范围
    for dim in range(n_components):
        col = coords[:, dim]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 1e-10:
            coords[:, dim] = (col - col_min) / (col_max - col_min) * 20 - 10
    
    pca_coords = [
        {"x": round(float(coords[i, 0]), 4),
         "y": round(float(coords[i, 1]), 4),
         "z": round(float(coords[i, 2]) if n_components > 2 else 0, 4)}
        for i in range(len(layers))
    ]
    
    per_layer_data = []
    for l in layers:
        m = per_layer_metrics.get(l, {})
        per_layer_data.append({
            "layer": l,
            "norm": m.get("norm", 0),
            "cos_with_target": m.get("cos_with_target", 0),
            "cos_with_source": m.get("cos_with_source", 0),
            "delta_cos": m.get("delta_cos", 0),
        })
    
    return export_trajectory(
        phase=phase, model=model, experiment_id=experiment_id,
        token=token, source_token=source_token, template=template,
        per_layer_data=per_layer_data,
        correction_layers=correction_layers,
        pca_coords=pca_coords,
    )


def export_point_cloud(phase, model, experiment_id, layer, points_data,
                       categories_colors=None):
    """导出point_cloud类型数据
    
    Args:
        layer: 层号
        points_data: list of dict, 每项包含:
            - token: str, 概念词
            - category: str, 类别
            - x, y, z: float, 3D坐标(PCA降维后)
            - norm: float, 残差流范数
            - activation: float, 概念激活强度(0-1)
        categories_colors: dict, {category: color_hex}
    
    Returns:
        dict: 标准point_cloud可视化对象
    """
    return {
        "type": "point_cloud",
        "id": f"layer{layer}_concepts_{experiment_id}",
        "label": f"L{layer} 概念空间 ({model})",
        "layer": layer,
        "points": points_data,
        "categories": categories_colors or CATEGORY_COLORS,
    }


def export_point_cloud_from_vectors(phase, model, experiment_id, layer,
                                     token_vectors, token_categories,
                                     token_activations=None):
    """从原始向量导出point_cloud (自动PCA降维)
    
    Args:
        token_vectors: dict {token: numpy_array(d_model,)}
        token_categories: dict {token: category_str}
        token_activations: 可选, dict {token: float}
    
    Returns:
        dict: 标准point_cloud可视化对象
    """
    tokens = list(token_vectors.keys())
    vectors = np.stack([token_vectors[t] for t in tokens])
    categories = {t: token_categories.get(t, "abstract") for t in tokens}
    
    # PCA降维
    pca = PCA(n_components=3)
    coords = pca.fit_transform(vectors)
    
    # 归一化
    for dim in range(3):
        col = coords[:, dim]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 1e-10:
            coords[:, dim] = (col - col_min) / (col_max - col_min) * 16 - 8
    
    points_data = []
    for i, token in enumerate(tokens):
        vec_norm = float(np.linalg.norm(token_vectors[token]))
        points_data.append({
            "token": token,
            "category": categories[token],
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "z": round(float(coords[i, 2]), 4),
            "norm": round(vec_norm, 2),
            "activation": round(token_activations.get(token, 0.5) if token_activations else 0.5, 4),
        })
    
    # 收集出现的类别
    used_categories = set(categories.values())
    cat_colors = {c: CATEGORY_COLORS.get(c, "#888888") for c in used_categories}
    
    return export_point_cloud(
        phase=phase, model=model, experiment_id=experiment_id,
        layer=layer, points_data=points_data, categories_colors=cat_colors,
    )


def export_heatmap_3d(phase, model, experiment_id, x_label, y_label, z_label,
                      x_values, y_values, cells, z_range=None):
    """导出heatmap_3d类型数据
    
    Args:
        x_label, y_label, z_label: 轴标签
        x_values: list, X轴值
        y_values: list, Y轴值
        cells: list of {x_index, y_index, value}
        z_range: [min, max] or None (自动)
    
    Returns:
        dict: 标准heatmap_3d可视化对象
    """
    if z_range is None:
        values = [c["value"] for c in cells]
        z_range = [min(values), max(values)]
    
    # 为每个cell添加颜色
    for cell in cells:
        v = cell["value"]
        normalized = (v - z_range[0]) / (z_range[1] - z_range[0] + 1e-10)
        cell["color"] = _delta_cos_to_color(normalized)
    
    return {
        "type": "heatmap_3d",
        "id": experiment_id,
        "label": f"{z_label}矩阵 ({model})",
        "x_axis": {"label": x_label, "values": x_values},
        "y_axis": {"label": y_label, "values": y_values},
        "z_axis": {"label": z_label, "range": z_range},
        "cells": cells,
    }


def export_flow(phase, model, experiment_id, layer, flows, node_positions,
                source_label="Token位置", target_label="Token位置"):
    """导出flow类型数据(注意力流)
    
    Args:
        layer: 层号
        flows: list of {source, target, weight, head}
        node_positions: list of {id, token, x, y, z}
    
    Returns:
        dict: 标准flow可视化对象
    """
    # 为每个flow添加颜色(按head分组)
    head_colors = {}
    for flow in flows:
        h = flow["head"]
        if h not in head_colors:
            head_colors[h] = TRAJECTORY_COLORS[h % len(TRAJECTORY_COLORS)]
        flow["color"] = head_colors[h]
    
    return {
        "type": "flow",
        "id": f"attention_L{layer}_{experiment_id}",
        "label": f"L{layer} 注意力流 ({model})",
        "layer": layer,
        "source_label": source_label,
        "target_label": target_label,
        "flows": flows,
        "node_positions": node_positions,
    }


def export_layer_stack(phase, model, experiment_id, n_layers, d_model,
                       layer_summaries, trajectory_ids=None):
    """导出layer_stack类型数据
    
    Args:
        n_layers: 总层数
        d_model: 模型维度
        layer_summaries: list of dict, 每项包含:
            - layer: int
            - label: str (如 "Embedding", "Template Hotspot")
            - function: str ("lexical"|"semantic"|"syntactic"|"decision")
            - metrics: dict {avg_norm, avg_delta_cos, switch_rate, category_R2, ...}
        trajectory_ids: 可选, 关联的trajectory ID列表
    
    Returns:
        dict: 标准layer_stack可视化对象
    """
    for ls in layer_summaries:
        func = ls.get("function", _get_layer_function(ls["layer"], n_layers))
        ls["function"] = func
        ls["color"] = LAYER_FUNCTION_COLORS.get(func, "#888888")
        # v2.0: W_U/W_U⊥占比 (可选)
        if "w_u_ratio" not in ls:
            ls["w_u_ratio"] = None
        if "w_u_perp_ratio" not in ls:
            ls["w_u_perp_ratio"] = None
    
    return {
        "type": "layer_stack",
        "id": f"{model}_full_model_{experiment_id}",
        "label": f"{model} 全模型层结构",
        "n_layers": n_layers,
        "d_model": d_model,
        "layers": layer_summaries,
        "trajectories": trajectory_ids or [],
    }


# ==================== v2.0 新增导出函数 ====================

def export_subspace_decomposition(phase, model, experiment_id, layer_data):
    """导出subspace_decomposition类型数据 (Schema v2.0)
    
    Args:
        layer_data: list of dict, 每项包含:
            - layer: int, 层号
            - w_u_ratio: float, W_U可见部分占比 (0-1)
            - w_u_perp_ratio: float, W_U⊥部分占比 (0-1)
            - grammar_in_perp: dict, {角色: 占比} 各语法角色在W_U⊥中的占比
            - semantics_in_w_u: float, 语义能量在W_U top10奇异模式占比
            - concept_points: 可选, list of {token, category, subspace, x, y, z}
    
    Returns:
        dict: 标准subspace_decomposition可视化对象
    """
    layers = []
    for ld in layer_data:
        entry = {
            "layer": ld["layer"],
            "w_u_ratio": round(ld.get("w_u_ratio", 0.15), 4),
            "w_u_perp_ratio": round(ld.get("w_u_perp_ratio", 0.85), 4),
            "grammar_in_perp": ld.get("grammar_in_perp", {}),
            "semantics_in_w_u": round(ld.get("semantics_in_w_u", 0), 4),
        }
        if "concept_points" in ld:
            entry["concept_points"] = ld["concept_points"]
        layers.append(entry)
    
    return {
        "type": "subspace_decomposition",
        "id": f"{model}_subspace_{experiment_id}",
        "label": f"W_U/W_U⊥子空间分解 ({model})",
        "layers": layers,
    }


def export_force_line(phase, model, experiment_id, concepts_data):
    """导出force_line类型数据 (Schema v2.0)
    
    Args:
        concepts_data: list of dict, 每项包含:
            - concept: str, 概念名
            - per_layer: list of {layer, norm, cos_with_wu}
            - growth_rate: float, exp拟合系数
    
    Returns:
        dict: 标准force_line可视化对象
    """
    concepts = []
    for cd in concepts_data:
        per_layer = cd.get("per_layer", [])
        points = []
        for i, pl in enumerate(per_layer):
            # 3D坐标: X=层号均匀展开, Y=norm(对数缩放), Z=cos_with_wu
            angle = i * 0.25
            radius = 1 + np.log1p(pl.get("norm", 1)) * 0.8
            x = radius * np.cos(angle)
            y = np.log1p(pl.get("norm", 1)) * 2
            z = radius * np.sin(angle)
            exp_fit = np.exp(cd.get("growth_rate", 0.2) * pl["layer"]) if i > 0 else pl.get("norm", 1)
            points.append({
                "layer": pl["layer"],
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "z": round(float(z), 4),
                "norm": round(pl.get("norm", 0), 2),
                "cos_with_wu": round(pl.get("cos_with_wu", 0), 4),
                "exp_fit": round(float(exp_fit), 2),
            })
        concepts.append({
            "concept": cd["concept"],
            "points": points,
            "growth_rate": round(cd.get("growth_rate", 0), 4),
        })
    
    return {
        "type": "force_line",
        "id": f"{model}_force_line_{experiment_id}",
        "label": f"语义力线指数增长 ({model})",
        "concepts": concepts,
    }


def export_grammar_role_matrix(phase, model, experiment_id, roles, cosine_matrix,
                                lda_accuracy=None, causal_effect=None, transfer_kl=None):
    """导出grammar_role_matrix类型数据 (Schema v2.0)
    
    Args:
        roles: list of str, 语法角色名列表
        cosine_matrix: 2D list, 角色间余弦矩阵
        lda_accuracy: 可选, list of float, 每个角色的LDA分类准确率
        causal_effect: 可选, list of float, 每个角色的因果效应
        transfer_kl: 可选, list of float, 每个角色的迁移KL散度
    
    Returns:
        dict: 标准grammar_role_matrix可视化对象
    """
    return {
        "type": "grammar_role_matrix",
        "id": f"{model}_grammar_matrix_{experiment_id}",
        "label": f"语法角色余弦矩阵 ({model})",
        "roles": roles,
        "cosine_matrix": [[round(v, 4) for v in row] for row in cosine_matrix],
        "lda_accuracy": [round(v, 4) for v in (lda_accuracy or [])],
        "causal_effect": [round(v, 4) for v in (causal_effect or [])],
        "transfer_kl": [round(v, 4) for v in (transfer_kl or [])],
    }


def export_causal_chain(phase, model, experiment_id, intervention, propagation):
    """导出causal_chain类型数据 (Schema v2.0)
    
    Args:
        intervention: dict, {layer, subspace, direction} 干预信息
        propagation: list of dict, 每项包含:
            - layer: int
            - kl_divergence: float
            - classification_flip: float
            - top_token: 可选, str
            - prob_change: 可选, float
    
    Returns:
        dict: 标准causal_chain可视化对象
    """
    return {
        "type": "causal_chain",
        "id": f"{model}_causal_chain_{experiment_id}",
        "label": f"因果链追踪: {intervention.get('subspace','')} {intervention.get('direction','')} ({model})",
        "intervention": intervention,
        "propagation": [
            {
                "layer": p["layer"],
                "kl_divergence": round(p.get("kl_divergence", 0), 4),
                "classification_flip": round(p.get("classification_flip", 0), 4),
                "top_token": p.get("top_token", ""),
                "prob_change": round(p.get("prob_change", 0), 4),
            }
            for p in propagation
        ],
    }


def export_dark_matter_flow(phase, model, experiment_id, signal_path, cascade_transfer=None):
    """导出dark_matter_flow类型数据 (Schema v2.0)
    
    Args:
        signal_path: list of dict, 每项包含:
            - layer: int
            - w_u_signal: float, W_U可见信号占比
            - w_u_perp_signal: float, W_U⊥信号占比
            - total_norm: float, 总范数
        cascade_transfer: 可选, list of dict, 级联转导细节:
            - from_layer: int
            - to_layer: int
            - transfer_ratio: float
            - nonlinear_component: float
    
    Returns:
        dict: 标准dark_matter_flow可视化对象
    """
    return {
        "type": "dark_matter_flow",
        "id": f"{model}_dark_matter_{experiment_id}",
        "label": f"暗物质非线性转导 ({model})",
        "signal_path": [
            {
                "layer": sp["layer"],
                "w_u_signal": round(sp.get("w_u_signal", 0), 4),
                "w_u_perp_signal": round(sp.get("w_u_perp_signal", 0), 4),
                "total_norm": round(sp.get("total_norm", 0), 2),
            }
            for sp in signal_path
        ],
        "cascade_transfer": [
            {
                "from_layer": ct["from_layer"],
                "to_layer": ct["to_layer"],
                "transfer_ratio": round(ct.get("transfer_ratio", 0), 4),
                "nonlinear_component": round(ct.get("nonlinear_component", 0), 4),
            }
            for ct in (cascade_transfer or [])
        ],
    }


# ==================== 保存函数 ====================

def save_vis_file(phase, model, experiment, visualizations, model_info, summary=None, schema_version="2.0"):
    """保存标准可视化数据文件
    
    Args:
        phase: Phase编号
        model: 模型名
        experiment: 实验名
        visualizations: list of visualization objects
        model_info: dict {class, n_layers, d_model, n_heads}
        summary: 可选, 人类可读摘要
        schema_version: "1.0" 或 "2.0" (默认2.0, 向后兼容)
    
    Returns:
        Path: 保存的文件路径
    """
    VIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 检查是否包含v2.0类型
    v2_types = {"subspace_decomposition", "force_line", "grammar_role_matrix", "causal_chain", "dark_matter_flow"}
    has_v2 = any(v.get("type") in v2_types for v in visualizations)
    if has_v2:
        schema_version = "2.0"
    
    data = {
        "schema_version": schema_version,
        "phase": phase,
        "experiment": experiment,
        "model": model,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "model_info": model_info,
        "visualizations": visualizations,
        "summary": summary or {},
    }
    
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"vis_{phase}_{model}_{experiment}_{timestamp_str}.json"
    filepath = VIS_DATA_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"  [VIS] 可视化数据已保存: {filepath}")
    print(f"  [VIS] 包含 {len(visualizations)} 个可视化对象")
    
    # 同时更新manifest
    _update_manifest(filepath, phase, model, experiment)
    
    return filepath


def _update_manifest(new_filepath, phase, model, experiment):
    """更新manifest.json文件列表"""
    manifest_path = VIS_DATA_DIR / "manifest.json"
    
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    else:
        manifest = {"files": [], "last_updated": ""}
    
    # 添加新文件
    entry = {
        "filename": new_filepath.name,
        "phase": phase,
        "model": model,
        "experiment": experiment,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "label": f"{phase} {model} {experiment}",
    }
    manifest["files"].append(entry)
    manifest["last_updated"] = datetime.now().isoformat(timespec='seconds')
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"  [VIS] manifest已更新: {len(manifest['files'])} 个文件")


# ==================== 快捷批量导出 ====================

def export_experiment_results(phase, model, model_info, raw_results, 
                               experiment_type="auto"):
    """从原始实验结果自动批量导出可视化数据
    
    Args:
        raw_results: dict, 实验脚本的原始结果字典
        experiment_type: "auto"(自动推断) | "trajectory" | "point_cloud" | "heatmap"
    
    Returns:
        list: 可视化对象列表
    """
    visualizations = []
    n_layers = model_info.get("n_layers", 36)
    
    # 检测是否包含逐层cos数据 (trajectory类型)
    for key, value in raw_results.items():
        if isinstance(value, dict):
            has_layer_cos = any(
                k.startswith("per_layer_") and isinstance(v, dict)
                for k, v in value.items()
            )
            
            if has_layer_cos:
                # 这是trajectory类型的数据
                source = value.get("source", "?")
                target = value.get("target", "?")
                template = value.get("template", "The {} is")
                
                per_layer_data = []
                layers_str = sorted(
                    value.get("per_layer_delta_cos", {}).keys(), 
                    key=lambda x: int(x)
                )
                
                for li_str in layers_str:
                    li = int(li_str)
                    per_layer_data.append({
                        "layer": li,
                        "norm": value.get("per_layer_norm", {}).get(li_str, 0),
                        "cos_with_target": value.get("per_layer_cos_with_target", {}).get(li_str, 0),
                        "cos_with_source": value.get("per_layer_cos_with_source", {}).get(li_str, 0),
                        "delta_cos": value.get("per_layer_delta_cos", {}).get(li_str, 0),
                    })
                
                correction = value.get("correction_layers", [])
                if correction and isinstance(correction[0], (list, tuple)):
                    correction = [c[0] for c in correction]
                
                vis = export_trajectory(
                    phase=phase, model=model,
                    experiment_id=f"{source}_to_{target}",
                    token=target, source_token=source,
                    template=template,
                    per_layer_data=per_layer_data,
                    correction_layers=correction,
                )
                visualizations.append(vis)
    
    return visualizations
