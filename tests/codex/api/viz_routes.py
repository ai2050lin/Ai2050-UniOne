#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化路由
扩展FastAPI服务器，添加可视化相关接口
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json

from api.visualization_helpers import get_visualization_helper
from api.data_api import get_api


router = APIRouter(prefix="/api/visualization", tags=["Visualization"])

# 获取实例
viz_helper = get_visualization_helper()
data_api = get_api()


# 请求模型
class SharedBearingRequest(BaseModel):
    family_type: str
    model: str = "deepseek7b"


class CrossModelVisualizationRequest(BaseModel):
    concept_ids: List[str]
    models: List[str]


class TemporalVisualizationRequest(BaseModel):
    concept_id: str
    checkpoint_range: List[int]


class InterventionVisualizationRequest(BaseModel):
    param_id: str
    intervention_type: str = "ablation"


@router.post("/shared-bearing/heatmap")
async def get_shared_bearing_heatmap(request: SharedBearingRequest):
    """获取共享承载机制热图数据"""
    result = viz_helper.prepare_shared_bearing_heatmap_data(
        family_type=request.family_type,
        model=request.model
    )
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


@router.post("/shared-bearing/scatter")
async def get_shared_bearing_scatter(request: SharedBearingRequest):
    """获取承载机制散点图数据"""
    result = viz_helper.prepare_bearing_scatter_plot_data(
        family_type=request.family_type,
        model=request.model
    )
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


@router.post("/shared-bearing/network")
async def get_bearing_network_graph(request: SharedBearingRequest):
    """获取承载关系网络图数据"""
    result = viz_helper.prepare_network_graph_data(
        family_type=request.family_type,
        model=request.model
    )
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


@router.post("/cross-model/comparison")
async def get_cross_model_comparison(request: CrossModelVisualizationRequest):
    """获取跨模型对比可视化数据"""
    result = viz_helper.prepare_cross_model_comparison_data(
        concept_ids=request.concept_ids,
        models=request.models
    )
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


@router.post("/temporal/trajectory")
async def get_temporal_trajectory(request: TemporalVisualizationRequest):
    """获取时间演化轨迹可视化数据"""
    result = viz_helper.prepare_temporal_trajectory_data(
        concept_id=request.concept_id,
        checkpoint_range=request.checkpoint_range
    )
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


@router.post("/intervention/result")
async def get_intervention_visualization(request: InterventionVisualizationRequest):
    """获取干预结果可视化数据"""
    result = viz_helper.prepare_intervention_result_data(
        param_id=request.param_id,
        intervention_type=request.intervention_type
    )
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


@router.post("/demo-data")
async def generate_demo_data():
    """生成可视化示例数据"""
    from api.visualization_demo import save_demo_data
    save_demo_data()
    return {"status": "success", "message": "示例数据已生成"}


@router.get("/chart-types")
async def get_chart_types():
    """获取支持的图表类型"""
    return {
        "chart_types": [
            {
                "id": "heatmap",
                "name": "热图",
                "description": "展示高维矩阵数据，适合展示激活模式"
            },
            {
                "id": "scatter",
                "name": "散点图",
                "description": "展示两个维度之间的关系"
            },
            {
                "id": "bar",
                "name": "柱状图",
                "description": "对比不同类别或条件的数据"
            },
            {
                "id": "line",
                "name": "折线图",
                "description": "展示数据随时间的变化"
            },
            {
                "id": "network",
                "name": "网络图",
                "description": "展示参数间的连接关系"
            }
        ]
    }
