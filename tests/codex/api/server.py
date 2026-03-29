#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI后端服务
提供RESTful API供前端访问
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.data_api import get_api, AGIDataAPI
from api.collectors import get_collector, TestResultCollector
from api.quality_checker import get_checker, DataQualityChecker
from api.viz_routes import router as viz_router
from api.layer_association_analyzer import LayerAssociationAnalyzer


app = FastAPI(
    title="AGI研究数据API",
    description="为AGI研究提供统一的数据访问接口",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册可视化路由
app.include_router(viz_router)

# 获取API实例
data_api: AGIDataAPI = get_api()
collector: TestResultCollector = get_collector()
checker: DataQualityChecker = get_checker()


# 请求和响应模型
class ConceptActivationRequest(BaseModel):
    model: str
    concept_id: str
    layer: Optional[int] = None


class CrossModelComparisonRequest(BaseModel):
    concept_ids: List[str]
    models: List[str]


class InterventionResultRequest(BaseModel):
    param_id: str
    intervention_type: str


class TemporalTrajectoryRequest(BaseModel):
    concept_id: str
    checkpoint_range: List[int]


# 数据源管理
@app.get("/api/data-sources")
async def get_data_sources():
    """获取数据源列表"""
    return data_api.get_data_source_list()


@app.get("/api/data-puzzle-categories")
async def get_data_puzzle_categories():
    """获取数据拼图分类"""
    return data_api.get_data_puzzle_categories()


# 概念激活数据
@app.post("/api/concept-activation")
async def get_concept_activation(request: ConceptActivationRequest):
    """获取概念激活数据"""
    result = data_api.get_concept_activation_data(
        model=request.model,
        concept_id=request.concept_id,
        layer=request.layer
    )
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


# 跨模型对比
@app.post("/api/cross-model-comparison")
async def get_cross_model_comparison(request: CrossModelComparisonRequest):
    """获取跨模型对比数据"""
    return data_api.get_cross_model_comparison(
        concept_ids=request.concept_ids,
        models=request.models
    )


# 因果干预结果
@app.post("/api/intervention-result")
async def get_intervention_result(request: InterventionResultRequest):
    """获取因果干预结果"""
    result = data_api.get_intervention_result(
        param_id=request.param_id,
        intervention_type=request.intervention_type
    )
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


# 时间演化轨迹
@app.post("/api/temporal-trajectory")
async def get_temporal_trajectory(request: TemporalTrajectoryRequest):
    """获取时间演化轨迹"""
    return data_api.get_temporal_trajectory(
        concept_id=request.concept_id,
        checkpoint_range=request.checkpoint_range
    )


# 共享承载机制数据
@app.get("/api/shared-bearing/{family_type}")
async def get_shared_bearing_mechanism(family_type: str, model: str = "deepseek7b"):
    """获取共享承载机制数据"""
    result = data_api.get_shared_bearing_mechanism_data(
        family_type=family_type,
        model=model
    )
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


# 跨模型同构数据
@app.post("/api/cross-model-isomorphism")
async def get_cross_model_isomorphism(models: List[str]):
    """获取跨模型同构数据"""
    return data_api.get_cross_model_isomorphism_data(models=models)


# 测试结果收集
@app.get("/api/test-results/{stage_id}")
async def get_test_result(stage_id: int):
    """获取测试结果"""
    return collector.get_result_by_stage(stage_id)


@app.get("/api/test-results")
async def list_all_test_results():
    """列出所有测试结果"""
    return collector.list_all_results()


# 数据质量检查
@app.post("/api/quality-check")
async def check_data_quality(data: Dict[str, Any]):
    """检查数据质量"""
    return checker.check_all_metrics(data)


@app.get("/api/quality-report/{stage_id}")
async def get_quality_report(stage_id: int, data: Dict[str, Any]):
    """获取质量报告"""
    report_path = checker.generate_quality_report(stage_id, data)
    return {"report_path": report_path}


# 健康检查
@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "api_version": "1.0.0"
    }


# 启动服务
if __name__ == "__main__":
    import uvicorn
    
    print("启动AGI研究数据API服务...")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
