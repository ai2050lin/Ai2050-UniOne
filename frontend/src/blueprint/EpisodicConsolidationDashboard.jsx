import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer,
  Cell
} from 'recharts';
import { CheckCircle2, XCircle } from 'lucide-react';

const EpisodicConsolidationDashboard = () => {
  const [data, setData] = useState(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const json = JSON.parse(e.target.result);
          setData(json);
        } catch (err) {
          console.error("Failed to parse JSON", err);
        }
      };
      reader.readAsText(file);
    }
  };

  const loadDefaultData = () => {
    const defaultData = {
      "meta": {
        "timestamp": "2026-03-11 15:00:00",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "entities_count": 20,
        "macro_concept": "fruit",
        "runtime_sec": 12.45
      },
      "metrics": {
        "sequence_crowded_norm": 124.58,
        "average_micro_norm": 42.15,
        "abstract_macro_norm": 45.22,
        "total_micro_variance": 5420.76,
        "post_consolidation_residual_variance": 315.42,
        "compression_ratio": 17.18,
        "semantic_retention_cosine": 0.965
      },
      "hypotheses": {
        "H1_sequence_overload_detected": true,
        "H2_consolidation_reduces_variance": true,
        "H3_semantic_identity_retained": true
      }
    };
    setData(defaultData);
  };

  if (!data) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>重绑定期 (Episodic Consolidation) 记忆切分测试看板</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4">
            <p className="text-sm text-gray-400">
              请导入 `episodic_consolidation_*.json` 文件，或使用默认演示数据（基于 AGI_GEMINI_MEMO.md 最新理论推导生成）。
            </p>
            <div className="flex gap-4">
              <input
                type="file"
                accept=".json"
                onChange={handleFileUpload}
                className="file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-600 cursor-pointer"
              />
              <button
                onClick={loadDefaultData}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-md text-sm font-semibold"
              >
                加载最新测试集
              </button>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { meta, metrics, hypotheses } = data;

  const normData = [
    { name: '单体平均微观特征模长', value: metrics.average_micro_norm, color: '#3b82f6' },
    { name: '拥挤序列向量总模长', value: metrics.sequence_crowded_norm, color: '#ef4444' },
    { name: '宏观单体抽象向量模长', value: metrics.abstract_macro_norm, color: '#10b981' },
  ];

  const varianceData = [
    { name: '微观明细拥挤总方差', value: metrics.total_micro_variance, color: '#f59e0b' },
    { name: '重绑定折叠后残差方差', value: metrics.post_consolidation_residual_variance, color: '#8b5cf6' },
  ];

  return (
    <Card className="w-full bg-slate-900 border-slate-800">
      <CardHeader>
        <CardTitle className="text-xl text-blue-400">重绑定期 (Episodic Consolidation) 抽象算子测试</CardTitle>
        <div className="text-xs text-slate-400 mt-2 space-y-1">
          <p>模型底座: <span className="text-slate-200">{meta.model_id}</span></p>
          <p>实体序列容量过载测试数: <span className="text-slate-200">{meta.entities_count}</span> -> 抽象宏观锚点: <span className="text-slate-200">{meta.macro_concept}</span></p>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-slate-800 p-4 rounded-lg flex flex-col items-center justify-center">
            <span className="text-slate-400 text-sm mb-1">降维压缩比 (Release Ratio)</span>
            <span className="text-3xl font-bold text-emerald-400">{metrics.compression_ratio.toFixed(2)}x</span>
          </div>
          <div className="bg-slate-800 p-4 rounded-lg flex flex-col items-center justify-center">
            <span className="text-slate-400 text-sm mb-1">语义特征保真度 (Cosine)</span>
            <span className="text-3xl font-bold text-blue-400">{metrics.semantic_retention_cosine.toFixed(4)}</span>
          </div>
          <div className="bg-slate-800 p-4 rounded-lg flex flex-col justify-center">
            <span className="text-slate-400 text-sm mb-2 text-center">核心假设验证</span>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                {hypotheses.H1_sequence_overload_detected ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <XCircle className="w-4 h-4 text-red-400" />}
                <span className={hypotheses.H1_sequence_overload_detected ? "text-slate-200" : "text-slate-500"}>H1: 序列超载诱发维度坍缩</span>
              </div>
              <div className="flex items-center gap-2">
                {hypotheses.H2_consolidation_reduces_variance ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <XCircle className="w-4 h-4 text-red-400" />}
                <span className={hypotheses.H2_consolidation_reduces_variance ? "text-slate-200" : "text-slate-500"}>H2: 重绑定折叠释放信噪方差空间</span>
              </div>
              <div className="flex items-center gap-2">
                {hypotheses.H3_semantic_identity_retained ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <XCircle className="w-4 h-4 text-red-400" />}
                <span className={hypotheses.H3_semantic_identity_retained ? "text-slate-200" : "text-slate-500"}>H3: 折叠后语义身份高阶维系</span>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-slate-800">
          <div className="h-64">
             <h3 className="text-sm text-slate-400 mb-4 text-center">序列超载模长 vs 抽象概念模长对比</h3>
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={normData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                 <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                 <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                 <YAxis stroke="#94a3b8" />
                 <RechartsTooltip cursor={{ fill: '#1e293b' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                 <Bar dataKey="value" name="向量模长 (Norm L2)">
                   {normData.map((entry, index) => (
                     <Cell key={`cell-${index}`} fill={entry.color} />
                   ))}
                 </Bar>
               </BarChart>
             </ResponsiveContainer>
          </div>

          <div className="h-64">
             <h3 className="text-sm text-slate-400 mb-4 text-center">微观细节方差压缩前后对比</h3>
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={varianceData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                 <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                 <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                 <YAxis stroke="#94a3b8" />
                 <RechartsTooltip cursor={{ fill: '#1e293b' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                 <Bar dataKey="value" name="方差 (Variance)">
                   {varianceData.map((entry, index) => (
                     <Cell key={`cell-${index}`} fill={entry.color} />
                   ))}
                 </Bar>
               </BarChart>
             </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-blue-900/20 p-4 rounded-lg border border-blue-800/50 mt-4">
          <h4 className="text-sm font-semibold text-blue-400 mb-2">💡 原理说明：为何要引入“重绑定期 (Episodic Consolidation)”</h4>
          <p className="text-xs text-slate-300 leading-relaxed">
            长上下文在流形特征空间里的简单张量叠加（即使使用了全息简化的 HRR）也必然受到维度方差爆发的影响。这就好比在一张 {4096} 像素的桌布上堆放大量的“具体商品细节”，没放几件桌布就会被塞满产生信息碰撞。
            <br/><br/>
            该现象通过左侧“向量集模长爆炸”与右侧“总方差过盈”清晰呈现。而此处的“摘要算法切片算子”（利用模型顶层宏观聚类头的输出倒提投影）能在短时间内将细杂概念如“apple, banana, grape ...”折叠吸纳为单个宏观词汇“fruit”，从而令剩余的无用方差直接坍缩掉，把信息密度腾空并交还给网络，避免算力与容量墙被强行突破。
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default EpisodicConsolidationDashboard;
