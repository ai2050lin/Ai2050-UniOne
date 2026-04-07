import React, { useState } from 'react';
import { Compass, Zap, Shield, Target, Layers, ArrowRight, RotateCcw, Activity, Sigma } from 'lucide-react';

const AxiomCard = ({ icon: Icon, title, description, math, reasoning, color }) => {
  const [showMath, setShowMath] = useState(false);

  return (
    <div style={{
      background: 'rgba(255, 255, 255, 0.03)',
      border: `1px solid ${color}33`,
      borderRadius: '16px',
      padding: '24px',
      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      cursor: 'pointer',
      position: 'relative',
      overflow: 'hidden'
    }}
    onMouseEnter={(e) => {
      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.07)';
      e.currentTarget.style.transform = 'translateY(-4px)';
      e.currentTarget.style.boxShadow = `0 12px 24px -10px ${color}44`;
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)';
      e.currentTarget.style.transform = 'translateY(0)';
      e.currentTarget.style.boxShadow = 'none';
    }}
    onClick={() => setShowMath(!showMath)}
    >
      <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start', marginBottom: '16px' }}>
        <div style={{ padding: '10px', borderRadius: '12px', background: `${color}22`, boxShadow: `inset 0 0 10px ${color}33` }}>
          <Icon size={22} color={color} />
        </div>
        <div>
          <div style={{ color: '#fff', fontSize: '16px', fontWeight: 'bold', letterSpacing: '0.01em' }}>{title}</div>
          <div style={{ color: '#94a3b8', fontSize: '13px', lineHeight: '1.6', marginTop: '6px' }}>{description}</div>
        </div>
      </div>
      
      <div style={{ 
        background: 'rgba(0,0,0,0.4)', 
        padding: '12px 16px', 
        borderRadius: '12px', 
        fontFamily: '"SF Mono", "Fira Code", monospace', 
        fontSize: '12px', 
        color: color,
        border: `1px dashed ${color}44`,
        transition: 'all 0.4s ease',
        opacity: showMath ? 1 : 0.6,
        transform: showMath ? 'scale(1.02)' : 'scale(1)',
        boxShadow: showMath ? `0 0 15px ${color}22` : 'none'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '4px', opacity: 0.8, fontSize: '10px', textTransform: 'uppercase' }}>Formula:</div>
        {math}
      </div>

      {showMath && (
        <div style={{ 
          marginTop: '16px', 
          padding: '12px', 
          background: 'rgba(255,255,255,0.05)', 
          borderRadius: '8px', 
          fontSize: '11px', 
          color: '#d1d5db',
          lineHeight: '1.5',
          animation: 'fadeIn 0.3s ease'
        }}>
          <div style={{ fontWeight: 'bold', color: color, marginBottom: '6px', fontSize: '10px' }}>推导过程：</div>
          {reasoning}
        </div>
      )}
    </div>
  );
};

const GLM5TheorySimplifiedDashboard = () => {
  const axioms = [
    {
      icon: Compass,
      title: "高维几何公理",
      description: "揭示 4096 维空间的“社交距离”。随机方向自然正交，确保语义互不干扰。",
      math: "E[|cos(θ)|] ≈ O(1/√d)",
      reasoning: "基于测度集中原理（Concentration of Measure），d 维球面上任意两点几乎必定位于对方的“赤道”带（垂直面）。对于 4096 维，正交偏离度仅为 0.015。",
      color: "#60a5fa"
    },
    {
      icon: Layers,
      title: "信息域公理",
      description: "输入分布决定编码策略。多模态通过密集纠缠换容量，纯文本在 d 维下保持低压正交。",
      math: "Entanglement ≈ C · √(K/d)",
      reasoning: "根据 JL 引理，当特征数 K 超过维度 d 时，相干性强制上升。这代表了知识域的“堆积压力”，解释了为何复杂任务会导致模型模糊。",
      color: "#a855f7"
    },
    {
      icon: Shield,
      title: "归一化隔离公理",
      description: "RMSNorm 充当系统内部的“防洪堤”，逐层重置能量，防止信号发散与淹没。",
      math: "Output = Norm(h + Δh)",
      reasoning: "通过将范数重置为 √d，系统强制每一层只计算“方向增量”而不仅仅是积累。这实现了本征语义在不同层间的软隔离。",
      color: "#10b981"
    },
    {
      icon: Target,
      title: "信号聚焦公理",
      description: "架构在训练中演化为“自发透镜”，将漫反射信号过滤并汇聚至概率重心的 Logit 方向。",
      math: "SNR_late = SNR_early · Π (Focus_l)",
      reasoning: "深度权重链充当各向异性的语义矩阵滤波器。非相关噪音偏向奇异值低位被衰减，而语义目标被持续投影对齐，最终实现“一语中的”。",
      color: "#fbbf24"
    },
    {
      icon: Sigma,
      title: "Logit 精确公理",
      description: "输出并非玄学猜测，而是向量在词空间上的完美线性投影。最后一步证明无秘密遗留。",
      math: "Margin = h_L · (w_i - w_j)",
      reasoning: "Unembedding 层是一个纯粹的投影操作。这证明了 AGI 的全部逻辑必须在进入该层前被压缩成一个线性可读的“决策向量”。",
      color: "#f87171"
    }
  ];

  return (
    <div style={{
      padding: '32px',
      background: 'rgba(15, 23, 42, 0.4)',
      borderRadius: '28px',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      backdropFilter: 'blur(30px)',
      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'baseline',
        marginBottom: '40px',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        paddingBottom: '20px'
      }}>
        <h2 style={{ margin: 0, color: '#fff', fontSize: '24px', fontWeight: '800', display: 'flex', alignItems: 'center', gap: '12px' }}>
           <Activity color="#6366f1" size={28} /> GLM5 统一理论 v5.0：智能物理还原
        </h2>
        <span style={{ fontSize: '12px', color: '#6366f1', fontWeight: '600', letterSpacing: '0.1em' }}>PRECISION MATH ANALYSIS</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '28px', marginBottom: '40px' }}>
        <div style={{
          background: 'linear-gradient(145deg, rgba(99, 102, 241, 0.15) 0%, rgba(30, 41, 59, 0.5) 100%)',
          padding: '28px',
          borderRadius: '24px',
          border: '1px solid rgba(99, 102, 241, 0.3)',
          transition: 'all 0.3s ease'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#818cf8', marginBottom: '20px' }}>
            <RotateCcw size={22} />
            <span style={{ fontWeight: '800', fontSize: '15px' }}>1. 陀螺与进动：旋转动力学还原</span>
          </div>
          <div style={{ position: 'relative', height: '140px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ 
              width: '90px', height: '90px', 
              borderRadius: '50%', border: '5px solid #818cf8', 
              borderTopColor: 'transparent',
              animation: 'spin 1.5s cubic-bezier(0.5, 0.1, 0.5, 0.9) infinite',
              boxShadow: '0 0 30px rgba(129, 140, 248, 0.2)'
            }} />
            <div style={{ position: 'absolute', textAlign: 'center' }}>
              <div style={{ fontSize: '28px', fontWeight: '900', color: '#fff', textShadow: '0 0 10px rgba(129, 140, 248, 0.6)' }}>92%</div>
              <div style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 'bold' }}>宏观旋转 (Macro-Align)</div>
            </div>
            <div style={{ position: 'absolute', right: '5%', top: '20%', borderLeft: '3px solid #f43f5e', paddingLeft: '12px', background: 'rgba(244, 63, 94, 0.05)', borderRadius: '0 8px 8px 0' }}>
               <div style={{ color: '#f43f5e', fontWeight: '900', fontSize: '16px' }}> {"< 8.0%"}</div>
               <div style={{ color: '#94a3b8', fontSize: '11px', fontWeight: '500' }}>真正承载智力的语义进动 (delta-h)</div>
            </div>
          </div>
          <p style={{ color: '#94a3b8', fontSize: '13px', lineHeight: '1.7', marginTop: '20px' }}>
            深度网络并非通过改变整体方向来思考。宏观大背景（92% 的旋转）在启动时已对齐，而真正的逻辑推理发生在极其微妙的<span style={{color:'#f43f5e',fontWeight:'bold'}}>进动微调层面</span>。
          </p>
        </div>

        <div style={{
          background: 'linear-gradient(145deg, rgba(16, 185, 129, 0.15) 0%, rgba(30, 41, 59, 0.5) 100%)',
          padding: '28px',
          borderRadius: '24px',
          border: '1px solid rgba(16, 185, 129, 0.3)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#34d399', marginBottom: '20px' }}>
            <Activity size={22} />
            <span style={{ fontWeight: '800', fontSize: '15px' }}>3. 编码基元：因果幅度的真相</span>
          </div>
          <div style={{ display: 'flex', height: '140px', alignItems: 'flex-end', gap: '24px', padding: '0 30px' }}>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                <div style={{ width: '40px', height: '40px', background: 'rgba(52, 211, 153, 0.2)', borderRadius: '6px', border: '1px solid rgba(52, 211, 153, 0.3)' }} />
                <span style={{ fontSize: '11px', color: '#94a3b8', fontWeight: '600' }}>方向 (A)</span>
            </div>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                <div style={{ position: 'relative', width: '40px', height: '100px', background: 'linear-gradient(180deg, #34d399 0%, rgba(52, 211, 153, 0.1) 100%)', borderRadius: '6px', border: '1px solid #34d399', boxShadow: '0 0 20px rgba(52, 211, 153, 0.3)' }}>
                   <div style={{ position: 'absolute', top: '-10px', left: '-10px', right: '-10px', background: '#34d399', color: '#000', fontSize: '9px', fontWeight: '900', textAlign: 'center', borderRadius: '2px' }}>DOMINANT</div>
                </div>
                <span style={{ fontSize: '11px', color: '#34d399', fontWeight: '800' }}>呐喊 (E_gate)</span>
            </div>
          </div>
          <p style={{ color: '#94a3b8', fontSize: '13px', lineHeight: '1.7', marginTop: '20px' }}>
            最新的因果审计表明：指向哪个方位（方向）固然重要，但指向后喊出的那一嗓子<span style={{color:'#34d399',fontWeight:'bold'}}>呐喊能量 (E_gate)</span>才是下游系统能否捕捉语义的关键。
          </p>
        </div>
      </div>

      <div>
        <div style={{ color: '#fff', fontSize: '18px', fontWeight: '800', marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Sigma size={20} color="#6366f1" /> 2. 统一理论五大公理 (Mathematical Axioms)
          <span style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 'normal', marginLeft: 'auto' }}>点击卡片查看深层推理 →</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
          {axioms.map((axiom, idx) => (
            <AxiomCard key={idx} {...axiom} />
          ))}
        </div>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
};

export default GLM5TheorySimplifiedDashboard;
