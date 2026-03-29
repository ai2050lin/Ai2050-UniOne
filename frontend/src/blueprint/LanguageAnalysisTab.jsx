import { BookOpen, Code, Database, Brain, Sparkles, CheckCircle, AlertCircle, Layers, Target, Puzzle, Trophy, TrendingUp, Zap, ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';

/**
 * 语言分析标签页
 * 展示语言特性分析和当前研究进展
 * 采用从上到下的垂直布局，不使用tab切换
 */
export const LanguageAnalysisTab = () => {
  const [expandedSection, setExpandedSection] = useState(null);
  const [expandedPuzzle, setExpandedPuzzle] = useState(null);
  const [expandedPreparation, setExpandedPreparation] = useState(null);

  // 语言核心特性数据
  const languageCharacteristics = [
    {
      id: 'encoding_mechanism',
      title: '编码机制 Encoding Mechanism',
      icon: <Database size={20} />,
      status: 'verified',
      statusText: '已验证 Verified',
      description: '语言通过参数级的编码机制实现语义表示',
      keyFindings: [
        '共享承载 -> 偏置偏转 -> 逐层放大 Shared Carrier -> Bias Deflection -> Layer Amplification',
        '名词形成家族片区 Family Patch，属性形成属性纤维 Attribute Fiber',
        '颜色编码：约61.1%参数共享 + 38.9%对象特异性 Color Encoding: 61.1% Shared + 38.9% Object-Specific',
        '路径机制：共享纤维层相同，对象路由和上下文绑定层分叉 Path Mechanism: Shared at fiber layer, divergent at route and context layers',
      ],
      progress: 0.85,
      confidence: 0.92,
      evidence: '多空间角色对齐图 Stage337, 偏转流形轨迹图 Stage339',
      latestStage: 'Stage413',
    },
    {
      id: 'semantic_structure',
      title: '语义结构 Semantic Structure',
      icon: <Brain size={20} />,
      status: 'in_progress',
      statusText: '研究中 In Progress',
      description: '语言的语义通过分层的数学结构表示',
      keyFindings: [
        '基础编码层：静态特征表示 Base Encoding Layer: Static Feature Representation',
        '动态路径层：语义推理 Dynamic Path Layer: Semantic Reasoning',
        '结果回收层：输出整合 Result Recycling Layer: Output Integration',
        '传播编码层：跨层传播 Propagation Encoding Layer: Cross-layer Propagation',
        '语义角色层：语法结构 Semantic Role Layer: Syntactic Structure',
      ],
      progress: 0.68,
      confidence: 0.75,
      evidence: '多空间对齐分析 Stage337, 模糊承载簇复核 Stage338',
      latestStage: 'Stage338',
    },
    {
      id: 'multi_modal_semantics',
      title: '多模态语义 Multi-Modal Semantics',
      icon: <Layers size={20} />,
      status: 'research_needed',
      statusText: '待研究 To Research',
      description: '语言语义在不同模态间的关联和映射机制',
      keyFindings: [
        '语言指令可以直接关联到图片区域 "修改左边苹果颜色"',
        '编程任务中"重构xx文件"可以进行重构操作',
        '说明语言本身存在跨模态的语义绑定',
        '跨模态语义绑定的神经元级机制尚未阐明',
      ],
      progress: 0.25,
      confidence: 0.45,
      evidence: '理论假设，需要实验验证',
      latestStage: 'Planning Phase',
    },
    {
      id: 'dynamic_learning',
      title: '动态学习 Dynamic Learning',
      icon: <Sparkles size={20} />,
      status: 'in_progress',
      statusText: '研究中 In Progress',
      description: '新概念和新属性如何被编码和学习',
      keyFindings: [
        '可容许更新 Admissible Update机制',
        '受限读取 Restricted Readout机制',
        '阶段条件传输 Stage-Conditioned Transport',
        '继承对齐传输 Successor-Aligned Transport',
        '协议桥接 Protocol Bridge机制',
      ],
      progress: 0.55,
      confidence: 0.68,
      evidence: 'ICSPB理论框架，在线学习实验',
      latestStage: 'Stage400-450',
    },
  ];

  // 当前研究进展数据
  const researchProgress = [
    {
      id: 'color_encoding',
      title: '颜色编码分析 Color Encoding Analysis',
      stage: 'Stage413',
      status: 'completed',
      date: '2026-03-29 15:16',
      summary: '分析了"苹果的红色"和"路灯的红色"的编码机制，确认了共享参数 + 对象特异性参数的混合编码模式',
      keyMetrics: {
        'Shared Param Ratio': '61.1%',
        'Object-Specific Ratio': '38.9%',
        'Shared Fiber Strength': '0.8420',
        'Path Similarity': '0.7120',
      },
      conclusions: [
        '不同对象的红色共享核心编码，但不是完全相同的参数',
        '路径机制相同在共享纤维层，不同在对象路由和上下文绑定层',
        '支持"共享属性纤维 + 对象路由分叉 + 上下文绑定分叉"的理论预期',
      ],
      artifacts: [
        'tests/codex_temp/test_color_pathway_mechanism_analysis.py',
        'research/gpt5/docs/COLOR_ENCODING_MECHANISM_DEEP_ANALYSIS.md',
        'research/gpt5/docs/COLOR_ENCODING_SUMMARY_PLAIN_CHINESE.md',
      ],
    },
    {
      id: 'noun_attribute',
      title: '名词属性神经元特性 Noun-Attribute Neuron Characteristics',
      stage: 'Stage414',
      status: 'completed',
      date: '2026-03-29 15:32',
      summary: '分析了名词和属性在神经元参数层面的特性差异，明确了功能分化和数学特征',
      keyMetrics: {
        'Noun Patch Strength': '0.7542',
        'Attribute Fiber Strength': '0.8726',
        'Noun Active Neurons': '847',
        'Attribute Active Neurons': '324',
        'Within-Family Similarity': '0.8234',
      },
      conclusions: [
        '名词形成稳定的局部密集片区，编码实体本身',
        '属性形成稀疏的纤维方向，跨对象共享',
        '支持了神经元功能分化的明确性',
      ],
      artifacts: [
        'tests/codex_temp/test_noun_attribute_neuron_param_analysis.py',
        'research/gpt5/docs/NOUN_ATTRIBUTE_NEURON_CHARACTERISTICS.md',
      ],
    },
    {
      id: 'multi_space_alignment',
      title: '多空间角色对齐 Multi-Space Role Alignment',
      stage: 'Stage337',
      status: 'completed',
      date: '2026-03-24',
      summary: '分析了对象空间、任务空间、传播空间的原始对齐情况',
      keyMetrics: {
        'Overall Alignment': '0.3784',
        'Object Space': '0.1349',
        'Task Space': '0.4993',
        'Propagation Space': '0.3577',
      },
      conclusions: [
        '对象空间的原始对齐最清楚',
        '任务空间和传播空间已经显影',
        '整体厚度仍然不够，离统一多空间结构还差一段',
      ],
      artifacts: [
        'tests/codex/stage337_multi_space_role_raw_alignment.py',
      ],
    },
  ];

  // 分析拼图数据
  const puzzleCategories = [
    {
      id: 'foundation_structure',
      title: '基础结构拼图',
      icon: <Layers size={20} />,
      description: '沉淀基础神经元和对象族的稳定结构证据',
      puzzles: [
        {
          id: 'shared_carrier',
          title: '共享承载机制',
          stage: 'Stage294-298',
          evidenceStrength: 3,
          completeness: 0.65,
          keyData: ['跨家族共享量化', 'base_load指标分布', '因果效应初步建立'],
          completenessCheck: ['✅ 现象描述和量化', '❌ 承载覆盖范围验证', '❌ 跨模型稳定性证据'],
          conclusions: '不同对象族共享基础编码载体，但覆盖范围和稳定性需更大规模验证',
        },
        {
          id: 'bias_deflection',
          title: '偏置偏转机制',
          stage: 'Stage295-299',
          evidenceStrength: 3,
          completeness: 0.60,
          keyData: ['对象切换中的作用', 'selectivity杠杆指标', '品牌/类内/细粒度方向区分'],
          completenessCheck: ['✅ 现象描述和量化', '❌ 因果干预验证', '❌ 独立性验证不足'],
          conclusions: '偏置位实现对象间区分，但缺少真正的反事实推理验证',
        },
        {
          id: 'family_patch',
          title: '家族片区机制',
          stage: 'Stage414',
          evidenceStrength: 4,
          completeness: 0.75,
          keyData: ['名词片区强度: 0.7542', '同族相似度: 0.8234', '847个激活神经元'],
          completenessCheck: ['✅ 量化指标成熟', '✅ 多名词验证', '⚠ 跨模型验证不足'],
          conclusions: '名词在参数空间形成稳定的局部密集片区，编码实体本身',
        },
        {
          id: 'attribute_fiber',
          title: '属性纤维机制',
          stage: 'Stage414',
          evidenceStrength: 4,
          completeness: 0.78,
          keyData: ['属性纤维强度: 0.8726', '跨对象相似度: 0.7915', '324个激活神经元'],
          completenessCheck: ['✅ 量化指标成熟', '✅ 多对象验证', '✅ 颜色编码深度分析'],
          conclusions: '属性形成稀疏的纤维方向，跨对象共享，编码修饰性特征',
        },
      ],
    },
    {
      id: 'parameter_coupling',
      title: '参数耦合拼图',
      icon: <Target size={20} />,
      description: '定位参数位的可分性、塌缩和耦合裂缝',
      puzzles: [
        {
          id: 'color_encoding',
          title: '颜色编码机制',
          stage: 'Stage413',
          evidenceStrength: 5,
          completeness: 0.85,
          keyData: ['共享参数比例: 61.1%', '对象特异性: 38.9%', '路径相似度: 0.7120'],
          completenessCheck: ['✅ 参数级量化', '✅ 路径机制分析', '✅ 理论预期验证'],
          conclusions: '不同对象的红色共享核心编码(61.1%)，但路径在对象路由和上下文绑定层分叉',
        },
        {
          id: 'layer_amplification',
          title: '逐层放大机制',
          stage: 'Stage319-343',
          evidenceStrength: 2,
          completeness: 0.45,
          keyData: ['早层第一次放大', '中层主放大', '后层持续放大'],
          completenessCheck: ['✅ 层次化描述', '❌ 层次划分粗糙', '❌ 每层功能验证不足'],
          conclusions: '编码逐层放大，但层次划分过于粗糙，缺少对每层具体功能的验证',
        },
      ],
    },
    {
      id: 'multi_space_mapping',
      title: '多空间映射拼图',
      icon: <Brain size={20} />,
      description: '分析对象空间、任务空间、传播空间等不同空间的角色',
      puzzles: [
        {
          id: 'space_alignment',
          title: '多空间角色对齐',
          stage: 'Stage337',
          evidenceStrength: 3,
          completeness: 0.60,
          keyData: ['整体对齐度: 0.3784', '对象空间: 0.1349', '任务空间: 0.4993'],
          completenessCheck: ['✅ 空间映射概念', '❌ 空间间相互作用分析不足', '❌ 因果验证深度有限'],
          conclusions: '对象空间对齐最清楚，任务空间和传播空间已显影，整体厚度仍不够',
        },
        {
          id: 'cross_model',
          title: '跨模型验证',
          stage: 'Stage141-159',
          evidenceStrength: 4,
          completeness: 0.80,
          keyData: ['3层同构', 'embedding级共享核概念', '层同构评分算法'],
          completenessCheck: ['✅ 跨模型验证方法成熟', '✅ 量化指标严谨', '❌ 模型数量较少'],
          conclusions: 'GPT-2、Qwen3-4B、DeepSeek-R1-Distill-Qwen-7B三层同构，验证编码机制的普适性',
        },
      ],
    },
    {
      id: 'semantic_structure',
      title: '语义结构拼图',
      icon: <Code size={20} />,
      description: '分析语言的分层语义结构',
      puzzles: [
        {
          id: 'five_layer_structure',
          title: '五层语义结构',
          stage: 'Stage400-450',
          evidenceStrength: 3,
          completeness: 0.68,
          keyData: ['基础编码层: 静态特征', '动态路径层: 语义推理', '结果回收层: 输出整合'],
          completenessCheck: ['✅ 五层结构识别', '⚠ 层间交互机制需深入研究', '⚠ 动态机制未阐明'],
          conclusions: '识别五层结构：基础编码、动态路径、结果回收、传播编码、语义角色',
        },
        {
          id: 'multi_modal',
          title: '多模态语义',
          stage: 'Planning',
          evidenceStrength: 1,
          completeness: 0.25,
          keyData: ['语言指令关联图片区域', '编程任务执行重构', '跨模态语义绑定'],
          completenessCheck: ['⚠ 现象已观察到', '❌ 神经元级机制未阐明', '❌ 需要实验验证'],
          conclusions: '语言本身存在跨模态的语义绑定，但神经元级机制尚未阐明',
        },
      ],
    },
  ];

  // 突破准备建议
  const breakthroughPreparation = [
    {
      id: 'math_formalization',
      title: '数学形式化',
      icon: <Trophy size={20} />,
      priority: 'high',
      description: '建立严格的数学框架，定义编码空间、片区、纤维的拓扑性质',
      tasks: [
        '定义编码空间的拓扑结构和度量',
        '建立编码方程（名词、属性、耦合）',
        '证明理论性质（稀疏性上界、正交性、可加性条件）',
      ],
      timeline: '3-6个月',
      dependencies: ['foundation_structure', 'parameter_coupling'],
    },
    {
      id: 'large_scale_validation',
      title: '大规模验证',
      icon: <TrendingUp size={20} />,
      priority: 'high',
      description: '在数百个概念和多个模型上验证编码机制',
      tasks: [
        '在数百个名词和属性上验证',
        '在多个模型上验证（GPT-2, GPT-3, Claude等）',
        '在多模态模型上验证（CLIP等）',
      ],
      timeline: '6-12个月',
      dependencies: ['foundation_structure', 'multi_space_mapping'],
    },
    {
      id: 'causal_mechanism',
      title: '因果机制研究',
      icon: <Target size={20} />,
      priority: 'medium',
      description: '从相关性走向因果关系，建立真正的因果推理',
      tasks: [
        '设计参数级别的因果干预实验',
        '使用Do-calculus等因果推理框架',
        '建立反事实推理能力',
      ],
      timeline: '12-18个月',
      dependencies: ['parameter_coupling', 'multi_space_mapping'],
    },
    {
      id: 'brain_side_verification',
      title: '脑侧验证',
      icon: <Brain size={20} />,
      priority: 'medium',
      description: '在真实大脑中验证理论预测',
      tasks: [
        '设计可证伪的脑实验',
        '进行fMRI实验测量响应',
        '进行单神经元记录验证',
      ],
      timeline: '18-24个月',
      dependencies: ['semantic_structure', 'math_formalization'],
    },
  ];

  const toggleSection = (sectionId) => {
    setExpandedSection(expandedSection === sectionId ? null : sectionId);
  };

  const togglePuzzle = (puzzleId) => {
    setExpandedPuzzle(expandedPuzzle === puzzleId ? null : puzzleId);
  };

  const togglePreparation = (prepId) => {
    setExpandedPreparation(expandedPreparation === prepId ? null : prepId);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'verified':
      case 'completed':
        return '#10b981';
      case 'in_progress':
        return '#f59e0b';
      case 'research_needed':
        return '#6366f1';
      default:
        return '#6b7280';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'verified':
      case 'completed':
        return <CheckCircle size={16} />;
      case 'in_progress':
        return <Sparkles size={16} />;
      case 'research_needed':
        return <AlertCircle size={16} />;
      default:
        return null;
    }
  };

  const getEvidenceStars = (strength) => {
    const stars = '⭐'.repeat(strength) + '☆'.repeat(5 - strength);
    return stars;
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high':
        return '#ef4444';
      case 'medium':
        return '#f59e0b';
      case 'low':
        return '#10b981';
      default:
        return '#6b7280';
    }
  };

  return (
    <div style={{
      padding: '20px',
      maxWidth: '1600px',
      margin: '0 auto',
    }}>
      {/* Page Header */}
      <div style={{
        marginBottom: '30px',
        borderBottom: '1px solid rgba(255,255,255,0.1)',
        paddingBottom: '20px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
          <BookOpen size={32} color="#00d2ff" />
          <h1 style={{
            fontSize: '28px',
            fontWeight: 'bold',
            margin: 0,
            color: '#fff',
          }}>
            语言分析 Language Analysis
          </h1>
        </div>
        <p style={{
          fontSize: '14px',
          color: '#888',
          margin: 0,
          lineHeight: '1.6',
        }}>
          深入分析语言的数学结构特性、以及背后的编码机制
          <br />
          In-depth analysis of the mathematical structure characteristics of language, current research progress, and puzzle accumulation
        </p>
      </div>

      {/* Language Core Framework */}
      <div style={{
        background: 'rgba(0, 50, 100, 0.2)',
        borderRadius: '16px',
        padding: '24px',
        border: '1px solid rgba(0, 210, 255, 0.2)',
        marginBottom: '40px',
      }}>
        <h2 style={{
          fontSize: '20px',
          fontWeight: 'bold',
          color: '#00d2ff',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}>
          <Brain size={24} />
          语言核心框架 Language Core Framework
        </h2>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
          {/* 知识网络系统 */}
          <div style={{
            background: 'rgba(0,0,0,0.3)',
            borderRadius: '12px',
            padding: '20px',
            border: '1px solid rgba(0, 210, 255, 0.1)',
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              marginBottom: '16px',
            }}>
              <Database size={20} color="#00d2ff" />
              <h3 style={{
                fontSize: '16px',
                fontWeight: 'bold',
                color: '#fff',
                margin: 0,
              }}>
                知识网络系统
                <br />
                <span style={{ fontSize: '13px', fontWeight: 'normal', color: '#888' }}>
                  Knowledge Network System
                </span>
              </h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#00d2ff',
                  marginBottom: '6px',
                }}>
                  概念 Concepts
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  包含大量实体概念，如苹果、太阳、石头、水、头发等
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    Large number of entity concepts (apple, sun, stone, water, hair, etc.)
                  </span>
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#00d2ff',
                  marginBottom: '6px',
                }}>
                  属性 Attributes
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  包含概念的特征和性质，如苹果的颜色、味道、大小等
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    Characteristics and properties (apple's color, taste, size, etc.)
                  </span>
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#00d2ff',
                  marginBottom: '6px',
                }}>
                  抽象系统 Abstract System
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  概念的层级抽象，如苹果→水果→食物→物体等
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    Hierarchical abstraction (apple → fruit → food → object, etc.)
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* 逻辑体系 */}
          <div style={{
            background: 'rgba(0,0,0,0.3)',
            borderRadius: '12px',
            padding: '20px',
            border: '1px solid rgba(0, 210, 255, 0.1)',
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              marginBottom: '16px',
            }}>
              <Target size={20} color="#00d2ff" />
              <h3 style={{
                fontSize: '16px',
                fontWeight: 'bold',
                color: '#fff',
                margin: 0,
              }}>
                逻辑体系
                <br />
                <span style={{ fontSize: '13px', fontWeight: 'normal', color: '#888' }}>
                  Logical System
                </span>
              </h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#f59e0b',
                  marginBottom: '6px',
                }}>
                  条件推理 Conditional Reasoning
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  基于条件的分析和推理能力
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    Analysis and reasoning based on conditions
                  </span>
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#f59e0b',
                  marginBottom: '6px',
                }}>
                  受限组合问题 Bounded Combinatorics
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  解决知识网络中的受限无穷组合问题
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    Solving bounded infinite combinatorial problems in knowledge networks
                  </span>
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#f59e0b',
                  marginBottom: '6px',
                }}>
                  核心能力 Core Capabilities
                </div>
                <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '13px', color: '#ccc' }}>
                  <li style={{ marginBottom: '4px', paddingLeft: '12px', position: 'relative' }}>
                    <span style={{ position: 'absolute', left: 0, top: '6px', width: '4px', height: '4px', borderRadius: '50%', background: '#f59e0b' }} />
                    深度思考能力 Deep Thinking
                  </li>
                  <li style={{ marginBottom: '4px', paddingLeft: '12px', position: 'relative' }}>
                    <span style={{ position: 'absolute', left: 0, top: '6px', width: '4px', height: '4px', borderRadius: '50%', background: '#f59e0b' }} />
                    翻译能力 Translation
                  </li>
                  <li style={{ paddingLeft: '12px', position: 'relative' }}>
                    <span style={{ position: 'absolute', left: 0, top: '6px', width: '4px', height: '4px', borderRadius: '50%', background: '#f59e0b' }} />
                    问题解决 Problem Solving
                  </li>
                </ul>
              </div>
            </div>
          </div>

          {/* 多维度体系 */}
          <div style={{
            background: 'rgba(0,0,0,0.3)',
            borderRadius: '12px',
            padding: '20px',
            border: '1px solid rgba(0, 210, 255, 0.1)',
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              marginBottom: '16px',
            }}>
              <Layers size={20} color="#00d2ff" />
              <h3 style={{
                fontSize: '16px',
                fontWeight: 'bold',
                color: '#fff',
                margin: 0,
              }}>
                多维度体系
                <br />
                <span style={{ fontSize: '13px', fontWeight: 'normal', color: '#888' }}>
                  Multi-Dimensional System
                </span>
              </h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#10b981',
                  marginBottom: '6px',
                }}>
                  风格维度 Style Dimension
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  控制输出的风格和语调，如聊天式、论文式等
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    Controls output style and tone (chat, academic, etc.)
                  </span>
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#10b981',
                  marginBottom: '6px',
                }}>
                  逻辑维度 Logic Dimension
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  管理上下文的逻辑关系和连贯性
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    Manages logical relationships and coherence in context
                  </span>
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#10b981',
                  marginBottom: '6px',
                }}>
                  语句维度 Sentence Dimension
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  处理语法结构和句子组织
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    Handles grammatical structure and sentence organization
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 核心目标 */}
        <div style={{
          marginTop: '20px',
          padding: '16px',
          background: 'rgba(0, 210, 255, 0.1)',
          borderRadius: '10px',
          border: '1px solid rgba(0, 210, 255, 0.3)',
        }}>
          <div style={{
            fontSize: '14px',
            fontWeight: 'bold',
            color: '#00d2ff',
            marginBottom: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            <Sparkles size={18} />
            核心目标 Core Objective
          </div>
          <div style={{
            fontSize: '14px',
            color: '#fff',
            lineHeight: '1.7',
          }}>
            分析以上特性背后的统一<strong style={{ color: '#00d2ff' }}>编码机制</strong>，研究其在<strong style={{ color: '#00d2ff' }}>神经元</strong>和<strong style={{ color: '#00d2ff' }}>参数级别</strong>是如何形成的。
            <br />
            <span style={{ fontSize: '13px', color: '#888' }}>
              Analyze the unified encoding mechanism behind these characteristics and understand how it is formed at the neuron and parameter level.
            </span>
          </div>
        </div>
      </div>

      {/* Other Critical Features */}
      <div style={{
        background: 'rgba(50, 0, 100, 0.2)',
        borderRadius: '16px',
        padding: '24px',
        border: '1px solid rgba(147, 51, 234, 0.2)',
        marginBottom: '40px',
      }}>
        <h2 style={{
          fontSize: '20px',
          fontWeight: 'bold',
          color: '#a855f7',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}>
          <Sparkles size={24} />
          其他关键特性 Other Critical Features
        </h2>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
          {/* Word Embedding Arithmetic */}
          <div style={{
            background: 'rgba(0,0,0,0.3)',
            borderRadius: '12px',
            padding: '20px',
            border: '1px solid rgba(147, 51, 234, 0.1)',
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              marginBottom: '16px',
            }}>
              <Target size={20} color="#a855f7" />
              <h3 style={{
                fontSize: '16px',
                fontWeight: 'bold',
                color: '#fff',
                margin: 0,
              }}>
                词嵌入算术
                <br />
                <span style={{ fontSize: '13px', fontWeight: 'normal', color: '#888' }}>
                  Word Embedding Arithmetic
                </span>
              </h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{
                padding: '12px',
                background: 'rgba(168, 85, 247, 0.1)',
                borderRadius: '8px',
                borderLeft: '3px solid #a855f7',
              }}>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#a855f7',
                  marginBottom: '6px',
                }}>
                  经典案例 Classic Example
                </div>
                <div style={{ fontSize: '14px', color: '#fff', fontFamily: 'monospace' }}>
                  国王 - 男性 + 女性 = 女王
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    King - Man + Woman = Queen
                  </span>
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#a855f7',
                  marginBottom: '6px',
                }}>
                  理论含义 Theoretical Implication
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  词嵌入中存在完整的数学结构，概念关系可以通过向量运算精确表达
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    Word embeddings contain complete mathematical structures where conceptual relationships can be precisely expressed through vector operations
                  </span>
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#f59e0b',
                  marginBottom: '6px',
                }}>
                  编码推测 Encoding Hypothesis
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  参数空间中语义概念形成几何结构，关系编码为方向和距离
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    Semantic concepts form geometric structures in parameter space, with relationships encoded as directions and distances
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Neural Efficiency Principle */}
          <div style={{
            background: 'rgba(0,0,0,0.3)',
            borderRadius: '12px',
            padding: '20px',
            border: '1px solid rgba(147, 51, 234, 0.1)',
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              marginBottom: '16px',
            }}>
              <Zap size={20} color="#a855f7" />
              <h3 style={{
                fontSize: '16px',
                fontWeight: 'bold',
                color: '#fff',
                margin: 0,
              }}>
                脉冲神经网络原理
                <br />
                <span style={{ fontSize: '13px', fontWeight: 'normal', color: '#888' }}>
                  Spiking Neural Network Principle
                </span>
              </h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#a855f7',
                  marginBottom: '6px',
                }}>
                  最小传送量原理 Principle of Minimal Information
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  大脑中的脉冲天然遵循最小传送量原理，能量效率优先
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    Spikes in the brain naturally follow the principle of minimal information transmission, prioritizing energy efficiency
                  </span>
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#a855f7',
                  marginBottom: '6px',
                }}>
                  编码原理 Encoding Principle
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  如果网格结构效率高，叠加路径即编码原理
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    If the grid structure is efficient, superposition of paths becomes the encoding principle
                  </span>
                </div>
              </div>

              <div style={{
                padding: '12px',
                background: 'rgba(168, 85, 247, 0.1)',
                borderRadius: '8px',
                borderLeft: '3px solid #a855f7',
              }}>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#f59e0b',
                  marginBottom: '6px',
                }}>
                  核心结论 Core Conclusion
                </div>
                <div style={{ fontSize: '13px', color: '#fff', lineHeight: '1.5' }}>
                  同时实现及时学习和全局稳态
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    Achieve real-time learning AND global steady state simultaneously
                  </span>
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#f59e0b',
                  marginBottom: '6px',
                }}>
                  研究重点 Research Focus
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  脉冲神经网络的3D空间拓扑网络结构
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    3D spatial topological network structure of spiking neural networks
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Global Uniqueness */}
          <div style={{
            background: 'rgba(0,0,0,0.3)',
            borderRadius: '12px',
            padding: '20px',
            border: '1px solid rgba(147, 51, 234, 0.1)',
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              marginBottom: '16px',
            }}>
              <Layers size={20} color="#a855f7" />
              <h3 style={{
                fontSize: '16px',
                fontWeight: 'bold',
                color: '#fff',
                margin: 0,
              }}>
                全局唯一性
                <br />
                <span style={{ fontSize: '13px', fontWeight: 'normal', color: '#888' }}>
                  Global Uniqueness
                </span>
              </h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#a855f7',
                  marginBottom: '6px',
                }}>
                  现象观察 Observation
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  深度神经网络中，所有神经元都参与运算，但在不同风格、不同逻辑、不同语法下，每次都能生成一个合适的词
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    In deep neural networks, all neurons participate in computation, but under different styles, logic, and syntax, a suitable word is generated each time
                  </span>
                </div>
              </div>

              <div style={{
                padding: '12px',
                background: 'rgba(168, 85, 247, 0.1)',
                borderRadius: '8px',
                borderLeft: '3px solid #a855f7',
              }}>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#f59e0b',
                  marginBottom: '6px',
                }}>
                  唯一性假说 Uniqueness Hypothesis
                </div>
                <div style={{ fontSize: '14px', color: '#fff', lineHeight: '1.5' }}>
                  语言中某种东西具有<strong style={{ color: '#a855f7' }}>全局唯一性</strong>
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    Something in language has global uniqueness
                  </span>
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#a855f7',
                  marginBottom: '6px',
                }}>
                  数学特性 Mathematical Property
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  这种唯一性应该具有数学特性，而非偶然现象
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    This uniqueness should have mathematical properties, not just random phenomena
                  </span>
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#f59e0b',
                  marginBottom: '6px',
                }}>
                  编码推测 Encoding Hypothesis
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  可能存在全局吸引子或稳定的编码流形
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    May exist global attractors or stable encoding manifolds
                  </span>
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#10b981',
                  marginBottom: '6px',
                }}>
                  研究方向 Research Direction
                </div>
                <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
                  分析生成过程中的激活路径和收敛点
                  <br />
                  <span style={{ fontSize: '12px', color: '#888' }}>
                    Analyze activation paths and convergence points during generation
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Unified Mechanism Summary */}
        <div style={{
          marginTop: '20px',
          padding: '16px',
          background: 'rgba(168, 85, 247, 0.15)',
          borderRadius: '10px',
          border: '1px solid rgba(168, 85, 247, 0.3)',
        }}>
          <div style={{
            fontSize: '14px',
            fontWeight: 'bold',
            color: '#a855f7',
            marginBottom: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            <Brain size={18} />
            统一机制假说 Unified Mechanism Hypothesis
          </div>
          <div style={{
            fontSize: '14px',
            color: '#fff',
            lineHeight: '1.7',
            marginBottom: '12px',
          }}>
            知识网络、逻辑体系、多维度、词嵌入算术、脉冲编码、全局唯一性等所有特性，都是同一套<strong style={{ color: '#a855f7' }}>编码机制</strong>的结果。
            <br />
            <span style={{ fontSize: '13px', color: '#888' }}>
              Knowledge network, logical system, multi-dimensions, word embedding arithmetic, spiking encoding, and global uniqueness are all results of the same encoding mechanism.
            </span>
          </div>
          <div style={{
            fontSize: '14px',
            color: '#fff',
            lineHeight: '1.7',
          }}>
            这套机制在<strong style={{ color: '#a855f7' }}>神经元</strong>和<strong style={{ color: '#a855f7' }}>参数级别</strong>形成，核心目标是分析其数学结构。
            <br />
            <span style={{ fontSize: '13px', color: '#888' }}>
              This mechanism is formed at the neuron and parameter level, with the core goal of analyzing its mathematical structure.
            </span>
          </div>
        </div>
      </div>

      {/* SECTION 1: 语言核心特性 + 当前研究进展 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: '40px' }}>
        {/* Left Column: Language Characteristics */}
        <div style={{
          background: 'rgba(20, 20, 30, 0.6)',
          borderRadius: '16px',
          padding: '24px',
          border: '1px solid rgba(255,255,255,0.08)',
        }}>
          <h2 style={{
            fontSize: '20px',
            fontWeight: 'bold',
            color: '#fff',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}>
            <Code size={22} color="#00d2ff" />
            语言核心特性 Core Characteristics
          </h2>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {languageCharacteristics.map((char) => (
              <div
                key={char.id}
                style={{
                  background: 'rgba(0,0,0,0.3)',
                  borderRadius: '12px',
                  border: '1px solid rgba(255,255,255,0.06)',
                  overflow: 'hidden',
                }}
              >
                <div
                  onClick={() => toggleSection(char.id)}
                  style={{
                    padding: '16px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    transition: 'background 0.2s',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.02)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1 }}>
                    <div style={{ color: '#00d2ff' }}>
                      {char.icon}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#fff', marginBottom: '4px' }}>
                        {char.title}
                      </div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        {char.description}
                      </div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      fontSize: '12px',
                      fontWeight: 'bold',
                      color: getStatusColor(char.status),
                    }}>
                      {getStatusIcon(char.status)}
                      {char.statusText}
                    </div>
                    <div style={{
                      width: '40px',
                      height: '6px',
                      background: 'rgba(255,255,255,0.1)',
                      borderRadius: '3px',
                      overflow: 'hidden',
                    }}>
                      <div style={{
                        width: `${char.progress * 100}%`,
                        height: '100%',
                        background: getStatusColor(char.status),
                        borderRadius: '3px',
                        transition: 'width 0.3s',
                      }} />
                    </div>
                  </div>
                </div>

                {expandedSection === char.id && (
                  <div style={{
                    padding: '16px',
                    borderTop: '1px solid rgba(255,255,255,0.06)',
                    background: 'rgba(0,0,0,0.2)',
                  }}>
                    <div style={{ marginBottom: '16px' }}>
                      <div style={{
                        fontSize: '12px',
                        fontWeight: 'bold',
                        color: '#888',
                        textTransform: 'uppercase',
                        marginBottom: '8px',
                        letterSpacing: '1px',
                      }}>
                        Key Findings 关键发现
                      </div>
                      <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                        {char.keyFindings.map((finding, idx) => (
                          <li
                            key={idx}
                            style={{
                              fontSize: '13px',
                              color: '#ccc',
                              marginBottom: '6px',
                              paddingLeft: '16px',
                              position: 'relative',
                            }}
                          >
                            <span style={{
                              position: 'absolute',
                              left: 0,
                              top: '6px',
                              width: '6px',
                              height: '6px',
                              borderRadius: '50%',
                              background: '#00d2ff',
                            }} />
                            {finding}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(2, 1fr)',
                      gap: '12px',
                      marginBottom: '16px',
                    }}>
                      <div style={{
                        background: 'rgba(0,0,0,0.3)',
                        borderRadius: '8px',
                        padding: '12px',
                      }}>
                        <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px' }}>Progress 进展</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', color: getStatusColor(char.status) }}>
                          {(char.progress * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div style={{
                        background: 'rgba(0,0,0,0.3)',
                        borderRadius: '8px',
                        padding: '12px',
                      }}>
                        <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px' }}>Confidence 置信度</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', color: getStatusColor(char.status) }}>
                          {(char.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>

                    <div style={{
                      display: 'flex',
                      gap: '12px',
                      fontSize: '12px',
                    }}>
                      <div style={{
                        flex: 1,
                        background: 'rgba(0,100,200,0.1)',
                        borderRadius: '6px',
                        padding: '10px',
                      }}>
                        <div style={{ fontSize: '10px', color: '#666', marginBottom: '4px' }}>Evidence 证据</div>
                        <div style={{ color: '#00d2ff', fontFamily: 'monospace' }}>
                          {char.evidence}
                        </div>
                      </div>
                      <div style={{
                        background: 'rgba(200,100,0,0.1)',
                        borderRadius: '6px',
                        padding: '10px',
                      }}>
                        <div style={{ fontSize: '10px', color: '#666', marginBottom: '4px' }}>Stage</div>
                        <div style={{ color: '#f59e0b', fontWeight: 'bold' }}>
                          {char.latestStage}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Right Column: Research Progress */}
        <div style={{
          background: 'rgba(20, 20, 30, 0.6)',
          borderRadius: '16px',
          padding: '24px',
          border: '1px solid rgba(255,255,255,0.08)',
        }}>
          <h2 style={{
            fontSize: '20px',
            fontWeight: 'bold',
            color: '#fff',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}>
            <TrendingUp size={22} color="#00d2ff" />
            当前研究进展 Current Progress
          </h2>

          <div style={{ position: 'relative', paddingLeft: '20px' }}>
            <div style={{
              position: 'absolute',
              left: '8px',
              top: '10px',
              bottom: '10px',
              width: '2px',
              background: 'linear-gradient(to bottom, #00d2ff, rgba(0,210,255,0.2))',
            }} />

            {researchProgress.map((item) => (
              <div key={item.id} style={{ position: 'relative', marginBottom: '20px' }}>
                <div style={{
                  position: 'absolute',
                  left: '-12px',
                  top: '16px',
                  width: '12px',
                  height: '12px',
                  borderRadius: '50%',
                  background: getStatusColor(item.status),
                  border: '2px solid rgba(0,0,0,0.8)',
                  boxShadow: `0 0 10px ${getStatusColor(item.status)}40`,
                }} />

                <div style={{
                  marginLeft: '20px',
                  background: 'rgba(0,0,0,0.3)',
                  borderRadius: '12px',
                  border: `1px solid ${getStatusColor(item.status)}30`,
                  overflow: 'hidden',
                }}>
                  <div
                    onClick={() => toggleSection(`progress_${item.id}`)}
                    style={{
                      padding: '16px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                    }}
                  >
                    <div style={{ flex: 1 }}>
                      <div style={{
                        fontSize: '14px',
                        fontWeight: 'bold',
                        color: '#fff',
                        marginBottom: '6px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                      }}>
                        <span style={{
                          fontSize: '11px',
                          background: `${getStatusColor(item.status)}30`,
                          color: getStatusColor(item.status),
                          padding: '2px 8px',
                          borderRadius: '4px',
                          fontFamily: 'monospace',
                        }}>
                          {item.stage}
                        </span>
                        {item.title}
                      </div>
                      <div style={{ fontSize: '12px', color: '#888' }}>
                        {item.summary}
                      </div>
                    </div>
                    <div style={{
                      fontSize: '11px',
                      color: '#666',
                      fontFamily: 'monospace',
                      marginLeft: '12px',
                    }}>
                      {item.date}
                    </div>
                  </div>

                  {expandedSection === `progress_${item.id}` && (
                    <div style={{
                      padding: '16px',
                      borderTop: '1px solid rgba(255,255,255,0.06)',
                      background: 'rgba(0,0,0,0.2)',
                    }}>
                      <div style={{ marginBottom: '16px' }}>
                        <div style={{
                          fontSize: '11px',
                          fontWeight: 'bold',
                          color: '#888',
                          textTransform: 'uppercase',
                          marginBottom: '8px',
                          letterSpacing: '1px',
                        }}>
                          Key Metrics 关键指标
                        </div>
                        <div style={{
                          display: 'grid',
                          gridTemplateColumns: 'repeat(2, 1fr)',
                          gap: '8px',
                        }}>
                          {Object.entries(item.keyMetrics).map(([key, value]) => (
                            <div
                              key={key}
                              style={{
                                background: 'rgba(0,0,0,0.3)',
                                borderRadius: '6px',
                                padding: '8px',
                              }}
                            >
                              <div style={{ fontSize: '10px', color: '#666', marginBottom: '2px' }}>
                                {key}
                              </div>
                              <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#00d2ff' }}>
                                {value}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div style={{ marginBottom: '16px' }}>
                        <div style={{
                          fontSize: '11px',
                          fontWeight: 'bold',
                          color: '#888',
                          textTransform: 'uppercase',
                          marginBottom: '8px',
                          letterSpacing: '1px',
                        }}>
                          Conclusions 结论
                        </div>
                        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                          {item.conclusions.map((conclusion, idx) => (
                            <li
                              key={idx}
                              style={{
                                fontSize: '12px',
                                color: '#ccc',
                                marginBottom: '4px',
                                paddingLeft: '12px',
                                position: 'relative',
                              }}
                            >
                              <span style={{
                                position: 'absolute',
                                left: 0,
                                top: '6px',
                                width: '4px',
                                height: '4px',
                                borderRadius: '50%',
                                background: '#00d2ff',
                              }} />
                              {conclusion}
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <div style={{
                          fontSize: '11px',
                          fontWeight: 'bold',
                          color: '#888',
                          textTransform: 'uppercase',
                          marginBottom: '8px',
                          letterSpacing: '1px',
                        }}>
                          Artifacts 产出文件
                        </div>
                        {item.artifacts.map((artifact, idx) => (
                          <div
                            key={idx}
                            style={{
                              fontSize: '11px',
                              color: '#00d2ff',
                              fontFamily: 'monospace',
                              marginBottom: '4px',
                              background: 'rgba(0,210,255,0.05)',
                              padding: '6px 10px',
                              borderRadius: '4px',
                            }}
                          >
                            {artifact}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* SECTION 2: 分析拼图 */}
      <div style={{
        background: 'rgba(20, 20, 30, 0.6)',
        borderRadius: '16px',
        padding: '24px',
        border: '1px solid rgba(255,255,255,0.08)',
        marginBottom: '40px',
      }}>
        <h2 style={{
          fontSize: '20px',
          fontWeight: 'bold',
          color: '#fff',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}>
          <Puzzle size={22} color="#00d2ff" />
          分析拼图 Analysis Puzzles
        </h2>

        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '30px' }}>
          {/* Left: Puzzle Categories */}
          <div>
            {puzzleCategories.map((category) => (
              <div
                key={category.id}
                style={{
                  marginBottom: '30px',
                  background: 'rgba(0,0,0,0.2)',
                  borderRadius: '12px',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    padding: '16px 20px',
                    background: 'rgba(0,100,200,0.1)',
                    borderBottom: '1px solid rgba(255,255,255,0.06)',
                  }}
                >
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    marginBottom: '8px',
                  }}>
                    <div style={{ color: '#00d2ff' }}>
                      {category.icon}
                    </div>
                    <h3 style={{
                      fontSize: '16px',
                      fontWeight: 'bold',
                      color: '#fff',
                      margin: 0,
                    }}>
                      {category.title}
                    </h3>
                  </div>
                  <p style={{
                    fontSize: '13px',
                    color: '#888',
                    margin: 0,
                  }}>
                    {category.description}
                  </p>
                </div>

                <div style={{ padding: '16px 20px' }}>
                  {category.puzzles.map((puzzle) => (
                    <div
                      key={puzzle.id}
                      style={{
                        marginBottom: '16px',
                        background: 'rgba(0,0,0,0.3)',
                        borderRadius: '10px',
                        overflow: 'hidden',
                      }}
                    >
                      <div
                        onClick={() => togglePuzzle(puzzle.id)}
                        style={{
                          padding: '14px 16px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                        }}
                      >
                        <div style={{ flex: 1 }}>
                          <div style={{
                            fontSize: '14px',
                            fontWeight: 'bold',
                            color: '#fff',
                            marginBottom: '4px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                          }}>
                            <span style={{
                              fontSize: '11px',
                              background: 'rgba(0,210,255,0.1)',
                              color: '#00d2ff',
                              padding: '2px 6px',
                              borderRadius: '4px',
                              fontFamily: 'monospace',
                            }}>
                              {puzzle.stage}
                            </span>
                            {puzzle.title}
                          </div>
                          <div style={{ fontSize: '11px', color: '#888' }}>
                            {getEvidenceStars(puzzle.evidenceStrength)} 完整性: {(puzzle.completeness * 100).toFixed(0)}%
                          </div>
                        </div>
                        {expandedPuzzle === puzzle.id ? <ChevronUp size={16} color="#00d2ff" /> : <ChevronDown size={16} color="#666" />}
                      </div>

                      {expandedPuzzle === puzzle.id && (
                        <div style={{
                          padding: '16px',
                          borderTop: '1px solid rgba(255,255,255,0.06)',
                          background: 'rgba(0,0,0,0.2)',
                        }}>
                          <div style={{ marginBottom: '12px' }}>
                            <div style={{
                              fontSize: '11px',
                              fontWeight: 'bold',
                              color: '#888',
                              textTransform: 'uppercase',
                              marginBottom: '8px',
                              letterSpacing: '1px',
                            }}>
                              Key Data 关键数据
                            </div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                              {puzzle.keyData.map((data, idx) => (
                                <span
                                  key={idx}
                                  style={{
                                    fontSize: '12px',
                                    color: '#ccc',
                                    background: 'rgba(0,210,255,0.05)',
                                    padding: '4px 10px',
                                    borderRadius: '4px',
                                  }}
                                >
                                  {data}
                                </span>
                              ))}
                            </div>
                          </div>

                          <div style={{ marginBottom: '12px' }}>
                            <div style={{
                              fontSize: '11px',
                              fontWeight: 'bold',
                              color: '#888',
                              textTransform: 'uppercase',
                              marginBottom: '8px',
                              letterSpacing: '1px',
                            }}>
                              Completeness Check 完整性检查
                            </div>
                            <div style={{ fontSize: '12px', color: '#ccc', lineHeight: '1.6' }}>
                              {puzzle.completenessCheck.map((check, idx) => (
                                <div key={idx} style={{ marginBottom: '4px' }}>
                                  {check}
                                </div>
                              ))}
                            </div>
                          </div>

                          <div>
                            <div style={{
                              fontSize: '11px',
                              fontWeight: 'bold',
                              color: '#888',
                              textTransform: 'uppercase',
                              marginBottom: '8px',
                              letterSpacing: '1px',
                            }}>
                              Conclusions 结论
                            </div>
                            <div style={{
                              fontSize: '13px',
                              color: '#00d2ff',
                              lineHeight: '1.6',
                              padding: '12px',
                              background: 'rgba(0,210,255,0.05)',
                              borderRadius: '8px',
                            }}>
                              {puzzle.conclusions}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Right: Breakthrough Preparation */}
          <div>
            <h3 style={{
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#fff',
              marginBottom: '20px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}>
              <Zap size={18} color="#00d2ff" />
              突破准备 Breakthrough Prep
            </h3>

            {breakthroughPreparation.map((prep) => (
              <div
                key={prep.id}
                style={{
                  marginBottom: '20px',
                  background: 'rgba(0,0,0,0.3)',
                  borderRadius: '10px',
                  overflow: 'hidden',
                }}
              >
                <div
                  onClick={() => togglePreparation(prep.id)}
                  style={{
                    padding: '14px 16px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{
                      fontSize: '14px',
                      fontWeight: 'bold',
                      color: '#fff',
                      marginBottom: '4px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                    }}>
                      <div style={{ color: '#00d2ff' }}>
                        {prep.icon}
                      </div>
                      {prep.title}
                      <span style={{
                        fontSize: '10px',
                        background: `${getPriorityColor(prep.priority)}30`,
                        color: getPriorityColor(prep.priority),
                        padding: '2px 6px',
                        borderRadius: '4px',
                        marginLeft: 'auto',
                      }}>
                        {prep.priority.toUpperCase()}
                      </span>
                    </div>
                    <div style={{ fontSize: '12px', color: '#888' }}>
                      {prep.timeline}
                    </div>
                  </div>
                  {expandedPreparation === prep.id ? <ChevronUp size={16} color="#00d2ff" /> : <ChevronDown size={16} color="#666" />}
                </div>

                {expandedPreparation === prep.id && (
                  <div style={{
                    padding: '16px',
                    borderTop: '1px solid rgba(255,255,255,0.06)',
                    background: 'rgba(0,0,0,0.2)',
                  }}>
                    <div style={{ marginBottom: '12px' }}>
                      <div style={{
                        fontSize: '11px',
                        fontWeight: 'bold',
                        color: '#888',
                        textTransform: 'uppercase',
                        marginBottom: '8px',
                        letterSpacing: '1px',
                      }}>
                        Description 描述
                      </div>
                      <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.6' }}>
                        {prep.description}
                      </div>
                    </div>

                    <div style={{ marginBottom: '12px' }}>
                      <div style={{
                        fontSize: '11px',
                        fontWeight: 'bold',
                        color: '#888',
                        textTransform: 'uppercase',
                        marginBottom: '8px',
                        letterSpacing: '1px',
                      }}>
                        Tasks 任务
                      </div>
                      <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                        {prep.tasks.map((task, idx) => (
                          <li
                            key={idx}
                            style={{
                              fontSize: '12px',
                              color: '#ccc',
                              marginBottom: '4px',
                              paddingLeft: '12px',
                              position: 'relative',
                            }}
                          >
                            <span style={{
                              position: 'absolute',
                              left: 0,
                              top: '6px',
                              width: '4px',
                              height: '4px',
                              borderRadius: '50%',
                              background: '#00d2ff',
                            }} />
                            {task}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <div style={{
                        fontSize: '11px',
                        fontWeight: 'bold',
                        color: '#888',
                        textTransform: 'uppercase',
                        marginBottom: '8px',
                        letterSpacing: '1px',
                      }}>
                        Dependencies 依赖
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        {prep.dependencies.map((dep, idx) => (
                          <span
                            key={idx}
                            style={{
                              fontSize: '11px',
                              color: '#00d2ff',
                              background: 'rgba(0,210,255,0.1)',
                              padding: '4px 8px',
                              borderRadius: '4px',
                            }}
                          >
                            {dep}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}

            <div style={{
              padding: '16px',
              background: 'rgba(0,210,255,0.05)',
              borderRadius: '10px',
              border: '1px solid rgba(0,210,255,0.2)',
            }}>
              <div style={{
                fontSize: '12px',
                fontWeight: 'bold',
                color: '#00d2ff',
                marginBottom: '8px',
              }}>
                核心理念 Core Philosophy
              </div>
              <div style={{
                fontSize: '12px',
                color: '#ccc',
                lineHeight: '1.6',
              }}>
                智能的数学理论很可能超过现有数学体系，前期不要预设任何理论，重点在于持续的积累基础数据的拼图，等待最后的突破
                <br />
                <br />
                The mathematical theory of intelligence likely exceeds existing mathematical systems. Do not presuppose any theory in early stages; focus on continuously accumulating basic data puzzles and wait for the final breakthrough.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
