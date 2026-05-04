import { BookOpen, Code, Database, Brain, Sparkles, AlertCircle, Layers, Target, Puzzle, Trophy, TrendingUp, Zap, ChevronDown, ChevronUp, Box, Flame, Atom, Microscope, Sigma, GitBranch, TestTube } from 'lucide-react';
import { useState, useEffect } from 'react';

/**
 * 语言分析标签页
 * 展示语言特性分析和当前研究进展
 * 采用从上到下的垂直布局，不使用tab切换
 */
export const LanguageAnalysisTab = () => {
  const [expandedPuzzle, setExpandedPuzzle] = useState(null);
  const [expandedPreparation, setExpandedPreparation] = useState(null);
  const [expandedCategory, setExpandedCategory] = useState(null);
  const [puzzleData, setPuzzleData] = useState(null);

  useEffect(() => {
    fetch('/data/language_analysis_puzzle.json')
      .then(res => res.json())
      .then(data => setPuzzleData(data))
      .catch(err => console.error('Failed to load puzzle data:', err));
  }, []);

  const iconMap = { brain: <Brain size={20} />, target: <Target size={20} />, code: <Code size={20} />, layers: <Layers size={20} />, zap: <Zap size={20} />, database: <Database size={20} />, box: <Box size={20} />, sparkles: <Sparkles size={20} /> };

  const evidenceLevelColor = { E0: '#6b7280', E1: '#6366f1', E2: '#3b82f6', E3: '#f59e0b', E4: '#10b981', E5: '#00d2ff' };
  const evidenceLevelLabel = { E0: '无证据', E1: '相关性', E2: '可预测', E3: '干预有效', E4: '跨模型复现', E5: '机制闭环' };

  // 分析拼图数据 — 从JSON加载
  const puzzleCategories = puzzleData ? puzzleData.categories.map(cat => ({
    id: cat.id,
    title: `${cat.id}: ${cat.name} ${cat.nameEn}`,
    icon: iconMap[cat.icon] || <Puzzle size={20} />,
    description: cat.description,
    goal: cat.goal,
    principle: cat.principle,
    fillRate: cat.fillRate,
    knowledgeRate: cat.knowledgeRate,
    causalRate: cat.causalRate,
    cellCount: cat.cellCount,
    interpretPower: cat.interpretPower,
    puzzles: cat.cells.map(cell => ({
      id: cell.id,
      title: cell.title,
      priority: cell.priority,
      status: cell.status,
      completeness: cell.fillRate,
      knowledgeRate: cell.knowledgeRate,
      causalRate: cell.causalRate,
      evidenceLevel: cell.evidenceLevel || 'E0',
      goal: cell.goal,
      principle: cell.principle,
      evidenceStrength: cell.priority === 'P0' ? 5 : cell.priority === 'P1' ? 3 : 2,
      keyData: cell.keyData,
      evidence: cell.evidence,
      completenessCheck: cell.status === 'filled'
        ? ['✅ 已填充', '✅ 三模型验证', '✅ 因果验证']
        : cell.status === 'partial'
        ? ['✅ 部分现象描述', '⚠ 因果验证不足', '❌ 跨模型验证待做']
        : ['❌ 未填充', '❌ 无因果验证', '❌ 无跨模型验证'],
      conclusions: cell.evidence,
    })),
  })) : [];

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

  const togglePuzzle = (puzzleId) => {
    setExpandedPuzzle(expandedPuzzle === puzzleId ? null : puzzleId);
  };

  const togglePreparation = (prepId) => {
    setExpandedPreparation(expandedPreparation === prepId ? null : prepId);
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

        {/* 总览进度条 */}
        {puzzleData && (
          <div style={{
            marginBottom: '24px',
            padding: '16px 20px',
            background: 'rgba(0,100,200,0.1)',
            borderRadius: '12px',
            border: '1px solid rgba(0,210,255,0.2)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#00d2ff' }}>
                总填充率 Overall Fill Rate
              </span>
              <span style={{ fontSize: '24px', fontWeight: 'bold', color: '#00d2ff' }}>
                {(puzzleData.overallProgress.fillRate * 100).toFixed(0)}%
              </span>
            </div>
            <div style={{
              width: '100%',
              height: '12px',
              background: 'rgba(255,255,255,0.1)',
              borderRadius: '6px',
              overflow: 'hidden',
              marginBottom: '12px',
            }}>
              <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                <div style={{
                  position: 'absolute', left: 0, top: 0, height: '100%',
                  width: `${puzzleData.overallProgress.knowledgeRate * 100}%`,
                  background: 'rgba(59,130,246,0.5)',
                  borderRadius: '6px 0 0 6px',
                }} />
                <div style={{
                  position: 'absolute', left: 0, top: 0, height: '100%',
                  width: `${puzzleData.overallProgress.causalRate * 100}%`,
                  background: 'linear-gradient(90deg, #f59e0b, #ef4444)',
                  borderRadius: '6px 0 0 6px',
                  zIndex: 1,
                }} />
                <div style={{
                  position: 'absolute', left: 0, top: 0, height: '100%',
                  width: `${puzzleData.overallProgress.fillRate * 100}%`,
                  background: 'linear-gradient(90deg, #00d2ff, #a855f7)',
                  borderRadius: '6px',
                  zIndex: 2,
                  opacity: 0.85,
                }} />
              </div>
            </div>
            <div style={{ display: 'flex', gap: '16px', fontSize: '12px', color: '#888', flexWrap: 'wrap', marginBottom: '10px' }}>
              <span>✓ 已填充: {puzzleData.overallProgress.filled}</span>
              <span>◐ 部分填充: {puzzleData.overallProgress.partial}</span>
              <span>□ 未填充: {puzzleData.overallProgress.empty}</span>
              <span>总格子: {puzzleData.overallProgress.totalCells}</span>
              <span>P0填充率: {(puzzleData.overallProgress.p0FillRate * 100).toFixed(0)}%</span>
            </div>
            <div style={{ display: 'flex', gap: '16px', fontSize: '12px', flexWrap: 'wrap', marginBottom: '10px' }}>
              <span style={{ color: '#3b82f6' }}>知识率: {(puzzleData.overallProgress.knowledgeRate * 100).toFixed(0)}%</span>
              <span style={{ color: '#f59e0b' }}>因果率: {(puzzleData.overallProgress.causalRate * 100).toFixed(0)}%</span>
              <span style={{ color: '#00d2ff' }}>解释力: {(puzzleData.overallProgress.interpretPower * 100).toFixed(0)}%</span>
            </div>
            {/* 成熟度 */}
            {puzzleData.maturity && (
              <div style={{ display: 'flex', gap: '12px', fontSize: '11px', padding: '8px 12px', background: 'rgba(0,0,0,0.3)', borderRadius: '6px', flexWrap: 'wrap' }}>
                <span style={{ color: '#10b981' }}>经验规律: {(puzzleData.maturity.empiricalDiscovery * 100).toFixed(0)}%</span>
                <span style={{ color: '#f59e0b' }}>统一理论: {(puzzleData.maturity.unifiedTheory * 100).toFixed(0)}%</span>
                <span style={{ color: '#ef4444' }}>可预测科学: {(puzzleData.maturity.predictiveScience * 100).toFixed(0)}%</span>
                <span style={{ color: '#888' }}>{puzzleData.maturity.stageLabel}</span>
              </div>
            )}
            {/* 子空间分解地图 */}
            <div style={{ marginTop: '12px', padding: '10px 14px', background: 'rgba(0,0,0,0.3)', borderRadius: '8px' }}>
              <div style={{ fontSize: '11px', color: '#888', marginBottom: '6px' }}>子空间分解地图 Subspace Decomposition</div>
              <div style={{ fontSize: '12px', color: '#00d2ff', fontFamily: 'monospace', lineHeight: '1.6' }}>
                R^d ≈ V_WU∩V_sem ⊕ V_WU⊥∩V_syn ⊕ V_WU⊥∩V_dark ⊕ V_⊥logic ⊕ V_residual
              </div>
              <div style={{ display: 'flex', gap: '8px', marginTop: '6px', flexWrap: 'wrap' }}>
                {puzzleData.subspaceDecomposition.subspaces.map((ss, idx) => (
                  <span key={idx} style={{ fontSize: '11px', color: '#ccc', background: 'rgba(0,210,255,0.08)', padding: '3px 8px', borderRadius: '4px' }}>
                    {ss.label}({ss.ratio})
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '30px' }}>
          {/* Left: Puzzle Categories */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {puzzleCategories.map((category) => (
              <div
                key={category.id}
                style={{
                  background: 'rgba(0,0,0,0.3)',
                  borderRadius: '12px',
                  border: '1px solid rgba(255,255,255,0.06)',
                  overflow: 'hidden',
                }}
              >
                <div
                  onClick={() => setExpandedCategory(expandedCategory === category.id ? null : category.id)}
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
                      {category.icon}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#fff', marginBottom: '4px' }}>
                        {category.title}
                      </div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        {category.description}
                      </div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '3px', minWidth: '100px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <span style={{ fontSize: '9px', color: '#3b82f6', minWidth: '28px' }}>知识</span>
                        <div style={{ flex: 1, height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                          <div style={{ width: `${(category.knowledgeRate || 0) * 100}%`, height: '100%', background: '#3b82f6', borderRadius: '2px' }} />
                        </div>
                        <span style={{ fontSize: '9px', color: '#3b82f6', minWidth: '24px', textAlign: 'right' }}>{((category.knowledgeRate || 0) * 100).toFixed(0)}%</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <span style={{ fontSize: '9px', color: '#f59e0b', minWidth: '28px' }}>因果</span>
                        <div style={{ flex: 1, height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                          <div style={{ width: `${(category.causalRate || 0) * 100}%`, height: '100%', background: '#f59e0b', borderRadius: '2px' }} />
                        </div>
                        <span style={{ fontSize: '9px', color: '#f59e0b', minWidth: '24px', textAlign: 'right' }}>{((category.causalRate || 0) * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      fontSize: '12px',
                      fontWeight: 'bold',
                      color: category.fillRate > 0.3 ? '#10b981' : category.fillRate > 0.15 ? '#f59e0b' : '#ef4444',
                    }}>
                      {(category.fillRate * 100).toFixed(0)}%
                    </div>
                    {expandedCategory === category.id ? <ChevronUp size={16} color="#00d2ff" /> : <ChevronDown size={16} color="#666" />}
                  </div>
                </div>

                {expandedCategory === category.id && (
                  <div style={{
                    padding: '16px',
                    borderTop: '1px solid rgba(255,255,255,0.06)',
                    background: 'rgba(0,0,0,0.2)',
                  }}>
                    {/* 目标与原理 */}
                    {category.goal && (
                      <div style={{ marginBottom: '14px', display: 'flex', gap: '10px' }}>
                        <div style={{
                          flex: 1,
                          padding: '10px 12px',
                          background: 'rgba(0,210,255,0.06)',
                          borderRadius: '8px',
                          borderLeft: '3px solid #00d2ff',
                        }}>
                          <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#00d2ff', marginBottom: '4px', letterSpacing: '0.5px' }}>
                            目标 GOAL
                          </div>
                          <div style={{ fontSize: '12px', color: '#ccc', lineHeight: '1.5' }}>
                            {category.goal}
                          </div>
                        </div>
                        {category.principle && (
                          <div style={{
                            flex: 1,
                            padding: '10px 12px',
                            background: 'rgba(168,85,247,0.06)',
                            borderRadius: '8px',
                            borderLeft: '3px solid #a855f7',
                          }}>
                            <div style={{ fontSize: '10px', fontWeight: 'bold', color: '#a855f7', marginBottom: '4px', letterSpacing: '0.5px' }}>
                              原理 PRINCIPLE
                            </div>
                            <div style={{ fontSize: '12px', color: '#ccc', lineHeight: '1.5' }}>
                              {category.principle}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                    {category.puzzles.map((puzzle) => (
                      <div
                        key={puzzle.id}
                        style={{
                          marginBottom: '12px',
                          background: 'rgba(0,0,0,0.3)',
                          borderRadius: '10px',
                          overflow: 'hidden',
                          border: '1px solid rgba(255,255,255,0.04)',
                        }}
                      >
                        <div
                          onClick={() => togglePuzzle(puzzle.id)}
                          style={{
                            padding: '12px 14px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                          }}
                        >
                          <div style={{ flex: 1 }}>
                            <div style={{
                              fontSize: '13px',
                              fontWeight: 'bold',
                              color: '#fff',
                              marginBottom: '4px',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '8px',
                            }}>
                              <span style={{
                                fontSize: '11px',
                                background: puzzle.priority === 'P0' ? 'rgba(239,68,68,0.2)' : puzzle.priority === 'P1' ? 'rgba(245,158,11,0.2)' : 'rgba(107,114,128,0.2)',
                                color: puzzle.priority === 'P0' ? '#ef4444' : puzzle.priority === 'P1' ? '#f59e0b' : '#6b7280',
                                padding: '2px 6px',
                                borderRadius: '4px',
                                fontFamily: 'monospace',
                              }}>
                                {puzzle.priority}
                              </span>
                              <span style={{
                                fontSize: '12px',
                                color: puzzle.status === 'filled' ? '#10b981' : puzzle.status === 'partial' ? '#f59e0b' : '#6b7280',
                              }}>
                                {puzzle.status === 'filled' ? '✓' : puzzle.status === 'partial' ? '◐' : '□'}
                              </span>
                              {puzzle.id} {puzzle.title}
                            </div>
                            <div style={{ fontSize: '11px', color: '#888' }}>
                              {getEvidenceStars(puzzle.evidenceStrength)} 知识:{(puzzle.knowledgeRate * 100).toFixed(0)}% 因果:{(puzzle.causalRate * 100).toFixed(0)}%
                            </div>
                            <div style={{ display: 'flex', gap: '4px', marginTop: '2px' }}>
                              <span style={{
                                fontSize: '9px',
                                background: `${evidenceLevelColor[puzzle.evidenceLevel]}20`,
                                color: evidenceLevelColor[puzzle.evidenceLevel],
                                padding: '1px 5px',
                                borderRadius: '3px',
                                fontFamily: 'monospace',
                              }}>
                                {puzzle.evidenceLevel}: {evidenceLevelLabel[puzzle.evidenceLevel]}
                              </span>
                            </div>
                          </div>
                          {expandedPuzzle === puzzle.id ? <ChevronUp size={14} color="#00d2ff" /> : <ChevronDown size={14} color="#666" />}
                        </div>

                        {expandedPuzzle === puzzle.id && (
                          <div style={{
                            padding: '14px',
                            borderTop: '1px solid rgba(255,255,255,0.06)',
                            background: 'rgba(0,0,0,0.2)',
                          }}>
                            {/* 格子目标与原理 */}
                            {(puzzle.goal || puzzle.principle) && (
                              <div style={{ marginBottom: '12px', display: 'flex', gap: '8px' }}>
                                {puzzle.goal && (
                                  <div style={{
                                    flex: 1,
                                    padding: '8px 10px',
                                    background: 'rgba(0,210,255,0.04)',
                                    borderRadius: '6px',
                                    borderLeft: '2px solid #00d2ff',
                                  }}>
                                    <div style={{ fontSize: '9px', fontWeight: 'bold', color: '#00d2ff', marginBottom: '2px', letterSpacing: '0.5px' }}>
                                      目标 GOAL
                                    </div>
                                    <div style={{ fontSize: '11px', color: '#ccc', lineHeight: '1.5' }}>
                                      {puzzle.goal}
                                    </div>
                                  </div>
                                )}
                                {puzzle.principle && (
                                  <div style={{
                                    flex: 1,
                                    padding: '8px 10px',
                                    background: 'rgba(168,85,247,0.04)',
                                    borderRadius: '6px',
                                    borderLeft: '2px solid #a855f7',
                                  }}>
                                    <div style={{ fontSize: '9px', fontWeight: 'bold', color: '#a855f7', marginBottom: '2px', letterSpacing: '0.5px' }}>
                                      原理 PRINCIPLE
                                    </div>
                                    <div style={{ fontSize: '11px', color: '#ccc', lineHeight: '1.5' }}>
                                      {puzzle.principle}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}
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
                                Evidence 证据
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
                )}
              </div>
            ))}
          </div>

          {/* Right: Framework & Mainlines */}
          <div>
            {/* 四层理论框架 */}
            <h3 style={{
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#fff',
              marginBottom: '20px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}>
              <Layers size={18} color="#00d2ff" />
              四层理论框架 Four-Layer Framework
            </h3>

            {puzzleData && puzzleData.fourLayerFramework.map((layer) => (
              <div
                key={layer.id}
                style={{
                  marginBottom: '12px',
                  background: 'rgba(0,0,0,0.3)',
                  borderRadius: '10px',
                  overflow: 'hidden',
                }}
              >
                <div
                  onClick={() => togglePreparation(`layer_${layer.id}`)}
                  style={{
                    padding: '12px 14px',
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
                      marginBottom: '2px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                    }}>
                      {layer.name} {layer.nameEn}
                      <span style={{
                        fontSize: '12px',
                        fontWeight: 'bold',
                        color: layer.fillRate > 0.3 ? '#10b981' : layer.fillRate > 0.15 ? '#f59e0b' : '#ef4444',
                      }}>
                        {(layer.fillRate * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div style={{ fontSize: '11px', color: '#888' }}>
                      {layer.question}
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{
                      width: '50px',
                      height: '5px',
                      background: 'rgba(255,255,255,0.1)',
                      borderRadius: '3px',
                      overflow: 'hidden',
                    }}>
                      <div style={{
                        width: `${layer.fillRate * 100}%`,
                        height: '100%',
                        background: layer.fillRate > 0.3 ? '#10b981' : layer.fillRate > 0.15 ? '#f59e0b' : '#ef4444',
                        borderRadius: '3px',
                      }} />
                    </div>
                    {expandedPreparation === `layer_${layer.id}` ? <ChevronUp size={14} color="#00d2ff" /> : <ChevronDown size={14} color="#666" />}
                  </div>
                </div>

                {expandedPreparation === `layer_${layer.id}` && (
                  <div style={{
                    padding: '12px 14px',
                    borderTop: '1px solid rgba(255,255,255,0.06)',
                    background: 'rgba(0,0,0,0.2)',
                  }}>
                    <div style={{ fontSize: '12px', color: '#888', marginBottom: '6px' }}>
                      工具: {layer.tools}
                    </div>
                    <div style={{ fontSize: '12px', color: '#ccc', marginBottom: '6px' }}>
                      规律: {layer.relatedRules.join(', ')}
                    </div>
                    {layer.note && (
                      <div style={{ fontSize: '12px', color: '#f59e0b', padding: '6px 10px', background: 'rgba(245,158,11,0.1)', borderRadius: '6px' }}>
                        {layer.note}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {/* 三条研究主线 */}
            <h3 style={{
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#fff',
              margin: '24px 0 16px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}>
              <Target size={18} color="#00d2ff" />
              三条研究主线 Research Mainlines
            </h3>

            {puzzleData && puzzleData.mainlines.map((mainline) => (
              <div
                key={mainline.id}
                style={{
                  marginBottom: '12px',
                  background: 'rgba(0,0,0,0.3)',
                  borderRadius: '10px',
                  overflow: 'hidden',
                }}
              >
                <div
                  onClick={() => togglePreparation(`mainline_${mainline.id}`)}
                  style={{
                    padding: '12px 14px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <div>
                    <div style={{
                      fontSize: '14px',
                      fontWeight: 'bold',
                      color: '#fff',
                      marginBottom: '2px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                    }}>
                      <span style={{
                        fontSize: '12px',
                        background: mainline.id === 'A' ? 'rgba(239,68,68,0.2)' : mainline.id === 'B' ? 'rgba(16,185,129,0.2)' : 'rgba(168,85,247,0.2)',
                        color: mainline.id === 'A' ? '#ef4444' : mainline.id === 'B' ? '#10b981' : '#a855f7',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontFamily: 'monospace',
                      }}>
                        主线{mainline.id}
                      </span>
                      {mainline.name}
                      <span style={{ fontSize: '11px', color: '#888', fontWeight: 'normal' }}>
                        {mainline.nameEn}
                      </span>
                    </div>
                    <div style={{ fontSize: '11px', color: '#888' }}>
                      目标: {mainline.target}
                    </div>
                  </div>
                  {expandedPreparation === `mainline_${mainline.id}` ? <ChevronUp size={14} color="#00d2ff" /> : <ChevronDown size={14} color="#666" />}
                </div>

                {expandedPreparation === `mainline_${mainline.id}` && (
                  <div style={{
                    padding: '12px 14px',
                    borderTop: '1px solid rgba(255,255,255,0.06)',
                    background: 'rgba(0,0,0,0.2)',
                  }}>
                    {mainline.steps.map((step) => (
                      <div key={step.id} style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '10px',
                        padding: '6px 0',
                        borderBottom: '1px solid rgba(255,255,255,0.04)',
                      }}>
                        <span style={{
                          fontSize: '11px',
                          fontFamily: 'monospace',
                          color: '#00d2ff',
                          minWidth: '28px',
                        }}>
                          {step.id}
                        </span>
                        <span style={{ fontSize: '12px', color: '#ccc', flex: 1 }}>
                          {step.title}
                        </span>
                        <span style={{ fontSize: '10px', color: '#888' }}>
                          {step.priority}
                        </span>
                        <span style={{
                          fontSize: '10px',
                          color: step.status === 'completed' ? '#10b981' : step.status === 'in_progress' ? '#f59e0b' : '#6b7280',
                          background: step.status === 'completed' ? 'rgba(16,185,129,0.1)' : step.status === 'in_progress' ? 'rgba(245,158,11,0.1)' : 'rgba(107,114,128,0.1)',
                          padding: '1px 6px',
                          borderRadius: '3px',
                        }}>
                          {step.status === 'completed' ? '✓' : step.status === 'in_progress' ? '◐' : '□'}
                        </span>
                      </div>
                    ))}
                    <div style={{ marginTop: '8px', fontSize: '11px', color: '#888' }}>
                      填充格子: {mainline.steps.flatMap(s => s.cells).join(', ')}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* 10大硬伤 */}
            <h3 style={{
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#fff',
              margin: '24px 0 16px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}>
              <AlertCircle size={18} color="#ef4444" />
              10大硬伤 Hard Issues
            </h3>

            {puzzleData && puzzleData.hardIssues.map((issue) => (
              <div key={issue.id} style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: '8px',
                marginBottom: '8px',
                padding: '6px 10px',
                background: 'rgba(239,68,68,0.05)',
                borderRadius: '6px',
                borderLeft: `3px solid ${issue.severity >= 5 ? '#ef4444' : issue.severity >= 4 ? '#f59e0b' : '#6b7280'}`,
              }}>
                <span style={{
                  fontSize: '11px',
                  fontWeight: 'bold',
                  color: issue.severity >= 5 ? '#ef4444' : issue.severity >= 4 ? '#f59e0b' : '#6b7280',
                  minWidth: '16px',
                }}>
                  {issue.severity >= 5 ? '★★★★★' : issue.severity >= 4 ? '★★★★' : '★★★'}
                </span>
                <span style={{ fontSize: '12px', color: '#ccc' }}>
                  {issue.title}
                </span>
              </div>
            ))}

            {/* 5条定律候选池 */}
            <h3 style={{
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#fff',
              margin: '24px 0 16px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}>
              <Sigma size={18} color="#a855f7" />
              定律候选池 Candidate Laws
            </h3>

            {puzzleData && puzzleData.candidateLaws && puzzleData.candidateLaws.map((law) => (
              <div key={law.id} style={{
                marginBottom: '10px',
                background: 'rgba(168,85,247,0.08)',
                borderRadius: '10px',
                overflow: 'hidden',
                borderLeft: '3px solid #a855f7',
              }}>
                <div
                  onClick={() => togglePreparation(`law_${law.id}`)}
                  style={{ padding: '10px 14px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                >
                  <div>
                    <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#fff', marginBottom: '2px' }}>
                      {law.name}
                      <span style={{ fontSize: '10px', color: '#888', fontWeight: 'normal', marginLeft: '4px' }}>{law.nameEn}</span>
                    </div>
                    <div style={{ fontSize: '10px', color: '#a855f7', fontFamily: 'monospace' }}>{law.formula}</div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ fontSize: '9px', background: `${evidenceLevelColor[law.evidenceLevel] || '#6b7280'}20`, color: evidenceLevelColor[law.evidenceLevel] || '#6b7280', padding: '2px 5px', borderRadius: '3px', fontFamily: 'monospace' }}>
                      {law.evidenceLevel}
                    </span>
                    {expandedPreparation === `law_${law.id}` ? <ChevronUp size={14} color="#a855f7" /> : <ChevronDown size={14} color="#666" />}
                  </div>
                </div>
                {expandedPreparation === `law_${law.id}` && (
                  <div style={{ padding: '10px 14px', borderTop: '1px solid rgba(168,85,247,0.15)', background: 'rgba(0,0,0,0.2)' }}>
                    <div style={{ fontSize: '11px', color: '#ccc', marginBottom: '6px' }}>{law.description}</div>
                    {law.testablePredictions && (
                      <div style={{ fontSize: '10px', color: '#10b981', marginBottom: '4px' }}>
                        预测: {law.testablePredictions.join('; ')}
                      </div>
                    )}
                    <div style={{ fontSize: '10px', color: '#888' }}>支撑: {law.supportedBy.join(', ')} | 格子: {law.cellsAffected.join(', ')}</div>
                    {law.note && <div style={{ fontSize: '10px', color: '#f59e0b', marginTop: '4px', padding: '4px 8px', background: 'rgba(245,158,11,0.1)', borderRadius: '4px' }}>{law.note}</div>}
                  </div>
                )}
              </div>
            ))}

            {/* 预测系统 */}
            <h3 style={{
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#fff',
              margin: '24px 0 16px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}>
              <TestTube size={18} color="#10b981" />
              预测系统 Predictions
            </h3>

            {puzzleData && puzzleData.predictions && puzzleData.predictions.map((pred) => (
              <div key={pred.id} style={{
                display: 'flex', flexDirection: 'column', gap: '4px',
                marginBottom: '8px', padding: '8px 10px',
                background: 'rgba(16,185,129,0.05)',
                borderRadius: '6px',
                borderLeft: `3px solid ${pred.priority === 'P0' ? '#ef4444' : pred.priority === 'P1' ? '#f59e0b' : '#6b7280'}`,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '10px', fontFamily: 'monospace', color: '#10b981' }}>{pred.id} [{pred.lawRef}]</span>
                  <span style={{ fontSize: '9px', background: pred.status === 'untested' ? 'rgba(107,114,128,0.2)' : 'rgba(16,185,129,0.2)', color: pred.status === 'untested' ? '#6b7280' : '#10b981', padding: '1px 5px', borderRadius: '3px' }}>
                    {pred.status === 'untested' ? '待验证' : '已验证'}
                  </span>
                </div>
                <div style={{ fontSize: '11px', color: '#ccc' }}>
                  <span style={{ color: '#10b981' }}>IF</span> {pred.if} → <span style={{ color: '#f59e0b' }}>THEN</span> {pred.then}
                </div>
              </div>
            ))}

            {/* 未解悖论 */}
            <h3 style={{
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#fff',
              margin: '24px 0 16px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}>
              <Flame size={18} color="#ef4444" />
              未解悖论 Unresolved Paradoxes
            </h3>

            {puzzleData && puzzleData.unresolvedParadoxes && puzzleData.unresolvedParadoxes.map((px) => (
              <div key={px.id} style={{
                marginBottom: '8px', padding: '8px 10px',
                background: 'rgba(239,68,68,0.06)',
                borderRadius: '6px',
                borderLeft: `3px solid ${px.severity >= 5 ? '#ef4444' : '#f59e0b'}`,
              }}>
                <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#fff', marginBottom: '2px' }}>
                  {px.title}
                  <span style={{ fontSize: '10px', color: px.severity >= 5 ? '#ef4444' : '#f59e0b', marginLeft: '6px' }}>
                    {'★'.repeat(px.severity)}
                  </span>
                </div>
                <div style={{ fontSize: '11px', color: '#ccc' }}>{px.description}</div>
              </div>
            ))}

            {/* 跨模型验证矩阵 */}
            <h3 style={{
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#fff',
              margin: '24px 0 16px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}>
              <GitBranch size={18} color="#3b82f6" />
              跨模型矩阵 Cross-Model
            </h3>

            {puzzleData && puzzleData.crossModelMatrix && (
              <div style={{ marginBottom: '12px' }}>
                <div style={{ display: 'flex', gap: '4px', marginBottom: '6px', flexWrap: 'wrap' }}>
                  {puzzleData.crossModelMatrix.models.map((m, i) => (
                    <span key={i} style={{ fontSize: '10px', color: '#3b82f6', background: 'rgba(59,130,246,0.1)', padding: '2px 6px', borderRadius: '3px' }}>{m}</span>
                  ))}
                </div>
                {puzzleData.crossModelMatrix.findings.map((f, i) => (
                  <div key={i} style={{
                    display: 'flex', alignItems: 'center', gap: '6px',
                    padding: '5px 8px', marginBottom: '3px',
                    background: 'rgba(0,0,0,0.2)', borderRadius: '4px',
                  }}>
                    <span style={{ fontSize: '10px', color: f.universal ? '#10b981' : '#f59e0b', fontFamily: 'monospace', minWidth: '40px' }}>
                      {f.universal ? '✓ Universal' : '✗ Specific'}
                    </span>
                    <span style={{ fontSize: '11px', color: '#ccc', flex: 1 }}>{f.finding}</span>
                  </div>
                ))}
              </div>
            )}

            {/* 训练动态 */}
            <h3 style={{
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#fff',
              margin: '24px 0 16px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}>
              <Atom size={18} color="#f59e0b" />
              训练动态 Training Dynamics
            </h3>

            {puzzleData && puzzleData.trainingDynamics && Object.entries(puzzleData.trainingDynamics).map(([key, phase]) => (
              <div key={key} style={{
                marginBottom: '8px', padding: '8px 10px',
                background: 'rgba(245,158,11,0.05)',
                borderRadius: '6px',
                borderLeft: '3px solid #f59e0b',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2px' }}>
                  <span style={{ fontSize: '12px', fontWeight: 'bold', color: '#fff' }}>{phase.name}</span>
                  <span style={{ fontSize: '9px', background: `${evidenceLevelColor[phase.confidence] || '#6b7280'}20`, color: evidenceLevelColor[phase.confidence] || '#6b7280', padding: '1px 5px', borderRadius: '3px', fontFamily: 'monospace' }}>
                    {phase.confidence}
                  </span>
                </div>
                <div style={{ fontSize: '10px', color: '#888', marginBottom: '2px' }}>{phase.nameEn} | {phase.layerRange}</div>
                <div style={{ fontSize: '11px', color: '#ccc' }}>{phase.description}</div>
                {phase.note && <div style={{ fontSize: '10px', color: '#f59e0b', marginTop: '2px' }}>{phase.note}</div>}
              </div>
            ))}

            {/* 证据等级说明 */}
            <h3 style={{
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#fff',
              margin: '24px 0 16px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}>
              <Microscope size={18} color="#6366f1" />
              证据等级 Evidence Levels
            </h3>

            {puzzleData && puzzleData.evidenceLevel && Object.entries(puzzleData.evidenceLevel).map(([key, level]) => (
              <div key={key} style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                padding: '4px 8px', marginBottom: '3px',
                background: `${evidenceLevelColor[key] || '#6b7280'}10`,
                borderRadius: '4px',
              }}>
                <span style={{ fontSize: '10px', fontFamily: 'monospace', color: evidenceLevelColor[key] || '#6b7280', fontWeight: 'bold', minWidth: '20px' }}>{key}</span>
                <span style={{ fontSize: '11px', color: '#ccc', flex: 1 }}>{level.label}</span>
                <span style={{ fontSize: '9px', color: '#888' }}>{'★'.repeat(level.strength)}</span>
              </div>
            ))}
            <div style={{
              marginTop: '20px',
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
