# 语言有效性的数学模型 (Mathematical Model of Language Validity)

为了从整体上建立“语言有效性”的数学模型，我们定义一个标量函数 $V(M, x)$，其中 $M$ 是大语言模型，$x$ 是输入序列。
该模型将“有效性”解构为三个维度的加权和：

$$ V(M, x) = \alpha \cdot V_{info}(M, x) + \beta \cdot V_{geo}(M, x) + \gamma \cdot V_{struct}(M, x) $$

## 1. 信息论有效性 (Information Theoretic Validity) - $V_{info}$
衡量模型输出作为“语言”的统计合理性。

*   **困惑度 (Perplexity, PPL)**:
    $$ V_{PPL}(x) = \exp\left( -\frac{1}{N} \sum_{i=1}^N \log P(x_i | x_{<i}) \right) $$
    *有效性与 PPL 成反比。*

*   **熵 (Entropy)**: 衡量预测的不确定性。
    $$ H(x) = - \sum_{k} P(x_{i+1}=k|x) \log P(x_{i+1}=k|x) $$
    *极低熵可能意味着过拟合或退化，极高熵意味着混乱。有效语言通常处于“边缘混沌”状态。*

## 2. 几何有效性 (Geometric Validity) - $V_{geo}$
衡量内部表征 (Internal Representations) 的空间结构质量。

*   **各向异性 (Anisotropy)**:
    计算嵌入向量在空间中的分布均匀度。
    $$ A(L) = \frac{1}{N^2} \sum_{i,j} \cos(h_i, h_j) $$
    *严重的各向异性（Representation Collapse）通常意味着表征退化。*

*   **本征维度 (Intrinsic Dimensionality, ID)**: 
    使用 PCA 或 Participation Ratio 估算流形维度。
    $$ D_{int} = \frac{(\sum \lambda_i)^2}{\sum \lambda_i^2} $$
    *有效表征应具备适度的本征维度（既不冗余也不过度压缩）。*

## 3. 结构有效性 (Structural Validity) - $V_{struct}$
衡量模型内部计算流是否遵循特定任务的因果逻辑。

*   **回路稀疏性 (Circuit Sparsity)**:
    类似于大脑的能量最小化原则，有效的推理应依赖于稀疏的子图。
    $$ S(G) = \frac{|E_{dominant}|}{|E_{total}|} $$
    *越稀疏的回路通常意味着越健壮的机制。*

*   **模块化程度 (Modularity)**:
    使用图论中的 Q-modularity 来衡量计算图的簇结构。

---

## 4. 建议的实现路径 (Implementation Path)

我们将在 `structure_analyzer.py` 中增加一个新的类 `LanguageValidity` 来计算上述指标。

### 新增 API

```python
class LanguageValidity:
    def __init__(self, model):
        self.model = model
    
    def compute_perplexity(self, text: str) -> float:
        pass
        
    def compute_entropy_profile(self, text: str) -> List[float]:
        pass
        
    def check_anisotropy(self, layer_idx: int) -> float:
        pass
        
    def analyze_holistic_validity(self, text: str) -> Dict[str, float]:
        """Aggregate all metrics into a holistic report"""
        pass
```
