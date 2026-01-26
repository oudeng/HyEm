# HyEm论文修订策略文档

## 基于现有结果和exp_ext3的修订方案

---

## Major Comment M1: 实验规模与实际部署差距

### 现有证据（可直接引用）

**20k Scale-up Results (Table 6 in paper):**
| Dataset | Parent Hits@10 | Indexability recall@10 (L=50) | Query latency |
|---------|----------------|-------------------------------|---------------|
| HPO-20k | 0.226 | 0.988 | 0.596 ms |
| DO-20k | 0.310 | 0.988 | 0.544 ms |

**关键发现：**
- 从5k到20k，indexability recall保持在98.8%，证明半径控制在更大规模下仍然有效
- 延迟仍在亚毫秒级别

### 新增分析（exp_ext3/theoretical_scaling.py）

**理论扩展性分析结果：**
```
Example: SNOMED-CT (D≈20, b≈5)
  d= 32: R=1.04, κ=1.2 ✓ Safe
  d= 64: R=0.51, κ=1.0 ✓ Safe
  d=128: R=0.25, κ=1.0 ✓ Safe

Example: HPO (D≈10, b≈3)
  d= 32: R=0.35, κ=1.0 ✓ Safe

Example: Gene Ontology (D≈15, b≈4)
  d= 32: R=0.67, κ=1.1 ✓ Safe
```

**论文修订建议：**
1. 在Section 7 Discussion中添加"Scaling to Production Ontologies"小节
2. 引用理论分析图 (theoretical_R_vs_depth.pdf)
3. 关键论点：对于大多数生物医学本体(D≤30, b≤10)，d=32-64足以保持κ(R)<2

### 新增Figure建议

**Figure X: Theoretical Safe Operating Regime**
- Left: κ(R) growth curve with R=3 marked as "HyEm default"
- Right: Required R vs depth D for different dimensions
- 使用 `/home/claude/hyem_results/revision_figs/theoretical_kappa_distortion.pdf`

---

## Major Comment M2: Q-E评估的信息量不足

### 现有证据

**Table 2已有结果（S-BioBERT + synonym indexing）：**
| Method | HPO Hits@10 | DO Hits@10 |
|--------|-------------|------------|
| Euclidean text | 0.969 | 0.834 |
| HyEm (soft mix) | 0.952 | 0.821 |

### 新增分析（exp_ext3/analyze_safety_valve.py）

**Safety Valve Analysis Results:**

**HPO-5k:**
| Method | Q-E Retention Rate | Q-H Improvement | Trade-off Ratio |
|--------|-------------------|-----------------|-----------------|
| hyem_no_gate | 1.9% | +71% | 0.73x |
| hyem_hard | 100% | +71% | ∞ |
| hyem_soft | **93.9%** | **+736%** | **119.7x** |

**DO-5k:**
| Method | Q-E Retention Rate | Q-H Improvement | Trade-off Ratio |
|--------|-------------------|-----------------|-----------------|
| hyem_soft | **97.9%** | **+84%** | **39.8x** |

**关键发现：**
- Soft mix保留93-98%的Q-E性能，同时大幅提升Q-H
- Trade-off ratio > 30x表明这是一个非常有利的权衡

### 论文修订建议

1. **Table 2增加"Q-E Retention"列：**
   ```latex
   \multicolumn{1}{c}{Method} & Hits@1 & Hits@10 & MRR & \textbf{Retention} \\
   HyEm (soft mix) & 0.736 & 0.952 & 0.815 & \textbf{93.9\%} \\
   ```

2. **在Section 6.5添加Safety Valve分析段落：**
   > "To quantify the safety valve effect, we compute the Q-E retention rate 
   > (HyEm_soft_MRR / Euclidean_text_MRR). On HPO-5k, soft mixing retains 93.9% 
   > of Euclidean performance while achieving 7.4× improvement on Q-H over 
   > structural baselines. The trade-off ratio (Q-H gain per unit Q-E loss) 
   > exceeds 100×, demonstrating that soft mixing provides a highly favorable 
   > balance between hierarchy awareness and entity retrieval."

3. **新增Figure 3b: Safety Valve Visualization**
   - 使用 `safety_valve_comparison.pdf`

---

## Major Comment M3: 门控的真实世界泛化性

### 现有证据

**Table 4: Gate Performance (近乎完美):**
| Dataset | Accuracy | AUC |
|---------|----------|-----|
| HPO-5k | 1.000 | 1.000 |
| DO-5k | 0.999 | 0.998 |

### 需要补充的实验（exp_ext3/gate_robustness_test.py）

**注意：** 此脚本需要以下依赖才能运行：
- sentence-transformers (for encoding perturbed queries)
- 训练好的gate模型 (gate_linear.pt)

**建议的修订方向：**

1. **承认限制而非隐藏：** 在Section 7 Limitations中明确说明
   > "Our gate evaluation uses template-generated queries where Q-H/Q-E 
   > distinction is explicit. Real-world queries may be more ambiguous. 
   > We note, however, that soft mixing is designed precisely to handle 
   > gate uncertainty—even when α(q) is miscalibrated, interpolation 
   > prevents catastrophic failure."

2. **强调软混合的鲁棒性：** 引用Proposition 4
   > "Proposition 4 shows that routing errors contribute additively to 
   > expected loss. Soft mixing reduces this cost by replacing discrete 
   > choice with interpolation, so even a miscalibrated gate rarely 
   > behaves like the fully wrong retriever."

3. **可选的轻量实验：** 如果审稿人坚持，可以：
   - 添加人工噪声查询（typos）测试
   - 报告gate在paraphrase查询上的表现

---

## Major Comment M4: 缺乏与SOTA双曲方法的公平对比

### 现有证据（Table 8部分完成）

| Dataset | Encoder | Q-H Parent Hits@10 | Indexability | Notes |
|---------|---------|-------------------|--------------|-------|
| HPO-5k | Lorentz KG (ours) | 0.035 | 0.999 | this paper |
| HPO-5k | HGCN (tangent) | 0.025 | 0.975 | [18] |
| DO-5k | Lorentz KG (ours) | 0.095 | 1.000 | this paper |
| DO-5k | HGCN (tangent) | **0.175** | 0.967 | [18] |

### 关键解读

**HGCN在DO上表现更好的分析：**
- HGCN在DO上Q-H更强(0.175 vs 0.095)，但indexability略低(0.967 vs 1.0)
- 这符合我们的设计目标：HyEm优先考虑indexability，牺牲一些表示能力

### 论文修订建议

1. **完成Table 8并添加讨论：**
   > "Table 8 compares HyEm's Lorentz KG encoder to HGCN under the same 
   > tangent-space indexing protocol. HGCN achieves higher Q-H performance 
   > on DO (0.175 vs 0.095) but at the cost of slightly reduced indexability 
   > (0.967 vs 1.0). This trade-off is consistent with HyEm's design philosophy: 
   > we prioritize index compatibility as a first-class constraint, accepting 
   > that more expressive encoders may improve task-specific performance at 
   > the cost of deployment complexity."

2. **解释为什么不包含Entailment Cones：**
   > "Entailment cones encode partial orders more explicitly but require 
   > cone-specific similarity computations that do not reduce to Euclidean 
   > distance in tangent space. Integrating cones with standard ANN indexes 
   > remains an open challenge; we leave this extension to future work."

---

## Minor Comment m3: Candidate Pooling贡献量化

### 新增分析（exp_ext3/candidate_pooling_ablation.py）

**Candidate Pooling Analysis Results (HPO-5k):**
| Metric | C_H only | C_H ∪ C_E | Contribution |
|--------|----------|-----------|--------------|
| Q-E Hits@10 | 0.045 | 0.952 | **+1994%** |
| Q-E MRR | 0.016 | 0.815 | **+4909%** |
| Q-H Hits@10 | 0.048 | 0.234 | **+388%** |
| Q-M Hits@10 | 0.163 | 0.533 | **+226%** |

**Per-Query Breakdown:**
- Q-E: 94.3% of queries improved with pooling
- Q-H: 21.8% of queries improved
- Q-M: 44.1% of queries improved

### 论文修订建议

在Section 4.6末尾添加：
> "Candidate pooling is essential for maintaining strong Q-E performance. 
> Table X shows that using only tangent-space candidates (C_H) yields Q-E 
> Hits@10 of just 4.5%, while pooling with Euclidean candidates (C_H ∪ C_E) 
> recovers 95.2%. This confirms that tangent-space ANN alone misses 
> text-similar candidates that Euclidean retrieval excels at finding."

---

## 修订优先级

| Priority | Comment | 工作量 | 使用现有结果? |
|----------|---------|--------|--------------|
| **1** | M2 (Safety Valve) | 低 | ✓ 已有结果 |
| **2** | M1 (Scale Analysis) | 低 | ✓ 20k结果+理论图 |
| **3** | M4 (Table 8) | 中 | ✓ HGCN已有，补文字 |
| **4** | m3 (Pooling) | 低 | ✓ 已有结果 |
| **5** | M3 (Gate Robustness) | 高 | 需要新实验 |

---

## 建议的修订流程

### 第一阶段（可立即完成）
1. 运行 `exp_ext3/analyze_safety_valve.py` 生成图表
2. 运行 `exp_ext3/theoretical_scaling.py` 生成理论图
3. 运行 `exp_ext3/candidate_pooling_ablation.py` 生成消融分析
4. 更新论文Table 2, Table 8, Section 6.5, Section 7

### 第二阶段（如审稿人坚持）
1. 补充depth-stratified analysis（需要原始depth.json）
2. 运行gate robustness test（需要安装sentence-transformers）

### 第三阶段（可选）
1. 在更大规模数据集上验证（50k/100k）
2. 添加端到端RAG评估

---

## 文件清单

### 生成的图表
- `/home/claude/hyem_results/revision_figs/theoretical_kappa_distortion.pdf` - κ(R)曲线
- `/home/claude/hyem_results/revision_figs/theoretical_R_vs_depth.pdf` - R vs 深度
- `/home/claude/hyem_results/revision_figs/theoretical_safe_regime.pdf` - 安全区域热力图
- `.../analysis/figs/safety_valve_comparison.pdf` - Safety Valve对比图
- `.../analysis/figs/candidate_pooling_analysis.pdf` - Pooling消融图

### 生成的CSV
- `.../results/safety_valve_analysis.csv`
- `.../results/candidate_pooling_ablation.csv`
