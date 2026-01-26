# HyEm Revision Analysis Summary

## 1. Safety Valve Analysis (M2)

| Dataset | Q-E Retention | Q-H Hits@10 | Trade-off Ratio |
|---------|---------------|-------------|-----------------|
| HPO | 93.9% | 0.234 | 119.7x |
| DO | 97.9% | 0.202 | 39.8x |

**Key Finding:** Soft mixing retains >90% of Q-E performance while significantly improving Q-H retrieval.

## 2. Candidate Pooling Ablation (m3)

| Dataset | Metric | C_H only | C_H∪C_E | Improvement |
|---------|--------|----------|---------|-------------|
| HPO | QE_hits10 | 0.045 | 0.952 | +1994% |
| HPO | QE_mrr | 0.016 | 0.815 | +4909% |
| HPO | QH_parent_hits10 | 0.048 | 0.234 | +388% |
| HPO | QM_hits10 | 0.163 | 0.533 | +226% |
| DO | QE_hits10 | 0.045 | 0.821 | +1737% |
| DO | QE_mrr | 0.019 | 0.673 | +3358% |
| DO | QH_parent_hits10 | 0.088 | 0.202 | +130% |
| DO | QM_hits10 | 0.221 | 0.469 | +112% |

**Key Finding:** Candidate pooling is essential for maintaining Q-E performance.

## 3. Depth-Stratified Analysis (M1)

### QE Results
| Dataset | Depth | MRR | n_queries |
|---------|-------|-----|-----------|
| HPO | D1 | 0.833 | 247 |
| HPO | D2 | 0.805 | 389 |
| HPO | D3 | 0.813 | 134 |
| DO | D1 | 0.660 | 291 |
| DO | D2 | 0.664 | 340 |
| DO | D3 | 0.706 | 219 |

### QH Results
| Dataset | Depth | MRR | n_queries |
|---------|-------|-----|-----------|
| HPO | D1 | 0.108 | 160 |
| HPO | D2 | 0.090 | 251 |
| HPO | D3 | 0.075 | 88 |
| DO | D1 | 0.062 | 170 |
| DO | D2 | 0.103 | 197 |
| DO | D3 | 0.074 | 133 |

### QM Results
| Dataset | Depth | MRR | n_queries |
|---------|-------|-----|-----------|
| HPO | D1 | 0.223 | 157 |
| HPO | D2 | 0.266 | 247 |
| HPO | D3 | 0.269 | 86 |
| DO | D1 | 0.216 | 169 |
| DO | D2 | 0.216 | 196 |
| DO | D3 | 0.274 | 128 |


## Generated Files

### Figures
- `fig_combined_safety_valve.pdf` - Safety valve comparison across datasets
- `fig_combined_pooling.pdf` - Pooling ablation across datasets
- `fig_safety_valve_*.pdf` - Per-dataset safety valve figures
- `fig_pooling_*.pdf` - Per-dataset pooling figures

### LaTeX Tables
- `table_safety_valve.tex` - Safety valve results for Table 2
- `table_pooling_ablation.tex` - Pooling ablation results
