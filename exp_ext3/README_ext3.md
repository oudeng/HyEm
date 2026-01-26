# Extended Experiments 3 (exp_ext3)

This folder contains **revision experiments** designed to address Major Comments from peer review.

## Overview of Experiments

| Script | Addresses | Description |
|--------|-----------|-------------|
| `analyze_safety_valve.py` | M2 | Quantify soft mixing's "safety valve" effect on Q-E |
| `depth_stratified_analysis.py` | M1, M2 | Stratify results by depth buckets |
| `theoretical_scaling.py` | M1 | Plot κ(R) vs depth and theoretical scale limits |
| `gate_robustness_test.py` | M3 | Test gate calibration on perturbed queries |
| `candidate_pooling_ablation.py` | m3 | Quantify contribution of C_H ∪ C_E pooling |
| `make_revision_figures.py` | All | Aggregate results into combined figures & LaTeX |

---

## Quick Start

```bash
# Run all exp_ext3 analyses on existing 5k results
bash exp_ext3/run_all_ext3.sh hpo 5000 0
bash exp_ext3/run_all_ext3.sh do 5000 0

# Generate combined revision figures (after running both datasets)
python exp_ext3/make_figures.py --data_dir data/processed --out_dir paper_artifacts/ext3
```

---

## Experiment Details

### 1. Safety Valve Analysis (M2)

**Goal:** Quantify how much soft mixing preserves Q-E performance compared to pure hyperbolic routing.

```bash
python exp_ext3/analyze_safety_valve.py \
  --dataset hpo --subset_size 5000 --seed 0 \
  --data_dir data/processed
```

**Output:**
- `results/safety_valve_analysis.csv`: Per-method Q-E degradation metrics
- `analysis/figs/safety_valve_comparison.pdf`: Bar chart comparing methods

**Key Metric:** "Q-E retention rate" = HyEm_soft_MRR / Euclidean_text_MRR

---

### 2. Depth-Stratified Analysis (M1, M2)

**Goal:** Show whether hyperbolic benefits concentrate at deeper levels.

```bash
python exp_ext3/depth_stratified_analysis.py \
  --dataset hpo --subset_size 5000 --seed 0 \
  --data_dir data/processed --depth_buckets 4
```

**Output:**
- `results/depth_stratified_qh.csv`: Q-H metrics per depth quartile
- `results/depth_stratified_qm.csv`: Q-M metrics per depth quartile
- `analysis/figs/depth_stratified_heatmap.pdf`: Heatmap visualization

---

### 3. Theoretical Scaling Analysis (M1)

**Goal:** Plot κ(R) = sinh(R)/R and show safe operating regime for different ontology depths.

```bash
python exp_ext3/theoretical_scaling.py \
  --max_depth 50 \
  --dimensions 16,32,64,128 \
  --out_dir paper_artifacts/ext3
```

**Output:**
- `theoretical_R_vs_depth.pdf`: Shows R needed for depth D at different d
- `theoretical_kappa_distortion.pdf`: Shows κ(R) growth and "danger zone"
- `theoretical_safe_regime.pdf`: 2D heatmap of safe operating regime

---

### 4. Gate Robustness Test (M3)

**Goal:** Test gate calibration robustness by adding noise to embeddings (simulates encoding variation from typos/paraphrases).

```bash
python exp_ext3/gate_robustness_test.py \
  --dataset hpo --subset_size 5000 --seed 0 \
  --data_dir data/processed \
  --noise_levels "0.0,0.1,0.2,0.3"
```

**How it works:**
- Uses pre-computed query embeddings (no re-encoding needed)
- Adds Gaussian noise at different levels to simulate encoding variation
- Reports accuracy degradation at each noise level

**Output:**
- `results/gate_robustness.csv`: Gate accuracy under different noise levels
- `analysis/figs/gate_robustness_bar.pdf`: Visualization

---

### 5. Candidate Pooling Ablation (m3)

**Goal:** Quantify contribution of using C_H ∪ C_E vs C_H only.

```bash
python exp_ext3/candidate_pooling_ablation.py \
  --dataset hpo --subset_size 5000 --seed 0 \
  --data_dir data/processed
```

**Output:**
- `results/candidate_pooling_ablation.csv`: Comparison metrics

---

### 6. Generate Combined Revision Figures

**Goal:** Aggregate all exp_ext3 results into combined figures and LaTeX tables for paper.

```bash
# Run after processing all datasets (hpo, do)
python exp_ext3/make_figures.py \
  --data_dir data/processed \
  --out_dir paper_artifacts/ext3 \
  --datasets hpo,do
```

**Output:**
- `fig_combined_safety_valve.pdf` - Safety valve comparison across datasets
- `fig_combined_pooling.pdf` - Pooling ablation comparison
- `table_safety_valve.tex` - LaTeX table for Table 2 revision
- `table_pooling_ablation.tex` - LaTeX table for pooling ablation
- `REVISION_SUMMARY.md` - Markdown summary report

---

## Expected Paper Revisions

After running exp_ext3, update the paper as follows:

1. **Table 2 revision:** Add "Q-E Retention" column showing soft mixing preserves >90% of Euclidean baseline
2. **New Figure (M1):** Add theoretical scaling plot showing safe R regime for depths up to 50
3. **Table 3 revision:** Add depth-stratified breakdown showing hyperbolic helps more at deeper levels
4. **Section 6.5 revision:** Add gate robustness discussion
5. **Table 8 completion:** Fill in HGCN results and discuss trade-offs

---

## Notes

- All scripts assume existing results from exp_basic/exp_ext1/exp_ext2 are available
- Outputs are saved alongside existing results for easy integration
- Scripts are designed to be lightweight (<1 min each on CPU)