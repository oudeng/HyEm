# Extended Experiments 1 (exp_ext1)

This folder contains **extended experiment scripts** for the HyEm paper, addressing reviewer concerns and providing additional empirical evidence.

## Prerequisites

Before running these scripts, ensure:

1. Environment is set up (see main [README.md](../README.md))
2. **Basic experiments completed** (see [exp_basic/README_basic.md](../exp_basic/README_basic.md))

---

## Experiment 1: Biomedical Encoder for Q-E

**Goal:** Address the "0.000 Q-E" concern by using a biomedical sentence encoder (S-BioBERT) instead of the default general-purpose encoder.

### Run Commands

```bash
# HPO dataset
bash exp_ext1/run_qe_biomed_encoder.sh hpo 5000 0 \
  pritamdeka/S-BioBERT-snli-multinli-stsb cuda

# DO dataset
bash exp_ext1/run_qe_biomed_encoder.sh do 5000 0 \
  pritamdeka/S-BioBERT-snli-multinli-stsb cuda
```

### Regenerate Paper Artifacts

```bash
python scripts/11_make_paper_artifacts.py \
  --datasets hpo do --subset_size 5000 --seed 0 \
  --data_dir data/processed --out_dir paper_artifacts
```

---

## Experiment 2: Scale-up (20k Nodes)

**Goal:** Provide evidence beyond 5k nodes by running the complete pipeline on 20k-node subsets, including indexability and efficiency benchmarks.

### Run Commands

```bash
# HPO dataset (20k nodes)
bash exp_ext1/run_scale_20k.sh hpo 20000 0 cuda

# DO dataset (20k nodes)
bash exp_ext1/run_scale_20k.sh do 20000 0 cuda
```

**Output:**
- `data/processed/<dataset>/20000_seed0/results/` — Retrieval metrics
- `data/processed/<dataset>/20000_seed0/analysis/` — Indexability & efficiency

---

## Experiment 3: Radius-Budget Sweep

**Goal:** Sweep radius budget R ∈ {2, 3, 4, 5} and compare how the indexability recall-vs-L curve shifts relative to the theory-guided line L_th = ⌈κ(R)·k⌉.

### Run Commands

```bash
# HPO dataset
bash exp_ext1/run_radius_sweep.sh hpo 5000 0 cuda "2.0 3.0 4.0 5.0"

# DO dataset
bash exp_ext1/run_radius_sweep.sh do 5000 0 cuda "2.0 3.0 4.0 5.0"
```

**Output:**
- `data/processed/<dataset>/5000_seed0/analysis/indexability_recall_curve_R*.csv`

---

## Experiment 4: Theory-Guided Indexability Plot

**Goal:** Overlay the theory-guided vertical line on the indexability recall curve.

### Run Commands

```bash
# Ensure analysis directory exists
mkdir -p paper_artifacts/figs

# HPO plot
python exp_ext1/plot_indexability_with_theory.py \
  --in_csv data/processed/hpo/5000_seed0/analysis/indexability_recall_curve.csv \
  --out_pdf paper_artifacts/figs/recall_curve_hpo_5000_with_theory.pdf \
  --R 3.0 --k 10 --title "HPO-5k"

# DO plot
python exp_ext1/plot_indexability_with_theory.py \
  --in_csv data/processed/do/5000_seed0/analysis/indexability_recall_curve.csv \
  --out_pdf paper_artifacts/figs/recall_curve_do_5000_with_theory.pdf \
  --R 3.0 --k 10 --title "DO-5k"
```

---

## Complete Extended Experiments

To run all extended experiments:

```bash
# Exp 1: Biomedical encoder
bash exp_ext1/run_qe_biomed_encoder.sh hpo 5000 0 pritamdeka/S-BioBERT-snli-multinli-stsb cuda
bash exp_ext1/run_qe_biomed_encoder.sh do  5000 0 pritamdeka/S-BioBERT-snli-multinli-stsb cuda

# Exp 2: Scale-up
bash exp_ext1/run_scale_20k.sh hpo 20000 0 cuda
bash exp_ext1/run_scale_20k.sh do  20000 0 cuda

# Exp 3: Radius sweep
bash exp_ext1/run_radius_sweep.sh hpo 5000 0 cuda "2.0 3.0 4.0 5.0"
bash exp_ext1/run_radius_sweep.sh do  5000 0 cuda "2.0 3.0 4.0 5.0"

# Exp 4: Theory plots
mkdir -p paper_artifacts/figs
python exp_ext1/plot_indexability_with_theory.py \
  --in_csv data/processed/hpo/5000_seed0/analysis/indexability_recall_curve.csv \
  --out_pdf paper_artifacts/figs/recall_curve_hpo_5000_with_theory.pdf \
  --R 3.0 --k 10 --title "HPO-5k"
python exp_ext1/plot_indexability_with_theory.py \
  --in_csv data/processed/do/5000_seed0/analysis/indexability_recall_curve.csv \
  --out_pdf paper_artifacts/figs/recall_curve_do_5000_with_theory.pdf \
  --R 3.0 --k 10 --title "DO-5k"
```

---

## Script Reference

| Script | Purpose | Key Arguments |
|--------|---------|---------------|
| `run_qe_biomed_encoder.sh` | Re-encode with biomedical encoder | dataset, subset_size, seed, model_name, device |
| `run_scale_20k.sh` | Full pipeline on 20k subsets | dataset, subset_size, seed, device |
| `run_radius_sweep.sh` | Sweep radius budget values | dataset, subset_size, seed, device, R_list |
| `plot_indexability_with_theory.py` | Plot recall curve with theory line | --in_csv, --out_pdf, --R, --k |

---

## Theory Background

The radius-budget sweep validates the theoretical bound:

$$\kappa(R) = \frac{\sinh(R)}{R}$$

| R | κ(R) | L_th (k=10) |
|---|------|-------------|
| 2.0 | 1.81 | 19 |
| 3.0 | 3.34 | 34 |
| 4.0 | 6.83 | 69 |
| 5.0 | 14.8 | 148 |

> L_th = ⌈κ(R)·k⌉ is the minimum oversampling factor needed for near-perfect recall.
