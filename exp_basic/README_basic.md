# Basic Experiments (exp_basic)

This folder contains shell scripts for running the **core HyEm experiments** on HPO and DO ontologies with 5k-node subsets.

## Prerequisites

Before running these scripts, ensure:

1. Environment is set up (see main [README.md](../README.md))
2. Raw ontology data is downloaded:
   ```bash
   python scripts/00_download_data.py --datasets hpo do
   ```

---

## Experiment Workflow

### Step 1: Run Basic Pipeline (Single Dataset)

Runs the complete pipeline: subset → queries → embeddings → training → evaluation.

```bash
# HPO dataset (5k nodes, seed=0, GPU)
bash exp_basic/run_basic_pipeline.sh hpo 5000 0 cuda

# DO dataset (5k nodes, seed=0, GPU)
bash exp_basic/run_basic_pipeline.sh do 5000 0 cuda

# (Option) CPU-only (slower)
# bash exp_basic/run_basic_pipeline.sh hpo 5000 0 cpu
```

**Output:**
- `data/processed/<dataset>/5000_seed0/results/summary.csv` — Main retrieval metrics
- `data/processed/<dataset>/5000_seed0/results/per_query_*.csv` — Per-query results

---

### Step 2: Run Analysis Benchmarks

Runs indexability stress test and efficiency benchmarks.

```bash
# HPO dataset
bash exp_basic/run_basic_analysis.sh hpo 5000 0 cuda

# DO dataset
bash exp_basic/run_basic_analysis.sh do 5000 0 cuda
```

**Output:**
- `data/processed/<dataset>/5000_seed0/analysis/indexability_recall_curve.csv`
- `data/processed/<dataset>/5000_seed0/analysis/efficiency.csv`
- `data/processed/<dataset>/5000_seed0/analysis/figs/recall_curve.pdf`

---

### Step 3: Generate Paper Artifacts

Generates LaTeX table rows and figures for the paper.

```bash
# Generate for both HPO and DO
bash exp_basic/run_paper_artifacts.sh "hpo do" 5000 0
```

**Output:**
- `paper_artifacts/<dataset>/rows_*.tex` — LaTeX table rows
- `paper_artifacts/<dataset>/mixing_bar.pdf` — Mixing visualization

---

## Complete Reproduction (Both Datasets)

To reproduce all basic experiments:

```bash
# 1. Download data
python scripts/00_download_data.py --datasets hpo do

# 2. Run pipelines
bash exp_basic/run_basic_pipeline.sh hpo 5000 0 cuda
bash exp_basic/run_basic_pipeline.sh do  5000 0 cuda

# 3. Run analysis
bash exp_basic/run_basic_analysis.sh hpo 5000 0 cuda
bash exp_basic/run_basic_analysis.sh do  5000 0 cuda

# 4. Generate paper artifacts
bash exp_basic/run_paper_artifacts.sh "hpo do" 5000 0
```

---

## Script Parameters

| Script | Arg 1 | Arg 2 | Arg 3 | Arg 4 |
|--------|-------|-------|-------|-------|
| `run_basic_pipeline.sh` | dataset | subset_size | seed | device |
| `run_basic_analysis.sh` | dataset | subset_size | seed | device |
| `run_paper_artifacts.sh` | "datasets" | subset_size | seed | — |

**Defaults:**
- dataset: `hpo`
- subset_size: `5000`
- seed: `0`
- device: `cpu`

---

## Expected Results

After running the basic experiments, you should observe:

| Metric | HPO-5k | DO-5k |
|--------|--------|-------|
| Q-H MRR (hyem_soft) | ~0.21 | ~0.16 |
| Q-M MRR (hyem_soft) | ~0.45 | ~0.36 |
| Gate AUC (MLP) | ~1.00 | ~1.00 |

> Note: Exact numbers may vary slightly due to random initialization.
