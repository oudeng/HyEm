# Extended Experiments 2 (exp_ext2)

This folder contains **additional extended experiments** for HyEm, mainly to address reviewer requests that require (i) non-trivial entity-centric (Q-E) baselines, (ii) adapter expressivity ablations, and (iii) comparisons to stronger hyperbolic encoders.

## Prerequisites

Before running these scripts, ensure:

1. Environment is set up (see main [README.md](../README.md))
2. Basic experiments completed (see [exp_basic/README_basic.md](../exp_basic/README_basic.md))

---

## Experiment 1: Realistic Q-E Baseline (BioEncoder + Synonym Indexing)

**Goal:** Replace the degenerate all-zero Q-E setting by (a) indexing at least one synonym per entity, and (b) using a biomedical sentence encoder (S-BioBERT).

### Run Commands

```bash
# HPO dataset
bash exp_ext2/run_qe_high_baseline.sh hpo 5000 0 \
  pritamdeka/S-BioBERT-snli-multinli-stsb cuda

# DO dataset
bash exp_ext2/run_qe_high_baseline.sh do 5000 0 \
  pritamdeka/S-BioBERT-snli-multinli-stsb cuda
```

---

## Experiment 2: Adapter Expressivity Ablation (Linear vs 2-layer MLP)

**Goal:** Test whether a non-linear adapter improves retrieval quality without hurting indexability.

### Run Commands

```bash
# HPO dataset
bash exp_ext2/run_adapter_ablation.sh hpo 5000 0 cuda

# DO dataset
bash exp_ext2/run_adapter_ablation.sh do 5000 0 cuda
```

**Output:**
- `data/processed/<dataset>/<subset>_seed<seed>/results/adapter_ablation.csv`

---

## Experiment 3: Stronger Hyperbolic Encoder Baseline (Tangent-space HGCN)

**Goal:** Compare the default Lorentz KG embedding to a lightweight tangent-space HGCN encoder under the same tangent-index + reranking pipeline.

### Run Commands

```bash
# HPO dataset
bash exp_ext2/run_hyper_encoder_compare.sh hpo 5000 0 cuda

# DO dataset
bash exp_ext2/run_hyper_encoder_compare.sh do 5000 0 cuda

# Generate paper artifacts
python scripts/11_make_paper_artifacts.py \
  --datasets hpo do --subset_size 5000 --seed 0 \
  --data_dir data/processed --out_dir paper_artifacts
```

**Output:**
- `data/processed/<dataset>/<subset>_seed<seed>/results/hyper_encoder_compare.csv`
- `data/processed/<dataset>/<subset>_seed<seed>/u_hgcn.npy` (trained embeddings)
- `data/processed/<dataset>/<subset>_seed<seed>/indexes/index_hgcn.bin` (HNSW index)

---

## Notes

- These scripts assume you run them from the repository root.
- Some settings overwrite existing `emb_nodes.npy` / `emb_queries_*.npy` if you change the encoder or node text mode. If you want to keep multiple runs side-by-side, copy the target `data/processed/<dataset>/<subset>_seed<seed>/` directory before rerunning.
