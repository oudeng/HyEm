# HyEm: Query-Adaptive Hyperbolic Retrieval for Biomedical Ontologies via Euclidean Vector Indexing

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the **reproducibility package** for the submitted paper.

---

## Overview

HyEm is a **query-adaptive hyperbolic retrieval layer** for biomedical ontology grounding in RAG systems. Biomedical ontologies (HPO, DO, MeSH) are dominated by deep *is-a* hierarchies—a natural fit for hyperbolic geometry—yet production retrieval stacks rely on Euclidean vector databases.

HyEm bridges this gap by solving two practical frictions:

1. **Deployment friction**: Hyperbolic nearest-neighbor search requires specialized indexing primitives unavailable in standard vector databases (FAISS, HNSW, etc.)

2. **Query heterogeneity**: Real query streams mix hierarchy-navigation tasks (*"subtypes of cardiomyopathy"*) with entity-linking tasks (*"what does cardiomyopathy mean"*). Forcing hyperbolic distance on all queries risks regressing against strong Euclidean baselines.

### Key Contributions

- **Indexable hyperbolic retrieval**: Radius-controlled hyperbolic embeddings deployable via standard Euclidean ANN through tangent-space indexing—no custom vector database needed

- **Query-adaptive safety valve**: Lightweight gating mechanism softly mixes Euclidean semantic similarity with hyperbolic hierarchy distance, preserving **94–98%** of entity-centric baseline performance while enabling substantial gains on taxonomy-navigation queries

- **Theory-guided engineering**: Bi-Lipschitz analysis translates a radius budget into explicit guidance for (i) tangent-space indexability and ANN oversampling, and (ii) hierarchical representational capacity

- **Rigorous evaluation protocol**: Stratified query taxonomy isolates hierarchy-navigation vs. entity-linking performance, addressing the "hyperbolic is not always better" critique with controlled ablations

**Bottom line**: HyEm makes hyperbolic geometry deployable in production RAG stacks without modifying retrieval infrastructure, while remaining robust under mixed query intents.

### Method Overview

<img src="https://github.com/oudeng/HyEm/blob/main/Fig/Fig1_HyEm_pipeline.png" alt="Figure 1: HyEm pipeline architecture" style="width:90%;" />

**Offline training and deployment-friendly retrieval in HyEm**
**Offline (left)**: We train hyperbolic ontology embeddings $\{\mathbf{x}_v\}$ under an explicit radius budget $R$ (Section~\ref{sec:entity_embed}), and store only origin log-mapped vectors $\mathbf{u}_v=\log_{\mathbf{0}}(\mathbf{x}_v)$ in a standard Euclidean ANN index.
In addition, we build a Euclidean text ANN index over entity texts.
**Online (right)**: Given a query $q$, we compute its Euclidean embedding $\mathbf{e}_q$ and map it into hyperbolic space via a compact adapter to obtain $\mathbf{x}_q$. We then retrieve candidates from both indexes and pool them by union ($C = C_H \cup C_E$).
This candidate pooling acts as a robustness **safety net**: it preserves the strong recall of Euclidean baselines on entity-centric queries while still enabling hierarchy-aware reranking in hyperbolic space.
Finally, we rerank the pooled candidates by combining hyperbolic hierarchy distance and Euclidean semantic similarity with a query-adaptive soft mixing weight $\alpha(q)$. Algorithm 1 summarizes the indexing and query-time steps. Note that during query processing (right), the adapter outputs the tangent vector $u_q$ directly without applying $\exp_0$; the exponential map to hyperbolic space is only computed during reranking.

---

## Quick Start

### 1. Environment Setup

**Tested with:** Python 3.9–3.11, PyTorch 2.0+

```bash
# Option A: conda (recommended)
conda create -n hyem python=3.10 -y
conda activate hyem
pip install -r requirements.txt
pip install -e .

# Option B: venv
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### 2. Download Data

```bash
python scripts/00_download_data.py --datasets hpo do
# Optional (may require manual download):
# python scripts/00_download_data.py --datasets mesh
```

### 3. Run Experiments

See the **Experiments** section below for detailed instructions.

---

## Repository Structure

```
HyEm/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Package configuration
│
├── scripts/                    # Numbered pipeline scripts (00-11)
│   ├── 00_download_data.py     # Download HPO/DO/MeSH ontologies
│   ├── 01_build_subset.py      # Build size-controlled subsets
│   ├── 02_build_queries.py     # Generate Q-E/Q-H/Q-M benchmarks
│   ├── 03_encode_text.py       # Encode texts with SentenceTransformers
│   ├── 04_train_embeddings.py  # Train hyperbolic & Euclidean embeddings
│   ├── 05_train_adapters.py    # Train query adapters
│   ├── 06_train_gate.py        # Train query-adaptive gate
│   ├── 07_build_indexes.py     # Build HNSW indexes
│   ├── 08_eval_retrieval.py    # Evaluate retrieval (main results)
│   ├── 09_indexability_test.py # Tangent-space recall stress test
│   ├── 10_efficiency_benchmark.py # Latency & memory benchmarks
│   └── 11_make_paper_artifacts.py # Export LaTeX tables & figures
│
├── exp_basic/                  # Basic experiments (main paper)
│   ├── README_basic.md         # Instructions for basic experiments
│   ├── run_basic_pipeline.sh   # Full pipeline (single dataset)
│   ├── run_basic_analysis.sh   # Indexability & efficiency analysis
│   └── run_paper_artifacts.sh  # Generate LaTeX tables & figures
│
├── exp_ext1/                   # Extended experiments 1
│   ├── README_ext1.md          # Instructions for extended experiments
│   ├── run_qe_biomed_encoder.sh    # Biomedical encoder experiment
│   ├── run_scale_20k.sh        # Scale-up to 20k nodes
│   ├── run_radius_sweep.sh     # Radius-budget sweep
│   └── plot_indexability_with_theory.py # Theory-guided plotting
│
├── exp_ext2/                   # Extended experiments 2 (reviewer-motivated)
│   ├── README_ext2.md          # Instructions for extended experiments
│   ├── run_qe_high_baseline.sh # Biomedical encoder + synonym indexing for Q-E
│   ├── run_adapter_ablation.sh # Linear vs 2-layer MLP adapter
│   └── run_hyper_encoder_compare.sh # Tangent-space HGCN baseline
│
├── exp_ext3/                   # Extended experiments 3 (depth, theory, robustness)
│   ├── README_ext3.md          # Instructions for depth and robustness analysis
│   ├── analyze_safety_valve.py # Safety valve analysis for Q-E preservation
│   ├── depth_stratified_analysis.py # Depth-stratified performance analysis
│   ├── theoretical_scaling.py  # Theoretical κ(R) vs depth plots
│   ├── gate_robustness_test.py # Gate calibration under noise
│   ├── candidate_pooling_ablation.py # C_H ∪ C_E pooling contribution
│   ├── make_figures.py         # Aggregate analysis figures and tables
│   └── run_all_ext3.sh         # Run all exp_ext3 analyses
│
├── exp_ext4/                   # Extended experiments 4 (noise robustness)
│   ├── README_ext4.md          # Instructions for noise robustness experiments
│   ├── noise_retrieval_test.py # End-to-end retrieval under noise
│   ├── run_noise_retrieval.sh  # Run noise test for single dataset
│   ├── aggregate_ext4_results.py # Aggregate results for Table 9
│   └── run_all_ext4.sh         # Run complete exp_ext4 pipeline
│
├── src/hyem/                   # Core library
│   ├── ontology/               # OBO parsing & graph utilities
│   ├── text/                   # Text embedding (SentenceTransformers)
│   ├── models/                 # Hyperbolic embeddings, adapters, gates
│   ├── indexing/               # HNSW wrapper (hnswlib)
│   ├── retrieval/              # Retrieval methods & soft mixing
│   └── eval/                   # Metrics & evaluation tasks
│
├── data/                       # Data directory (created during experiments)
│   ├── raw/<dataset>/          # Downloaded .obo files
│   └── processed/<dataset>/<subset>_seed<seed>/
│       ├── nodes.jsonl, edges.csv  # Graph structure
│       ├── queries_*.jsonl     # Benchmark queries
│       ├── emb_*.npy           # Text embeddings
│       ├── u_hyem.npy          # Hyperbolic graph embeddings
│       ├── adapter_*.pt        # Trained adapters
│       ├── indexes/            # HNSW index files
│       ├── results/            # Evaluation results
│       └── analysis/           # Indexability & efficiency analysis
│
└── paper_artifacts/            # Generated LaTeX snippets & figures
```

---

## Experiments

### Basic Experiments (Main Paper)

The basic experiments reproduce the main results on HPO-5k and DO-5k.

**Quick Start:**
```bash
# Run complete pipeline for HPO
bash exp_basic/run_basic_pipeline.sh hpo 5000 0 cuda

# Run analysis benchmarks
bash exp_basic/run_basic_analysis.sh hpo 5000 0 cuda

# Generate paper artifacts
bash exp_basic/run_paper_artifacts.sh "hpo do" 5000 0
```

📖 **Full instructions:** [exp_basic/README_basic.md](exp_basic/README_basic.md)

---

### Extended Experiments 1

Extended experiments address reviewer concerns and provide additional evidence:

| Experiment | Purpose | Command |
|------------|---------|---------|
| **Biomedical Encoder** | Non-trivial Q-E baseline | `bash exp_ext1/run_qe_biomed_encoder.sh ...` |
| **Scale-up (20k)** | Evidence beyond 5k nodes | `bash exp_ext1/run_scale_20k.sh ...` |
| **Radius Sweep** | Validate theory bounds | `bash exp_ext1/run_radius_sweep.sh ...` |
| **Theory Plot** | Visualize κ(R) bound | `python exp_ext1/plot_indexability_with_theory.py ...` |

📖 **Full instructions:** [exp_ext1/README_ext1.md](exp_ext1/README_ext1.md)

---

### Extended Experiments 2

Additional extended experiments address reviewer requests for stronger baselines and ablations:

| Experiment | Purpose | Command |
|------------|---------|---------|
| **Realistic Q-E Baseline** | BioEncoder + synonym indexing | `bash exp_ext2/run_qe_high_baseline.sh ...` |
| **Adapter Ablation** | Linear vs 2-layer MLP | `bash exp_ext2/run_adapter_ablation.sh ...` |
| **Hyperbolic Encoder** | Tangent-space HGCN baseline | `bash exp_ext2/run_hyper_encoder_compare.sh ...` |

📖 **Full instructions:** [exp_ext2/README_ext2.md](exp_ext2/README_ext2.md)

---

### Extended Experiments 3

Depth-stratified analysis, theoretical scaling, and robustness validation:

| Experiment | Purpose | Command |
|------------|---------|---------|
| **Safety Valve Analysis** | Quantify soft mixing's Q-E preservation | `python exp_ext3/analyze_safety_valve.py ...` |
| **Depth-Stratified Analysis** | Stratify results by depth buckets | `python exp_ext3/depth_stratified_analysis.py ...` |
| **Theoretical Scaling** |  Plot κ(R) vs depth and safe regime | `python exp_ext3/theoretical_scaling.py ...` |
| **Gate Robustness** | Test gate calibration on perturbed queries | `python exp_ext3/gate_robustness_test.py ...` |
| **Candidate Pooling Ablation** | Quantify C_H ∪ C_E pooling contribution | `python exp_ext3/candidate_pooling_ablation.py ...` |

**Quick Start:**
```bash
# Run all exp_ext3 analyses
bash exp_ext3/run_all_ext3.sh hpo 5000 0
bash exp_ext3/run_all_ext3.sh do 5000 0

# Generate combined analysis figures
python exp_ext3/make_figures.py --data_dir data/processed --out_dir paper_artifacts/ext3
```

📖 **Full instructions:** [exp_ext3/README_ext3.md](exp_ext3/README_ext3.md)

---

### Extended Experiments 4

End-to-end retrieval robustness under query embedding noise:

**Purpose:** Demonstrates that soft mixing maintains >90% retrieval performance even when gate accuracy drops to 70%, while hard routing suffers catastrophic failures on misrouted queries.

| Noise Level | Simulates | Gate Accuracy |
|-------------|-----------|---------------|
| σ=0.0 | Clean baseline | 100% |
| σ=0.1 | Minor typos | ~91% |
| σ=0.2 | Moderate paraphrase | ~78% |
| σ=0.3 | Highly uncertain | ~70% |

**Quick Start:**
```bash
# Run complete pipeline for both datasets
bash exp_ext4/run_all_ext4.sh 5000 0

# Or run individually
bash exp_ext4/run_noise_retrieval.sh hpo 5000 0
bash exp_ext4/run_noise_retrieval.sh do 5000 0

# Aggregate results and generate Table 9
python exp_ext4/aggregate_ext4_results.py \
  --data_dir data/processed \
  --out_dir paper_artifacts/ext4 \
  --subset_size 5000 --seed 0
```

📖 **Full instructions:** [exp_ext4/README_ext4.md](exp_ext4/README_ext4.md)

---

## Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| **HPO** | Human Phenotype Ontology | [hpo.jax.org](https://hpo.jax.org/) |
| **DO** | Disease Ontology | [disease-ontology.org](https://disease-ontology.org/) |
| **MeSH** | Medical Subject Headings (optional) | [nlm.nih.gov/mesh](https://www.nlm.nih.gov/mesh/) |

> **Note:** SNOMED CT and ICD are not included due to licensing constraints.

---

## Query Taxonomy

HyEm evaluates on three query families:

| Type | Description | Example | Primary Signal |
|------|-------------|---------|----------------|
| **Q-E** | Entity-centric | "What is dilated cardiomyopathy?" | Euclidean similarity |
| **Q-H** | Taxonomy-navigation | "What are subtypes of cardiomyopathy?" | Hyperbolic distance |
| **Q-M** | Mixed-intent | "Diseases similar to X at same specificity" | Soft mixing |

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dim` | 32 | Embedding dimension |
| `--radius_budget` | 3.0 | Maximum radius in tangent space (controls indexability) |
| `--L_h` | 200 | Hyperbolic candidate oversampling factor |
| `--L_e` | 200 | Euclidean candidate oversampling factor |
| `--epochs` | 10 (graph), 5 (adapter/gate) | Training epochs |

**Guidance from theory:**
- Larger `radius_budget` → better hierarchy representation, worse tangent-space approximation
- Distortion factor κ(R) = sinh(R)/R bounds the approximation error

---

## Output Files

### Main Results
- `results/summary.csv` — Aggregated metrics (Hits@k, MRR, F1) per method and query type
- `results/per_query_*.csv` — Per-query results for significance tests

### Analysis
- `analysis/indexability_recall_curve.csv` — Recall@k vs. oversampling L
- `analysis/efficiency.csv` — Query latency and index size

### Paper Artifacts
- `paper_artifacts/<dataset>/rows_*.tex` — LaTeX table rows

---

## Troubleshooting

### Common Issues

1. **SentenceTransformers download fails**
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. **CUDA out of memory**
   ```bash
   # Use CPU instead
   bash exp_basic/run_basic_pipeline.sh hpo 5000 0 cpu
   ```

3. **MeSH download fails**
   - Download manually from [NLM](https://www.nlm.nih.gov/mesh/filelist.html)
   - Place at `data/raw/mesh/mesh.obo`

---

## Citation

If you use this code, please cite:

```bibtex
@article{Deng2026hyem,
  title={HyEm: Query-Adaptive Hyperbolic Retrieval for Biomedical Ontologies via Euclidean Vector Indexing},
  author={Deng, Ou and Nishimura, Shoji and Ogihara, Atsushi and Jin, Qun},
  journal={[arXiv]},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
