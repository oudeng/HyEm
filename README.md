# HyEm: Query-Adaptive Hyperbolic Retrieval for Biomedical Ontologies via Euclidean Vector Indexing

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/1142128401.svg)](https://doi.org/10.5281/zenodo.18371905)

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
@article{deng2026hyem,
  title={HyEm: Radius-Controlled Hyperbolic Retrieval with Tangent-Space Indexing for Biomedical Ontologies},
  author={Deng, Ou and Nishimura, Shoji and Ogihara, Atsushi and Jin, Qun},
  journal={[arXiv]},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
