#!/usr/bin/env python
"""
End-to-End Retrieval Performance Under Noise (Table 9)

This script tests how soft mixing vs hard routing degrades under query embedding noise.
- Adds Gaussian noise to query embeddings at different levels
- Runs retrieval with both hard routing and soft mixing
- Reports Hits@10 for Q-E, Q-H, Q-M queries
- Demonstrates soft mixing degrades gracefully while hard routing fails catastrophically
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyem.models.adapter import LinearAdapter
from hyem.models.gate import LinearGate, predict_gate
from hyem.indexing.hnsw import load_hnsw
from hyem.io import read_jsonl, load_npy
from hyem.retrieval.retriever import RetrievalConfig
from hyem.eval.run_eval import evaluate_method


def load_trained_models_and_indexes(root: Path, device: str = 'cpu'):
    """Load trained models, embeddings, and indexes."""
    
    # Load node IDs
    with open(root / "node_ids.txt", "r", encoding="utf-8") as f:
        node_ids = [line.strip() for line in f if line.strip()]
    
    # Load entity embeddings
    e_nodes = np.load(root / "emb_nodes.npy").astype(np.float32)
    u_hyem = np.load(root / "u_hyem.npy").astype(np.float32) if (root / "u_hyem.npy").exists() else None
    z_euckg = np.load(root / "z_euckg.npy").astype(np.float32) if (root / "z_euckg.npy").exists() else None
    
    # Load indexes
    idx_dir = root / "indexes"
    idx_E = load_hnsw(idx_dir / "index_text.bin", dim=e_nodes.shape[1], space="cosine", ef_search=100)
    idx_H = load_hnsw(idx_dir / "index_hyem.bin", dim=u_hyem.shape[1], space="l2", ef_search=100) if u_hyem is not None else None
    idx_Z = load_hnsw(idx_dir / "index_euckg.bin", dim=z_euckg.shape[1], space="l2", ef_search=100) if z_euckg is not None else None
    
    # Load adapters
    in_dim = e_nodes.shape[1]
    
    adapter_hyem = None
    if (root / "adapter_hyem.pt").exists() and u_hyem is not None:
        adapter_hyem = LinearAdapter(in_dim, u_hyem.shape[1]).to(device)
        adapter_hyem.load_state_dict(torch.load(root / "adapter_hyem.pt", map_location=device))
        adapter_hyem.eval()
    
    adapter_euc = None
    if (root / "adapter_euckg.pt").exists() and z_euckg is not None:
        adapter_euc = LinearAdapter(in_dim, z_euckg.shape[1]).to(device)
        adapter_euc.load_state_dict(torch.load(root / "adapter_euckg.pt", map_location=device))
        adapter_euc.eval()
    
    # Load gate
    gate = None
    if (root / "gate_linear.pt").exists():
        gate = LinearGate(in_dim=in_dim).to(device)
        gate.load_state_dict(torch.load(root / "gate_linear.pt", map_location=device))
        gate.eval()
    
    return {
        'node_ids': node_ids,
        'e_nodes': e_nodes,
        'u_hyem': u_hyem,
        'z_euckg': z_euckg,
        'idx_E': idx_E,
        'idx_H': idx_H,
        'idx_Z': idx_Z,
        'adapter_hyem': adapter_hyem,
        'adapter_euc': adapter_euc,
        'gate': gate,
        'device': device,
    }


def load_test_queries(root: Path) -> List[dict]:
    """Load test queries with embeddings attached."""
    items = read_jsonl(root / "queries_test.jsonl")
    X = load_npy(root / "emb_queries_test.npy")
    with open(root / "query_ids_test.txt", "r", encoding="utf-8") as f:
        qids = [line.strip() for line in f if line.strip()]
    
    assert len(items) == X.shape[0] == len(qids)
    
    for it, qid, x in zip(items, qids, X):
        it["qid"] = qid
        it["e_q"] = x
    
    return items


def add_noise_to_queries(
    queries: List[dict],
    noise_level: float,
    seed: int = 42
) -> List[dict]:
    """Add Gaussian noise to query embeddings."""
    
    np.random.seed(seed)
    noisy_queries = []
    
    for q in queries:
        # Add Gaussian noise
        emb = q['e_q'].copy()
        noise = np.random.randn(*emb.shape).astype(np.float32) * noise_level
        noisy_emb = emb + noise
        
        # Renormalize (sentence embeddings are typically normalized)
        noisy_emb = noisy_emb / (np.linalg.norm(noisy_emb) + 1e-8)
        
        # Create new query with noisy embedding
        q_noisy = q.copy()
        q_noisy['e_q'] = noisy_emb
        noisy_queries.append(q_noisy)
    
    return noisy_queries


def compute_gate_scores_from_queries(queries: List[dict], gate, device: str = 'cpu') -> Dict[str, float]:
    """Compute gate scores for queries."""
    
    if gate is None:
        # Default to hierarchy (α=1) if no gate
        return {q['qid']: 1.0 for q in queries}
    
    X = np.stack([q['e_q'] for q in queries], axis=0).astype(np.float32)
    scores = predict_gate(gate, X, device=device)
    
    gate_scores = {q['qid']: float(s) for q, s in zip(queries, scores)}
    return gate_scores


def compute_gate_accuracy(queries: List[dict], gate_scores: Dict[str, float]) -> float:
    """Compute gate classification accuracy."""
    
    preds = []
    labels = []
    
    for q in queries:
        if q['type'] not in ('QE', 'QH'):
            continue
        
        alpha = gate_scores.get(q['qid'], 1.0)
        pred = 1 if alpha > 0.5 else 0  # 1=QH, 0=QE
        label = 1 if q['type'] == 'QH' else 0
        
        preds.append(pred)
        labels.append(label)
    
    if not labels:
        return 0.0
    
    return float(np.mean(np.array(preds) == np.array(labels)))


def evaluate_retrieval_under_noise(
    queries: List[dict],
    models: dict,
    noise_levels: List[float],
    cfg: RetrievalConfig,
    seed: int = 42
) -> pd.DataFrame:
    """Evaluate retrieval performance under different noise levels."""
    
    results = []
    
    for noise_level in noise_levels:
        print(f"\n=== Testing noise level: {noise_level:.2f} ===")
        
        # Add noise to query embeddings
        if noise_level == 0:
            noisy_queries = queries
        else:
            noisy_queries = add_noise_to_queries(queries, noise_level, seed)
        
        # Compute gate scores on noisy embeddings
        gate_scores = compute_gate_scores_from_queries(noisy_queries, models['gate'], models['device'])
        
        # Compute gate accuracy
        gate_acc = compute_gate_accuracy(noisy_queries, gate_scores)
        
        # Evaluate hard routing
        print("  Evaluating: hard routing")
        _, metrics_hard = evaluate_method(
            queries=noisy_queries,
            method_name="hyem_hard",
            node_ids=models['node_ids'],
            e_nodes=models['e_nodes'],
            u_nodes=models['u_hyem'],
            z_nodes=models['z_euckg'],
            idx_E=models['idx_E'],
            idx_H=models['idx_H'],
            idx_Z=models['idx_Z'],
            adapter_hyp=models['adapter_hyem'],
            adapter_euc=models['adapter_euc'],
            gate_scores=gate_scores,
            cfg=cfg,
        )
        
        # Evaluate soft mixing
        print("  Evaluating: soft mixing")
        _, metrics_soft = evaluate_method(
            queries=noisy_queries,
            method_name="hyem_soft",
            node_ids=models['node_ids'],
            e_nodes=models['e_nodes'],
            u_nodes=models['u_hyem'],
            z_nodes=models['z_euckg'],
            idx_E=models['idx_E'],
            idx_H=models['idx_H'],
            idx_Z=models['idx_Z'],
            adapter_hyp=models['adapter_hyem'],
            adapter_euc=models['adapter_euc'],
            gate_scores=gate_scores,
            cfg=cfg,
        )
        
        # Record results
        results.append({
            'noise_level': noise_level,
            'gate_acc': gate_acc,
            'QE_hits10_hard': metrics_hard.get('QE_hits10', 0),
            'QE_hits10_soft': metrics_soft.get('QE_hits10', 0),
            'QH_parent_hits10_hard': metrics_hard.get('QH_parent_hits10', 0),
            'QH_parent_hits10_soft': metrics_soft.get('QH_parent_hits10', 0),
            'QM_hits10_hard': metrics_hard.get('QM_hits10', 0),
            'QM_hits10_soft': metrics_soft.get('QM_hits10', 0),
        })
        
        print(f"  Gate Accuracy: {gate_acc:.3f}")
        print(f"  Q-E Hits@10: Hard={metrics_hard.get('QE_hits10', 0):.3f}, Soft={metrics_soft.get('QE_hits10', 0):.3f}")
        print(f"  Q-H Parent Hits@10: Hard={metrics_hard.get('QH_parent_hits10', 0):.3f}, Soft={metrics_soft.get('QH_parent_hits10', 0):.3f}")
        print(f"  Q-M Hits@10: Hard={metrics_hard.get('QM_hits10', 0):.3f}, Soft={metrics_soft.get('QM_hits10', 0):.3f}")
    
    return pd.DataFrame(results)


def format_ext4_latex(df: pd.DataFrame, dataset: str) -> str:
    """Format results as LaTeX table matching ext4 format."""
    
    latex = f"% ext4 data for {dataset.upper()}\n"
    latex += "\\begin{tabular}{c|c|cc|cc|cc}\n"
    latex += "\\hline\n"
    latex += "Noise $\\sigma$ & Gate Acc. & \\multicolumn{2}{c|}{Q-E Hits@10} & \\multicolumn{2}{c|}{Q-H Hits@10} & \\multicolumn{2}{c}{Q-M Hits@10} \\\\\n"
    latex += " & & Hard & Soft & Hard & Soft & Hard & Soft \\\\\n"
    latex += "\\hline\n"
    
    for _, row in df.iterrows():
        noise = row['noise_level']
        gate_acc = row['gate_acc']
        
        label = {0.0: "0.00 (clean)", 0.1: "0.10 (typos)", 0.2: "0.20 (paraphrase)", 0.3: "0.30 (ambiguous)"}.get(noise, f"{noise:.2f}")
        
        latex += f"{label} & {gate_acc:.1%} & "
        latex += f"{row['QE_hits10_hard']:.3f} & {row['QE_hits10_soft']:.3f} & "
        latex += f"{row['QH_parent_hits10_hard']:.3f} & {row['QH_parent_hits10_soft']:.3f} & "
        latex += f"{row['QM_hits10_hard']:.3f} & {row['QM_hits10_soft']:.3f} \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    
    return latex


def main():
    ap = argparse.ArgumentParser(
        description="Test end-to-end retrieval performance under embedding noise"
    )
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--noise_levels", type=str, default="0.0,0.1,0.2,0.3",
                   help="Comma-separated noise levels to test")
    ap.add_argument("--L_h", type=int, default=50,
                   help="Number of candidates to retrieve from hyperbolic index")
    ap.add_argument("--L_e", type=int, default=50,
                   help="Number of candidates to retrieve from Euclidean index")
    args = ap.parse_args()
    
    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    
    print(f"\n{'='*60}")
    print(f"End-to-End Retrieval Under Noise (Table of ext4)")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Subset: {args.subset_size} nodes, seed={args.seed}")
    
    # Load models and indexes
    print("\n[1/4] Loading trained models, embeddings, and indexes...")
    try:
        models = load_trained_models_and_indexes(root, args.device)
        print(f"   ✓ Loaded {len(models['node_ids'])} entities")
        print(f"   ✓ Entity embedding dim: {models['e_nodes'].shape[1]}")
        print(f"   ✓ Hyperbolic embedding dim: {models['u_hyem'].shape[1] if models['u_hyem'] is not None else 'N/A'}")
        if models['gate'] is None:
            print("   ⚠ No gate found. Will use default hierarchy routing (α=1.0)")
        else:
            print(f"   ✓ Gate loaded")
    except Exception as e:
        print(f"   ✗ Error loading models: {e}")
        print("   Please run the baseline pipeline first (exp_basic/run_basic_pipeline.sh)")
        return
    
    # Load test queries
    print("\n[2/4] Loading test queries...")
    try:
        queries = load_test_queries(root)
        print(f"   ✓ Loaded {len(queries)} queries")
        print(f"   ✓ Q-E: {sum(1 for q in queries if q['type'] == 'QE')}")
        print(f"   ✓ Q-H: {sum(1 for q in queries if q['type'] == 'QH')}")
        print(f"   ✓ Q-M: {sum(1 for q in queries if q['type'] == 'QM')}")
    except Exception as e:
        print(f"   ✗ Error loading queries: {e}")
        return
    
    # Setup retrieval config
    cfg = RetrievalConfig(
        k=10,
        L_h=args.L_h,
        L_e=args.L_e,
        anc_k=20,
        device=args.device,
    )
    
    # Parse noise levels
    noise_levels = [float(x) for x in args.noise_levels.split(',')]
    print(f"\n[3/4] Testing noise levels: {noise_levels}")
    print(f"   (σ=0.0: clean, σ=0.1: typos, σ=0.2: paraphrase, σ=0.3: ambiguous)")
    
    # Run evaluation
    print("\n[4/4] Running retrieval evaluation...")
    print("   This may take a few minutes depending on dataset size...")
    
    results_df = evaluate_retrieval_under_noise(
        queries=queries,
        models=models,
        noise_levels=noise_levels,
        cfg=cfg,
        seed=args.seed,
    )
    
    # Save results
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "ext4_noise_retrieval.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n{'='*60}")
    print(f"✓ Saved results: {csv_path}")
    
    # Generate LaTeX table
    latex_path = out_dir / "ext4_noise_retrieval.tex"
    latex = format_ext4_latex(results_df, args.dataset)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"✓ Saved LaTeX table: {latex_path}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print(f"Table of ext4 Results ({args.dataset.upper()})")
    print(f"{'='*60}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    
    # Key findings
    print(f"\n{'='*60}")
    print("Key Findings")
    print(f"{'='*60}")
    
    baseline = results_df[results_df['noise_level'] == 0.0].iloc[0]
    print(f"\nClean queries (σ=0.0):")
    print(f"  Gate Accuracy: {baseline['gate_acc']:.1%}")
    print(f"  Q-E Hits@10: Hard={baseline['QE_hits10_hard']:.3f}, Soft={baseline['QE_hits10_soft']:.3f}")
    print(f"  Q-H Parent Hits@10: Hard={baseline['QH_parent_hits10_hard']:.3f}, Soft={baseline['QH_parent_hits10_soft']:.3f}")
    
    for noise_level in [0.1, 0.2, 0.3]:
        if noise_level not in noise_levels:
            continue
        
        noisy = results_df[results_df['noise_level'] == noise_level].iloc[0]
        label = {0.1: "typos", 0.2: "paraphrase", 0.3: "ambiguous"}.get(noise_level, "")
        
        print(f"\nNoise σ={noise_level:.1f} ({label}):")
        gate_deg = baseline['gate_acc'] - noisy['gate_acc']
        print(f"  Gate Accuracy: {noisy['gate_acc']:.1%} (degradation: {gate_deg:.1%})")
        
        # Q-E retention (most important metric for Table 9)
        hard_retention = noisy['QE_hits10_hard'] / max(baseline['QE_hits10_hard'], 0.001)
        soft_retention = noisy['QE_hits10_soft'] / max(baseline['QE_hits10_soft'], 0.001)
        print(f"  Q-E Retention: Hard={hard_retention:.1%}, Soft={soft_retention:.1%}")
        
        # Interpretation
        if soft_retention > 0.9 and hard_retention < 0.7:
            print(f"  ✓ Soft mixing maintains stability (>{90:.0f}%) while hard routing degrades (<{70:.0f}%)")
        elif abs(soft_retention - hard_retention) < 0.05:
            print(f"  → Similar performance at this noise level")
        else:
            if soft_retention > hard_retention:
                print(f"  ✓ Soft mixing more stable (+{(soft_retention - hard_retention)*100:.1f}% retention)")
            else:
                print(f"  ⚠ Hard routing more stable at this noise level")
    
    print(f"\n{'='*60}")
    print("Done! Results saved to:")
    print(f"  - {csv_path}")
    print(f"  - {latex_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
