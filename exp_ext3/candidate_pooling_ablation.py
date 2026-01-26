#!/usr/bin/env python
"""
Candidate Pooling Ablation (m3)

Quantify the contribution of using C_H ∪ C_E pooling vs C_H only.

This addresses minor comment about the contribution of candidate pooling
to the final retrieval performance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def extract_pooling_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and compare pooling strategies from existing results."""
    
    # hyem_no_gate uses only C_H (hyperbolic tangent-space candidates)
    # hyem_soft uses C_H ∪ C_E (pooled candidates)
    # The difference shows the contribution of candidate pooling
    
    methods = summary_df.set_index('method')
    
    rows = []
    
    # Compare no_gate (C_H only) vs soft (C_H ∪ C_E)
    hyem_no_gate = methods.loc['hyem_no_gate']
    hyem_soft = methods.loc['hyem_soft']
    euclid_text = methods.loc['euclid_text']
    
    for metric in ['QE_hits10', 'QE_mrr', 'QH_parent_hits10', 'QM_hits10']:
        no_gate_val = hyem_no_gate[metric]
        soft_val = hyem_soft[metric]
        euclid_val = euclid_text[metric]
        
        # Pooling contribution = soft - no_gate
        pooling_contribution = soft_val - no_gate_val
        
        # Also compute relative to Euclidean baseline
        euclid_gap_no_gate = euclid_val - no_gate_val
        euclid_gap_soft = euclid_val - soft_val
        
        rows.append({
            'metric': metric,
            'C_H_only': no_gate_val,
            'C_H_union_C_E': soft_val,
            'pooling_contribution': pooling_contribution,
            'pct_improvement': pooling_contribution / max(no_gate_val, 0.001) * 100,
            'euclidean_baseline': euclid_val,
            'gap_to_euclid_no_pool': euclid_gap_no_gate,
            'gap_to_euclid_with_pool': euclid_gap_soft,
        })
    
    return pd.DataFrame(rows)


def analyze_pooling_by_query_type(root: Path) -> dict:
    """Analyze how pooling helps different query types."""
    
    per_query_no_gate = pd.read_csv(root / "results" / "per_query_hyem_no_gate.csv")
    per_query_soft = pd.read_csv(root / "results" / "per_query_hyem_soft.csv")
    
    analysis = {}
    
    for qtype in ['QE', 'QH', 'QM']:
        no_gate_q = per_query_no_gate[per_query_no_gate['type'] == qtype]
        soft_q = per_query_soft[per_query_soft['type'] == qtype]
        
        if len(no_gate_q) == 0:
            continue
        
        # Compute per-query improvement
        merged = no_gate_q.merge(soft_q, on='qid', suffixes=('_no_gate', '_soft'))
        merged['rr_improvement'] = merged['rr_soft'] - merged['rr_no_gate']
        
        # Count queries that improved, stayed same, or degraded
        n_improved = (merged['rr_improvement'] > 0.001).sum()
        n_same = (np.abs(merged['rr_improvement']) <= 0.001).sum()
        n_degraded = (merged['rr_improvement'] < -0.001).sum()
        
        analysis[qtype] = {
            'n_queries': len(merged),
            'n_improved': n_improved,
            'n_same': n_same,
            'n_degraded': n_degraded,
            'pct_improved': n_improved / len(merged) * 100,
            'mean_rr_no_gate': no_gate_q['rr'].mean(),
            'mean_rr_soft': soft_q['rr'].mean(),
            'mean_improvement': merged['rr_improvement'].mean(),
        }
    
    return analysis


def plot_pooling_contribution(pooling_df: pd.DataFrame, dataset: str, out_path: Path):
    """Visualize pooling contribution."""
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Left: Absolute values comparison
    ax1 = axes[0]
    x = np.arange(len(pooling_df))
    width = 0.35
    
    ax1.bar(x - width/2, pooling_df['C_H_only'], width, label='C_H only', color='coral')
    ax1.bar(x + width/2, pooling_df['C_H_union_C_E'], width, label='C_H ∪ C_E', color='steelblue')
    
    # Add Euclidean baseline as reference line
    for i, val in enumerate(pooling_df['euclidean_baseline']):
        ax1.hlines(val, i - 0.4, i + 0.4, colors='gray', linestyles='--', alpha=0.7)
    
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Metric')
    ax1.set_title('Candidate Pooling Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Q-E\nHits@10', 'Q-E\nMRR', 'Q-H\nHits@10', 'Q-M\nHits@10'], fontsize=9)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Contribution analysis
    ax2 = axes[1]
    colors = ['red' if x < 0 else 'green' for x in pooling_df['pooling_contribution']]
    ax2.bar(x, pooling_df['pooling_contribution'], color=colors, edgecolor='black')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    
    ax2.set_ylabel('Score Improvement')
    ax2.set_xlabel('Metric')
    ax2.set_title('Pooling Contribution (C_H∪C_E - C_H)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Q-E\nHits@10', 'Q-E\nMRR', 'Q-H\nHits@10', 'Q-M\nHits@10'], fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add percentage annotations
    for i, (contrib, pct) in enumerate(zip(pooling_df['pooling_contribution'], 
                                          pooling_df['pct_improvement'])):
        if contrib > 0:
            ax2.text(i, contrib + 0.01, f'+{pct:.0f}%', ha='center', fontsize=9)
        else:
            ax2.text(i, contrib - 0.02, f'{pct:.0f}%', ha='center', fontsize=9)
    
    fig.suptitle(f'Candidate Pooling Analysis: {dataset.upper()}', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")
    args = ap.parse_args()
    
    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    
    # Load summary results
    summary_df = pd.read_csv(root / "results" / "summary.csv")
    
    # Extract pooling comparison
    pooling_df = extract_pooling_comparison(summary_df)
    
    print(f"\n=== Candidate Pooling Analysis ({args.dataset.upper()}) ===")
    print(pooling_df.to_string(index=False))
    
    # Save results
    out_path = root / "results" / "candidate_pooling_ablation.csv"
    pooling_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    
    # Per-query type analysis
    print("\n=== Per-Query-Type Breakdown ===")
    query_analysis = analyze_pooling_by_query_type(root)
    for qtype, stats in query_analysis.items():
        print(f"\n{qtype}:")
        print(f"  Improved: {stats['n_improved']}/{stats['n_queries']} ({stats['pct_improved']:.1f}%)")
        print(f"  Mean RR improvement: {stats['mean_improvement']:.4f}")
    
    # Create visualization
    fig_dir = root / "analysis" / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_pooling_contribution(pooling_df, args.dataset, fig_dir / "candidate_pooling_analysis.pdf")
    
    # Print key findings
    qe_row = pooling_df[pooling_df['metric'] == 'QE_hits10'].iloc[0]
    qh_row = pooling_df[pooling_df['metric'] == 'QH_parent_hits10'].iloc[0]
    
    print(f"\n=== Key Findings for Paper ===")
    print(f"Q-E: Pooling contributes +{qe_row['pct_improvement']:.0f}% improvement")
    print(f"Q-H: Pooling contributes +{qh_row['pct_improvement']:.0f}% improvement")
    print(f"Main benefit: Pooling recovers Q-E candidates missed by tangent-space ANN")


if __name__ == "__main__":
    main()
