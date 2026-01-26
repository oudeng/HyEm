#!/usr/bin/env python
"""
Safety Valve Analysis (M2)

Quantify how well soft mixing preserves Q-E performance while enabling Q-H benefits.
This directly addresses the reviewer concern that "Q-E evaluation is uninformative".

Key metrics:
- Q-E Retention Rate: HyEm_soft_MRR / Euclidean_text_MRR
- Q-H Improvement: (HyEm_soft - baseline) / baseline
- Trade-off Ratio: Q-H_gain / Q-E_loss
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_safety_valve_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute safety valve metrics from summary results."""
    
    # Extract key methods
    methods = summary_df.set_index('method')
    
    euclid_text = methods.loc['euclid_text']
    hyem_no_gate = methods.loc['hyem_no_gate']
    hyem_soft = methods.loc['hyem_soft']
    hyem_hard = methods.loc['hyem_hard']
    
    rows = []
    
    # Q-E retention analysis
    qe_baseline_mrr = euclid_text['QE_mrr']
    for method_name, method_row in [('hyem_no_gate', hyem_no_gate), 
                                     ('hyem_hard', hyem_hard),
                                     ('hyem_soft', hyem_soft)]:
        qe_mrr = method_row['QE_mrr']
        qe_retention = qe_mrr / qe_baseline_mrr if qe_baseline_mrr > 0 else 0
        
        # Q-H improvement over pure structure baselines
        qh_baseline = methods.loc['euclid_kg']['QH_parent_hits10']
        qh_score = method_row['QH_parent_hits10']
        qh_improvement = (qh_score - qh_baseline) / qh_baseline if qh_baseline > 0 else 0
        
        # Trade-off ratio: how much Q-H gain per unit Q-E loss
        qe_loss = 1 - qe_retention
        trade_off = qh_improvement / qe_loss if qe_loss > 0.001 else float('inf')
        
        rows.append({
            'method': method_name,
            'QE_MRR': qe_mrr,
            'QE_retention_rate': qe_retention,
            'QH_parent_hits10': qh_score,
            'QH_improvement_over_kg': qh_improvement,
            'trade_off_ratio': trade_off if trade_off != float('inf') else 999.9,
        })
    
    return pd.DataFrame(rows)


def plot_safety_valve(analysis_df: pd.DataFrame, dataset: str, out_path: Path):
    """Create visualization of safety valve effect."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    methods = analysis_df['method'].tolist()
    x = np.arange(len(methods))
    
    # Plot 1: Q-E Retention Rate
    ax1 = axes[0]
    bars1 = ax1.bar(x, analysis_df['QE_retention_rate'], color=['#ff7f0e', '#2ca02c', '#1f77b4'])
    ax1.axhline(y=1.0, color='gray', linestyle='--', label='Euclidean baseline')
    ax1.axhline(y=0.9, color='red', linestyle=':', alpha=0.7, label='90% threshold')
    ax1.set_ylabel('Q-E Retention Rate')
    ax1.set_title('Q-E Preservation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['No Gate', 'Hard Route', 'Soft Mix'], rotation=15)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=8)
    
    # Plot 2: Q-H Improvement
    ax2 = axes[1]
    bars2 = ax2.bar(x, analysis_df['QH_parent_hits10'], color=['#ff7f0e', '#2ca02c', '#1f77b4'])
    ax2.set_ylabel('Q-H Parent Hits@10')
    ax2.set_title('Q-H Performance')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['No Gate', 'Hard Route', 'Soft Mix'], rotation=15)
    
    # Plot 3: Trade-off visualization
    ax3 = axes[2]
    qe_loss = 1 - analysis_df['QE_retention_rate']
    qh_gain = analysis_df['QH_parent_hits10'] - analysis_df['QH_parent_hits10'].min()
    
    for i, (method, loss, gain) in enumerate(zip(methods, qe_loss, qh_gain)):
        color = ['#ff7f0e', '#2ca02c', '#1f77b4'][i]
        label = ['No Gate', 'Hard Route', 'Soft Mix'][i]
        ax3.scatter(loss, gain, s=200, c=color, label=label, marker='o', edgecolors='black')
    
    ax3.set_xlabel('Q-E Loss (1 - retention)')
    ax3.set_ylabel('Q-H Gain (relative)')
    ax3.set_title('Trade-off: Q-E Loss vs Q-H Gain')
    ax3.legend(fontsize=8)
    
    # Add annotation for soft mix
    soft_idx = methods.index('hyem_soft')
    ax3.annotate('Soft Mix\n(best trade-off)', 
                xy=(qe_loss.iloc[soft_idx], qh_gain.iloc[soft_idx]),
                xytext=(qe_loss.iloc[soft_idx] + 0.05, qh_gain.iloc[soft_idx] + 0.02),
                fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))
    
    fig.suptitle(f'Safety Valve Analysis: {dataset.upper()}', fontsize=12)
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
    
    # Compute safety valve metrics
    analysis_df = compute_safety_valve_metrics(summary_df)
    
    # Save results
    out_path = root / "results" / "safety_valve_analysis.csv"
    analysis_df.to_csv(out_path, index=False)
    print(f"\n=== Safety Valve Analysis ({args.dataset.upper()}) ===")
    print(analysis_df.to_string(index=False))
    
    # Create visualization
    fig_dir = root / "analysis" / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_safety_valve(analysis_df, args.dataset, fig_dir / "safety_valve_comparison.pdf")
    
    # Print key findings for paper
    soft_row = analysis_df[analysis_df['method'] == 'hyem_soft'].iloc[0]
    print(f"\n=== Key Findings for Paper ===")
    print(f"Q-E Retention Rate (soft mix): {soft_row['QE_retention_rate']:.1%}")
    print(f"Q-H Parent Hits@10 (soft mix): {soft_row['QH_parent_hits10']:.3f}")
    print(f"Trade-off Ratio: {soft_row['trade_off_ratio']:.2f}x Q-H gain per unit Q-E loss")


if __name__ == "__main__":
    main()
