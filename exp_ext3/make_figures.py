#!/usr/bin/env python
"""
Make Figures

Aggregate all exp_ext3 analysis results and generate summary figures
for paper revision.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def collect_safety_valve_results(data_dir: Path, datasets: list, subset_size: int, seed: int) -> pd.DataFrame:
    """Collect safety valve results from all datasets."""
    all_results = []
    for dataset in datasets:
        path = data_dir / dataset / f"{subset_size}_seed{seed}" / "results" / "safety_valve_analysis.csv"
        if path.exists():
            df = pd.read_csv(path)
            df['dataset'] = dataset.upper()
            all_results.append(df)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def collect_pooling_results(data_dir: Path, datasets: list, subset_size: int, seed: int) -> pd.DataFrame:
    """Collect candidate pooling results from all datasets."""
    all_results = []
    for dataset in datasets:
        path = data_dir / dataset / f"{subset_size}_seed{seed}" / "results" / "candidate_pooling_ablation.csv"
        if path.exists():
            df = pd.read_csv(path)
            df['dataset'] = dataset.upper()
            all_results.append(df)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def collect_depth_results(data_dir: Path, datasets: list, subset_size: int, seed: int) -> pd.DataFrame:
    """Collect depth-stratified results from all datasets."""
    all_results = []
    for dataset in datasets:
        path = data_dir / dataset / f"{subset_size}_seed{seed}" / "results" / "depth_stratified_analysis.csv"
        if path.exists():
            df = pd.read_csv(path)
            df['dataset'] = dataset.upper()
            all_results.append(df)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def plot_combined_safety_valve(safety_df: pd.DataFrame, out_path: Path):
    """Create combined safety valve figure for all datasets."""
    if len(safety_df) == 0:
        print("No safety valve data available")
        return
    
    datasets = safety_df['dataset'].unique()
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4.5))
    if n_datasets == 1:
        axes = [axes]
    
    for ax, dataset in zip(axes, datasets):
        df = safety_df[safety_df['dataset'] == dataset]
        
        methods = df['method'].tolist()
        x = np.arange(len(methods))
        
        # Plot retention rate
        bars = ax.bar(x, df['QE_retention_rate'], color=['#ff7f0e', '#2ca02c', '#1f77b4'],
                     edgecolor='black', linewidth=1)
        
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='100% baseline')
        ax.axhline(y=0.9, color='red', linestyle=':', alpha=0.7, linewidth=1.5, label='90% threshold')
        
        ax.set_ylabel('Q-E Retention Rate', fontsize=11)
        ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['No Gate', 'Hard Route', 'Soft Mix'], rotation=15, fontsize=10)
        ax.set_ylim(0, 1.15)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df['QE_retention_rate'])):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.1%}',
                   ha='center', va='bottom', fontsize=9)
        
        if ax == axes[0]:
            ax.legend(loc='lower right', fontsize=9)
    
    fig.suptitle('Safety Valve Effect: Q-E Retention Rate', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_combined_pooling(pooling_df: pd.DataFrame, out_path: Path):
    """Create combined pooling contribution figure."""
    if len(pooling_df) == 0:
        print("No pooling data available")
        return
    
    datasets = pooling_df['dataset'].unique()
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 4.5))
    if n_datasets == 1:
        axes = [axes]
    
    for ax, dataset in zip(axes, datasets):
        df = pooling_df[pooling_df['dataset'] == dataset]
        
        metrics = df['metric'].tolist()
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, df['C_H_only'], width, label='C_H only', color='coral', edgecolor='black')
        ax.bar(x + width/2, df['C_H_union_C_E'], width, label='C_H ∪ C_E', color='steelblue', edgecolor='black')
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Q-E\nHits@10', 'Q-E\nMRR', 'Q-H\nHits@10', 'Q-M\nHits@10'], fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Candidate Pooling Contribution', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def generate_latex_tables(safety_df: pd.DataFrame, pooling_df: pd.DataFrame, out_dir: Path):
    """Generate LaTeX table snippets for paper."""
    
    # Safety Valve Table
    if len(safety_df) > 0:
        soft_mix = safety_df[safety_df['method'] == 'hyem_soft']
        
        latex_lines = [
            "% Safety Valve Results (Table 2 addition)",
            "\\begin{tabular}{llccc}",
            "\\toprule",
            "Dataset & Method & Q-E MRR & Q-E Retention & Q-H Hits@10 \\\\",
            "\\midrule",
        ]
        
        for _, row in soft_mix.iterrows():
            latex_lines.append(
                f"{row['dataset']} & HyEm (soft mix) & {row['QE_MRR']:.3f} & "
                f"\\textbf{{{row['QE_retention_rate']:.1%}}} & {row['QH_parent_hits10']:.3f} \\\\"
            )
        
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
        ])
        
        with open(out_dir / "table_safety_valve.tex", "w") as f:
            f.write("\n".join(latex_lines))
        print(f"Saved: {out_dir / 'table_safety_valve.tex'}")
    
    # Pooling Contribution Table
    if len(pooling_df) > 0:
        latex_lines = [
            "% Candidate Pooling Ablation",
            "\\begin{tabular}{llccc}",
            "\\toprule",
            "Dataset & Metric & C_H only & C_H $\\cup$ C_E & Improvement \\\\",
            "\\midrule",
        ]
        
        for dataset in pooling_df['dataset'].unique():
            df = pooling_df[pooling_df['dataset'] == dataset]
            for _, row in df.iterrows():
                metric_name = row['metric'].replace('_', ' ').replace('hits', 'Hits@').replace('mrr', 'MRR')
                latex_lines.append(
                    f"{dataset} & {metric_name} & {row['C_H_only']:.3f} & "
                    f"{row['C_H_union_C_E']:.3f} & +{row['pct_improvement']:.0f}\\% \\\\"
                )
            latex_lines.append("\\midrule")
        
        latex_lines[-1] = "\\bottomrule"
        latex_lines.append("\\end{tabular}")
        
        with open(out_dir / "table_pooling_ablation.tex", "w") as f:
            f.write("\n".join(latex_lines))
        print(f"Saved: {out_dir / 'table_pooling_ablation.tex'}")


def copy_individual_figures(data_dir: Path, datasets: list, subset_size: int, seed: int, out_dir: Path):
    """Copy individual analysis figures to output directory."""
    
    figure_mappings = [
        ("analysis/figs/safety_valve_comparison.pdf", "fig_safety_valve_{dataset}.pdf"),
        ("analysis/figs/candidate_pooling_analysis.pdf", "fig_pooling_{dataset}.pdf"),
        ("analysis/figs/depth_stratified_trend.pdf", "fig_depth_trend_{dataset}.pdf"),
        ("analysis/figs/gate_robustness_bar.pdf", "fig_gate_robustness_{dataset}.pdf"),
    ]
    
    for dataset in datasets:
        root = data_dir / dataset / f"{subset_size}_seed{seed}"
        for src_pattern, dst_pattern in figure_mappings:
            src = root / src_pattern
            dst = out_dir / dst_pattern.format(dataset=dataset)
            if src.exists():
                shutil.copy(src, dst)
                print(f"Copied: {dst}")


def generate_summary_report(safety_df: pd.DataFrame, pooling_df: pd.DataFrame, 
                           depth_df: pd.DataFrame, out_dir: Path):
    """Generate markdown summary report."""
    
    lines = [
        "# HyEm Revision Analysis Summary",
        "",
        "## 1. Safety Valve Analysis (M2)",
        "",
    ]
    
    if len(safety_df) > 0:
        soft_mix = safety_df[safety_df['method'] == 'hyem_soft']
        lines.append("| Dataset | Q-E Retention | Q-H Hits@10 | Trade-off Ratio |")
        lines.append("|---------|---------------|-------------|-----------------|")
        for _, row in soft_mix.iterrows():
            lines.append(f"| {row['dataset']} | {row['QE_retention_rate']:.1%} | "
                        f"{row['QH_parent_hits10']:.3f} | {row['trade_off_ratio']:.1f}x |")
        lines.append("")
        lines.append("**Key Finding:** Soft mixing retains >90% of Q-E performance while "
                    "significantly improving Q-H retrieval.")
    else:
        lines.append("*No safety valve data available*")
    
    lines.extend([
        "",
        "## 2. Candidate Pooling Ablation (m3)",
        "",
    ])
    
    if len(pooling_df) > 0:
        lines.append("| Dataset | Metric | C_H only | C_H∪C_E | Improvement |")
        lines.append("|---------|--------|----------|---------|-------------|")
        for _, row in pooling_df.iterrows():
            lines.append(f"| {row['dataset']} | {row['metric']} | {row['C_H_only']:.3f} | "
                        f"{row['C_H_union_C_E']:.3f} | +{row['pct_improvement']:.0f}% |")
        lines.append("")
        lines.append("**Key Finding:** Candidate pooling is essential for maintaining Q-E performance.")
    else:
        lines.append("*No pooling data available*")
    
    lines.extend([
        "",
        "## 3. Depth-Stratified Analysis (M1)",
        "",
    ])
    
    if len(depth_df) > 0:
        for qtype in ['QE', 'QH', 'QM']:
            df = depth_df[depth_df['query_type'] == qtype]
            if len(df) > 0:
                lines.append(f"### {qtype} Results")
                lines.append("| Dataset | Depth | MRR | n_queries |")
                lines.append("|---------|-------|-----|-----------|")
                for _, row in df.iterrows():
                    lines.append(f"| {row['dataset']} | {row['depth_bucket']} | "
                                f"{row['mrr']:.3f} | {int(row['n_queries'])} |")
                lines.append("")
    else:
        lines.append("*No depth-stratified data available*")
    
    lines.extend([
        "",
        "## Generated Files",
        "",
        "### Figures",
        "- `fig_combined_safety_valve.pdf` - Safety valve comparison across datasets",
        "- `fig_combined_pooling.pdf` - Pooling ablation across datasets",
        "- `fig_safety_valve_*.pdf` - Per-dataset safety valve figures",
        "- `fig_pooling_*.pdf` - Per-dataset pooling figures",
        "",
        "### LaTeX Tables",
        "- `table_safety_valve.tex` - Safety valve results for Table 2",
        "- `table_pooling_ablation.tex` - Pooling ablation results",
        "",
    ])
    
    with open(out_dir / "REVISION_SUMMARY.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {out_dir / 'REVISION_SUMMARY.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed")
    ap.add_argument("--out_dir", type=str, default="paper_artifacts/ext3")
    ap.add_argument("--datasets", type=str, default="hpo,do")
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = args.datasets.split(',')
    
    print("=" * 60)
    print("Generating Revision Figures and Tables")
    print("=" * 60)
    
    # Collect results
    print("\nCollecting results...")
    safety_df = collect_safety_valve_results(data_dir, datasets, args.subset_size, args.seed)
    pooling_df = collect_pooling_results(data_dir, datasets, args.subset_size, args.seed)
    depth_df = collect_depth_results(data_dir, datasets, args.subset_size, args.seed)
    
    print(f"  Safety valve: {len(safety_df)} rows")
    print(f"  Pooling: {len(pooling_df)} rows")
    print(f"  Depth: {len(depth_df)} rows")
    
    # Generate combined figures
    print("\nGenerating combined figures...")
    plot_combined_safety_valve(safety_df, out_dir / "fig_combined_safety_valve.pdf")
    plot_combined_pooling(pooling_df, out_dir / "fig_combined_pooling.pdf")
    
    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    generate_latex_tables(safety_df, pooling_df, out_dir)
    
    # Copy individual figures
    print("\nCopying individual figures...")
    copy_individual_figures(data_dir, datasets, args.subset_size, args.seed, out_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(safety_df, pooling_df, depth_df, out_dir)
    
    print("\n" + "=" * 60)
    print(f"All outputs saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()