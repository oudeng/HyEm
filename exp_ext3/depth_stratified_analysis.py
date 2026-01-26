#!/usr/bin/env python
"""
Depth-Stratified Analysis (M1, M2)

Stratify retrieval results by node depth to show:
1. Where hyperbolic structure helps most (deeper levels)
2. Whether improvements are concentrated or uniform

This addresses reviewer concern about "5k is too small" by showing
the method works across depth levels, suggesting scalability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_depth_info(root: Path) -> dict:
    """Load node depth information."""
    with open(root / "depth.json", "r", encoding="utf-8") as f:
        depth = {k: int(v) for k, v in json.load(f).items()}
    return depth


def load_queries_with_depth(root: Path, split: str = "test") -> list:
    """Load queries and attach depth info."""
    import json
    queries = []
    with open(root / f"queries_{split}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line.strip())
            queries.append(q)
    return queries


def stratify_by_depth(queries: list, per_query_df: pd.DataFrame, 
                      depth_info: dict, n_buckets: int = 4) -> pd.DataFrame:
    """Stratify per-query results by depth buckets."""
    
    # Build qid -> depth mapping
    qid_to_depth = {}
    for q in queries:
        qid = q['qid']
        qtype = q['type']
        
        if qtype == 'QE':
            # For Q-E, use the target node's depth
            target_id = q['pos_ids'][0] if q.get('pos_ids') else None
            if target_id and target_id in depth_info:
                qid_to_depth[qid] = depth_info[target_id]
        elif qtype in ('QH', 'QM'):
            # For Q-H/Q-M, use the focus node's depth
            focus_id = q.get('focus_id')
            if focus_id and focus_id in depth_info:
                qid_to_depth[qid] = depth_info[focus_id]
            # Also check if depth is stored in query
            if 'depth' in q:
                qid_to_depth[qid] = q['depth']
    
    # Merge depth info with per-query results
    per_query_df = per_query_df.copy()
    per_query_df['depth'] = per_query_df['qid'].map(qid_to_depth)
    per_query_df = per_query_df.dropna(subset=['depth'])
    
    # Create depth buckets (quartiles)
    if len(per_query_df) > 0:
        try:
            # Try qcut first (equal-frequency bins)
            per_query_df['depth_bucket'] = pd.qcut(
                per_query_df['depth'], 
                q=n_buckets, 
                labels=False,  # Use numeric labels first
                duplicates='drop'
            )
            # Get actual number of buckets created
            actual_buckets = per_query_df['depth_bucket'].nunique()
            # Relabel with D1, D2, etc.
            per_query_df['depth_bucket'] = per_query_df['depth_bucket'].map(
                lambda x: f'D{int(x)+1}' if pd.notna(x) else None
            )
        except ValueError:
            # If qcut fails, fall back to cut (equal-width bins)
            try:
                per_query_df['depth_bucket'] = pd.cut(
                    per_query_df['depth'],
                    bins=n_buckets,
                    labels=[f'D{i+1}' for i in range(n_buckets)],
                    include_lowest=True
                )
            except ValueError:
                # Last resort: use unique depth values as buckets
                unique_depths = sorted(per_query_df['depth'].unique())
                if len(unique_depths) <= n_buckets:
                    per_query_df['depth_bucket'] = per_query_df['depth'].map(
                        lambda d: f'D{unique_depths.index(d)+1}'
                    )
                else:
                    # Group depths into n_buckets ranges
                    per_query_df['depth_bucket'] = pd.cut(
                        per_query_df['depth'],
                        bins=n_buckets,
                        labels=[f'D{i+1}' for i in range(n_buckets)]
                    )
    
    return per_query_df.dropna(subset=['depth_bucket'])


def compute_stratified_metrics(per_query_df: pd.DataFrame, query_type: str) -> pd.DataFrame:
    """Compute metrics per depth bucket for a given query type."""
    
    df = per_query_df[per_query_df['type'] == query_type].copy()
    if len(df) == 0:
        return pd.DataFrame()
    
    # Drop rows with missing depth_bucket
    df = df.dropna(subset=['depth_bucket'])
    if len(df) == 0:
        return pd.DataFrame()
    
    # Group by depth bucket and compute metrics
    grouped = df.groupby('depth_bucket').agg({
        'rr': ['mean', 'std', 'count'],
        'depth': ['min', 'max', 'mean']
    }).round(4)
    
    # Flatten column names
    grouped.columns = ['mrr', 'mrr_std', 'n_queries', 'depth_min', 'depth_max', 'depth_mean']
    grouped = grouped.reset_index()
    grouped['query_type'] = query_type
    
    # Fill NaN std with 0 (happens when only 1 sample in bucket)
    grouped['mrr_std'] = grouped['mrr_std'].fillna(0)
    
    return grouped


def plot_depth_stratified_heatmap(results: dict, methods: list, out_path: Path):
    """Create heatmap showing performance across depths and methods."""
    
    # Focus on Q-H results for heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, (dataset, method_data) in zip(axes, results.items()):
        # Build matrix: rows = depth buckets, cols = methods
        buckets = sorted(set(b for m in method_data.values() for b in m.get('buckets', [])))
        
        if not buckets:
            continue
            
        matrix = np.zeros((len(buckets), len(methods)))
        
        for j, method in enumerate(methods):
            if method in method_data:
                for i, bucket in enumerate(buckets):
                    if bucket in method_data[method].get('mrr', {}):
                        matrix[i, j] = method_data[method]['mrr'][bucket]
        
        # Create heatmap
        sns.heatmap(matrix, ax=ax, annot=True, fmt='.3f', 
                   xticklabels=methods, yticklabels=buckets,
                   cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f'{dataset.upper()}: Q-H MRR by Depth')
        ax.set_xlabel('Method')
        ax.set_ylabel('Depth Bucket')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_depth_trend(stratified_results: list, dataset: str, out_path: Path):
    """Plot performance trends across depth buckets."""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for ax, qtype in zip(axes, ['QE', 'QH', 'QM']):
        # Filter results for this query type
        matching_dfs = []
        for r in stratified_results:
            if len(r) > 0 and 'query_type' in r.columns:
                if r['query_type'].iloc[0] == qtype:
                    matching_dfs.append(r)
        
        if not matching_dfs:
            ax.text(0.5, 0.5, f'No {qtype} data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{qtype} Performance by Depth')
            continue
        
        df = pd.concat(matching_dfs, ignore_index=True)
        
        if len(df) == 0:
            ax.text(0.5, 0.5, f'No {qtype} data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{qtype} Performance by Depth')
            continue
        
        # Sort by depth bucket for consistent ordering
        df = df.sort_values('depth_bucket')
        
        # Plot MRR with error bars
        buckets = df['depth_bucket'].tolist()
        x = np.arange(len(buckets))
        
        ax.bar(x, df['mrr'].values, yerr=df['mrr_std'].values, capsize=3, 
               color='steelblue', alpha=0.7, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.set_xlabel('Depth Bucket (D1=shallow, D4=deep)')
        ax.set_ylabel('MRR')
        ax.set_title(f'{qtype} Performance by Depth')
        ax.set_ylim(0, 1)
        
        # Add sample sizes
        for i, (_, row) in enumerate(df.iterrows()):
            y_pos = min(row['mrr'] + row['mrr_std'] + 0.02, 0.95)
            ax.text(i, y_pos, f'n={int(row["n_queries"])}', ha='center', fontsize=8)
    
    fig.suptitle(f'Depth-Stratified Analysis: {dataset.upper()} (HyEm soft mix)', fontsize=12)
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
    ap.add_argument("--depth_buckets", type=int, default=4)
    args = ap.parse_args()
    
    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    
    # Load data
    depth_info = load_depth_info(root)
    queries = load_queries_with_depth(root, "test")
    
    # Focus on HyEm soft mix for primary analysis
    per_query_df = pd.read_csv(root / "results" / "per_query_hyem_soft.csv")
    
    # Stratify by depth
    stratified_df = stratify_by_depth(queries, per_query_df, depth_info, args.depth_buckets)
    
    if len(stratified_df) == 0:
        print(f"\nWarning: No queries could be mapped to depth buckets.")
        print(f"This may happen if depth.json doesn't contain the test query nodes.")
        print(f"Skipping depth-stratified analysis.")
        return
    
    print(f"\nStratified {len(stratified_df)} queries into depth buckets")
    
    # Compute metrics per query type and depth bucket
    all_results = []
    for qtype in ['QE', 'QH', 'QM']:
        result = compute_stratified_metrics(stratified_df, qtype)
        if len(result) > 0:
            all_results.append(result)
            print(f"\n=== {qtype} Depth-Stratified Results ===")
            print(result.to_string(index=False))
    
    # Save combined results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_path = root / "results" / "depth_stratified_analysis.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")
        
        # Create visualization
        fig_dir = root / "analysis" / "figs"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_depth_trend(all_results, args.dataset, fig_dir / "depth_stratified_trend.pdf")
    
    # Print key findings
    print(f"\n=== Key Findings for Paper ===")
    print(f"Dataset: {args.dataset.upper()}")
    if depth_info:
        print(f"Depth range: {min(depth_info.values())} - {max(depth_info.values())}")
    else:
        print(f"Depth range: N/A (no depth info available)")
    print(f"Analysis suggests: Check if Q-H performance improves at deeper levels")


if __name__ == "__main__":
    main()