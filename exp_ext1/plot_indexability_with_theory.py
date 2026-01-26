#!/usr/bin/env python3
"""Plot indexability recall-vs-L curve and overlay the theory-guided L_th line.

This script is repo-agnostic: it reads the CSV produced by scripts/09_indexability_test.py
(indexability_recall_curve.csv) and writes a PDF.

Example:
  python new_experiments/plot_indexability_with_theory.py \
    --in_csv data/processed/hpo/5000_seed0/analysis/indexability_recall_curve.csv \
    --out_pdf paper_artifacts/figs/recall_curve_hpo_5000_with_theory.pdf \
    --R 3.0 --k 10 --title "HPO-5k"
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def kappa(R: float) -> float:
    return math.sinh(R) / R


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_csv', type=str, required=True)
    parser.add_argument('--out_pdf', type=str, required=True)
    parser.add_argument('--R', type=float, required=True)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--ymin', type=float, default=0.9)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    if not {'L', 'recall_at_k'}.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {args.in_csv}: {df.columns}")

    L_th = math.ceil(kappa(args.R) * args.k)

    plt.figure(figsize=(4.2, 3.2))
    x = df['L'].values
    y = df['recall_at_k'].values
    yerr = df['std'].values if 'std' in df.columns else None

    if yerr is not None:
        plt.errorbar(x, y, yerr=yerr, marker='o', linestyle='-')
    else:
        plt.plot(x, y, marker='o', linestyle='-')

    plt.xscale('log')
    plt.ylim(args.ymin, 1.01)
    plt.xlabel('Oversampling L (log scale)')
    plt.ylabel(f'Recall@{args.k}')

    plt.axvline(L_th, linestyle='--', linewidth=1)
    plt.text(L_th * 1.05, args.ymin + 0.005, f"L_th={L_th}", rotation=90, va='bottom', fontsize=8)

    if args.title:
        plt.title(args.title, fontsize=10)

    plt.tight_layout()
    out_path = Path(args.out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    print(f"Saved: {out_path} (L_th={L_th}, kappa(R)={kappa(args.R):.4f})")


if __name__ == '__main__':
    main()
