#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hyem.io import load_csv
from hyem.utils import ensure_dir


def fmt(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hpo", "do"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")
    ap.add_argument("--out_dir", type=str, default="paper_artifacts")
    args = ap.parse_args()

    out_root = ensure_dir(Path(args.out_dir))

    for ds in args.datasets:
        root = Path(args.data_dir) / ds / f"{args.subset_size}_seed{args.seed}"
        if not root.exists():
            print(f"[skip] {root} not found")
            continue
        ds_out = ensure_dir(out_root / ds)

        # dataset stats
        with open(root / "stats.json", "r", encoding="utf-8") as f:
            st = json.load(f)
        stats_tex = (
            f"{ds.upper()} & {st['n_nodes']} & {st['n_edges']} & {st['max_depth']} & {st['avg_branching']:.2f} \\\\n"
        )
        with open(ds_out / "row_dataset_stats.tex", "w", encoding="utf-8") as f:
            f.write(stats_tex)

        # summary
        sum_path = root / "results" / "summary.csv"
        if not sum_path.exists():
            print(f"[warn] missing {sum_path}")
            continue
        df = load_csv(sum_path).set_index("method")

        # QE table rows
        def row(method_key: str, label: str):
            r = df.loc[method_key]
            return f"{label} & {fmt(r.get('QE_hits1',0))} & {fmt(r.get('QE_hits10',0))} & {fmt(r.get('QE_mrr',0))} \\\\n"

        qe_rows = []
        qe_rows.append(row("euclid_text", "Euclidean text retrieval"))
        qe_rows.append(row("euclid_kg", "Euclidean KG embedding"))
        qe_rows.append(row("hyp_noR", "Hyperbolic (no radius ctrl)"))
        qe_rows.append(row("hyem_no_gate", "HYEM (no gate)"))
        qe_rows.append(row("hyem_soft", "HYEM (soft mix)"))
        with open(ds_out / "rows_qe.tex", "w", encoding="utf-8") as f:
            f.writelines(qe_rows)

        # QH table rows
        def row_qh(method_key: str, label: str):
            r = df.loc[method_key]
            return (
                f"{label} & {fmt(r.get('QH_parent_hits5',0))} & {fmt(r.get('QH_parent_hits10',0))} & {fmt(r.get('QH_anc_f1_macro',0))} & {fmt(r.get('QH_anc_f1_micro',0))} \\\\n"
            )

        qh_rows = []
        qh_rows.append(row_qh("euclid_text", "Euclidean text retrieval"))
        qh_rows.append(row_qh("euclid_kg", "Euclidean KG embedding"))
        qh_rows.append(row_qh("hyp_noR", "Hyperbolic (no radius ctrl)"))
        qh_rows.append(row_qh("hyem_no_gate", "HYEM (no gate)"))
        qh_rows.append(row_qh("hyem_soft", "HYEM (soft mix)"))
        with open(ds_out / "rows_qh.tex", "w", encoding="utf-8") as f:
            f.writelines(qh_rows)

        # gate table
        gate_path = root / "gate_metrics.csv"
        if gate_path.exists():
            gdf = load_csv(gate_path).set_index("gate")
            gate_rows = []
            for g in ["rule", "linear", "mlp"]:
                r = gdf.loc[g]
                gate_rows.append(
                    f"{g} & {fmt(r['accuracy'])} & {fmt(r['auc'])} & {fmt(r['precision_qh'])} & {fmt(r['recall_qh'])} \\\\n"
                )
            with open(ds_out / "rows_gate.tex", "w", encoding="utf-8") as f:
                f.writelines(gate_rows)

        # efficiency table
        eff_path = root / "analysis" / "efficiency.csv"
        if eff_path.exists():
            edf = load_csv(eff_path).set_index("method")
            eff_rows = []
            for m, label, extra in [
                ("euclid_text", "Euclidean text", "--"),
                ("hyem_no_gate", "HYEM (no gate)", "adapter + rerank"),
                ("hyem_soft", "HYEM (soft mix)", "adapter + rerank + optional 2nd ANN"),
            ]:
                r = edf.loc[m]
                eff_rows.append(f"{label} & {fmt(r['index_size_mb'],2)} & {fmt(r['latency_ms'],2)} & {extra} \\\\n")
            with open(ds_out / "rows_efficiency.tex", "w", encoding="utf-8") as f:
                f.writelines(eff_rows)

        # mixing figure (bar chart for QE/QH/QM hits@10)
        try:
            methods = ["hyem_no_gate", "hyem_hard", "hyem_soft"]
            labels = ["no gate", "hard route", "soft mix"]
            qe = [float(df.loc[m].get("QE_hits10", 0.0)) for m in methods]
            qh = [float(df.loc[m].get("QH_parent_hits10", 0.0)) for m in methods]
            qm = [float(df.loc[m].get("QM_hits10", 0.0)) for m in methods]

            x = np.arange(len(methods))
            width = 0.25

            plt.figure()
            plt.bar(x - width, qe, width, label="Q-E Hits@10")
            plt.bar(x, qh, width, label="Q-H Parent Hits@10")
            plt.bar(x + width, qm, width, label="Q-M Hits@10")
            plt.xticks(x, labels)
            plt.ylim(0.0, 1.0)
            plt.ylabel("Score")
            plt.title(f"Query-adaptive mixing ({ds.upper()}, {args.subset_size})")
            plt.legend()
            plt.tight_layout()
            fig_path = ds_out / "mixing_bar.pdf"
            plt.savefig(fig_path)
        except Exception as e:
            print(f"[warn] failed to make mixing figure for {ds}: {e}")

        print(f"[done] wrote artifacts to {ds_out}")


if __name__ == "__main__":
    main()
