#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from hyem.indexing.hnsw import load_hnsw, query_hnsw
from hyem.io import read_jsonl, load_npy, save_csv
from hyem.models.adapter import LinearAdapter
from hyem.models.hyperbolic import lorentz_distance_to_many
from hyem.utils import ensure_dir


def _attach_embeddings(root: Path, split: str) -> list:
    items = read_jsonl(root / f"queries_{split}.jsonl")
    X = load_npy(root / f"emb_queries_{split}.npy")
    with open(root / f"query_ids_{split}.txt", "r", encoding="utf-8") as f:
        qids = [line.strip() for line in f if line.strip()]
    assert len(items) == X.shape[0] == len(qids)
    out = []
    for it, qid, x in zip(items, qids, X):
        it = dict(it)
        it["qid"] = qid
        it["e_q"] = x
        out.append(it)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do", "mesh"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")

    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--Ls", type=int, nargs="+", default=[20, 50, 100, 200, 500, 1000])
    ap.add_argument("--max_queries", type=int, default=200)
    ap.add_argument("--query_type", type=str, default="QE", choices=["QE", "QH", "QM"])
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    out_dir = ensure_dir(root / "analysis")
    ensure_dir(out_dir / "figs")

    # node ids
    with open(root / "node_ids.txt", "r", encoding="utf-8") as f:
        node_ids = [line.strip() for line in f if line.strip()]

    u_nodes = np.load(root / "u_hyem.npy").astype(np.float32)
    e_nodes = np.load(root / "emb_nodes.npy").astype(np.float32)

    # ANN index for tangent vectors
    idx_H = load_hnsw(root / "indexes" / "index_hyem.bin", dim=u_nodes.shape[1], space="l2", ef_search=200)

    # adapter
    adapter = LinearAdapter(e_nodes.shape[1], u_nodes.shape[1]).to(args.device)
    adapter.load_state_dict(torch.load(root / "adapter_hyem.pt", map_location=args.device))
    adapter.eval()

    queries = _attach_embeddings(root, "test")
    queries = [q for q in queries if q["type"] == args.query_type]
    queries = queries[: args.max_queries]
    print(f"[queries] using {len(queries)} queries of type {args.query_type}")

    # brute-force true top-k in hyperbolic distance
    U = torch.tensor(u_nodes, dtype=torch.float32, device=args.device)

    def true_topk(u_q: np.ndarray) -> np.ndarray:
        uq = torch.tensor(u_q, dtype=torch.float32, device=args.device)
        d = lorentz_distance_to_many(uq, U).detach().cpu().numpy()
        return np.argsort(d)[: args.k]

    def approx_topk(u_q: np.ndarray, L: int) -> np.ndarray:
        lab, _ = query_hnsw(idx_H, u_q, k=L)
        cand = lab.astype(int)
        u_c = u_nodes[cand]
        uq = torch.tensor(u_q, dtype=torch.float32, device=args.device)
        uc = torch.tensor(u_c, dtype=torch.float32, device=args.device)
        d = lorentz_distance_to_many(uq, uc).detach().cpu().numpy()
        order = np.argsort(d)[: args.k]
        return cand[order]

    rows = []
    for L in args.Ls:
        recalls = []
        for q in queries:
            e_q = q["e_q"]
            with torch.no_grad():
                u_q = adapter(torch.tensor(e_q, dtype=torch.float32, device=args.device)).detach().cpu().numpy().astype(np.float32)
            tk = true_topk(u_q)
            ak = approx_topk(u_q, L)
            rec = len(set(tk.tolist()) & set(ak.tolist())) / float(args.k)
            recalls.append(rec)
        rows.append({"L": int(L), "recall_at_k": float(np.mean(recalls)), "std": float(np.std(recalls))})
        print(f"L={L} recall={rows[-1]['recall_at_k']:.3f}±{rows[-1]['std']:.3f}")

    df = pd.DataFrame(rows)
    save_csv(df, out_dir / "indexability_recall_curve.csv")

    # plot
    plt.figure()
    plt.plot(df["L"], df["recall_at_k"], marker="o")
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Oversampling L (log scale)")
    plt.ylabel(f"Recall@{args.k} vs exact hyperbolic top-{args.k}")
    plt.title(f"Indexability stress test ({args.dataset}, {args.subset_size})")
    plt.tight_layout()
    fig_path = out_dir / "figs" / "recall_curve.pdf"
    plt.savefig(fig_path)
    print(f"[done] saved {fig_path}")


if __name__ == "__main__":
    main()
