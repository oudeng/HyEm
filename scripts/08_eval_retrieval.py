#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hyem.eval.run_eval import evaluate_method
from hyem.indexing.hnsw import load_hnsw
from hyem.io import read_jsonl, load_npy, save_csv
from hyem.models.adapter import LinearAdapter
from hyem.models.gate import LinearGate, predict_gate
from hyem.retrieval.retriever import RetrievalConfig
from hyem.utils import ensure_dir


def _attach_embeddings(root: Path, split: str) -> list:
    items = read_jsonl(root / f"queries_{split}.jsonl")
    X = load_npy(root / f"emb_queries_{split}.npy")
    with open(root / f"query_ids_{split}.txt", "r", encoding="utf-8") as f:
        qids = [line.strip() for line in f if line.strip()]
    assert len(items) == X.shape[0] == len(qids)
    for it, qid, x in zip(items, qids, X):
        it["qid"] = qid
        it["e_q"] = x
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do", "mesh"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")

    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--L_h", type=int, default=200)
    ap.add_argument("--L_e", type=int, default=200)
    ap.add_argument("--anc_k", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    ensure_dir(root / "results")

    # node ids
    with open(root / "node_ids.txt", "r", encoding="utf-8") as f:
        node_ids = [line.strip() for line in f if line.strip()]

    # load node embeddings
    e_nodes = np.load(root / "emb_nodes.npy").astype(np.float32)
    u_hyem = np.load(root / "u_hyem.npy").astype(np.float32)
    u_noR = np.load(root / "u_noR.npy").astype(np.float32)
    z_euckg = np.load(root / "z_euckg.npy").astype(np.float32)

    # load indexes
    idx_dir = root / "indexes"
    idx_E = load_hnsw(idx_dir / "index_text.bin", dim=e_nodes.shape[1], space="cosine", ef_search=100)
    idx_H = load_hnsw(idx_dir / "index_hyem.bin", dim=u_hyem.shape[1], space="l2", ef_search=100)
    idx_H2 = load_hnsw(idx_dir / "index_noR.bin", dim=u_noR.shape[1], space="l2", ef_search=100)
    idx_Z = load_hnsw(idx_dir / "index_euckg.bin", dim=z_euckg.shape[1], space="l2", ef_search=100)

    # adapters
    adapter_hyem = LinearAdapter(e_nodes.shape[1], u_hyem.shape[1]).to(args.device)
    adapter_hyem.load_state_dict(torch.load(root / "adapter_hyem.pt", map_location=args.device))

    adapter_noR = LinearAdapter(e_nodes.shape[1], u_noR.shape[1]).to(args.device)
    adapter_noR.load_state_dict(torch.load(root / "adapter_noR.pt", map_location=args.device))

    adapter_euc = LinearAdapter(e_nodes.shape[1], z_euckg.shape[1]).to(args.device)
    adapter_euc.load_state_dict(torch.load(root / "adapter_euckg.pt", map_location=args.device))

    # gate scores (linear gate trained on QE/QH)
    gate_scores = None
    gate_path = root / "gate_linear.pt"
    if gate_path.exists():
        # predict on test queries
        gate = LinearGate(in_dim=e_nodes.shape[1]).to(args.device)
        gate.load_state_dict(torch.load(gate_path, map_location=args.device))
        test_items = _attach_embeddings(root, "test")
        Xte = np.stack([it["e_q"] for it in test_items], axis=0).astype(np.float32)
        scores = predict_gate(gate, Xte, device=args.device)
        gate_scores = {it["qid"]: float(s) for it, s in zip(test_items, scores)}
    else:
        test_items = _attach_embeddings(root, "test")

    cfg = RetrievalConfig(k=args.k, L_h=args.L_h, L_e=args.L_e, anc_k=args.anc_k, device=args.device)

    methods = [
        ("euclid_text", dict(u_nodes=None, z_nodes=None, idx_H=None, idx_Z=None, adapter_hyp=None, adapter_euc=None, gate_scores=None)),
        ("euclid_kg", dict(u_nodes=None, z_nodes=z_euckg, idx_H=None, idx_Z=idx_Z, adapter_hyp=None, adapter_euc=adapter_euc, gate_scores=None)),
        ("hyp_noR", dict(u_nodes=u_noR, z_nodes=None, idx_H=idx_H2, idx_Z=None, adapter_hyp=adapter_noR, adapter_euc=None, gate_scores=None)),
        ("hyem_no_gate", dict(u_nodes=u_hyem, z_nodes=None, idx_H=idx_H, idx_Z=None, adapter_hyp=adapter_hyem, adapter_euc=None, gate_scores=None)),
        ("hyem_hard", dict(u_nodes=u_hyem, z_nodes=z_euckg, idx_H=idx_H, idx_Z=idx_Z, adapter_hyp=adapter_hyem, adapter_euc=adapter_euc, gate_scores=gate_scores)),
        ("hyem_soft", dict(u_nodes=u_hyem, z_nodes=z_euckg, idx_H=idx_H, idx_Z=idx_Z, adapter_hyp=adapter_hyem, adapter_euc=adapter_euc, gate_scores=gate_scores)),
    ]

    summary_rows = []
    for mname, kw in methods:
        print(f"\n== Evaluate {mname} ==")
        per_df, summary = evaluate_method(
            queries=test_items,
            method_name=mname,
            node_ids=node_ids,
            e_nodes=e_nodes,
            u_nodes=kw.get("u_nodes"),
            z_nodes=kw.get("z_nodes"),
            idx_E=idx_E,
            idx_H=kw.get("idx_H") if kw.get("idx_H") is not None else idx_H,
            idx_Z=kw.get("idx_Z") if kw.get("idx_Z") is not None else idx_Z,
            adapter_hyp=kw.get("adapter_hyp"),
            adapter_euc=kw.get("adapter_euc"),
            gate_scores=kw.get("gate_scores"),
            cfg=cfg,
        )
        per_df.to_csv(root / "results" / f"per_query_{mname}.csv", index=False)
        row = {"method": mname, **summary}
        summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows)
    save_csv(df_sum, root / "results" / "summary.csv")
    print(df_sum)

    print(f"[done] results saved under {root / 'results'}")


if __name__ == "__main__":
    main()
