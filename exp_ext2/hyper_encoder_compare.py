#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from hyem.eval.run_eval import evaluate_method
from hyem.indexing.hnsw import HNSWConfig, build_hnsw_index, load_hnsw, query_hnsw, save_hnsw
from hyem.io import read_jsonl, load_npy, save_csv
from hyem.models.adapter import AdapterTrainConfig, LinearAdapter, train_hyperbolic_adapter
from hyem.models.hyperbolic import lorentz_distance_to_many
from hyem.models.hgcn import HGCNTrainConfig, train_hgcn_hyperbolic_embeddings
from hyem.retrieval.retriever import RetrievalConfig
from hyem.utils import ensure_dir, set_seed


def _attach_embeddings(root: Path, split: str) -> List[dict]:
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


def _load_train_pos_lists(root: Path, train_types: List[str]) -> Tuple[np.ndarray, List[List[int]]]:
    with open(root / "node2idx.json", "r", encoding="utf-8") as f:
        node2idx = json.load(f)

    q_items = read_jsonl(root / "queries_train.jsonl")
    q_emb = load_npy(root / "emb_queries_train.npy")
    assert len(q_items) == q_emb.shape[0]

    pos_lists: List[List[int]] = []
    q_keep: List[np.ndarray] = []
    for item, emb in zip(q_items, q_emb):
        if item.get("type") not in train_types:
            continue
        pos = []
        for nid in item.get("pos_ids", []):
            if nid in node2idx:
                pos.append(int(node2idx[nid]))
        if not pos:
            continue
        pos_lists.append(pos)
        q_keep.append(emb)
    if not q_keep:
        raise RuntimeError("No training queries kept for adapter training. Check train_types.")
    q_keep_arr = np.stack(q_keep, axis=0).astype(np.float32)
    return q_keep_arr, pos_lists


def _indexability_recall_at_L(
    *,
    u_nodes: np.ndarray,
    idx_H,
    adapter,
    queries: List[dict],
    k: int,
    L: int,
    device: str,
) -> float:
    """Recall@k of exact hyperbolic top-k when restricted to ANN candidate pool of size L."""
    U = torch.tensor(u_nodes, dtype=torch.float32, device=device)
    adapter.eval()

    def true_topk(u_q: np.ndarray) -> np.ndarray:
        uq = torch.tensor(u_q, dtype=torch.float32, device=device)
        d = lorentz_distance_to_many(uq, U).detach().cpu().numpy()
        return np.argsort(d)[:k]

    def approx_topk(u_q: np.ndarray) -> np.ndarray:
        lab, _ = query_hnsw(idx_H, u_q, k=L)
        cand = lab.astype(int)
        u_c = u_nodes[cand]
        uq = torch.tensor(u_q, dtype=torch.float32, device=device)
        uc = torch.tensor(u_c, dtype=torch.float32, device=device)
        d = lorentz_distance_to_many(uq, uc).detach().cpu().numpy()
        order = np.argsort(d)[:k]
        return cand[order]

    recalls = []
    for q in queries:
        e_q = q["e_q"]
        with torch.no_grad():
            u_q = adapter(torch.tensor(e_q, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
        tk = true_topk(u_q)
        ak = approx_topk(u_q)
        rec = len(set(tk.tolist()) & set(ak.tolist())) / float(k)
        recalls.append(rec)
    return float(np.mean(recalls)) if recalls else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do", "mesh"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")
    ap.add_argument("--device", type=str, default="cpu")

    # HGCN training
    ap.add_argument("--hgcn_epochs", type=int, default=50)
    ap.add_argument("--hgcn_layers", type=int, default=2)
    ap.add_argument("--hgcn_dropout", type=float, default=0.0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--radius_budget", type=float, default=3.0)
    ap.add_argument("--weight_radius", type=float, default=0.1)
    ap.add_argument("--force_train", action="store_true")

    # adapter training for HGCN
    ap.add_argument("--adapter_epochs", type=int, default=5)
    ap.add_argument("--adapter_batch_size", type=int, default=256)
    ap.add_argument("--adapter_neg_k", type=int, default=50)
    ap.add_argument("--adapter_lr", type=float, default=1e-3)
    ap.add_argument("--train_types", nargs="+", default=["QE", "QH"])

    # evaluation + indexability
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--L_index", type=int, default=50)
    ap.add_argument("--max_queries", type=int, default=200)
    ap.add_argument("--query_type", type=str, default="QH", choices=["QE", "QH", "QM"])
    args = ap.parse_args()

    set_seed(args.seed)
    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    ensure_dir(root / "results")
    ensure_dir(root / "adapters")
    ensure_dir(root / "indexes")

    with open(root / "node_ids.txt", "r", encoding="utf-8") as f:
        node_ids = [line.strip() for line in f if line.strip()]
    num_nodes = len(node_ids)

    e_nodes = np.load(root / "emb_nodes.npy").astype(np.float32)
    u_base = np.load(root / "u_hyem.npy").astype(np.float32)

    # Build/load baseline index + adapter
    idx_E = load_hnsw(root / "indexes" / "index_text.bin", dim=e_nodes.shape[1], space="cosine", ef_search=200)
    idx_base = load_hnsw(root / "indexes" / "index_hyem.bin", dim=u_base.shape[1], space="l2", ef_search=200)
    adapter_base = LinearAdapter(e_nodes.shape[1], u_base.shape[1]).to(args.device)
    adapter_base.load_state_dict(torch.load(root / "adapter_hyem.pt", map_location=args.device))

    # Prepare queries
    test_items = _attach_embeddings(root, "test")
    test_items = [q for q in test_items if q.get("type") == args.query_type]
    test_items = test_items[: args.max_queries]
    print(f"[queries] {len(test_items)} queries of type {args.query_type}")

    eval_cfg = RetrievalConfig(k=args.k, L_h=200, L_e=200, anc_k=50, device=args.device)

    # ---- Baseline evaluation ----
    per_df, summary = evaluate_method(
        queries=test_items,
        method_name="hyem_no_gate",
        node_ids=node_ids,
        e_nodes=e_nodes,
        u_nodes=u_base,
        z_nodes=None,
        idx_E=idx_E,
        idx_H=idx_base,
        idx_Z=None,
        adapter_hyp=adapter_base,
        adapter_euc=None,
        gate_scores=None,
        cfg=eval_cfg,
    )
    base_qh = float(summary.get("QH_parent_hits10", float("nan")))
    base_rec = _indexability_recall_at_L(
        u_nodes=u_base,
        idx_H=idx_base,
        adapter=adapter_base,
        queries=test_items,
        k=args.k,
        L=args.L_index,
        device=args.device,
    )

    rows = [
        {
            "encoder": "lorentz_kg",
            "QH_parent_hits10": base_qh,
            f"indexability_recall@{args.k}_L{args.L_index}": base_rec,
            "notes": "this paper",
        }
    ]

    # ---- Train/load HGCN embeddings ----
    hgcn_path = root / "u_hgcn.npy"
    if args.force_train or not hgcn_path.exists():
        import pandas as _pd

        edges = _pd.read_csv(root / "edges.csv")[["parent", "child"]].values.astype(np.int64)
        cfg_h = HGCNTrainConfig(
            dim=args.dim,
            num_layers=args.hgcn_layers,
            dropout=args.hgcn_dropout,
            epochs=args.hgcn_epochs,
            radius_budget=args.radius_budget,
            weight_radius=args.weight_radius,
            device=args.device,
        )
        print("\n== Train tangent-space HGCN embeddings ==")
        u_hgcn = train_hgcn_hyperbolic_embeddings(num_nodes, edges, cfg_h, seed=args.seed)
        np.save(hgcn_path, u_hgcn.astype(np.float32))
    else:
        u_hgcn = np.load(hgcn_path).astype(np.float32)

    # index for HGCN
    idx_hgcn_path = root / "indexes" / "index_hgcn.bin"
    if args.force_train or not idx_hgcn_path.exists():
        cfg_l2 = HNSWConfig(space="l2", ef_construction=200, M=16, ef_search=200)
        idx_hgcn = build_hnsw_index(u_hgcn, cfg_l2)
        save_hnsw(idx_hgcn, idx_hgcn_path)
    idx_hgcn = load_hnsw(idx_hgcn_path, dim=u_hgcn.shape[1], space="l2", ef_search=200)

    # adapter for HGCN
    adapter_hgcn_path = root / "adapters" / "adapter_hgcn.pt"
    if args.force_train or not adapter_hgcn_path.exists():
        q_train, pos_lists = _load_train_pos_lists(root, train_types=list(args.train_types))
        cfg_a = AdapterTrainConfig(
            lr=args.adapter_lr,
            epochs=args.adapter_epochs,
            batch_size=args.adapter_batch_size,
            neg_k=args.adapter_neg_k,
            radius_budget=args.radius_budget,
            weight_radius=args.weight_radius,
            device=args.device,
            adapter_type="linear",
        )
        print("\n== Train adapter for HGCN embeddings ==")
        adapter_hgcn = train_hyperbolic_adapter(u_hgcn, q_train, pos_lists, cfg_a, seed=args.seed)
        torch.save(adapter_hgcn.state_dict(), adapter_hgcn_path)
    adapter_hgcn = LinearAdapter(e_nodes.shape[1], u_hgcn.shape[1]).to(args.device)
    adapter_hgcn.load_state_dict(torch.load(adapter_hgcn_path, map_location=args.device))

    # Evaluate
    per_df_h, summary_h = evaluate_method(
        queries=test_items,
        method_name="hyem_no_gate",
        node_ids=node_ids,
        e_nodes=e_nodes,
        u_nodes=u_hgcn,
        z_nodes=None,
        idx_E=idx_E,
        idx_H=idx_hgcn,
        idx_Z=None,
        adapter_hyp=adapter_hgcn,
        adapter_euc=None,
        gate_scores=None,
        cfg=eval_cfg,
    )
    hgcn_qh = float(summary_h.get("QH_parent_hits10", float("nan")))
    hgcn_rec = _indexability_recall_at_L(
        u_nodes=u_hgcn,
        idx_H=idx_hgcn,
        adapter=adapter_hgcn,
        queries=test_items,
        k=args.k,
        L=args.L_index,
        device=args.device,
    )

    rows.append(
        {
            "encoder": "hgcn_tangent",
            "QH_parent_hits10": hgcn_qh,
            f"indexability_recall@{args.k}_L{args.L_index}": hgcn_rec,
            "notes": "tangent-space HGCN baseline",
        }
    )

    # Placeholder row for entailment cones (not implemented in this lightweight package)
    rows.append(
        {
            "encoder": "entailment_cones",
            "QH_parent_hits10": np.nan,
            f"indexability_recall@{args.k}_L{args.L_index}": np.nan,
            "notes": "planned (not implemented)",
        }
    )

    df = pd.DataFrame(rows)
    out_csv = root / "results" / "hyper_encoder_compare.csv"
    save_csv(df, out_csv)
    print("\n", df)
    print(f"\n[done] wrote {out_csv}")


if __name__ == "__main__":
    main()
