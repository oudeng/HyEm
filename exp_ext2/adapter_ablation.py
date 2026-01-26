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
from hyem.indexing.hnsw import load_hnsw
from hyem.io import read_jsonl, load_npy, save_csv
from hyem.models.adapter import AdapterTrainConfig, train_hyperbolic_adapter
from hyem.models.gate import LinearGate, predict_gate
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


def _predict_gate_scores(root: Path, device: str) -> Dict[str, float]:
    gate_path = root / "gate_linear.pt"
    if not gate_path.exists():
        raise FileNotFoundError(f"Missing {gate_path}. Run scripts/06_train_gate.py first.")

    e_nodes = np.load(root / "emb_nodes.npy").astype(np.float32)
    gate = LinearGate(in_dim=e_nodes.shape[1]).to(device)
    gate.load_state_dict(torch.load(gate_path, map_location=device))

    test_items = _attach_embeddings(root, "test")
    Xte = np.stack([it["e_q"] for it in test_items], axis=0).astype(np.float32)
    scores = predict_gate(gate, Xte, device=device)
    return {it["qid"]: float(s) for it, s in zip(test_items, scores)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do", "mesh"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")
    ap.add_argument("--device", type=str, default="cpu")

    # adapter training
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--neg_k", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--radius_budget", type=float, default=3.0)
    ap.add_argument("--weight_radius", type=float, default=0.1)
    ap.add_argument("--mlp_hidden_dim", type=int, default=128)
    ap.add_argument("--mlp_dropout", type=float, default=0.0)
    ap.add_argument("--train_types", nargs="+", default=["QE", "QH"], help="Query types used for adapter training")

    # retrieval config
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--L_h", type=int, default=200)
    ap.add_argument("--L_e", type=int, default=200)
    ap.add_argument("--anc_k", type=int, default=50)
    args = ap.parse_args()

    set_seed(args.seed)
    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    ensure_dir(root / "results")
    ensure_dir(root / "adapters")

    # Load fixed resources
    with open(root / "node_ids.txt", "r", encoding="utf-8") as f:
        node_ids = [line.strip() for line in f if line.strip()]
    e_nodes = np.load(root / "emb_nodes.npy").astype(np.float32)
    u_hyem = np.load(root / "u_hyem.npy").astype(np.float32)

    idx_dir = root / "indexes"
    idx_E = load_hnsw(idx_dir / "index_text.bin", dim=e_nodes.shape[1], space="cosine", ef_search=100)
    idx_H = load_hnsw(idx_dir / "index_hyem.bin", dim=u_hyem.shape[1], space="l2", ef_search=100)

    gate_scores = _predict_gate_scores(root, device=args.device)
    test_items = _attach_embeddings(root, "test")

    q_train, pos_lists = _load_train_pos_lists(root, train_types=list(args.train_types))

    cfg_base = AdapterTrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        neg_k=args.neg_k,
        temperature=args.temperature,
        radius_budget=args.radius_budget,
        weight_radius=args.weight_radius,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
        device=args.device,
    )

    r_cfg = RetrievalConfig(k=args.k, L_h=args.L_h, L_e=args.L_e, anc_k=args.anc_k, device=args.device)

    results = []
    for adapter_type in ["linear", "mlp"]:
        print(f"\n== Adapter ablation: {adapter_type} ==")
        cfg = AdapterTrainConfig(**{**cfg_base.__dict__})
        cfg.adapter_type = adapter_type

        adapter = train_hyperbolic_adapter(u_hyem, q_train, pos_lists, cfg, seed=args.seed)
        ckpt = root / "adapters" / f"adapter_hyem_{adapter_type}.pt"
        torch.save(adapter.state_dict(), ckpt)

        per_df, summary = evaluate_method(
            queries=test_items,
            method_name="hyem_soft",
            node_ids=node_ids,
            e_nodes=e_nodes,
            u_nodes=u_hyem,
            z_nodes=None,
            idx_E=idx_E,
            idx_H=idx_H,
            idx_Z=None,
            adapter_hyp=adapter,
            adapter_euc=None,
            gate_scores=gate_scores,
            cfg=r_cfg,
        )
        out_per = root / "results" / f"per_query_hyem_soft_adapter_{adapter_type}.csv"
        per_df.to_csv(out_per, index=False)

        row = {"adapter": adapter_type, **summary}
        results.append(row)
        print(row)

    df = pd.DataFrame(results)
    out_csv = root / "results" / "adapter_ablation.csv"
    save_csv(df, out_csv)
    print(f"\n[done] wrote {out_csv}")

    # Convenience: print the paper table fields
    def _pick(col: str) -> str:
        return ", ".join([f"{r['adapter']}={r.get(col, float('nan')):.3f}" for r in results if col in r])

    if results:
        print("\n[paper] Q-E Hits@10:", _pick("QE_hits10"))
        print("[paper] Q-H Parent Hits@10:", _pick("QH_parent_hits10"))
        print("[paper] Q-M Hits@10:", _pick("QM_hits10"))


if __name__ == "__main__":
    main()
