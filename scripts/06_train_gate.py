#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hyem.eval.metrics import compute_gate_metrics
from hyem.io import read_jsonl, load_npy, save_csv
from hyem.models.gate import (
    GateTrainConfig,
    LinearGate,
    MLPGate,
    predict_gate,
    rule_based_score,
    train_gate,
)
from hyem.utils import ensure_dir, set_seed


def _load(root: Path, split: str):
    items = read_jsonl(root / f"queries_{split}.jsonl")
    X = load_npy(root / f"emb_queries_{split}.npy")
    with open(root / f"query_ids_{split}.txt", "r", encoding="utf-8") as f:
        qids = [line.strip() for line in f if line.strip()]
    assert len(items) == X.shape[0] == len(qids)
    for it, qid in zip(items, qids):
        it["qid"] = qid
    return items, X


def _filter_qe_qh(items, X):
    keep_items = []
    keep_X = []
    keep_y = []
    for it, x in zip(items, X):
        if it["type"] not in ["QE", "QH"]:
            continue
        y = 1 if it["type"] == "QH" else 0
        keep_items.append(it)
        keep_X.append(x)
        keep_y.append(y)
    return keep_items, np.stack(keep_X, axis=0).astype(np.float32), np.array(keep_y, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do", "mesh"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--keywords", nargs="*", default=None, help="Override rule-based keywords.")
    args = ap.parse_args()

    set_seed(args.seed)
    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    ensure_dir(root)

    tr_items, Xtr = _load(root, "train")
    va_items, Xva = _load(root, "val")
    te_items, Xte = _load(root, "test")

    tr_items, Xtr, ytr = _filter_qe_qh(tr_items, Xtr)
    va_items, Xva, yva = _filter_qe_qh(va_items, Xva)
    te_items, Xte, yte = _filter_qe_qh(te_items, Xte)

    print(f"[data] train={len(tr_items)} val={len(va_items)} test={len(te_items)}")

    cfg = GateTrainConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, device=args.device)

    # Rule-based
    kw = args.keywords
    texts_te = [it["text"] for it in te_items]
    y_score_rule = rule_based_score(texts_te) if kw is None else rule_based_score(texts_te, keywords=kw)
    m_rule = compute_gate_metrics(yte, y_score_rule, threshold=0.5)

    # Linear
    lin = LinearGate(in_dim=Xtr.shape[1])
    lin = train_gate(lin, Xtr, ytr.astype(np.float32), cfg, seed=args.seed)
    y_score_lin = predict_gate(lin, Xte, device=args.device)
    m_lin = compute_gate_metrics(yte, y_score_lin, threshold=0.5)
    torch.save(lin.state_dict(), root / "gate_linear.pt")

    # MLP
    mlp = MLPGate(in_dim=Xtr.shape[1], hidden=128, dropout=0.1)
    mlp = train_gate(mlp, Xtr, ytr.astype(np.float32), cfg, seed=args.seed)
    y_score_mlp = predict_gate(mlp, Xte, device=args.device)
    m_mlp = compute_gate_metrics(yte, y_score_mlp, threshold=0.5)
    torch.save(mlp.state_dict(), root / "gate_mlp.pt")

    df = pd.DataFrame(
        [
            {"gate": "rule", **m_rule.__dict__},
            {"gate": "linear", **m_lin.__dict__},
            {"gate": "mlp", **m_mlp.__dict__},
        ]
    )
    save_csv(df, root / "gate_metrics.csv")
    print(df)

    # also save per-query scores for downstream evaluation
    gate_scores = {it["qid"]: float(s) for it, s in zip(te_items, y_score_lin)}
    with open(root / "gate_scores_test.json", "w", encoding="utf-8") as f:
        json.dump(gate_scores, f, indent=2)

    print(f"[done] saved gate_metrics.csv and gate_scores_test.json under {root}")


if __name__ == "__main__":
    main()
