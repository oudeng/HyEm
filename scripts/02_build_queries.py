#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyem.eval.tasks import (
    generate_qe_queries,
    generate_qh_queries,
    generate_qm_queries,
    split_nodes_by_seed,
)
from hyem.io import read_jsonl, write_jsonl
from hyem.utils import ensure_dir, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do", "mesh"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")
    ap.add_argument("--max_qe_per_node", type=int, default=2)
    ap.add_argument("--max_templates", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)

    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    ensure_dir(root)

    nodes_list = read_jsonl(root / "nodes.jsonl")
    nodes = {x["id"]: x for x in nodes_list}

    with open(root / "parents.json", "r", encoding="utf-8") as f:
        parents = json.load(f)
    with open(root / "children.json", "r", encoding="utf-8") as f:
        children = json.load(f)
    with open(root / "depth.json", "r", encoding="utf-8") as f:
        depth = {k: int(v) for k, v in json.load(f).items()}

    node_ids = sorted(list(nodes.keys()))
    splits = split_nodes_by_seed(node_ids, seed=args.seed, train=0.8, val=0.1)

    with open(root / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    def _make(split_name: str):
        ids = splits[split_name]
        qe = generate_qe_queries(nodes, ids, max_per_node=args.max_qe_per_node, seed=args.seed)
        qh = generate_qh_queries(nodes, ids, parents, seed=args.seed, max_templates=args.max_templates)
        qm = generate_qm_queries(nodes, ids, parents, children, depth, seed=args.seed, max_templates=args.max_templates)
        return qe + qh + qm

    train_q = _make("train")
    val_q = _make("val")
    test_q = _make("test")

    write_jsonl(train_q, root / "queries_train.jsonl")
    write_jsonl(val_q, root / "queries_val.jsonl")
    write_jsonl(test_q, root / "queries_test.jsonl")

    print(f"[done] queries saved under {root}")
    print(f"train={len(train_q)} val={len(val_q)} test={len(test_q)}")


if __name__ == "__main__":
    main()
