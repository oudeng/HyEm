#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hyem.ontology.obo import build_edges, parse_obo
from hyem.ontology.graph import (
    build_parent_child_maps,
    compute_min_depths,
    compute_stats,
    find_roots,
    induced_edges,
    sample_subset_bfs,
)
from hyem.io import write_jsonl
from hyem.utils import ensure_dir, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do", "mesh"])
    ap.add_argument("--raw_path", type=str, default=None, help="Path to raw OBO file. If omitted, uses data/raw/<dataset>/*.obo")
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="data/processed")
    args = ap.parse_args()

    set_seed(args.seed)
    out_root = Path(args.out_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    ensure_dir(out_root)

    if args.raw_path is None:
        # find .obo under data/raw/<dataset>/
        raw_dir = Path("data/raw") / args.dataset
        cand = list(raw_dir.glob("*.obo"))
        if not cand:
            raise FileNotFoundError(f"No .obo file found under {raw_dir}. Run scripts/00_download_data.py first.")
        raw_path = cand[0]
    else:
        raw_path = Path(args.raw_path)

    print(f"[parse] {raw_path}")
    terms, alt_map = parse_obo(raw_path)
    edges_all = build_edges(terms, alt_map)

    nodes_all = set(terms.keys())
    parents_all, children_all = build_parent_child_maps(edges_all)
    roots = find_roots(nodes_all, parents_all)
    if not roots:
        # if ontology has no explicit root, fall back to any node with minimal in-degree
        roots = sorted(list(nodes_all))[:1]

    subset_nodes = set(sample_subset_bfs(roots, children_all, target_size=args.subset_size, seed=args.seed))
    edges = induced_edges(edges_all, subset_nodes)
    parents, children = build_parent_child_maps(edges)
    roots_sub = find_roots(subset_nodes, parents)
    depth = compute_min_depths(roots_sub, children)

    stats = compute_stats(subset_nodes, edges, parents, children)
    with open(out_root / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats.__dict__, f, indent=2)

    # persist nodes in a stable order (for embedding matrices)
    node_ids = sorted(list(subset_nodes))
    with open(out_root / "node_ids.txt", "w", encoding="utf-8") as f:
        for nid in node_ids:
            f.write(nid + "\n")

    node2idx = {nid: i for i, nid in enumerate(node_ids)}
    with open(out_root / "node2idx.json", "w", encoding="utf-8") as f:
        json.dump(node2idx, f, indent=2)

    # write node metadata
    items = []
    for nid in node_ids:
        t = terms[nid]
        items.append(t.to_dict())
    write_jsonl(items, out_root / "nodes.jsonl")

    # edges as indices
    edge_idx = [(node2idx[p], node2idx[c]) for (p, c) in edges if p in node2idx and c in node2idx]
    df = pd.DataFrame(edge_idx, columns=["parent", "child"])
    df.to_csv(out_root / "edges.csv", index=False)

    # parent/child maps in ids
    with open(out_root / "parents.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in parents.items() if k in subset_nodes}, f, indent=2)
    with open(out_root / "children.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in children.items() if k in subset_nodes}, f, indent=2)
    with open(out_root / "depth.json", "w", encoding="utf-8") as f:
        json.dump({k: int(v) for k, v in depth.items()}, f, indent=2)

    print(f"[done] wrote subset to {out_root}")
    print(stats)


if __name__ == "__main__":
    main()
