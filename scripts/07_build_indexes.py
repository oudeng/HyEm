#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hyem.indexing.hnsw import HNSWConfig, build_hnsw_index, save_hnsw
from hyem.utils import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do", "mesh"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")

    ap.add_argument("--ef_construction", type=int, default=200)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef_search", type=int, default=100)
    args = ap.parse_args()

    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    out = ensure_dir(root / "indexes")

    e_nodes = np.load(root / "emb_nodes.npy").astype(np.float32)
    u_hyem = np.load(root / "u_hyem.npy").astype(np.float32)
    u_noR = np.load(root / "u_noR.npy").astype(np.float32)
    z_euc = np.load(root / "z_euckg.npy").astype(np.float32)

    cfg_cos = HNSWConfig(space="cosine", ef_construction=args.ef_construction, M=args.M, ef_search=args.ef_search)
    cfg_l2 = HNSWConfig(space="l2", ef_construction=args.ef_construction, M=args.M, ef_search=args.ef_search)

    print("[build] Euclidean text index")
    idx_E = build_hnsw_index(e_nodes, cfg_cos)
    save_hnsw(idx_E, out / "index_text.bin")

    print("[build] Hyperbolic tangent index (HYEM)")
    idx_H = build_hnsw_index(u_hyem, cfg_l2)
    save_hnsw(idx_H, out / "index_hyem.bin")

    print("[build] Hyperbolic tangent index (noR)")
    idx_H2 = build_hnsw_index(u_noR, cfg_l2)
    save_hnsw(idx_H2, out / "index_noR.bin")

    print("[build] Euclidean KG index")
    idx_Z = build_hnsw_index(z_euc, cfg_l2)
    save_hnsw(idx_Z, out / "index_euckg.bin")

    with open(out / "index_config.json", "w", encoding="utf-8") as f:
        json.dump({"cosine": cfg_cos.__dict__, "l2": cfg_l2.__dict__}, f, indent=2)

    print(f"[done] indexes saved under {out}")


if __name__ == "__main__":
    main()
