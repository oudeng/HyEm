#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hyem.io import read_jsonl, save_npy
from hyem.text.embedder import TextEmbedder, TextEmbedderConfig
from hyem.utils import ensure_dir, set_seed


def _first_synonym(node: dict) -> str:
    syns = node.get("synonyms", []) or []
    for s in syns:
        if isinstance(s, str) and s.strip():
            return s.strip()
    return ""


def node_to_text(node: dict, mode: str = "label_def") -> str:
    """Convert a node record to an indexed text string.

    Modes are designed to support both a strict synonym-only stress test
    (no synonym indexing) and more realistic entity-normalization settings
    where at least one synonym is indexed.
    """
    name = (node.get("name", "") or "").strip()
    definition = (node.get("definition", "") or "").strip()
    syn1 = _first_synonym(node)

    if mode == "label":
        return name

    if mode == "label_def":
        if definition:
            return f"{name}. {definition}"
        return name

    if mode == "label_def_1syn":
        # Index the preferred label + one definition sentence + one synonym.
        base = f"{name}. {definition}" if definition else name
        if syn1:
            return f"{base}. Synonym: {syn1}"
        return base

    if mode == "label_def_all_syn":
        base = f"{name}. {definition}" if definition else name
        syns = [s.strip() for s in (node.get("synonyms", []) or []) if isinstance(s, str) and s.strip()]
        if syns:
            # Avoid exploding text length for very synonym-rich nodes.
            syns = syns[:10]
            return f"{base}. Synonyms: {'; '.join(syns)}"
        return base

    raise ValueError(f"Unknown node_text_mode={mode}")


def encode_jsonl_texts(jsonl_path: Path, text_field: str = "text"):
    items = read_jsonl(jsonl_path)
    texts = [x[text_field] for x in items]
    qids = [x.get("qid", str(i)) for i, x in enumerate(items)]
    return items, texts, qids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do", "mesh"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")
    ap.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--node_text_mode",
        type=str,
        default="label_def",
        choices=["label", "label_def", "label_def_1syn", "label_def_all_syn"],
        help="How to build indexed node text (optionally include synonyms).",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    ensure_dir(root)

    nodes = read_jsonl(root / "nodes.jsonl")
    node_texts = [node_to_text(n, mode=args.node_text_mode) for n in nodes]

    embedder = TextEmbedder(TextEmbedderConfig(model_name=args.model_name, device=args.device, batch_size=args.batch_size, normalize=True))

    print(f"[encode] nodes={len(node_texts)}")
    emb_nodes = embedder.encode(node_texts)
    save_npy(emb_nodes, root / "emb_nodes.npy")

    for split in ["train", "val", "test"]:
        q_items, q_texts, qids = encode_jsonl_texts(root / f"queries_{split}.jsonl")
        print(f"[encode] queries_{split}={len(q_texts)}")
        emb_q = embedder.encode(q_texts)
        save_npy(emb_q, root / f"emb_queries_{split}.npy")
        # store qids for joining
        with open(root / f"query_ids_{split}.txt", "w", encoding="utf-8") as f:
            for qid in qids:
                f.write(qid + "\n")

    print(f"[done] embeddings saved under {root}")


if __name__ == "__main__":
    main()
