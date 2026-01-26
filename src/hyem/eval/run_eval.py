from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .metrics import compute_qe_metrics, compute_qh_metrics, hits_at_k, reciprocal_rank
from ..retrieval.retriever import (
    RetrievalConfig,
    retrieve_euclidean_text,
    retrieve_euclidean_kg,
    retrieve_hyem_hyperbolic,
    retrieve_hard_route,
    retrieve_soft_mix,
)


def _as_set(x) -> Set[str]:
    return set(list(x))


def evaluate_method(
    queries: List[dict],
    method_name: str,
    node_ids: Sequence[str],
    e_nodes: np.ndarray,
    u_nodes: Optional[np.ndarray],
    z_nodes: Optional[np.ndarray],
    idx_E,
    idx_H,
    idx_Z,
    adapter_hyp=None,
    adapter_euc=None,
    gate_scores: Optional[Dict[str, float]] = None,
    cfg: Optional[RetrievalConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate one method over a mixed query set.

    Returns
    -------
    per_query_df: DataFrame with per-query metrics for significance tests
    summary: dict aggregated metrics for tables
    """
    if cfg is None:
        cfg = RetrievalConfig()

    ranks_qe: List[List[str]] = []
    gts_qe: List[Set[str]] = []

    ranks_qh: List[List[str]] = []
    parents_qh: List[Set[str]] = []
    anc_qh: List[Set[str]] = []

    ranks_qm: List[List[str]] = []
    gts_qm: List[Set[str]] = []

    per_rows = []

    for q in queries:
        qid = q["qid"]
        qtype = q["type"]
        e_q = q["e_q"]  # pre-encoded
        focus = q.get("focus_id", None)
        exclude = {focus} if focus else None

        if method_name == "euclid_text":
            ranked = retrieve_euclidean_text(e_q, e_nodes, idx_E, node_ids, cfg, exclude=exclude)
        elif method_name == "euclid_kg":
            ranked = retrieve_euclidean_kg(e_q, adapter_euc, z_nodes, idx_Z, node_ids, cfg, exclude=exclude)
        elif method_name == "hyp_noR":
            ranked = retrieve_hyem_hyperbolic(e_q, adapter_hyp, u_nodes, idx_H, node_ids, cfg, exclude=exclude)
        elif method_name == "hyem_no_gate":
            ranked = retrieve_hyem_hyperbolic(e_q, adapter_hyp, u_nodes, idx_H, node_ids, cfg, exclude=exclude)
        elif method_name == "hyem_hard":
            alpha = float(gate_scores.get(qid, 1.0)) if gate_scores else 1.0
            ranked = retrieve_hard_route(
                e_q=e_q,
                adapter_hyp=adapter_hyp,
                adapter_euc=adapter_euc,
                gate_score=alpha,
                e_nodes=e_nodes,
                u_nodes=u_nodes,
                z_nodes=z_nodes,
                idx_E=idx_E,
                idx_H=idx_H,
                idx_Z=idx_Z,
                node_ids=node_ids,
                cfg=cfg,
                exclude=exclude,
            )
        elif method_name == "hyem_soft":
            alpha = float(gate_scores.get(qid, 1.0)) if gate_scores else 1.0
            ranked = retrieve_soft_mix(
                text=q.get("text",""),
                e_q=e_q,
                adapter_hyp=adapter_hyp,
                gate_score=alpha,
                e_nodes=e_nodes,
                u_nodes=u_nodes,
                idx_E=idx_E,
                idx_H=idx_H,
                node_ids=node_ids,
                cfg=cfg,
                exclude=exclude,
            )
        else:
            raise ValueError(f"Unknown method {method_name}")

        gt = _as_set(q["pos_ids"])

        if qtype == "QE":
            ranks_qe.append(ranked)
            gts_qe.append(gt)
            rr = reciprocal_rank(ranked, gt)
            per_rows.append({"qid": qid, "type": qtype, "rr": rr})
        elif qtype == "QH":
            ranks_qh.append(ranked)
            parents_qh.append(gt)
            anc_qh.append(_as_set(q.get("anc_ids", [])))
            rr = reciprocal_rank(ranked, gt)  # RR of first parent
            per_rows.append({"qid": qid, "type": qtype, "rr": rr})
        elif qtype == "QM":
            ranks_qm.append(ranked)
            gts_qm.append(gt)
            rr = reciprocal_rank(ranked, gt)
            per_rows.append({"qid": qid, "type": qtype, "rr": rr})
        else:
            continue

    summary: Dict[str, float] = {}
    if ranks_qe:
        m = compute_qe_metrics(ranks_qe, gts_qe)
        summary.update(
            {
                "QE_hits1": m.hits1,
                "QE_hits10": m.hits10,
                "QE_mrr": m.mrr,
                "QE_ndcg10": m.ndcg10,
            }
        )
    if ranks_qh:
        mh = compute_qh_metrics(ranks_qh, parents_qh, anc_qh, anc_k=cfg.anc_k)
        summary.update(
            {
                "QH_parent_hits5": mh.parent_hits5,
                "QH_parent_hits10": mh.parent_hits10,
                "QH_anc_f1_macro": mh.anc_f1_macro,
                "QH_anc_f1_micro": mh.anc_f1_micro,
            }
        )
    if ranks_qm:
        # for QM we report Hits@10 and MRR as diagnostic
        summary.update(
            {
                "QM_hits10": float(np.mean([hits_at_k(r, gt, 10) for r, gt in zip(ranks_qm, gts_qm)])),
                "QM_mrr": float(np.mean([reciprocal_rank(r, gt) for r, gt in zip(ranks_qm, gts_qm)])),
            }
        )

    per_df = pd.DataFrame(per_rows)
    return per_df, summary
