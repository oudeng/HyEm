from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def hits_at_k(ranked: Sequence[str], gt: Set[str], k: int) -> float:
    topk = ranked[:k]
    return 1.0 if any(x in gt for x in topk) else 0.0


def reciprocal_rank(ranked: Sequence[str], gt: Set[str]) -> float:
    for i, x in enumerate(ranked):
        if x in gt:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gt: Set[str], k: int) -> float:
    # binary relevance
    dcg = 0.0
    for i, x in enumerate(ranked[:k]):
        if x in gt:
            dcg += 1.0 / np.log2(i + 2)
    # ideal DCG
    ideal_hits = min(len(gt), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return float(dcg / idcg) if idcg > 0 else 0.0


def f1_from_sets(pred: Set[str], gt: Set[str]) -> Tuple[float, float, float]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


@dataclass
class QEMetrics:
    hits1: float
    hits10: float
    mrr: float
    ndcg10: float


def compute_qe_metrics(ranks: List[Sequence[str]], gts: List[Set[str]]) -> QEMetrics:
    assert len(ranks) == len(gts)
    h1 = np.mean([hits_at_k(r, gt, 1) for r, gt in zip(ranks, gts)])
    h10 = np.mean([hits_at_k(r, gt, 10) for r, gt in zip(ranks, gts)])
    mrr = np.mean([reciprocal_rank(r, gt) for r, gt in zip(ranks, gts)])
    ndcg10 = np.mean([ndcg_at_k(r, gt, 10) for r, gt in zip(ranks, gts)])
    return QEMetrics(float(h1), float(h10), float(mrr), float(ndcg10))


@dataclass
class QHMetrics:
    parent_hits5: float
    parent_hits10: float
    anc_f1_macro: float
    anc_f1_micro: float


def compute_qh_metrics(
    ranks: List[Sequence[str]],
    parent_sets: List[Set[str]],
    anc_sets: List[Set[str]],
    anc_k: int = 50,
) -> QHMetrics:
    assert len(ranks) == len(parent_sets) == len(anc_sets)
    ph5 = np.mean([hits_at_k(r, gt, 5) for r, gt in zip(ranks, parent_sets)])
    ph10 = np.mean([hits_at_k(r, gt, 10) for r, gt in zip(ranks, parent_sets)])

    # macro F1
    f1s = []
    tp_all = fp_all = fn_all = 0
    for r, anc in zip(ranks, anc_sets):
        pred = set([x for x in r[:anc_k] if x in anc])
        # predicted set for F1 should be the retrieved ancestors; count FP as retrieved non-ancestors
        pred_all = set(r[:anc_k])
        tp = len(pred_all & anc)
        fp = len(pred_all - anc)
        fn = len(anc - pred_all)
        tp_all += tp
        fp_all += fp
        fn_all += fn
        _, _, f1 = f1_from_sets(pred_all, anc)
        f1s.append(f1)

    macro = float(np.mean(f1s)) if f1s else 0.0
    prec_micro = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    rec_micro = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    micro = (2 * prec_micro * rec_micro / (prec_micro + rec_micro)) if (prec_micro + rec_micro) > 0 else 0.0
    return QHMetrics(float(ph5), float(ph10), float(macro), float(micro))


@dataclass
class GateMetrics:
    accuracy: float
    precision_qh: float
    recall_qh: float
    auc: float


def compute_gate_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> GateMetrics:
    y_pred = (y_score >= threshold).astype(int)
    acc = float(np.mean(y_pred == y_true))
    # precision/recall for positive class (QH)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = 0.0
    return GateMetrics(acc, float(prec), float(rec), auc)
