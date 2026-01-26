from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..ontology.graph import get_ancestors, get_siblings


QE_TEMPLATES = [
    "{label}",
    "{syn}",
    "{label} (medical term)",
    "definition: {definition}",
]

QH_PARENT_TEMPLATES = [
    "What is the parent category of {label}?",
    "Which broader term does {label} belong to?",
    "In an is-a taxonomy, what is {label} a type of?",
    "What are the superclasses of {label}?",
]

QM_SIBLING_TEMPLATES = [
    "conditions similar to {label} at the same specificity level",
    "phenotypes comparable to {label}",
    "sibling concepts of {label} in the ontology",
]


def _pick_nonempty(xs: Sequence[str]) -> Optional[str]:
    for x in xs:
        if x and x.strip():
            return x.strip()
    return None


def generate_qe_queries(
    nodes: Dict[str, dict],
    split_ids: Sequence[str],
    max_per_node: int = 2,
    seed: int = 0,
) -> List[dict]:
    """Entity-centric queries: map a surface string to the target node.
    
    Note: Unlike Q-H and Q-M, Q-E queries do NOT set focus_id because
    the goal is to retrieve the target node itself (not exclude it).
    """
    rnd = random.Random(seed)
    out: List[dict] = []
    for nid in split_ids:
        meta = nodes[nid]
        label = meta.get("name", "")
        definition = meta.get("definition", "")
        syns = [s for s in meta.get("synonyms", []) if s and len(s) >= 3]
        cand = [label] + syns
        rnd.shuffle(cand)
        cand = cand[:max_per_node]
        for j, s in enumerate(cand):
            text = s
            qid = f"QE:{nid}:{j}"
            out.append(
                {
                    "qid": qid,
                    "type": "QE",
                    "text": text,
                    "pos_ids": [nid],
                    # No focus_id for Q-E: we want to find the target node, not exclude it
                }
            )
    return out


def generate_qh_queries(
    nodes: Dict[str, dict],
    split_ids: Sequence[str],
    parents: Dict[str, List[str]],
    seed: int = 0,
    max_templates: int = 1,
) -> List[dict]:
    """Hierarchy-navigation queries: ask for parents/ancestors of a node."""
    rnd = random.Random(seed)
    out: List[dict] = []
    for nid in split_ids:
        ps = parents.get(nid, [])
        if not ps:
            continue
        meta = nodes[nid]
        label = meta.get("name", "")
        if not label:
            continue
        templates = list(QH_PARENT_TEMPLATES)
        rnd.shuffle(templates)
        templates = templates[:max_templates]
        anc = sorted(list(get_ancestors(nid, parents)))
        for j, tpl in enumerate(templates):
            text = tpl.format(label=label)
            qid = f"QH:{nid}:{j}"
            out.append(
                {
                    "qid": qid,
                    "type": "QH",
                    "text": text,
                    "pos_ids": list(ps),          # immediate parents for Hits@k
                    "anc_ids": anc,               # full ancestors for F1
                    "focus_id": nid,              # exclude the child itself at evaluation time
                }
            )
    return out


def generate_qm_queries(
    nodes: Dict[str, dict],
    split_ids: Sequence[str],
    parents: Dict[str, List[str]],
    children: Dict[str, List[str]],
    depth: Dict[str, int],
    seed: int = 0,
    max_templates: int = 1,
    min_siblings: int = 1,
    max_pos: int = 20,
) -> List[dict]:
    """Mixed-intent sibling-style queries.

    We treat siblings (nodes sharing a parent) as the ground-truth set.
    This is a proxy for ``same specificity level'' queries in ontology grounding.
    """
    rnd = random.Random(seed)
    out: List[dict] = []
    for nid in split_ids:
        sib = sorted(list(get_siblings(nid, parents, children)))
        if len(sib) < min_siblings:
            continue
        meta = nodes[nid]
        label = meta.get("name", "")
        if not label:
            continue
        templates = list(QM_SIBLING_TEMPLATES)
        rnd.shuffle(templates)
        templates = templates[:max_templates]
        # sample a bounded positive set for evaluation stability
        rnd.shuffle(sib)
        sib = sib[:max_pos]
        for j, tpl in enumerate(templates):
            text = tpl.format(label=label)
            qid = f"QM:{nid}:{j}"
            out.append(
                {
                    "qid": qid,
                    "type": "QM",
                    "text": text,
                    "pos_ids": sib,
                    "depth": int(depth.get(nid, 0)),
                    "focus_id": nid,
                }
            )
    return out


def split_nodes_by_seed(node_ids: Sequence[str], seed: int = 0, train: float = 0.8, val: float = 0.1) -> Dict[str, List[str]]:
    """Split node IDs into train/val/test by shuffling."""
    assert 0 < train < 1 and 0 < val < 1 and train + val < 1
    rnd = random.Random(seed)
    ids = list(node_ids)
    rnd.shuffle(ids)
    n = len(ids)
    n_train = int(train * n)
    n_val = int(val * n)
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }
