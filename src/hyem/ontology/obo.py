from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_SYN_RE = re.compile(r'^synonym:\s*"(?P<text>.*?)"\s+(?P<scope>\w+)\s*(\[(?P<xrefs>.*?)\])?')


@dataclass
class Term:
    tid: str
    name: str = ""
    definition: str = ""
    synonyms: List[str] = field(default_factory=list)
    alt_ids: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    is_obsolete: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.tid,
            "name": self.name,
            "definition": self.definition,
            "synonyms": list(self.synonyms),
            "alt_ids": list(self.alt_ids),
            "parents": list(self.parents),
        }


def parse_obo(path: str | Path) -> Tuple[Dict[str, Term], Dict[str, str]]:
    """Parse an OBO file into a dict of Terms.

    Returns
    -------
    terms: dict term_id -> Term
    alt_id_map: dict alt_id -> primary_id
    """
    path = Path(path)
    terms: Dict[str, Term] = {}
    alt_id_map: Dict[str, str] = {}

    cur: Optional[Term] = None

    def _commit() -> None:
        nonlocal cur
        if cur is None:
            return
        if cur.tid and (not cur.is_obsolete):
            terms[cur.tid] = cur
            for a in cur.alt_ids:
                alt_id_map[a] = cur.tid
        cur = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line == "[Term]":
                _commit()
                cur = Term(tid="")
                continue

            if line.startswith("[") and line != "[Term]":
                # [Typedef] or other blocks; commit current term and ignore until next [Term]
                _commit()
                cur = None
                continue

            if cur is None:
                continue

            if line.startswith("id:"):
                cur.tid = line.split("id:", 1)[1].strip()
                continue
            if line.startswith("name:"):
                cur.name = line.split("name:", 1)[1].strip()
                continue
            if line.startswith("def:"):
                # def: "text" [xrefs]
                val = line.split("def:", 1)[1].strip()
                if val.startswith('"'):
                    # extract content between first pair of quotes
                    end = val.find('"', 1)
                    if end != -1:
                        cur.definition = val[1:end]
                    else:
                        cur.definition = val.strip('"')
                else:
                    cur.definition = val
                continue
            if line.startswith("synonym:"):
                m = _SYN_RE.match(line)
                if m:
                    cur.synonyms.append(m.group("text"))
                else:
                    # fallback: keep raw content between first quotes
                    if '"' in line:
                        q1 = line.find('"')
                        q2 = line.find('"', q1 + 1)
                        if q1 != -1 and q2 != -1:
                            cur.synonyms.append(line[q1 + 1 : q2])
                continue
            if line.startswith("alt_id:"):
                cur.alt_ids.append(line.split("alt_id:", 1)[1].strip())
                continue
            if line.startswith("is_a:"):
                # is_a: HP:0000002 ! comment
                pid = line.split("is_a:", 1)[1].strip().split()[0]
                cur.parents.append(pid)
                continue
            if line.startswith("is_obsolete:"):
                cur.is_obsolete = ("true" in line.lower())
                continue

    _commit()
    return terms, alt_id_map


def canonicalize_id(tid: str, alt_id_map: Dict[str, str]) -> str:
    return alt_id_map.get(tid, tid)


def build_edges(terms: Dict[str, Term], alt_id_map: Dict[str, str]) -> List[Tuple[str, str]]:
    """Return parent->child edges for is_a relations, resolving alt_ids."""
    edges: List[Tuple[str, str]] = []
    for tid, t in terms.items():
        child = canonicalize_id(tid, alt_id_map)
        for p in t.parents:
            parent = canonicalize_id(p, alt_id_map)
            if parent == child:
                continue
            if parent in terms and child in terms:
                edges.append((parent, child))
    # remove duplicates deterministically
    edges = sorted(set(edges))
    return edges
