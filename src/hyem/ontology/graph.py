from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple


def build_parent_child_maps(edges: Iterable[Tuple[str, str]]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build parent and child adjacency lists from parent->child edges."""
    parents: Dict[str, List[str]] = defaultdict(list)
    children: Dict[str, List[str]] = defaultdict(list)
    for p, c in edges:
        parents[c].append(p)
        children[p].append(c)
    # sort for determinism
    for k in list(parents.keys()):
        parents[k] = sorted(set(parents[k]))
    for k in list(children.keys()):
        children[k] = sorted(set(children[k]))
    return dict(parents), dict(children)


def find_roots(nodes: Iterable[str], parents: Dict[str, List[str]]) -> List[str]:
    nodes = list(nodes)
    roots = [n for n in nodes if len(parents.get(n, [])) == 0]
    return sorted(roots)


def compute_min_depths(roots: List[str], children: Dict[str, List[str]]) -> Dict[str, int]:
    """Compute minimum depth from any root (BFS on DAG treated as graph)."""
    depth: Dict[str, int] = {}
    q = deque()
    for r in roots:
        depth[r] = 0
        q.append(r)
    while q:
        u = q.popleft()
        for v in children.get(u, []):
            nd = depth[u] + 1
            if v not in depth or nd < depth[v]:
                depth[v] = nd
                q.append(v)
    return depth


def sample_subset_bfs(
    roots: List[str],
    children: Dict[str, List[str]],
    target_size: int,
    seed: int = 0,
) -> List[str]:
    """Deterministically sample a connected-ish subset by BFS expansion.

    Notes
    -----
    - For ontologies with multiple roots, we start from all roots.
    - This keeps experiments lightweight and preserves depth growth patterns.
    """
    import random

    rnd = random.Random(seed)
    start = list(roots)
    rnd.shuffle(start)

    seen: Set[str] = set()
    q = deque(start)
    order: List[str] = []
    while q and len(order) < target_size:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        order.append(u)
        nbrs = list(children.get(u, []))
        rnd.shuffle(nbrs)
        for v in nbrs:
            if v not in seen:
                q.append(v)

    return order


def induced_edges(edges: Iterable[Tuple[str, str]], keep_nodes: Set[str]) -> List[Tuple[str, str]]:
    out = [(p, c) for (p, c) in edges if p in keep_nodes and c in keep_nodes]
    return sorted(set(out))


def avg_branching(children: Dict[str, List[str]], nodes: Set[str]) -> float:
    counts = [len([c for c in children.get(n, []) if c in nodes]) for n in nodes]
    return float(sum(counts) / max(1, len(counts)))


def get_ancestors(node: str, parents: Dict[str, List[str]]) -> Set[str]:
    anc: Set[str] = set()
    stack = list(parents.get(node, []))
    while stack:
        p = stack.pop()
        if p in anc:
            continue
        anc.add(p)
        stack.extend(parents.get(p, []))
    return anc


def get_siblings(node: str, parents: Dict[str, List[str]], children: Dict[str, List[str]]) -> Set[str]:
    sib: Set[str] = set()
    for p in parents.get(node, []):
        for c in children.get(p, []):
            if c != node:
                sib.add(c)
    return sib


@dataclass
class GraphStats:
    n_nodes: int
    n_edges: int
    max_depth: int
    avg_branching: float


def compute_stats(nodes: Set[str], edges: List[Tuple[str, str]], parents: Dict[str, List[str]], children: Dict[str, List[str]]) -> GraphStats:
    roots = find_roots(nodes, parents)
    depth = compute_min_depths(roots, children)
    max_depth = max(depth.values()) if depth else 0
    return GraphStats(
        n_nodes=len(nodes),
        n_edges=len(edges),
        max_depth=max_depth,
        avg_branching=avg_branching(children, nodes),
    )
