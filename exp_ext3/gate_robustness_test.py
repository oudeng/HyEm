#!/usr/bin/env python
"""
Gate Robustness Test (M3)

Test gate calibration on queries with added perturbations:
1. Typos - simulate realistic user input errors
2. Paraphrase - rephrase queries while preserving intent
3. Ambiguous - queries that could be interpreted as Q-E or Q-H

This addresses reviewer concern that "gate classification is too easy"
on template-generated queries.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_typos(text: str, typo_rate: float = 0.1, seed: int = 42) -> str:
    """Add random typos to text."""
    rnd = random.Random(seed)
    chars = list(text)
    for i in range(len(chars)):
        if rnd.random() < typo_rate and chars[i].isalpha():
            if rnd.random() < 0.5:
                # Swap adjacent characters
                if i < len(chars) - 1:
                    chars[i], chars[i+1] = chars[i+1], chars[i]
            else:
                # Replace with nearby key
                nearby = {'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr',
                         'f': 'dg', 'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk',
                         'k': 'jl', 'l': 'k', 'm': 'n', 'n': 'bm', 'o': 'ip',
                         'p': 'o', 'q': 'wa', 'r': 'et', 's': 'ad', 't': 'ry',
                         'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc', 'y': 'tu', 'z': 'x'}
                c = chars[i].lower()
                if c in nearby:
                    chars[i] = rnd.choice(nearby[c])
    return ''.join(chars)


def paraphrase_qh_query(text: str, seed: int = 42) -> str:
    """Paraphrase Q-H query to make it less obviously hierarchical."""
    rnd = random.Random(seed)
    
    # Remove explicit hierarchy keywords
    paraphrases = [
        (r"What is the parent category of", "What would you call"),
        (r"Which broader term does", "What category includes"),
        (r"what is .* a type of", "how would you classify"),
        (r"What are the superclasses of", "What encompasses"),
        (r"parent of", "category for"),
        (r"subtypes of", "specific kinds of"),
        (r"broader term", "general category"),
    ]
    
    result = text
    for pattern, replacement in paraphrases:
        import re
        if re.search(pattern, text, re.IGNORECASE):
            result = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            break
    
    return result


def create_ambiguous_query(text: str, query_type: str, seed: int = 42) -> Tuple[str, str]:
    """Create queries that are ambiguous between Q-E and Q-H."""
    rnd = random.Random(seed)
    
    # Extract the entity name from the query
    import re
    match = re.search(r'"([^"]+)"|\{([^}]+)\}|of (\w+[\w\s]*?)[\?\.]', text)
    if match:
        entity = match.group(1) or match.group(2) or match.group(3)
        entity = entity.strip()
    else:
        entity = text
    
    # Create ambiguous versions
    ambiguous_templates = [
        f"Tell me about {entity}",
        f"What can you say about {entity}",
        f"{entity} - explain",
        f"I need information on {entity}",
        f"Describe {entity}",
    ]
    
    return rnd.choice(ambiguous_templates), 'ambiguous'


def load_gate_and_embeddings(root: Path, device: str = 'cpu'):
    """Load trained gate and pre-computed query embeddings."""
    import torch
    from hyem.models.gate import LinearGate
    
    # Load gate
    gate_path = root / "gate_linear.pt"
    if not gate_path.exists():
        return None, None, None, None
    
    # Load pre-computed node embeddings to get dimension
    emb_nodes_path = root / "emb_nodes.npy"
    if not emb_nodes_path.exists():
        return None, None, None, None
    
    e_nodes = np.load(emb_nodes_path)
    in_dim = e_nodes.shape[1]
    
    # Load pre-computed test query embeddings
    emb_queries_path = root / "emb_queries_test.npy"
    query_ids_path = root / "query_ids_test.txt"
    
    if not emb_queries_path.exists() or not query_ids_path.exists():
        return None, None, None, None
    
    emb_queries = np.load(emb_queries_path).astype(np.float32)
    with open(query_ids_path, "r", encoding="utf-8") as f:
        query_ids = [line.strip() for line in f if line.strip()]
    
    # Create qid -> embedding mapping
    qid_to_emb = {qid: emb for qid, emb in zip(query_ids, emb_queries)}
    
    gate = LinearGate(in_dim=in_dim).to(device)
    gate.load_state_dict(torch.load(gate_path, map_location=device))
    gate.eval()
    
    return gate, in_dim, device, qid_to_emb


def predict_gate_scores(gate, embeddings: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """Predict gate scores for embeddings."""
    import torch
    
    with torch.no_grad():
        X = torch.from_numpy(embeddings).to(device)
        scores = torch.sigmoid(gate(X)).cpu().numpy().flatten()
    return scores


def evaluate_gate_on_original(queries: List[dict], gate, qid_to_emb: dict,
                               device: str = 'cpu') -> dict:
    """Evaluate gate on original (unperturbed) queries using pre-computed embeddings."""
    
    embeddings = []
    labels = []  # 1 = Q-H, 0 = Q-E
    valid_qids = []
    
    for q in queries:
        qid = q['qid']
        qtype = q['type']
        
        if qtype not in ('QE', 'QH'):
            continue
        
        if qid not in qid_to_emb:
            continue
        
        embeddings.append(qid_to_emb[qid])
        labels.append(1 if qtype == 'QH' else 0)
        valid_qids.append(qid)
    
    if not embeddings:
        return {'perturbation': 'original', 'accuracy': 0, 'n_samples': 0}
    
    embeddings = np.stack(embeddings, axis=0)
    scores = predict_gate_scores(gate, embeddings, device)
    
    # Evaluate (threshold at 0.5)
    predictions = (scores > 0.5).astype(int)
    labels = np.array(labels)
    
    accuracy = np.mean(predictions == labels)
    
    # Compute precision/recall for Q-H class
    qh_mask = labels == 1
    if qh_mask.sum() > 0:
        precision_qh = np.sum((predictions == 1) & (labels == 1)) / max(np.sum(predictions == 1), 1)
        recall_qh = np.sum((predictions == 1) & (labels == 1)) / np.sum(labels == 1)
    else:
        precision_qh = recall_qh = 0
    
    return {
        'perturbation': 'original',
        'accuracy': accuracy,
        'precision_qh': precision_qh,
        'recall_qh': recall_qh,
        'n_samples': len(labels),
        'n_qh': int(qh_mask.sum()),
        'n_qe': int((~qh_mask).sum()),
    }


def evaluate_gate_with_noise(queries: List[dict], gate, qid_to_emb: dict,
                              noise_level: float, device: str = 'cpu') -> dict:
    """Evaluate gate on queries with Gaussian noise added to embeddings."""
    
    embeddings = []
    labels = []
    
    for q in queries:
        qid = q['qid']
        qtype = q['type']
        
        if qtype not in ('QE', 'QH'):
            continue
        
        if qid not in qid_to_emb:
            continue
        
        # Add Gaussian noise to simulate encoding variation
        emb = qid_to_emb[qid].copy()
        noise = np.random.randn(*emb.shape).astype(np.float32) * noise_level
        emb = emb + noise
        # Renormalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        
        embeddings.append(emb)
        labels.append(1 if qtype == 'QH' else 0)
    
    if not embeddings:
        return {'perturbation': f'noise_{noise_level}', 'accuracy': 0, 'n_samples': 0}
    
    embeddings = np.stack(embeddings, axis=0)
    scores = predict_gate_scores(gate, embeddings, device)
    
    predictions = (scores > 0.5).astype(int)
    labels = np.array(labels)
    
    accuracy = np.mean(predictions == labels)
    
    qh_mask = labels == 1
    if qh_mask.sum() > 0:
        precision_qh = np.sum((predictions == 1) & (labels == 1)) / max(np.sum(predictions == 1), 1)
        recall_qh = np.sum((predictions == 1) & (labels == 1)) / np.sum(labels == 1)
    else:
        precision_qh = recall_qh = 0
    
    return {
        'perturbation': f'noise_{noise_level:.2f}',
        'accuracy': accuracy,
        'precision_qh': precision_qh,
        'recall_qh': recall_qh,
        'n_samples': len(labels),
        'n_qh': int(qh_mask.sum()),
        'n_qe': int((~qh_mask).sum()),
    }


def plot_robustness_results(results_df: pd.DataFrame, dataset: str, out_path: Path):
    """Plot gate robustness comparison."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(results_df))
    width = 0.25
    
    ax.bar(x - width, results_df['accuracy'], width, label='Accuracy', color='steelblue')
    ax.bar(x, results_df['precision_qh'], width, label='Precision (Q-H)', color='coral')
    ax.bar(x + width, results_df['recall_qh'], width, label='Recall (Q-H)', color='seagreen')
    
    ax.set_ylabel('Score')
    ax.set_xlabel('Perturbation Type')
    ax.set_title(f'Gate Robustness Test: {dataset.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['perturbation'], rotation=15)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add sample size annotations
    for i, row in results_df.iterrows():
        ax.text(i, 1.02, f'n={row["n_samples"]}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--noise_levels", type=str, default="0.0,0.1,0.2,0.3",
                   help="Comma-separated noise levels to test (simulates encoding variation)")
    args = ap.parse_args()
    
    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    
    # Load gate and pre-computed embeddings
    gate, in_dim, device, qid_to_emb = load_gate_and_embeddings(root, args.device)
    
    if gate is None:
        print(f"Warning: Gate or embeddings not found at {root}. Skipping robustness test.")
        print("Required files: gate_linear.pt, emb_nodes.npy, emb_queries_test.npy, query_ids_test.txt")
        return
    
    print(f"Loaded gate with input dim={in_dim}, {len(qid_to_emb)} query embeddings")
    
    # Load test queries
    queries = []
    with open(root / "queries_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    
    # Run robustness tests
    results = []
    
    # 1. Original (no perturbation)
    print("Testing: original (no perturbation)")
    result = evaluate_gate_on_original(queries, gate, qid_to_emb, device)
    results.append(result)
    print(f"  Accuracy: {result['accuracy']:.3f}, Precision: {result['precision_qh']:.3f}, "
          f"Recall: {result['recall_qh']:.3f}")
    
    # 2. Test with different noise levels (simulates encoding variation from typos/paraphrases)
    noise_levels = [float(x) for x in args.noise_levels.split(',') if float(x) > 0]
    for noise_level in noise_levels:
        print(f"Testing: noise={noise_level} (simulates encoding variation)")
        result = evaluate_gate_with_noise(queries, gate, qid_to_emb, noise_level, device)
        results.append(result)
        print(f"  Accuracy: {result['accuracy']:.3f}, Precision: {result['precision_qh']:.3f}, "
              f"Recall: {result['recall_qh']:.3f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    out_path = root / "results" / "gate_robustness.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    
    # Create visualization
    fig_dir = root / "analysis" / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_robustness_results(results_df, args.dataset, fig_dir / "gate_robustness_bar.pdf")
    
    # Print summary
    print(f"\n=== Gate Robustness Summary ({args.dataset.upper()}) ===")
    print(results_df.to_string(index=False))
    
    # Key findings
    original = results_df[results_df['perturbation'] == 'original'].iloc[0]
    
    print(f"\n=== Key Findings ===")
    print(f"Original accuracy: {original['accuracy']:.1%}")
    
    if len(noise_levels) > 0:
        # Find max noise that maintains >90% of original accuracy
        for noise_level in sorted(noise_levels):
            noisy = results_df[results_df['perturbation'] == f'noise_{noise_level:.2f}']
            if len(noisy) > 0:
                degradation = original['accuracy'] - noisy.iloc[0]['accuracy']
                print(f"Noise={noise_level:.2f}: degradation={degradation:.1%}")
                if degradation < 0.05:
                    print(f"  ✓ Gate is robust to noise level {noise_level}")
                else:
                    print(f"  ⚠ Gate degrades significantly at noise level {noise_level}")


if __name__ == "__main__":
    main()