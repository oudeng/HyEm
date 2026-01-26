#!/usr/bin/env python
"""
Theoretical Scaling Analysis (M1)

Plot theoretical bounds to show:
1. κ(R) = sinh(R)/R growth as a function of R
2. Required R for ontology depth D at different dimensions d
3. "Safe operating regime" where κ(R) < threshold

This addresses reviewer concern about scalability to larger ontologies
by showing the theoretical limits of tangent-space indexing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def kappa(R: np.ndarray) -> np.ndarray:
    """Compute κ(R) = sinh(R)/R (tangent-space distortion factor)."""
    # Handle R=0 case
    result = np.where(R > 1e-10, np.sinh(R) / R, 1.0)
    return result


def R_required(D: int, b: float, d: int, epsilon: float = 0.1) -> float:
    """
    Compute required radius R for depth-D, branching-b tree in d dimensions.
    
    From Proposition 3: R ≳ (D log b) / (d - 1) - O(log(1/ε)/(d-1))
    
    We use a simplified version: R ≈ (D log b) / (d - 1)
    """
    return (D * np.log(b)) / (d - 1)


def plot_kappa_vs_R(out_path: Path):
    """Plot κ(R) growth to show distortion behavior."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: κ(R) on linear scale
    R = np.linspace(0, 7, 200)
    kR = kappa(R)
    
    ax1 = axes[0]
    ax1.plot(R, kR, 'b-', linewidth=2, label=r'$\kappa(R) = \sinh(R)/R$')
    ax1.axhline(y=2, color='orange', linestyle='--', label=r'$\kappa(R) = 2$ (safe threshold)')
    ax1.axhline(y=5, color='red', linestyle=':', label=r'$\kappa(R) = 5$ (danger zone)')
    
    # Shade safe region
    ax1.fill_between(R, 0, kR, where=(kR <= 2), alpha=0.2, color='green', label='Safe region')
    ax1.fill_between(R, 0, kR, where=(kR > 5), alpha=0.2, color='red', label='Danger zone')
    
    ax1.set_xlabel('Radius R', fontsize=11)
    ax1.set_ylabel(r'Distortion factor $\kappa(R)$', fontsize=11)
    ax1.set_title('Tangent-Space Distortion Growth', fontsize=12)
    ax1.set_xlim(0, 7)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Mark key points
    for R_val in [2, 3, 4, 5]:
        k_val = kappa(np.array([R_val]))[0]
        ax1.plot(R_val, k_val, 'ko', markersize=6)
        ax1.annotate(f'R={R_val}\nκ={k_val:.1f}', 
                    xy=(R_val, k_val), xytext=(R_val+0.3, k_val+10),
                    fontsize=8)
    
    # Right: κ(R) on log scale to show exponential growth
    ax2 = axes[1]
    ax2.semilogy(R, kR, 'b-', linewidth=2)
    ax2.axhline(y=2, color='orange', linestyle='--')
    ax2.axhline(y=5, color='red', linestyle=':')
    
    # Add asymptotic approximation
    R_large = R[R > 1]
    asymptotic = np.exp(R_large) / (2 * R_large)
    ax2.semilogy(R_large, asymptotic, 'g--', linewidth=1.5, 
                label=r'Asymptotic: $e^R / (2R)$')
    
    ax2.set_xlabel('Radius R', fontsize=11)
    ax2.set_ylabel(r'$\kappa(R)$ (log scale)', fontsize=11)
    ax2.set_title('Distortion Growth (Log Scale)', fontsize=12)
    ax2.set_xlim(0, 7)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_R_vs_depth(out_path: Path, max_depth: int = 50, dimensions: list = None):
    """Plot required R as function of depth for different dimensions."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    depths = np.arange(1, max_depth + 1)
    if dimensions is None:
        dimensions = [16, 32, 64, 128]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(dimensions)))
    
    # Left: R required vs depth (for b=2, typical binary tree)
    ax1 = axes[0]
    b = 2  # binary branching
    
    for d, color in zip(dimensions, colors):
        R_vals = [R_required(D, b, d) for D in depths]
        ax1.plot(depths, R_vals, '-', color=color, linewidth=2, label=f'd={d}')
    
    # Add safe threshold lines
    ax1.axhline(y=3.0, color='orange', linestyle='--', label='R=3 (κ≈3.3)')
    ax1.axhline(y=5.0, color='red', linestyle=':', label='R=5 (κ≈29.5)')
    
    ax1.set_xlabel('Ontology Depth D', fontsize=11)
    ax1.set_ylabel('Required Radius R', fontsize=11)
    ax1.set_title(f'Required R for Binary Tree (b=2)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, max_depth)
    
    # Right: R required vs depth for different branching factors (fixed d=middle dimension)
    ax2 = axes[1]
    d = dimensions[len(dimensions)//2] if len(dimensions) > 1 else dimensions[0]  # Use middle dimension
    branching_factors = [2, 5, 10, 20]
    colors2 = plt.cm.plasma(np.linspace(0.2, 0.8, len(branching_factors)))
    
    for b, color in zip(branching_factors, colors2):
        R_vals = [R_required(D, b, d) for D in depths]
        ax2.plot(depths, R_vals, '-', color=color, linewidth=2, label=f'b={b}')
    
    ax2.axhline(y=3.0, color='orange', linestyle='--', label='R=3')
    ax2.axhline(y=5.0, color='red', linestyle=':', label='R=5')
    
    ax2.set_xlabel('Ontology Depth D', fontsize=11)
    ax2.set_ylabel('Required Radius R', fontsize=11)
    ax2.set_title(f'Required R at d={d} (varying branching)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, max_depth)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_safe_operating_regime(out_path: Path):
    """Create 2D heatmap showing safe operating regime."""
    
    depths = np.arange(5, 55, 5)
    dimensions = np.array([8, 16, 32, 64, 128, 256])
    
    # For each (depth, dim), compute R required and resulting κ(R)
    # Assume b=3 as typical branching factor
    b = 3
    
    kappa_matrix = np.zeros((len(depths), len(dimensions)))
    R_matrix = np.zeros((len(depths), len(dimensions)))
    
    for i, D in enumerate(depths):
        for j, d in enumerate(dimensions):
            R = R_required(D, b, d)
            R_matrix[i, j] = R
            kappa_matrix[i, j] = kappa(np.array([R]))[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Required R heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(R_matrix, aspect='auto', cmap='YlOrRd',
                     extent=[dimensions[0]-4, dimensions[-1]+4, depths[-1]+2.5, depths[0]-2.5])
    ax1.set_xlabel('Embedding Dimension d', fontsize=11)
    ax1.set_ylabel('Ontology Depth D', fontsize=11)
    ax1.set_title(f'Required Radius R (b={b})', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('R')
    
    # Add contour lines
    CS1 = ax1.contour(dimensions, depths, R_matrix, levels=[2, 3, 4, 5], colors='white', linewidths=1)
    ax1.clabel(CS1, inline=True, fontsize=8, fmt='R=%.0f')
    
    # Right: κ(R) heatmap with safe/danger zones
    ax2 = axes[1]
    
    # Use diverging colormap centered at κ=2
    im2 = ax2.imshow(np.log10(kappa_matrix), aspect='auto', cmap='RdYlGn_r',
                     vmin=0, vmax=2,
                     extent=[dimensions[0]-4, dimensions[-1]+4, depths[-1]+2.5, depths[0]-2.5])
    ax2.set_xlabel('Embedding Dimension d', fontsize=11)
    ax2.set_ylabel('Ontology Depth D', fontsize=11)
    ax2.set_title(f'Distortion Factor κ(R) (log scale)', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label(r'$\log_{10}(\kappa)$')
    
    # Add safe zone boundary (κ=2)
    CS2 = ax2.contour(dimensions, depths, kappa_matrix, levels=[2, 5, 10], 
                      colors=['green', 'orange', 'red'], linewidths=2)
    ax2.clabel(CS2, inline=True, fontsize=8, fmt='κ=%.0f')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def print_scale_analysis():
    """Print practical guidance for production ontologies."""
    
    print("\n" + "="*60)
    print("THEORETICAL SCALING ANALYSIS - KEY FINDINGS")
    print("="*60)
    
    # SNOMED-CT example
    print("\nExample: SNOMED-CT (D≈20, b≈5)")
    for d in [32, 64, 128]:
        R = R_required(20, 5, d)
        k = kappa(np.array([R]))[0]
        status = "✓ Safe" if k < 5 else "⚠ Caution" if k < 20 else "✗ Danger"
        print(f"  d={d:3d}: R={R:.2f}, κ={k:.1f} {status}")
    
    # HPO example
    print("\nExample: HPO (D≈10, b≈3)")
    for d in [32, 64]:
        R = R_required(10, 3, d)
        k = kappa(np.array([R]))[0]
        status = "✓ Safe" if k < 5 else "⚠ Caution"
        print(f"  d={d:3d}: R={R:.2f}, κ={k:.1f} {status}")
    
    # Gene Ontology example
    print("\nExample: Gene Ontology (D≈15, b≈4)")
    for d in [32, 64]:
        R = R_required(15, 4, d)
        k = kappa(np.array([R]))[0]
        status = "✓ Safe" if k < 5 else "⚠ Caution"
        print(f"  d={d:3d}: R={R:.2f}, κ={k:.1f} {status}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION: Use d≥32 for most biomedical ontologies")
    print("For extremely deep hierarchies (D>30), increase d to 64-128")
    print("="*60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_depth", type=int, default=50)
    ap.add_argument("--out_dir", type=str, default="paper_artifacts/revision")
    ap.add_argument("--dimensions", type=str, default="16,32,64,128",
                   help="Comma-separated list of dimensions to plot (e.g., '16,32,64,128')")
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse dimensions
    dimensions = [int(d.strip()) for d in args.dimensions.split(',')]
    
    # Generate all theoretical plots
    plot_kappa_vs_R(out_dir / "theoretical_kappa_distortion.pdf")
    plot_R_vs_depth(out_dir / "theoretical_R_vs_depth.pdf", max_depth=args.max_depth, dimensions=dimensions)
    plot_safe_operating_regime(out_dir / "theoretical_safe_regime.pdf")
    
    # Print analysis summary
    print_scale_analysis()


if __name__ == "__main__":
    main()