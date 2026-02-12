"""
09_generate_figures.py
======================
Generates all publication-quality figures for the BMC Genomics manuscript.

Figures:
    Figure 1: Two-module architecture
        (a) RQC pathway schematic with module annotations
        (b) Three-way Venn diagram of top 5% network overlap
        (c) Pairwise Jaccard similarity heatmap

    Figure 2: Gene Ontology enrichment of unique gene sets
        (a–d) Top GO:BP terms for PELO unique, LTN1 unique, NEMF unique,
               and shared RQC core

    Figure 3: Multi-region replication
        (a–c) Cross-region Spearman ρ heatmaps for PELO, LTN1, NEMF

Output:
    PNG (300 DPI) and PDF for each figure

Usage:
    pip install matplotlib-venn
    python 07_generate_figures.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_venn import venn3

# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════
RESULTS_DIR = "../results"
FIG_DIR = "../results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Google Colab paths (uncomment if using Colab):
# RESULTS_DIR = "/content/drive/MyDrive/Research/Results/Manuscript_Statistics"
# FIG_DIR = "/content/drive/MyDrive/Research/Results/Manuscript_Figures/MS1_RQC"

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.facecolor': 'white',
})

# Color palette (colorblind-friendly)
COLORS = {
    'PELO': '#2171B5',      # Blue
    'LTN1': '#CB181D',      # Red
    'NEMF': '#238B45',      # Green
    'shared': '#6A51A3',    # Purple
    'module_a': '#9ECAE1',  # Light blue
    'module_b': '#FCAE91',  # Light red
}


# ═══════════════════════════════════════════════════════
# FIGURE 1: Two-Module Architecture
# ═══════════════════════════════════════════════════════
def generate_figure_1():
    """Three-panel figure: schematic + Venn + Jaccard heatmap."""
    print("  Generating Figure 1...")

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.0, 0.8], wspace=0.35)

    # ── Panel (a): RQC Pathway Schematic ──
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 8)
    ax_a.axis('off')
    ax_a.set_title('(a)', fontweight='bold', fontsize=12, loc='left')

    # Ribosome
    ax_a.add_patch(FancyBboxPatch((0.5, 3.5), 2, 1.5,
        boxstyle="round,pad=0.2", facecolor='#D9D9D9', edgecolor='black', lw=1.5))
    ax_a.text(1.5, 4.25, 'Stalled\nRibosome', ha='center', va='center', fontsize=8, fontweight='bold')

    # PELO (Module A)
    ax_a.add_patch(FancyBboxPatch((3.5, 3.5), 1.8, 1.5,
        boxstyle="round,pad=0.2", facecolor=COLORS['module_a'], edgecolor=COLORS['PELO'], lw=2))
    ax_a.text(4.4, 4.25, 'PELO\n(splitting)', ha='center', va='center', fontsize=8, fontweight='bold')

    # LTN1 (Module B)
    ax_a.add_patch(FancyBboxPatch((6.2, 5), 1.8, 1.5,
        boxstyle="round,pad=0.2", facecolor=COLORS['module_b'], edgecolor=COLORS['LTN1'], lw=2))
    ax_a.text(7.1, 5.75, 'LTN1\n(ubiquitin)', ha='center', va='center', fontsize=8, fontweight='bold')

    # NEMF (Module B)
    ax_a.add_patch(FancyBboxPatch((6.2, 2), 1.8, 1.5,
        boxstyle="round,pad=0.2", facecolor=COLORS['module_b'], edgecolor=COLORS['NEMF'], lw=2))
    ax_a.text(7.1, 2.75, 'NEMF\n(CAT-tail)', ha='center', va='center', fontsize=8, fontweight='bold')

    # Arrows
    ax_a.annotate('', xy=(3.4, 4.25), xytext=(2.6, 4.25),
        arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax_a.annotate('', xy=(6.1, 5.75), xytext=(5.4, 4.6),
        arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax_a.annotate('', xy=(6.1, 2.75), xytext=(5.4, 3.9),
        arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Module labels
    ax_a.text(4.4, 6.5, 'Module A', ha='center', fontsize=9, fontweight='bold',
        color=COLORS['PELO'], style='italic')
    ax_a.text(4.4, 6.0, '(Surveillance)', ha='center', fontsize=8, color=COLORS['PELO'])
    ax_a.text(7.1, 7.2, 'Module B', ha='center', fontsize=9, fontweight='bold',
        color=COLORS['LTN1'], style='italic')
    ax_a.text(7.1, 6.7, '(Nascent chain QC)', ha='center', fontsize=8, color=COLORS['LTN1'])

    # Bracket for Module B
    ax_a.plot([8.3, 8.6, 8.6, 8.3], [5.75, 5.75, 2.75, 2.75],
        color=COLORS['LTN1'], lw=1.5, clip_on=False)
    ax_a.text(9.0, 4.25, 'ρ = 0.983', ha='center', va='center', fontsize=8,
        color=COLORS['LTN1'], fontweight='bold')

    # ── Panel (b): Venn Diagram ──
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_title('(b)', fontweight='bold', fontsize=12, loc='left')

    # Venn counts: (PELO_only, LTN1_only, P∩L, NEMF_only, P∩N, L∩N, all3)
    v = venn3(subsets=(534, 259, 53, 240, 72, 347, 153),
              set_labels=('PELO', 'LTN1', 'NEMF'), ax=ax_b)

    # Style labels
    for label_id in ['100', '010', '001', '110', '101', '011', '111']:
        if v.get_label_by_id(label_id):
            v.get_label_by_id(label_id).set_fontsize(9)

    for label in ['A', 'B', 'C']:
        if v.get_label_by_id(label):
            v.get_label_by_id(label).set_fontsize(11)
            v.get_label_by_id(label).set_fontweight('bold')

    # Color patches
    patch_colors = {'100': COLORS['PELO'], '010': COLORS['LTN1'], '001': COLORS['NEMF']}
    for pid, color in patch_colors.items():
        if v.get_patch_by_id(pid):
            v.get_patch_by_id(pid).set_alpha(0.3)
            v.get_patch_by_id(pid).set_facecolor(color)

    # ── Panel (c): Jaccard Heatmap ──
    ax_c = fig.add_subplot(gs[2])
    ax_c.set_title('(c)', fontweight='bold', fontsize=12, loc='left')

    jaccard_matrix = np.array([
        [1.000, 0.145, 0.161],
        [0.145, 1.000, 0.445],
        [0.161, 0.445, 1.000],
    ])
    genes = ['PELO', 'LTN1', 'NEMF']

    cmap = LinearSegmentedColormap.from_list('custom', ['#FFFFFF', '#2171B5'], N=256)
    im = ax_c.imshow(jaccard_matrix, cmap=cmap, vmin=0, vmax=0.5, aspect='equal')

    ax_c.set_xticks(range(3))
    ax_c.set_yticks(range(3))
    ax_c.set_xticklabels(genes, fontsize=10, fontweight='bold')
    ax_c.set_yticklabels(genes, fontsize=10, fontweight='bold')

    for i in range(3):
        for j in range(3):
            color = 'white' if jaccard_matrix[i, j] > 0.35 else 'black'
            ax_c.text(j, i, f'{jaccard_matrix[i, j]:.3f}',
                     ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label('Jaccard Index', fontsize=9)

    # Save
    for ext in ['png', 'pdf']:
        path = f"{FIG_DIR}/Figure_1_two_module_architecture.{ext}"
        fig.savefig(path, dpi=300)
    plt.close(fig)
    print("    ✓ Figure 1 saved")


# ═══════════════════════════════════════════════════════
# FIGURE 2: GO Enrichment
# ═══════════════════════════════════════════════════════
def generate_figure_2():
    """Four-panel GO:BP enrichment bar charts."""
    print("  Generating Figure 2...")

    panels = [
        ('PELO_unique', 'PELO unique (534 genes)', COLORS['PELO']),
        ('LTN1_unique', 'LTN1 unique (259 genes)', COLORS['LTN1']),
        ('NEMF_unique', 'NEMF unique (240 genes)', COLORS['NEMF']),
        ('RQC_shared_all3', 'Shared RQC core (153 genes)', COLORS['shared']),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    for idx, (name, title, color) in enumerate(panels):
        ax = axes[idx]
        ax.set_title(f'{panel_labels[idx]} {title}', fontweight='bold', fontsize=11, loc='left')

        # Try to load enrichment data
        filepath = f"{RESULTS_DIR}/{name}_GO_enrichment.csv"
        if not os.path.exists(filepath):
            ax.text(0.5, 0.5, 'Data not found\nRun 03_go_enrichment.py first',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        df = pd.read_csv(filepath)
        bp = df[df['source'] == 'GO:BP'].head(5).copy()

        if len(bp) == 0:
            ax.text(0.5, 0.5, 'No GO:BP terms', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, color='gray')
            continue

        bp['neg_log_p'] = -np.log10(bp['p_value'])
        bp = bp.iloc[::-1]  # Reverse for horizontal bars

        # Truncate long term names
        bp['short_name'] = bp['name'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)

        ax.barh(range(len(bp)), bp['neg_log_p'], color=color, alpha=0.8, edgecolor='black', lw=0.5)
        ax.set_yticks(range(len(bp)))
        ax.set_yticklabels(bp['short_name'], fontsize=8)
        ax.set_xlabel('$-\\log_{10}(p)$', fontsize=10)
        ax.axvline(x=-np.log10(0.05), color='gray', linestyle='--', lw=0.8, alpha=0.5)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{FIG_DIR}/Figure_2_GO_enrichment.{ext}", dpi=300)
    plt.close(fig)
    print("    ✓ Figure 2 saved")


# ═══════════════════════════════════════════════════════
# FIGURE 3: Multi-Region Replication
# ═══════════════════════════════════════════════════════
def generate_figure_3():
    """Three-panel cross-region correlation heatmaps."""
    print("  Generating Figure 3...")

    genes = ['PELO', 'LTN1', 'NEMF']
    panel_labels = ['(a)', '(b)', '(c)']
    gene_colors = [COLORS['PELO'], COLORS['LTN1'], COLORS['NEMF']]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cmap = LinearSegmentedColormap.from_list('rho', ['#FFF5EB', '#C6DBEF', '#08519C'], N=256)

    for idx, gene in enumerate(genes):
        ax = axes[idx]
        ax.set_title(f'{panel_labels[idx]} {gene}', fontweight='bold', fontsize=12, loc='left')

        filepath = f"{RESULTS_DIR}/{gene}_cross_region_matrix.csv"
        if not os.path.exists(filepath):
            ax.text(0.5, 0.5, 'Data not found\nRun 04_multi_region_replication.py',
                   ha='center', va='center', transform=ax.transAxes, fontsize=9, color='gray')
            continue

        matrix = pd.read_csv(filepath, index_col=0)
        regions = matrix.index.tolist()
        values = matrix.values

        im = ax.imshow(values, cmap=cmap, vmin=0.75, vmax=1.0, aspect='equal')

        ax.set_xticks(range(len(regions)))
        ax.set_yticks(range(len(regions)))
        ax.set_xticklabels(regions, fontsize=8, rotation=45, ha='right')
        ax.set_yticklabels(regions, fontsize=8)

        for i in range(len(regions)):
            for j in range(len(regions)):
                color = 'white' if values[i, j] > 0.90 else 'black'
                ax.text(j, i, f'{values[i, j]:.2f}',
                       ha='center', va='center', fontsize=8, fontweight='bold', color=color)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Spearman ρ', fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    for ext in ['png', 'pdf']:
        fig.savefig(f"{FIG_DIR}/Figure_3_multi_region_replication.{ext}", dpi=300)
    plt.close(fig)
    print("    ✓ Figure 3 saved")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("07 — Generate Publication Figures")
    print("=" * 60)

    generate_figure_1()
    generate_figure_2()
    generate_figure_3()

    print(f"\n{'=' * 60}")
    print(f"✓ All figures saved to {FIG_DIR}")
    print(f"{'=' * 60}")
