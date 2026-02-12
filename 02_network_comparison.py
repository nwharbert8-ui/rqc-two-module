"""
02_network_comparison.py
========================
Compares co-expression networks between PELO, LTN1, and NEMF using three
complementary metrics:
    1. Jaccard similarity index (set overlap)
    2. Fisher's exact test (overlap significance vs. chance)
    3. Spearman rank correlation (genome-wide profile similarity)

Also computes all seven three-way Venn diagram partitions and extracts
unique/shared gene lists for downstream enrichment analysis.

Input:
    Top 5% gene lists from 01_genome_wide_correlations.py
    Full genome-wide correlation rankings

Output:
    network_comparisons_rqc.csv      — Pairwise statistics (Table 1)
    venn_counts.csv                  — Seven-partition Venn counts (Table 2)
    {group}_top5pct.csv              — Gene lists for each partition

Usage:
    python 02_network_comparison.py
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════
RESULTS_DIR = "../results"
OUTPUT_DIR = "../results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_GENES = ['PELO', 'LTN1', 'NEMF']

# Google Colab paths (uncomment if using Colab):
# RESULTS_DIR = "/content/drive/MyDrive/Research/Results"
# OUTPUT_DIR = "/content/drive/MyDrive/Research/Results/Manuscript_Statistics"


print("=" * 60)
print("02 — Network Comparison Analysis")
print("=" * 60)


# ═══════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════
networks = {}        # Top 5% gene sets
full_rankings = {}   # Genome-wide Pearson r for rank correlations
genome_size = None

for gene in TARGET_GENES:
    # Top 5% network
    top5_path = f"{RESULTS_DIR}/{gene}_top5pct.csv"
    df = pd.read_csv(top5_path)
    gene_col = [c for c in df.columns if c.lower() in ['gene', 'gene_name', 'symbol']][0]
    networks[gene] = set(df[gene_col].tolist())
    print(f"  {gene}: {len(networks[gene])} genes in top 5%")

    # Full genome-wide rankings
    full_path = f"{RESULTS_DIR}/{gene}_genome_wide_correlations.csv"
    full_df = pd.read_csv(full_path)
    full_gene_col = [c for c in full_df.columns if c.lower() in ['gene', 'gene_name', 'symbol']][0]
    r_col = [c for c in full_df.columns if 'pearson' in c.lower() or c == 'r'][0]
    full_rankings[gene] = full_df.set_index(full_gene_col)[r_col]

    if genome_size is None:
        genome_size = len(full_df)

print(f"\n  Genome size (total ranked genes): {genome_size:,}")


# ═══════════════════════════════════════════════════════
# PAIRWISE COMPARISONS (Table 1)
# ═══════════════════════════════════════════════════════
print(f"\n{'─' * 50}")
print("Pairwise network comparisons:")

results = []
for g1, g2 in combinations(TARGET_GENES, 2):
    set1, set2 = networks[g1], networks[g2]
    intersection = set1 & set2
    union = set1 | set2

    # Jaccard similarity
    jaccard = len(intersection) / len(union)

    # Fisher's exact test (2×2 contingency)
    a = len(intersection)                     # In both
    b = len(set1 - set2)                      # In g1 only
    c = len(set2 - set1)                      # In g2 only
    d = genome_size - len(union)              # In neither
    odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c, d]], alternative='greater')

    # Spearman rank correlation of genome-wide profiles
    common = full_rankings[g1].index.intersection(full_rankings[g2].index)
    rho, rho_p = stats.spearmanr(
        full_rankings[g1].loc[common].values,
        full_rankings[g2].loc[common].values
    )

    results.append({
        'gene_a': g1, 'gene_b': g2,
        'set_a_size': len(set1), 'set_b_size': len(set2),
        'shared_genes': a, 'union_size': len(union),
        'jaccard': round(jaccard, 4),
        'fisher_OR': round(odds_ratio, 2),
        'fisher_p': fisher_p,
        'rank_spearman_rho': round(rho, 4),
        'rank_spearman_p': rho_p,
    })

    print(f"\n  {g1} vs {g2}:")
    print(f"    Jaccard      = {jaccard:.3f} ({a} shared / {len(union)} union)")
    print(f"    Fisher OR    = {odds_ratio:.2f}, p = {fisher_p:.2e}")
    print(f"    Spearman ρ   = {rho:.3f}, p = {rho_p:.2e}")

comp_df = pd.DataFrame(results)
comp_path = f"{OUTPUT_DIR}/network_comparisons_rqc.csv"
comp_df.to_csv(comp_path, index=False)
print(f"\n  ✓ Saved: {comp_path}")


# ═══════════════════════════════════════════════════════
# THREE-WAY VENN PARTITIONS (Table 2)
# ═══════════════════════════════════════════════════════
print(f"\n{'─' * 50}")
print("Three-way Venn partitions:")

P, L, N = networks['PELO'], networks['LTN1'], networks['NEMF']

all_three    = P & L & N
pelo_ltn1    = (P & L) - N
pelo_nemf    = (P & N) - L
ltn1_nemf    = (L & N) - P
pelo_unique  = P - L - N
ltn1_unique  = L - P - N
nemf_unique  = N - P - L
total_unique = P | L | N

partitions = [
    ('All three (PELO ∩ LTN1 ∩ NEMF)', all_three),
    ('LTN1 ∩ NEMF only',                ltn1_nemf),
    ('PELO ∩ NEMF only',                pelo_nemf),
    ('PELO ∩ LTN1 only',               pelo_ltn1),
    ('PELO unique',                      pelo_unique),
    ('LTN1 unique',                      ltn1_unique),
    ('NEMF unique',                      nemf_unique),
]

venn_rows = []
for label, gene_set in partitions:
    pct = len(gene_set) / len(total_unique) * 100
    venn_rows.append({'category': label, 'gene_count': len(gene_set),
                      'pct_of_total': round(pct, 1)})
    print(f"  {label:40s}  {len(gene_set):>4d}  ({pct:5.1f}%)")

print(f"  {'─' * 55}")
print(f"  {'Total unique genes':40s}  {len(total_unique):>4d}")

venn_df = pd.DataFrame(venn_rows)
venn_path = f"{OUTPUT_DIR}/venn_counts.csv"
venn_df.to_csv(venn_path, index=False)
print(f"\n  ✓ Saved: {venn_path}")


# ═══════════════════════════════════════════════════════
# EXPORT GENE LISTS FOR ENRICHMENT
# ═══════════════════════════════════════════════════════
print(f"\n{'─' * 50}")
print("Exporting gene lists:")

gene_lists = {
    'PELO_unique_top5pct':       pelo_unique,
    'LTN1_unique_top5pct':       ltn1_unique,
    'NEMF_unique_top5pct':       nemf_unique,
    'RQC_shared_all3_top5pct':   all_three,
    'LTN1_NEMF_shared_only':     ltn1_nemf,
    'PELO_LTN1_shared_only':     pelo_ltn1,
    'PELO_NEMF_shared_only':     pelo_nemf,
}

for name, gene_set in gene_lists.items():
    df = pd.DataFrame({'gene': sorted(gene_set)})
    path = f"{OUTPUT_DIR}/{name}.csv"
    df.to_csv(path, index=False)
    print(f"  {name}: {len(gene_set)} genes → {path}")


print(f"\n{'=' * 60}")
print("✓ All network comparisons complete")
print(f"{'=' * 60}")
