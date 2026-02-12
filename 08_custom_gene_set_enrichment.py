"""
08_custom_gene_set_enrichment.py
=================================
Custom gene set enrichment analysis using Fisher's exact test.

Tests enrichment of curated gene sets within PELO, LTN1, and NEMF
top 5% co-expression networks, including:
    - Ribosome Quality Control (11 genes)
    - Proteasome subunits (19 genes)
    - Epigenetic regulators (52 genes) — key finding for NEMF

Reports fold-enrichment, Fisher's exact p-value, and hit lists.

Output:
    custom_enrichment_results.csv  — enrichment table for all sets × targets × regions

Usage:
    python 08_custom_gene_set_enrichment.py
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

# ═══════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════
DATA_DIR = "data"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

TPM_FILE = os.path.join(DATA_DIR, "ba9_tpm.parquet")
TARGET_GENES = ["PELO", "LTN1", "NEMF"]
TOP_PCT = 0.05

# ═══════════════════════════════════════════════════════
# Curated gene sets
# ═══════════════════════════════════════════════════════
CUSTOM_SETS = {
    "ribosome_quality_control": [
        "PELO", "HBS1L", "LTN1", "NEMF", "ANKZF1",
        "VCP", "UFD1", "NPLOC4", "ZNF598", "RACK1", "ABCE1",
    ],
    "proteasome_subunits": [
        "PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7",
        "PSMB1", "PSMB2", "PSMB3", "PSMB4", "PSMB5",
        "PSMC1", "PSMC2", "PSMC3", "PSMC4", "PSMC5", "PSMC6",
        "PSMD1",
    ],
    "epigenetic_regulators": [
        # Chromatin remodelers
        "CHD1", "CHD2", "CHD3", "CHD4", "CHD6", "CHD7", "CHD8", "CHD9",
        "SMARCA2", "SMARCA4", "SMARCB1", "SMARCC1", "SMARCC2", "SMARCE1",
        "ARID1A", "ARID1B", "ARID2",
        # Histone acetyltransferases
        "EP300", "CREBBP", "KAT2A", "KAT2B", "KAT6A", "KAT6B",
        # Histone deacetylases
        "HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC5", "HDAC6",
        # Histone methyltransferases
        "EZH2", "KMT2A", "KMT2C", "KMT2D", "SETD2", "NSD1", "NSD2",
        "SUV39H1", "SUV39H2", "EHMT1", "EHMT2",
        # Histone demethylases
        "KDM1A", "KDM2A", "KDM3A", "KDM4A", "KDM5A", "KDM5B", "KDM6A", "KDM6B",
        # DNA methyltransferases
        "DNMT1", "DNMT3A", "DNMT3B",
        # Readers
        "BRD4",
    ],
}

# ═══════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════
print("Loading expression data...")
tpm = pd.read_parquet(TPM_FILE)
expr = np.log2(tpm + 1)
all_genes = list(expr.columns)
genome_size = len(all_genes)
n_top = int(genome_size * TOP_PCT)

print(f"  {genome_size} expressed genes, top 5% = {n_top} genes")

# ═══════════════════════════════════════════════════════
# Compute correlations and top 5% for each target
# ═══════════════════════════════════════════════════════
results = []

for target in TARGET_GENES:
    print(f"\n{'=' * 60}")
    print(f"TARGET: {target}")
    print(f"{'=' * 60}")

    other_genes = [g for g in all_genes if g != target]
    target_vals = expr[target].values

    corrs = {}
    for gene in other_genes:
        r, _ = stats.pearsonr(target_vals, expr[gene].values)
        corrs[gene] = r

    ranking = pd.Series(corrs).sort_values(ascending=False)
    top5_set = set(ranking.head(n_top).index)

    for set_name, gene_list in CUSTOM_SETS.items():
        # Filter to expressed genes
        expressed = [g for g in gene_list if g in set(other_genes)]
        # Count hits in top 5%
        hits = [g for g in expressed if g in top5_set]

        k = len(hits)          # hits
        K = len(expressed)     # set size (expressed)
        n = n_top              # top 5% size
        N = len(other_genes)   # genome

        # Expected
        expected = K * n / N
        fold = k / expected if expected > 0 else 0

        # Fisher's exact test (one-sided, enrichment)
        table = [[k, K - k], [n - k, N - n - K + k]]
        _, p_val = stats.fisher_exact(table, alternative="greater")

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        results.append({
            "target": target,
            "gene_set": set_name,
            "set_size_total": len(gene_list),
            "set_size_expressed": K,
            "hits_in_top5pct": k,
            "expected": round(expected, 2),
            "fold_enrichment": round(fold, 1),
            "fisher_p": p_val,
            "significant": sig,
            "hit_genes": "; ".join(sorted(hits)),
            "missed_genes": "; ".join(sorted(set(expressed) - set(hits))),
        })

        print(f"\n  {set_name} ({K} expressed / {len(gene_list)} total)")
        print(f"    Hits: {k}/{K} (fold = {fold:.1f}, p = {p_val:.2e}) {sig}")
        if hits:
            print(f"    → Hits: {', '.join(sorted(hits))}")

# ═══════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════
df = pd.DataFrame(results)
outpath = os.path.join(OUT_DIR, "custom_enrichment_results.csv")
df.to_csv(outpath, index=False)
print(f"\nSaved: {outpath}")
print("Done.")
