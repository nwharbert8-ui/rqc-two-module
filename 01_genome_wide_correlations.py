"""
01_genome_wide_correlations.py
==============================
Computes genome-wide Pearson correlations for PELO, LTN1, and NEMF
against all expressed genes in GTEx v8 Brain – Frontal Cortex (BA9).

Methodology (from manuscript):
    - Expression filtered at median TPM >= 1.0 (retains ~16,212 genes)
    - log2(TPM + 1) transformation applied before correlation
    - Pearson r computed for each target gene vs. all other genes
    - Top 5% of positively correlated genes defines each co-expression network

Input:
    GTEx v8 gene TPM matrix (.parquet or .gct.gz)
    GTEx v8 sample attributes file

Output:
    {gene}_genome_wide_correlations.csv  — Full ranked correlation table
    {gene}_top5pct.csv                   — Top 5% co-expressed genes

Usage:
    python 01_genome_wide_correlations.py

    For Google Colab: mount Drive, adjust DATA_DIR and OUTPUT_DIR below.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# ═══════════════════════════════════════════════════════
# CONFIGURATION — Adjust paths for your environment
# ═══════════════════════════════════════════════════════
DATA_DIR = "../data"                           # Directory containing GTEx files
GTEX_TPM = f"{DATA_DIR}/gtex_v8_tpm.parquet"   # or .gct.gz / .tsv
GTEX_ATTR = f"{DATA_DIR}/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
OUTPUT_DIR = "../results"

# Google Colab paths (uncomment if using Colab):
# DATA_DIR = "/content/drive/MyDrive/PCDH_SIGMAR1_Analysis"
# GTEX_TPM = f"{DATA_DIR}/gtex_v8_tpm.parquet"
# GTEX_ATTR = f"{DATA_DIR}/gtex_v8_sample_attrs.parquet"
# OUTPUT_DIR = "/content/drive/MyDrive/Research/Results"

TARGET_GENES = ['PELO', 'LTN1', 'NEMF']
BRAIN_REGION = "Brain - Frontal Cortex (BA9)"
MIN_MEDIAN_TPM = 1.0   # Manuscript threshold: median TPM >= 1.0
TOP_PERCENT = 5         # Top 5% defines co-expression network

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════
def load_expression_matrix(filepath):
    """Load GTEx TPM matrix from parquet, TSV, or GCT format."""
    if filepath.endswith('.parquet'):
        return pd.read_parquet(filepath)
    elif filepath.endswith('.gct.gz') or filepath.endswith('.gct'):
        df = pd.read_csv(filepath, sep='\t', skiprows=2,
                         index_col=1, compression='gzip' if filepath.endswith('.gz') else None)
        df = df.drop(columns=['Name'], errors='ignore')
        return df
    else:
        return pd.read_csv(filepath, sep='\t', index_col=0)

def load_sample_attributes(filepath):
    """Load GTEx sample attributes."""
    if filepath.endswith('.parquet'):
        return pd.read_parquet(filepath)
    else:
        return pd.read_csv(filepath, sep='\t')


print("=" * 60)
print("01 — Genome-Wide Co-expression Analysis")
print("=" * 60)

print("\nLoading GTEx v8 expression data...")
tpm = load_expression_matrix(GTEX_TPM)
attr = load_sample_attributes(GTEX_ATTR)
print(f"  TPM matrix: {tpm.shape[0]:,} genes × {tpm.shape[1]:,} samples")


# ═══════════════════════════════════════════════════════
# REGION FILTERING
# ═══════════════════════════════════════════════════════
# Identify tissue and sample ID columns (robust to different GTEx file versions)
smtsd_col = [c for c in attr.columns if 'SMTSD' in c.upper() or 'tissue_detail' in c.lower()][0]
sampid_col = [c for c in attr.columns if 'SAMPID' in c.upper() or 'sample_id' in c.lower()][0]

brain_samples = attr[attr[smtsd_col] == BRAIN_REGION][sampid_col].tolist()
brain_cols = [c for c in tpm.columns if c in brain_samples]

if len(brain_cols) < 50:
    print(f"  ERROR: Only {len(brain_cols)} samples found for {BRAIN_REGION}.")
    sys.exit(1)

brain_tpm = tpm[brain_cols]
print(f"  {BRAIN_REGION}: {len(brain_cols)} samples")


# ═══════════════════════════════════════════════════════
# GENE FILTERING (median TPM >= 1.0)
# ═══════════════════════════════════════════════════════
gene_medians = brain_tpm.median(axis=1)
expressed_genes = gene_medians[gene_medians >= MIN_MEDIAN_TPM].index
brain_expr = brain_tpm.loc[expressed_genes]
print(f"  Expressed genes (median TPM >= {MIN_MEDIAN_TPM}): {len(expressed_genes):,}")

# Verify target genes are present
for gene in TARGET_GENES:
    if gene not in brain_expr.index:
        print(f"  FATAL: {gene} not found in expression matrix after filtering")
        sys.exit(1)
    else:
        med = gene_medians.loc[gene]
        print(f"  ✓ {gene} present (median TPM = {med:.2f})")


# ═══════════════════════════════════════════════════════
# LOG2 TRANSFORMATION
# ═══════════════════════════════════════════════════════
brain_log = np.log2(brain_expr + 1)
print(f"\n  log2(TPM+1) transformation applied")


# ═══════════════════════════════════════════════════════
# GENOME-WIDE PEARSON CORRELATIONS
# ═══════════════════════════════════════════════════════
for gene in TARGET_GENES:
    print(f"\n{'─' * 50}")
    print(f"  Computing correlations for {gene}...")
    target_expr = brain_log.loc[gene].values

    results = []
    for other_gene in brain_log.index:
        if other_gene == gene:
            continue
        other_expr = brain_log.loc[other_gene].values
        r, pval = stats.pearsonr(target_expr, other_expr)
        results.append({
            'gene': other_gene,
            'pearson_r': round(r, 6),
            'p_value': pval,
            'median_tpm': round(gene_medians.loc[other_gene], 3),
        })

    df = pd.DataFrame(results)
    df = df.sort_values('pearson_r', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    df['percentile'] = round(df['rank'] / len(df) * 100, 2)

    # Save full genome-wide rankings
    full_path = f"{OUTPUT_DIR}/{gene}_genome_wide_correlations.csv"
    df.to_csv(full_path, index=False)
    print(f"  Saved: {full_path}")
    print(f"  Total genes ranked: {len(df):,}")

    # Extract and save top 5%
    n_top = int(np.ceil(len(df) * TOP_PERCENT / 100))
    top5 = df.head(n_top).copy()
    threshold_r = top5.iloc[-1]['pearson_r']
    top5_path = f"{OUTPUT_DIR}/{gene}_top5pct.csv"
    top5.to_csv(top5_path, index=False)
    print(f"  Top {TOP_PERCENT}% threshold: r >= {threshold_r:.4f} ({len(top5)} genes)")

    # Report top 5 partners
    print(f"  Top 5 co-expression partners:")
    for _, row in df.head(5).iterrows():
        print(f"    {row['gene']:>12s}: r = {row['pearson_r']:.4f}")


print(f"\n{'=' * 60}")
print("✓ All genome-wide correlations complete")
print(f"{'=' * 60}")
