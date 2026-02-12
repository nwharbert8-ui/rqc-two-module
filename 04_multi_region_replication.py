"""
04_multi_region_replication.py
==============================
Validates co-expression network stability by repeating genome-wide
correlation analysis across five GTEx brain regions and computing
cross-region Spearman rank correlations.

Methodology (from manuscript):
    - Complete genome-wide Pearson correlations computed independently
      in each of 5 brain regions for each target gene
    - Cross-region consistency: Spearman ρ of genome-wide ranking vectors
    - Replication criterion: ρ > 0.80 across all pairwise region comparisons

Regions:
    BA9 (primary), Putamen, Hippocampus, Nucleus Accumbens, BA24

Input:
    GTEx v8 TPM matrix and sample attributes

Output:
    {gene}_cross_region_matrix.csv  — Spearman ρ matrix (5×5)

Usage:
    python 04_multi_region_replication.py
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════
DATA_DIR = "../data"
GTEX_TPM = f"{DATA_DIR}/gtex_v8_tpm.parquet"
GTEX_ATTR = f"{DATA_DIR}/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
OUTPUT_DIR = "../results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Google Colab paths (uncomment if using Colab):
# DATA_DIR = "/content/drive/MyDrive/PCDH_SIGMAR1_Analysis"
# GTEX_TPM = f"{DATA_DIR}/gtex_v8_tpm.parquet"
# GTEX_ATTR = f"{DATA_DIR}/gtex_v8_sample_attrs.parquet"
# OUTPUT_DIR = "/content/drive/MyDrive/Research/Results/Manuscript_Statistics"

TARGET_GENES = ['PELO', 'LTN1', 'NEMF']
MIN_MEDIAN_TPM = 1.0

BRAIN_REGIONS = [
    "Brain - Frontal Cortex (BA9)",
    "Brain - Putamen (basal ganglia)",
    "Brain - Hippocampus",
    "Brain - Nucleus accumbens (basal ganglia)",
    "Brain - Anterior cingulate cortex (BA24)",
]

REGION_SHORT = ['BA9', 'Putamen', 'Hippocampus', 'Nuc_Acc', 'BA24']


# ═══════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("04 — Multi-Region Replication Analysis")
print("=" * 60)

print("\nLoading GTEx v8 data...")
if GTEX_TPM.endswith('.parquet'):
    tpm = pd.read_parquet(GTEX_TPM)
else:
    tpm = pd.read_csv(GTEX_TPM, sep='\t', index_col=0)

if GTEX_ATTR.endswith('.parquet'):
    attr = pd.read_parquet(GTEX_ATTR)
else:
    attr = pd.read_csv(GTEX_ATTR, sep='\t')

smtsd_col = [c for c in attr.columns if 'SMTSD' in c.upper() or 'tissue_detail' in c.lower()][0]
sampid_col = [c for c in attr.columns if 'SAMPID' in c.upper() or 'sample_id' in c.lower()][0]


# ═══════════════════════════════════════════════════════
# PER-REGION CORRELATION FUNCTION
# ═══════════════════════════════════════════════════════
def compute_region_correlations(gene, region):
    """
    Compute genome-wide Pearson correlations for a target gene
    in a specific brain region.

    Returns:
        pd.Series indexed by gene symbol with Pearson r values,
        or None if insufficient samples.
    """
    samples = attr[attr[smtsd_col] == region][sampid_col].tolist()
    cols = [c for c in tpm.columns if c in samples]

    if len(cols) < 30:
        print(f"    WARNING: {region} has only {len(cols)} samples — skipping")
        return None

    region_tpm = tpm[cols]
    gene_medians = region_tpm.median(axis=1)
    expressed = gene_medians[gene_medians >= MIN_MEDIAN_TPM].index
    region_log = np.log2(region_tpm.loc[expressed] + 1)

    if gene not in region_log.index:
        print(f"    WARNING: {gene} not expressed in {region}")
        return None

    target = region_log.loc[gene].values
    correlations = {}
    for other in region_log.index:
        if other == gene:
            continue
        r, _ = stats.pearsonr(target, region_log.loc[other].values)
        correlations[other] = r

    return pd.Series(correlations)


# ═══════════════════════════════════════════════════════
# CROSS-REGION ANALYSIS
# ═══════════════════════════════════════════════════════
for gene in TARGET_GENES:
    print(f"\n{'═' * 50}")
    print(f"  Cross-region analysis: {gene}")
    print(f"{'═' * 50}")

    region_profiles = {}
    for i, region in enumerate(BRAIN_REGIONS):
        print(f"  Computing {REGION_SHORT[i]:12s} (n samples pending)...")
        profile = compute_region_correlations(gene, region)
        if profile is not None:
            region_profiles[REGION_SHORT[i]] = profile

    if len(region_profiles) < 2:
        print(f"  ERROR: Fewer than 2 regions available for {gene}")
        continue

    # Compute pairwise Spearman ρ
    regions_available = list(region_profiles.keys())
    n = len(regions_available)
    cross_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            common = region_profiles[regions_available[i]].index.intersection(
                     region_profiles[regions_available[j]].index)
            rho, _ = stats.spearmanr(
                region_profiles[regions_available[i]].loc[common],
                region_profiles[regions_available[j]].loc[common]
            )
            cross_matrix[i, j] = round(rho, 3)
            cross_matrix[j, i] = round(rho, 3)

    cross_df = pd.DataFrame(cross_matrix,
                            index=regions_available,
                            columns=regions_available)

    outpath = f"{OUTPUT_DIR}/{gene}_cross_region_matrix.csv"
    cross_df.to_csv(outpath)
    print(f"\n  Saved: {outpath}")

    # Report range
    upper_tri = cross_matrix[np.triu_indices(n, k=1)]
    print(f"  Range: ρ = {upper_tri.min():.3f} – {upper_tri.max():.3f}")
    print(f"  Median: ρ = {np.median(upper_tri):.3f}")
    print(f"  All > 0.80: {'Yes' if upper_tri.min() > 0.80 else 'NO'}")
    print(f"\n{cross_df.to_string()}")


print(f"\n{'=' * 60}")
print("✓ Multi-region replication complete")
print(f"{'=' * 60}")
