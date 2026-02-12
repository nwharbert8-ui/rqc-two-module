"""
05_cell_type_deconvolution.py
=============================
Sensitivity analysis controlling for cell-type composition variation.

Methodology (from manuscript):
    - 53 established cell-type marker genes spanning 6 major brain cell types:
      neurons, astrocytes, oligodendrocytes, microglia, endothelial cells,
      and oligodendrocyte precursor cells (OPCs)
    - Expression of marker genes used to estimate cell-type proportions
    - Partial Pearson correlations computed controlling for 6 cell-type
      proportion variables
    - Top 5% networks re-derived from partial correlations and compared
      with unadjusted results

Purpose:
    Confirm that observed co-expression architecture is not an artifact
    of cell-type composition variation across post-mortem samples.

Input:
    GTEx v8 TPM matrix (same as 01)

Output:
    {gene}_deconvolved_correlations.csv  — Partial correlation rankings
    deconvolution_comparison.csv         — Rank correlation with unadjusted results

Usage:
    python 05_cell_type_deconvolution.py
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════
DATA_DIR = "../data"
GTEX_TPM = f"{DATA_DIR}/gtex_v8_tpm.parquet"
GTEX_ATTR = f"{DATA_DIR}/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
RESULTS_DIR = "../results"
OUTPUT_DIR = "../results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Google Colab paths (uncomment if using Colab):
# DATA_DIR = "/content/drive/MyDrive/PCDH_SIGMAR1_Analysis"
# GTEX_TPM = f"{DATA_DIR}/gtex_v8_tpm.parquet"
# GTEX_ATTR = f"{DATA_DIR}/gtex_v8_sample_attrs.parquet"
# RESULTS_DIR = "/content/drive/MyDrive/Research/Results"
# OUTPUT_DIR = "/content/drive/MyDrive/Research/Results/Manuscript_Statistics"

TARGET_GENES = ['PELO', 'LTN1', 'NEMF']
BRAIN_REGION = "Brain - Frontal Cortex (BA9)"
MIN_MEDIAN_TPM = 1.0

# ═══════════════════════════════════════════════════════
# CELL-TYPE MARKER GENES
# ═══════════════════════════════════════════════════════
# Established markers from Darmanis et al. 2015, Zhang et al. 2016, and
# Lake et al. 2018. Organized by cell type.
CELL_TYPE_MARKERS = {
    'neuron': [
        'SYT1', 'SNAP25', 'SLC17A7', 'GAD1', 'GAD2', 'RBFOX3', 'STMN2',
        'NRGN', 'ENO2',
    ],
    'astrocyte': [
        'GFAP', 'AQP4', 'SLC1A2', 'SLC1A3', 'ALDH1L1', 'GJA1', 'S100B',
        'ALDOC', 'SOX9',
    ],
    'oligodendrocyte': [
        'MBP', 'PLP1', 'MOG', 'MAG', 'CLDN11', 'MOBP', 'OPALIN',
        'CNP', 'TF',
    ],
    'microglia': [
        'CX3CR1', 'P2RY12', 'CSF1R', 'ITGAM', 'AIF1', 'TMEM119',
        'HEXB', 'C1QA', 'C1QB',
    ],
    'endothelial': [
        'CLDN5', 'FLT1', 'PECAM1', 'VWF', 'CDH5', 'ERG', 'ENG',
        'ESAM',
    ],
    'OPC': [
        'PDGFRA', 'CSPG4', 'GPR17', 'OLIG1', 'OLIG2', 'SOX10',
        'NKX2-2', 'VCAN',
    ],
}

ALL_MARKERS = []
for markers in CELL_TYPE_MARKERS.values():
    ALL_MARKERS.extend(markers)


# ═══════════════════════════════════════════════════════
# PARTIAL CORRELATION FUNCTION
# ═══════════════════════════════════════════════════════
def partial_corr(x, y, covariates):
    """
    Compute partial Pearson correlation between x and y,
    controlling for a matrix of covariates.

    Uses residualization: regress out covariates from both x and y,
    then correlate residuals.
    """
    C = np.column_stack(covariates)
    # Add intercept
    C_int = np.column_stack([np.ones(len(x)), C])

    # Residualize x
    beta_x, _, _, _ = np.linalg.lstsq(C_int, x, rcond=None)
    resid_x = x - C_int @ beta_x

    # Residualize y
    beta_y, _, _, _ = np.linalg.lstsq(C_int, y, rcond=None)
    resid_y = y - C_int @ beta_y

    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p


# ═══════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("05 — Cell-Type Deconvolution Sensitivity Analysis")
print("=" * 60)

# Load data
print("\nLoading GTEx v8 data...")
if GTEX_TPM.endswith('.parquet'):
    tpm = pd.read_parquet(GTEX_TPM)
else:
    tpm = pd.read_csv(GTEX_TPM, sep='\t', index_col=0)

if GTEX_ATTR.endswith('.parquet'):
    attr = pd.read_parquet(GTEX_ATTR)
else:
    attr = pd.read_csv(GTEX_ATTR, sep='\t')

smtsd_col = [c for c in attr.columns if 'SMTSD' in c.upper()][0]
sampid_col = [c for c in attr.columns if 'SAMPID' in c.upper()][0]

# Filter to BA9
brain_samples = attr[attr[smtsd_col] == BRAIN_REGION][sampid_col].tolist()
brain_cols = [c for c in tpm.columns if c in brain_samples]
brain_tpm = tpm[brain_cols]

gene_medians = brain_tpm.median(axis=1)
expressed = gene_medians[gene_medians >= MIN_MEDIAN_TPM].index
brain_log = np.log2(brain_tpm.loc[expressed] + 1)
print(f"  Samples: {len(brain_cols)}, Genes: {len(expressed):,}")

# Estimate cell-type proportions using marker gene expression
print(f"\n  Cell-type markers available in expression data:")
cell_type_proportions = {}
for cell_type, markers in CELL_TYPE_MARKERS.items():
    available = [m for m in markers if m in brain_log.index]
    if len(available) >= 2:
        # Use mean expression of available markers as proxy proportion
        cell_type_proportions[cell_type] = brain_log.loc[available].mean(axis=0).values
        print(f"    {cell_type:20s}: {len(available)}/{len(markers)} markers")
    else:
        print(f"    {cell_type:20s}: SKIPPED ({len(available)} markers)")

covariate_matrix = list(cell_type_proportions.values())
print(f"\n  Controlling for {len(covariate_matrix)} cell-type covariates")

# Compute partial correlations for each target gene
comparison_results = []

for gene in TARGET_GENES:
    print(f"\n{'─' * 50}")
    print(f"  Partial correlations for {gene}...")

    if gene not in brain_log.index:
        print(f"    ERROR: {gene} not found")
        continue

    target = brain_log.loc[gene].values
    results = []

    for other_gene in brain_log.index:
        if other_gene == gene:
            continue
        other = brain_log.loc[other_gene].values
        r_partial, p_partial = partial_corr(target, other, covariate_matrix)
        results.append({
            'gene': other_gene,
            'partial_pearson_r': round(r_partial, 6),
            'partial_p_value': p_partial,
        })

    df = pd.DataFrame(results)
    df = df.sort_values('partial_pearson_r', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    outpath = f"{OUTPUT_DIR}/{gene}_deconvolved_correlations.csv"
    df.to_csv(outpath, index=False)
    print(f"  Saved: {outpath}")

    # Compare with unadjusted rankings
    unadj_path = f"{RESULTS_DIR}/{gene}_genome_wide_correlations.csv"
    if os.path.exists(unadj_path):
        unadj = pd.read_csv(unadj_path)
        gene_col = [c for c in unadj.columns if c.lower() in ['gene', 'gene_name', 'symbol']][0]
        unadj_ranks = unadj.set_index(gene_col)['rank']
        deconv_ranks = df.set_index('gene')['rank']
        common = unadj_ranks.index.intersection(deconv_ranks.index)
        rho, _ = stats.spearmanr(unadj_ranks.loc[common], deconv_ranks.loc[common])
        print(f"  Rank correlation with unadjusted: ρ = {rho:.4f}")
        comparison_results.append({
            'gene': gene,
            'rank_correlation_unadj_vs_deconv': round(rho, 4),
            'n_genes_compared': len(common),
        })

# Save comparison
if comparison_results:
    comp_df = pd.DataFrame(comparison_results)
    comp_df.to_csv(f"{OUTPUT_DIR}/deconvolution_comparison.csv", index=False)
    print(f"\n  ✓ Saved: deconvolution_comparison.csv")


print(f"\n{'=' * 60}")
print("✓ Cell-type deconvolution analysis complete")
print(f"{'=' * 60}")
