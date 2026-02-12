"""
06_covariate_adjustment.py
==========================
Sensitivity analysis controlling for donor age and sex.

Methodology (from manuscript):
    - Donor age and sex obtained from GTEx subject phenotypes file
    - Partial Pearson correlations computed controlling for age and sex
    - Top 5% networks re-derived and compared with unadjusted results
    - Minimal changes expected, confirming patterns are not demographic artifacts

Input:
    GTEx v8 TPM matrix
    GTEx v8 subject phenotypes (age, sex)
    GTEx v8 sample attributes (to link samples → donors)

Output:
    {gene}_covariate_adjusted_correlations.csv  — Partial correlation rankings
    covariate_adjustment_comparison.csv         — Rank correlation with unadjusted

Usage:
    python 06_covariate_adjustment.py
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
GTEX_PHENO = f"{DATA_DIR}/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
RESULTS_DIR = "../results"
OUTPUT_DIR = "../results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Google Colab paths (uncomment if using Colab):
# DATA_DIR = "/content/drive/MyDrive/PCDH_SIGMAR1_Analysis"
# GTEX_TPM = f"{DATA_DIR}/gtex_v8_tpm.parquet"
# GTEX_ATTR = f"{DATA_DIR}/gtex_v8_sample_attrs.parquet"
# GTEX_PHENO = f"{DATA_DIR}/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
# RESULTS_DIR = "/content/drive/MyDrive/Research/Results"
# OUTPUT_DIR = "/content/drive/MyDrive/Research/Results/Manuscript_Statistics"

TARGET_GENES = ['PELO', 'LTN1', 'NEMF']
BRAIN_REGION = "Brain - Frontal Cortex (BA9)"
MIN_MEDIAN_TPM = 1.0


def partial_corr(x, y, covariates):
    """Partial Pearson correlation controlling for covariates via residualization."""
    C = np.column_stack(covariates)
    C_int = np.column_stack([np.ones(len(x)), C])
    beta_x, _, _, _ = np.linalg.lstsq(C_int, x, rcond=None)
    beta_y, _, _, _ = np.linalg.lstsq(C_int, y, rcond=None)
    resid_x = x - C_int @ beta_x
    resid_y = y - C_int @ beta_y
    return stats.pearsonr(resid_x, resid_y)


# ═══════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("06 — Covariate Adjustment (Age + Sex)")
print("=" * 60)

# Load expression data
print("\nLoading GTEx v8 data...")
if GTEX_TPM.endswith('.parquet'):
    tpm = pd.read_parquet(GTEX_TPM)
else:
    tpm = pd.read_csv(GTEX_TPM, sep='\t', index_col=0)

if GTEX_ATTR.endswith('.parquet'):
    attr = pd.read_parquet(GTEX_ATTR)
else:
    attr = pd.read_csv(GTEX_ATTR, sep='\t')

# Load phenotype data
pheno = pd.read_csv(GTEX_PHENO, sep='\t')
print(f"  Phenotype file: {pheno.shape[0]} donors")

# Identify columns
smtsd_col = [c for c in attr.columns if 'SMTSD' in c.upper()][0]
sampid_col = [c for c in attr.columns if 'SAMPID' in c.upper()][0]
subjid_col = [c for c in pheno.columns if 'SUBJID' in c.upper()][0]
age_col = [c for c in pheno.columns if 'AGE' in c.upper()][0]
sex_col = [c for c in pheno.columns if 'SEX' in c.upper()][0]

# Filter to BA9 samples
brain_attr = attr[attr[smtsd_col] == BRAIN_REGION].copy()
brain_samples = brain_attr[sampid_col].tolist()
brain_cols = [c for c in tpm.columns if c in brain_samples]
brain_tpm = tpm[brain_cols]

# Extract donor ID from sample ID (GTEx format: GTEX-XXXXX-...)
def sample_to_donor(sample_id):
    parts = sample_id.split('-')
    return '-'.join(parts[:2])  # e.g., GTEX-1117F

donor_map = {s: sample_to_donor(s) for s in brain_cols}

# Get age and sex for each sample
pheno_dict = pheno.set_index(subjid_col)
ages = []
sexes = []
valid_cols = []

for sample in brain_cols:
    donor = donor_map[sample]
    if donor in pheno_dict.index:
        age_val = pheno_dict.loc[donor, age_col]
        sex_val = pheno_dict.loc[donor, sex_col]
        # GTEx age is binned (e.g., "60-69"); use midpoint
        if isinstance(age_val, str) and '-' in age_val:
            lo, hi = age_val.split('-')
            age_num = (int(lo) + int(hi)) / 2
        else:
            age_num = float(age_val) if pd.notna(age_val) else np.nan
        sex_num = float(sex_val) if pd.notna(sex_val) else np.nan
        if not np.isnan(age_num) and not np.isnan(sex_num):
            ages.append(age_num)
            sexes.append(sex_num)
            valid_cols.append(sample)

print(f"  BA9 samples with age+sex data: {len(valid_cols)}")

# Filter expression to valid samples
brain_tpm_valid = brain_tpm[valid_cols]
gene_medians = brain_tpm_valid.median(axis=1)
expressed = gene_medians[gene_medians >= MIN_MEDIAN_TPM].index
brain_log = np.log2(brain_tpm_valid.loc[expressed] + 1)

ages_arr = np.array(ages)
sexes_arr = np.array(sexes)
covariates = [ages_arr, sexes_arr]

print(f"  Expressed genes: {len(expressed):,}")
print(f"  Age range: {ages_arr.min():.0f}–{ages_arr.max():.0f}")
print(f"  Sex (1=M, 2=F): {(sexes_arr==1).sum()} male, {(sexes_arr==2).sum()} female")

# Compute partial correlations
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
        r_adj, p_adj = partial_corr(target, other, covariates)
        results.append({
            'gene': other_gene,
            'adjusted_pearson_r': round(r_adj, 6),
            'adjusted_p_value': p_adj,
        })

    df = pd.DataFrame(results)
    df = df.sort_values('adjusted_pearson_r', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    outpath = f"{OUTPUT_DIR}/{gene}_covariate_adjusted_correlations.csv"
    df.to_csv(outpath, index=False)
    print(f"  Saved: {outpath}")

    # Compare with unadjusted
    unadj_path = f"{RESULTS_DIR}/{gene}_genome_wide_correlations.csv"
    if os.path.exists(unadj_path):
        unadj = pd.read_csv(unadj_path)
        gene_col = [c for c in unadj.columns if c.lower() in ['gene', 'gene_name', 'symbol']][0]
        unadj_ranks = unadj.set_index(gene_col)['rank']
        adj_ranks = df.set_index('gene')['rank']
        common = unadj_ranks.index.intersection(adj_ranks.index)
        rho, _ = stats.spearmanr(unadj_ranks.loc[common], adj_ranks.loc[common])
        print(f"  Rank correlation with unadjusted: ρ = {rho:.4f}")
        comparison_results.append({
            'gene': gene,
            'rank_correlation_unadj_vs_adjusted': round(rho, 4),
            'n_genes_compared': len(common),
        })

if comparison_results:
    comp_df = pd.DataFrame(comparison_results)
    comp_df.to_csv(f"{OUTPUT_DIR}/covariate_adjustment_comparison.csv", index=False)


print(f"\n{'=' * 60}")
print("✓ Covariate adjustment analysis complete")
print(f"{'=' * 60}")
