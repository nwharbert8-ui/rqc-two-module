"""
07_agonal_stress_sensitivity.py
================================
Sensitivity analysis controlling for agonal stress covariates: Hardy Scale
(DTHHRDY; death circumstance) and ischemic time (SMTSISCH).

Confirms that the two-module RQC architecture reflects constitutive brain
biology rather than post-mortem artifacts.

Models evaluated:
    (i)   Hardy Scale alone
    (ii)  Ischemic time alone
    (iii) Hardy Scale + ischemic time
    (iv)  Full model: age + sex + Hardy Scale + ischemic time

Output:
    agonal_stress_sensitivity.csv  — rank preservation for each model/gene
    agonal_stress_jaccard.csv      — Jaccard indices under each model

Usage:
    python 07_agonal_stress_sensitivity.py
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
PHENO_FILE = os.path.join(DATA_DIR,
    "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt")
SAMPLE_FILE = os.path.join(DATA_DIR,
    "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt")

TARGET_GENES = ["PELO", "LTN1", "NEMF"]
TOP_PCT = 0.05

# ═══════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════
print("Loading expression data...")
tpm = pd.read_parquet(TPM_FILE)
expr = np.log2(tpm + 1)

print("Loading phenotype data...")
pheno = pd.read_csv(PHENO_FILE, sep="\t")
pheno["SUBJID"] = pheno["SUBJID"].astype(str)

samples = pd.read_csv(SAMPLE_FILE, sep="\t")
samples["SUBJID"] = samples["SAMPID"].str.extract(r"(GTEX-[^-]+)")[0]

# Merge sample-level and subject-level metadata
meta = samples[samples["SAMPID"].isin(expr.index)][
    ["SAMPID", "SUBJID", "SMTSISCH"]
].merge(pheno[["SUBJID", "AGE", "SEX", "DTHHRDY"]], on="SUBJID", how="left")
meta = meta.set_index("SAMPID").loc[expr.index]

# Encode age as midpoint of GTEx decade bins
age_map = {"20-29": 25, "30-39": 35, "40-49": 45, "50-59": 55, "60-69": 65, "70-79": 75}
meta["AGE_NUM"] = meta["AGE"].map(age_map)

# Drop samples with missing covariates
valid = meta.dropna(subset=["AGE_NUM", "SEX", "DTHHRDY", "SMTSISCH"]).index
expr_valid = expr.loc[valid]
meta_valid = meta.loc[valid]
print(f"  {len(valid)} samples with complete covariates")

# ═══════════════════════════════════════════════════════
# Define sensitivity models
# ═══════════════════════════════════════════════════════
models = {
    "unadjusted": [],
    "hardy_only": ["DTHHRDY"],
    "ischemic_only": ["SMTSISCH"],
    "hardy_ischemic": ["DTHHRDY", "SMTSISCH"],
    "full_model": ["AGE_NUM", "SEX", "DTHHRDY", "SMTSISCH"],
}

def partial_corr(x, y, covariates):
    """Compute partial Pearson correlation controlling for covariates."""
    if len(covariates) == 0:
        return stats.pearsonr(x, y)[0]
    Z = np.column_stack(covariates)
    # Residualize x and y
    Z_aug = np.column_stack([Z, np.ones(len(Z))])
    x_resid = x - Z_aug @ np.linalg.lstsq(Z_aug, x, rcond=None)[0]
    y_resid = y - Z_aug @ np.linalg.lstsq(Z_aug, y, rcond=None)[0]
    return stats.pearsonr(x_resid, y_resid)[0]

# ═══════════════════════════════════════════════════════
# Compute correlations under each model
# ═══════════════════════════════════════════════════════
all_genes = [g for g in expr_valid.columns if g not in TARGET_GENES]
n_top = int(len(all_genes) * TOP_PCT)

results = []
rankings = {}  # {(model, gene): pd.Series of ranks}

for model_name, cov_cols in models.items():
    print(f"\nModel: {model_name}")
    covs = [meta_valid[c].values.astype(float) for c in cov_cols]

    for target in TARGET_GENES:
        target_vals = expr_valid[target].values
        corrs = {}
        for gene in all_genes:
            corrs[gene] = partial_corr(target_vals, expr_valid[gene].values, covs)

        ranking = pd.Series(corrs).sort_values(ascending=False)
        rankings[(model_name, target)] = ranking

        top_genes = set(ranking.head(n_top).index)

        print(f"  {target}: top partner = {ranking.index[0]} "
              f"(r = {ranking.iloc[0]:.4f}), threshold = {ranking.iloc[n_top-1]:.4f}")

for target in TARGET_GENES:
    unadj_rank = rankings[("unadjusted", target)].rank(ascending=False)
    for model_name in models:
        if model_name == "unadjusted":
            continue
        adj_rank = rankings[(model_name, target)].rank(ascending=False)
        common = unadj_rank.index.intersection(adj_rank.index)
        rho, _ = stats.spearmanr(unadj_rank[common], adj_rank[common])
        results.append({
            "gene": target,
            "model": model_name,
            "rank_preservation_rho": round(rho, 6),
        })
        print(f"  {target} | {model_name}: rank preservation ρ = {rho:.4f}")

# ═══════════════════════════════════════════════════════
# Jaccard under each model
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("JACCARD INDICES UNDER EACH MODEL")
print("=" * 60)

jaccard_results = []
for model_name in models:
    top_sets = {}
    for target in TARGET_GENES:
        ranking = rankings[(model_name, target)]
        top_sets[target] = set(ranking.head(n_top).index)

    for g1, g2 in [("PELO", "LTN1"), ("PELO", "NEMF"), ("LTN1", "NEMF")]:
        inter = len(top_sets[g1] & top_sets[g2])
        union = len(top_sets[g1] | top_sets[g2])
        j = inter / union if union > 0 else 0
        jaccard_results.append({
            "model": model_name,
            "pair": f"{g1}-{g2}",
            "jaccard": round(j, 4),
            "shared": inter,
        })
        print(f"  {model_name:20s} | {g1}-{g2}: J = {j:.4f} ({inter} shared)")

# ═══════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════
pd.DataFrame(results).to_csv(
    os.path.join(OUT_DIR, "agonal_stress_sensitivity.csv"), index=False)
pd.DataFrame(jaccard_results).to_csv(
    os.path.join(OUT_DIR, "agonal_stress_jaccard.csv"), index=False)

print(f"\nSaved: agonal_stress_sensitivity.csv")
print(f"Saved: agonal_stress_jaccard.csv")
print("Done.")
