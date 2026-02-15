"""
MS1_RQC_COMPLETE.py
====================
Complete reproducible pipeline for:
  "Two-module architecture of ribosome-associated quality control
   in human brain" — Drake H. Harbert, Inner Architecture LLC

Generates: ALL manuscript numbers, Figures 1-3, Tables 1-3,
           Supplementary Tables S1-S4, summary output.

Environment: Google Colab, ≤12 GB RAM, Google Drive mounted.
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL 0 — Install dependencies
# ═══════════════════════════════════════════════════════════════════════
# !pip install matplotlib-venn --quiet

# ═══════════════════════════════════════════════════════════════════════
# CELL 1 — Mount drive, load ALL brain regions (RAM-safe)
# ═══════════════════════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
from scipy import stats
import os, urllib.request, gc, json, time, requests, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_venn import venn3
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.dpi': 300,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})

OUT = '/content/drive/MyDrive/MS1_RQC_Output'
os.makedirs(OUT, exist_ok=True)
os.makedirs(f'{OUT}/figures', exist_ok=True)
os.makedirs(f'{OUT}/tables', exist_ok=True)
os.makedirs(f'{OUT}/supplementary', exist_ok=True)

GCT_FILE = '/content/drive/MyDrive/GenomicAnalysis/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz'
PHENO_FILE = '/content/drive/MyDrive/PCDH_SIGMAR1_Analysis/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt'

SAMPLE_FILE = '/content/GTEx_SampleAttributes.txt'
if not os.path.exists(SAMPLE_FILE):
    print("Downloading GTEx SampleAttributes...")
    url = "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
    urllib.request.urlretrieve(url, SAMPLE_FILE)
    print("  Done.")

TARGETS = ["PELO", "LTN1", "NEMF"]
TOP_PCT = 0.05

REGIONS = {
    "BA9":  "Brain - Frontal Cortex (BA9)",
    "PUT":  "Brain - Putamen (basal ganglia)",
    "HIP":  "Brain - Hippocampus",
    "NAC":  "Brain - Nucleus accumbens (basal ganglia)",
    "BA24": "Brain - Anterior cingulate cortex (BA24)",
}

# Identify all brain samples across 5 regions
samples_df = pd.read_csv(SAMPLE_FILE, sep="\t")

# Diagnostic: print actual brain tissue names in dataset
actual_brain = samples_df[samples_df["SMTSD"].str.contains("Brain", na=False)]["SMTSD"].unique()
print("  GTEx brain tissues found in SampleAttributes:")
for t in sorted(actual_brain):
    print(f"    '{t}'")

region_samples = {}
all_brain_ids = []
for abbr, full_name in REGIONS.items():
    ids = samples_df[samples_df["SMTSD"] == full_name]["SAMPID"].tolist()
    region_samples[abbr] = ids
    all_brain_ids.extend(ids)
    status = "OK" if len(ids) > 0 else "*** NOT FOUND — check tissue name ***"
    print(f"  {abbr}: {len(ids)} samples  {status}")

# Read only brain columns from GCT (single load for all regions)
header = pd.read_csv(GCT_FILE, sep='\t', skiprows=2, nrows=0)
brain_cols = [c for c in header.columns if c in all_brain_ids]
use_cols = ["Name", "Description"] + brain_cols
del header; gc.collect()

print(f"\nLoading {len(brain_cols)} brain samples from GCT...")
gct = pd.read_csv(GCT_FILE, sep='\t', skiprows=2, usecols=use_cols)
gene_names = gct["Description"].values
tpm_all = gct[brain_cols].T.copy()
tpm_all.columns = gene_names
del gct; gc.collect()

# Deduplicate: keep highest median per gene symbol
tpm_all = tpm_all.T.groupby(level=0).max().T
print(f"  Total: {tpm_all.shape[0]} samples, {tpm_all.shape[1]} genes (pre-filter)")

# Load phenotype data
pheno = pd.read_csv(PHENO_FILE, sep="\t")
samples_df["SUBJID"] = samples_df["SAMPID"].str.extract(r"(GTEX-[^-]+)")[0]
age_map = {"20-29": 25, "30-39": 35, "40-49": 45,
           "50-59": 55, "60-69": 65, "70-79": 75}

# Build per-region expression matrices (filtered, log-transformed)
region_data = {}
for abbr in REGIONS:
    ids = [s for s in region_samples[abbr] if s in tpm_all.index]
    tpm_r = tpm_all.loc[ids]
    medians = tpm_r.median(axis=0)
    expressed = medians[medians >= 1.0].index
    tpm_r = tpm_r[expressed]
    expr_r = np.log2(tpm_r + 1)

    # Merge covariates for primary region
    meta_r = samples_df[samples_df["SAMPID"].isin(ids)][
        ["SAMPID", "SUBJID", "SMTSISCH"]
    ].merge(
        pheno[["SUBJID", "AGE", "SEX", "DTHHRDY"]], on="SUBJID", how="left"
    ).set_index("SAMPID").reindex(expr_r.index)
    meta_r["AGE_NUM"] = meta_r["AGE"].map(age_map)

    region_data[abbr] = {
        "expr": expr_r, "tpm": tpm_r, "meta": meta_r,
        "n_samples": expr_r.shape[0], "n_genes": expr_r.shape[1]
    }
    missing = [t for t in TARGETS if t not in expr_r.columns]
    print(f"  {abbr}: {expr_r.shape[0]} samples, {expr_r.shape[1]} genes"
          + (f" [MISSING: {missing}]" if missing else ""))

del tpm_all, pheno; gc.collect()
print("\n  All regions loaded.\n")


# ═══════════════════════════════════════════════════════════════════════
# CELL 2 — Primary analysis: genome-wide correlations + overlaps (BA9)
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PRIMARY ANALYSIS — BA9")
print("=" * 70)

expr = region_data["BA9"]["expr"]
tpm  = region_data["BA9"]["tpm"]
meta = region_data["BA9"]["meta"]
N_SAMPLES = expr.shape[0]
N_GENES   = expr.shape[1]

# Subset to samples with complete covariates for sensitivity analyses
valid = meta.dropna(subset=["AGE_NUM", "SEX", "DTHHRDY", "SMTSISCH"]).index
expr_v = expr.loc[valid]
meta_v = meta.loc[valid]

for t in TARGETS:
    print(f"  {t}: median TPM = {tpm[t].median():.2f}")

other_genes = [g for g in expr_v.columns if g not in TARGETS]
n_other = len(other_genes)
n_top = int(n_other * TOP_PCT)
print(f"\n  {N_SAMPLES} samples, {N_GENES} expressed genes")
print(f"  {len(valid)} samples with complete covariates")
print(f"  {n_other} non-target genes, top 5% = {n_top}")

# Genome-wide Pearson correlations
print("\nComputing genome-wide correlations...")
correlations = {}
top5_sets = {}
thresholds = {}

for target in TARGETS:
    t_vals = expr_v[target].values
    corrs = {}
    for g in other_genes:
        corrs[g] = stats.pearsonr(t_vals, expr_v[g].values)[0]
    ranking = pd.Series(corrs).sort_values(ascending=False)
    correlations[target] = ranking
    top5_sets[target] = set(ranking.head(n_top).index)
    thresholds[target] = ranking.iloc[n_top - 1]
    print(f"  {target}: top = {ranking.index[0]} (r={ranking.iloc[0]:.4f}), "
          f"threshold r >= {thresholds[target]:.4f}")

# ── Pairwise statistics ──
print(f"\nPAIRWISE NETWORK COMPARISON:")
pairs = [("PELO", "LTN1"), ("PELO", "NEMF"), ("LTN1", "NEMF")]
pairwise = {}

for g1, g2 in pairs:
    s1, s2 = top5_sets[g1], top5_sets[g2]
    inter = len(s1 & s2)
    union = len(s1 | s2)
    j = inter / union
    a, b, c = inter, len(s1) - inter, len(s2) - inter
    d = n_other - len(s1 | s2)
    oddsratio, fisher_p = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
    r1, r2 = correlations[g1], correlations[g2]
    common = r1.index.intersection(r2.index)
    rho, _ = stats.spearmanr(r1[common], r2[common])
    pairwise[(g1, g2)] = {
        "jaccard": j, "shared": inter, "union": union,
        "or": oddsratio, "fisher_p": fisher_p, "spearman": rho
    }
    p_str = f"{fisher_p:.2e}" if fisher_p > 0 else "< 1e-324"
    print(f"  {g1}-{g2}: J={j:.4f}, shared={inter}, "
          f"OR={oddsratio:.2f}, p={p_str}, rho={rho:.4f}")

# ── Three-way overlap ──
P, L, N = top5_sets["PELO"], top5_sets["LTN1"], top5_sets["NEMF"]
all_three      = P & L & N
pelo_ltn1_only = (P & L) - N
pelo_nemf_only = (P & N) - L
ltn1_nemf_only = (L & N) - P
pelo_only      = P - L - N
ltn1_only      = L - P - N
nemf_only      = N - P - L

total_unique = len(P | L | N)
venn_regions = {
    "All three (PELO ∩ LTN1 ∩ NEMF)": (len(all_three), all_three),
    "LTN1 ∩ NEMF only": (len(ltn1_nemf_only), ltn1_nemf_only),
    "PELO ∩ NEMF only": (len(pelo_nemf_only), pelo_nemf_only),
    "PELO ∩ LTN1 only": (len(pelo_ltn1_only), pelo_ltn1_only),
    "PELO unique": (len(pelo_only), pelo_only),
    "LTN1 unique": (len(ltn1_only), ltn1_only),
    "NEMF unique": (len(nemf_only), nemf_only),
}

print(f"\nTHREE-WAY OVERLAP:")
for name, (count, _) in venn_regions.items():
    pct = count / total_unique * 100
    print(f"  {name:35s}: {count:5d}  ({pct:.1f}%)")
print(f"  {'Total unique':35s}: {total_unique:5d}")

# Store gene sets for enrichment
gene_sets_for_enrichment = {
    "PELO_full": list(top5_sets["PELO"]),
    "LTN1_full": list(top5_sets["LTN1"]),
    "NEMF_full": list(top5_sets["NEMF"]),
    "PELO_unique": list(pelo_only),
    "LTN1_unique": list(ltn1_only),
    "NEMF_unique": list(nemf_only),
    "shared_all_three": list(all_three),
    "LTN1_NEMF_shared": list(ltn1_nemf_only),
}
background_genes = list(expr_v.columns)


# ═══════════════════════════════════════════════════════════════════════
# CELL 3 — Agonal stress sensitivity
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("AGONAL STRESS SENSITIVITY")
print("="*70)

# Hardy Scale distribution
hardy_counts = meta_v["DTHHRDY"].value_counts().sort_index()
print(f"\n  Hardy Scale distribution (n={len(meta_v)}):")
hardy_labels = {0: "Ventilator", 1: "Violent/fast", 2: "Fast natural",
                3: "Intermediate", 4: "Slow illness"}
for val, cnt in hardy_counts.items():
    print(f"    {int(val)} ({hardy_labels.get(int(val), '?')}): n={cnt}")
print(f"  Ischemic time: median={meta_v['SMTSISCH'].median():.0f} min, "
      f"range={meta_v['SMTSISCH'].min():.0f}-{meta_v['SMTSISCH'].max():.0f}")

def partial_corr_vector(target_vals, expr_df, genes, covariates):
    """Compute partial Pearson correlations controlling for covariates."""
    if len(covariates) == 0:
        return pd.Series({g: stats.pearsonr(target_vals, expr_df[g].values)[0]
                          for g in genes})
    Z = np.column_stack(covariates + [np.ones(len(covariates[0]))])
    t_resid = target_vals - Z @ np.linalg.lstsq(Z, target_vals, rcond=None)[0]
    corrs = {}
    for g in genes:
        g_vals = expr_df[g].values
        g_resid = g_vals - Z @ np.linalg.lstsq(Z, g_vals, rcond=None)[0]
        corrs[g] = stats.pearsonr(t_resid, g_resid)[0]
    return pd.Series(corrs)

models = {
    "Hardy Scale": [meta_v["DTHHRDY"].values.astype(float)],
    "Ischemic time": [meta_v["SMTSISCH"].values.astype(float)],
    "Hardy + Ischemic": [meta_v["DTHHRDY"].values.astype(float),
                         meta_v["SMTSISCH"].values.astype(float)],
    "Full model": [meta_v["AGE_NUM"].values.astype(float),
                   meta_v["SEX"].values.astype(float),
                   meta_v["DTHHRDY"].values.astype(float),
                   meta_v["SMTSISCH"].values.astype(float)],
}

rp_results = []
adj_jaccards = {}

for model_name, covs in models.items():
    print(f"\n  Model: {model_name}")
    adj_rankings = {}
    for target in TARGETS:
        t_vals = expr_v[target].values
        ranking = partial_corr_vector(t_vals, expr_v, other_genes, covs)\
                  .sort_values(ascending=False)
        adj_rankings[target] = ranking
        # Rank preservation
        unadj_rank = correlations[target].rank(ascending=False)
        adj_rank = ranking.rank(ascending=False)
        common = unadj_rank.index.intersection(adj_rank.index)
        rho, _ = stats.spearmanr(unadj_rank[common], adj_rank[common])
        rp_results.append({"gene": target, "model": model_name, "rho": rho})
        print(f"    {target}: rank preservation rho = {rho:.6f}")

    # Jaccard under this model
    top_adj = {t: set(adj_rankings[t].head(n_top).index) for t in TARGETS}
    for g1, g2 in pairs:
        inter = len(top_adj[g1] & top_adj[g2])
        union = len(top_adj[g1] | top_adj[g2])
        adj_jaccards[(model_name, g1, g2)] = inter / union

# Partition test
print(f"\n  PARTITION TEST (LTN1-NEMF J / max PELO J):")
for model_name in ["Unadjusted"] + list(models.keys()):
    if model_name == "Unadjusted":
        pl = pairwise[("PELO","LTN1")]["jaccard"]
        pn = pairwise[("PELO","NEMF")]["jaccard"]
        ln = pairwise[("LTN1","NEMF")]["jaccard"]
    else:
        pl = adj_jaccards.get((model_name, "PELO", "LTN1"), 0)
        pn = adj_jaccards.get((model_name, "PELO", "NEMF"), 0)
        ln = adj_jaccards.get((model_name, "LTN1", "NEMF"), 0)
    ratio = ln / max(pl, pn)
    print(f"    {model_name:20s}: {ratio:.2f}x")


# ═══════════════════════════════════════════════════════════════════════
# CELL 4 — Multi-region replication
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("MULTI-REGION REPLICATION")
print("="*70)

region_correlations = {}

for abbr in REGIONS:
    print(f"\n  Processing {abbr}...")
    rd = region_data[abbr]
    expr_r = rd["expr"]
    other_r = [g for g in expr_r.columns if g not in TARGETS]
    n_top_r = int(len(other_r) * TOP_PCT)

    region_corr = {}
    for target in TARGETS:
        if target not in expr_r.columns:
            print(f"    {target}: NOT EXPRESSED")
            continue
        t_vals = expr_r[target].values
        corrs = {}
        for g in other_r:
            corrs[g] = stats.pearsonr(t_vals, expr_r[g].values)[0]
        region_corr[target] = pd.Series(corrs).sort_values(ascending=False)
    region_correlations[abbr] = region_corr

# Cross-region Spearman rank correlations
# Use genes expressed in ALL regions as common universe for consistency
region_abbrs = list(REGIONS.keys())
all_region_genes = None
for abbr in region_abbrs:
    rd = region_data[abbr]
    genes_in_region = set(rd["expr"].columns) - set(TARGETS)
    if all_region_genes is None:
        all_region_genes = genes_in_region
    else:
        all_region_genes = all_region_genes.intersection(genes_in_region)
print(f"\n  Common gene universe across all 5 regions: {len(all_region_genes)} genes")

print(f"\n  Cross-region rank correlations:")
region_abbrs = list(REGIONS.keys())
cross_region_rho = {t: pd.DataFrame(np.nan, index=region_abbrs,
                    columns=region_abbrs) for t in TARGETS}

for target in TARGETS:
    for i, r1 in enumerate(region_abbrs):
        for j, r2 in enumerate(region_abbrs):
            if i >= j:
                if r1 == r2:
                    cross_region_rho[target].loc[r1, r2] = 1.0
                    continue
                c1 = region_correlations[r1].get(target)
                c2 = region_correlations[r2].get(target)
                if c1 is None or c2 is None:
                    continue
                # Restrict to common universe across ALL regions
                common = sorted(all_region_genes.intersection(
                    c1.index).intersection(c2.index))
                rho, _ = stats.spearmanr(c1[common], c2[common])
                cross_region_rho[target].loc[r1, r2] = rho
                cross_region_rho[target].loc[r2, r1] = rho

    # Summary stats
    vals = []
    pair_details = []
    for i, r1 in enumerate(region_abbrs):
        for j, r2 in enumerate(region_abbrs):
            if i < j:
                v = cross_region_rho[target].loc[r1, r2]
                if not np.isnan(v):
                    vals.append(v)
                    pair_details.append((r1, r2, v))
    print(f"  {target}: min={min(vals):.3f}, max={max(vals):.3f}, "
          f"median={np.median(vals):.3f}, all>0.80={all(v>0.80 for v in vals)}")
    # Show any pairs below 0.80
    low_pairs = [(r1, r2, v) for r1, r2, v in pair_details if v < 0.80]
    if low_pairs:
        for r1, r2, v in low_pairs:
            n1 = len(region_correlations[r1].get(target, []))
            n2 = len(region_correlations[r2].get(target, []))
            print(f"    *** LOW: {r1}-{r2} rho={v:.3f} (genes: {r1}={n1}, {r2}={n2})")


# ═══════════════════════════════════════════════════════════════════════
# CELL 5 — Cell-type deconvolution
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("CELL-TYPE DECONVOLUTION")
print("="*70)

CELL_MARKERS = {
    "Neurons": ["SYT1","SNAP25","SLC17A7","GRIN1","GAD1","GAD2",
                "SLC32A1","RBFOX3","STMN2","NRGN","SYP"],
    "Astrocytes": ["GFAP","AQP4","SLC1A2","SLC1A3","ALDH1L1",
                   "GJA1","S100B","SOX9","GLUL"],
    "Oligodendrocytes": ["MBP","MOG","PLP1","OLIG1","OLIG2",
                         "MAG","CNP","CLDN11","MOBP"],
    "Microglia": ["CX3CR1","P2RY12","TMEM119","CSF1R","AIF1",
                  "CD68","ITGAM","TREM2","HEXB"],
    "Endothelial": ["CLDN5","FLT1","VWF","PECAM1","CDH5",
                    "TIE1","ERG","ESAM"],
    "OPCs": ["PDGFRA","CSPG4","VCAN","GPR17","NEU4",
             "PCDH15","SOX10"],
}

# Count total markers available
all_markers = []
for ct, genes in CELL_MARKERS.items():
    available = [g for g in genes if g in expr_v.columns]
    all_markers.extend(available)
    print(f"  {ct}: {len(available)}/{len(genes)} markers expressed")
print(f"  Total marker genes used: {len(all_markers)}")

# Estimate cell-type proportions (mean log2 expression of markers)
ct_props = pd.DataFrame(index=expr_v.index)
for ct, genes in CELL_MARKERS.items():
    available = [g for g in genes if g in expr_v.columns]
    if available:
        ct_props[ct] = expr_v[available].mean(axis=1)

# Partial correlations controlling for cell-type proportions
print(f"\n  Computing cell-type-adjusted correlations...")
ct_covariates = [ct_props[col].values for col in ct_props.columns]

ct_adj_correlations = {}
for target in TARGETS:
    t_vals = expr_v[target].values
    ranking = partial_corr_vector(t_vals, expr_v, other_genes, ct_covariates)\
              .sort_values(ascending=False)
    ct_adj_correlations[target] = ranking
    # Rank preservation
    unadj_rank = correlations[target].rank(ascending=False)
    adj_rank = ranking.rank(ascending=False)
    common = unadj_rank.index.intersection(adj_rank.index)
    rho, _ = stats.spearmanr(unadj_rank[common], adj_rank[common])
    print(f"    {target}: rank preservation rho = {rho:.4f}, "
          f"adj threshold = {ranking.iloc[n_top-1]:.4f}")

# Check LTN1-NEMF convergence after cell-type adjustment
ct_top5 = {t: set(ct_adj_correlations[t].head(n_top).index) for t in TARGETS}
for g1, g2 in pairs:
    inter = len(ct_top5[g1] & ct_top5[g2])
    union = len(ct_top5[g1] | ct_top5[g2])
    j = inter / union
    print(f"  Cell-type adjusted {g1}-{g2}: J={j:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 6 — gProfiler enrichment analysis
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("GENE ONTOLOGY ENRICHMENT (gProfiler)")
print("="*70)

# Install gprofiler-official (handles API formatting correctly)
import subprocess
subprocess.call(['pip', 'uninstall', '-y', '-q', 'gProfiler'], 
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.check_call(['pip', 'install', '-q', 'gprofiler-official'])
from gprofiler import GProfiler

gp = GProfiler(return_dataframe=True)

def run_gprofiler(gene_list, background, name="query"):
    """Run gProfiler enrichment using official Python package."""
    try:
        # Try with custom background first
        df = gp.profile(
            organism='hsapiens',
            query=gene_list,
            sources=['GO:BP', 'GO:MF', 'GO:CC', 'KEGG', 'REAC'],
            user_threshold=0.05,
            significance_threshold_method='g_SCS',
            background=background,
            no_evidences=True,
        )
        if df is not None and len(df) > 0:
            # Standardize column names
            if 'native' in df.columns and 'term_id' not in df.columns:
                df = df.rename(columns={'native': 'term_id'})
            if 'name' in df.columns and 'term_name' not in df.columns:
                df = df.rename(columns={'name': 'term_name'})
            df['gene_set'] = name
            print(f"    OK (custom bg, {len(df)} terms)")
            return df
    except Exception as e:
        print(f"    Custom bg failed: {e}")

    try:
        # Fallback: annotated background
        df = gp.profile(
            organism='hsapiens',
            query=gene_list,
            sources=['GO:BP', 'GO:MF', 'GO:CC', 'KEGG', 'REAC'],
            user_threshold=0.05,
            significance_threshold_method='g_SCS',
            no_evidences=True,
        )
        if df is not None and len(df) > 0:
            if 'native' in df.columns and 'term_id' not in df.columns:
                df = df.rename(columns={'native': 'term_id'})
            if 'name' in df.columns and 'term_name' not in df.columns:
                df = df.rename(columns={'name': 'term_name'})
            df['gene_set'] = name
            print(f"    OK (annotated bg, {len(df)} terms)")
            return df
    except Exception as e:
        print(f"    Annotated bg failed: {e}")

    return pd.DataFrame()

all_enrichment = []
enrichment_sets = {
    "PELO_full": list(top5_sets["PELO"]),
    "LTN1_full": list(top5_sets["LTN1"]),
    "NEMF_full": list(top5_sets["NEMF"]),
    "PELO_unique": list(pelo_only),
    "LTN1_unique": list(ltn1_only),
    "NEMF_unique": list(nemf_only),
    "Shared_all_three": list(all_three),
    "LTN1_NEMF_shared": list(ltn1_nemf_only),
    "PELO_LTN1_shared": list(pelo_ltn1_only),
    "PELO_NEMF_shared": list(pelo_nemf_only),
}

for set_name, genes in enrichment_sets.items():
    if len(genes) < 5:
        print(f"  {set_name}: {len(genes)} genes — skipping (too few)")
        continue
    print(f"  {set_name}: {len(genes)} genes — querying gProfiler...")
    df = run_gprofiler(genes, background_genes, name=set_name)
    if len(df) > 0:
        all_enrichment.append(df)
        sig = df[df['p_value'] < 0.05]
        print(f"    {len(sig)} significant terms")
        # Print top 3 per source
        for src in ['GO:BP', 'GO:MF', 'GO:CC', 'KEGG', 'REAC']:
            sub = sig[sig['source'] == src].sort_values('p_value').head(3)
            for _, row in sub.iterrows():
                print(f"      {src}: {row['term_name']} (p={row['p_value']:.2e})")
    else:
        print(f"    No significant terms")
    time.sleep(1)  # Rate limiting

if all_enrichment:
    enrichment_df = pd.concat(all_enrichment, ignore_index=True)
    enrichment_df.to_csv(f'{OUT}/supplementary/TableS2_enrichment_all.csv', index=False)
    print(f"\n  Saved {len(enrichment_df)} total enrichment terms to TableS2")


# ═══════════════════════════════════════════════════════════════════════
# CELL 7 — Custom gene set enrichment (epigenetic regulators + others)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("CUSTOM GENE SET ENRICHMENT")
print("="*70)

CUSTOM_SETS = {
    "Epigenetic regulators": [
        "CHD1","CHD2","CHD3","CHD4","CHD5","CHD6","CHD7","CHD8","CHD9",
        "SMARCA2","SMARCA4","SMARCB1","SMARCC1","SMARCC2","SMARCD1",
        "ARID1A","ARID1B","ARID2",
        "DNMT1","DNMT3A","DNMT3B",
        "TET1","TET2","TET3",
        "KDM1A","KDM2A","KDM2B","KDM3A","KDM4A","KDM5A","KDM5B","KDM6A","KDM6B",
        "KAT2A","KAT2B","KAT5","KAT6A","KAT7","KAT8",
        "HDAC1","HDAC2","HDAC3","HDAC4","HDAC5","HDAC6","HDAC7","HDAC8",
        "HDAC9","HDAC10","HDAC11",
        "EZH1","EZH2","SUZ12","EED",
    ],
    "Ribosome quality control": [
        "PELO","HBS1L","LTN1","NEMF","ANKZF1","VCP","UFD1","NPLOC4",
        "ZNF598","RACK1","ABCE1","TCF25",
    ],
    "Proteostasis network": [
        "HSPA1A","HSPA5","HSPA8","HSP90AA1","HSP90AB1","HSP90B1",
        "UBE2D1","UBE2D3","UBE2N","UBE3A","UBB","UBC",
        "PSMA1","PSMA3","PSMB1","PSMB5","PSMC2","PSMD4",
        "ATG5","ATG7","ATG12","BECN1","MAP1LC3B","SQSTM1",
    ],
    "Vascular markers (negative control)": [
        "PECAM1","CDH5","VWF","FLT1","KDR","ENG","CLDN5","ESAM",
        "ERG","TIE1","TEK","ANGPT1","ANGPT2","NOS3","MCAM",
        "PODXL","EMCN","ROBO4",
    ],
}

n_custom_tests = 0
custom_results = []

for set_name, gene_list in CUSTOM_SETS.items():
    expressed = [g for g in gene_list if g in expr_v.columns]
    # Remove targets from consideration if present
    expressed_clean = [g for g in expressed if g not in TARGETS]

    for target in TARGETS:
        top_set = top5_sets[target]
        in_top = [g for g in expressed_clean if g in top_set]
        k = len(in_top)
        n_set = len(expressed_clean)
        if n_set == 0:
            continue
        n_custom_tests += 1

        # Fisher's exact test (one-sided)
        a = k
        b = len(top_set) - k
        c = n_set - k
        d = n_other - len(top_set) - c
        d = max(d, 0)
        _, p = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
        fold = (k / n_set) / (n_top / n_other) if n_set > 0 else 0

        custom_results.append({
            "gene_set": set_name, "target": target,
            "in_top5": k, "total_expressed": n_set,
            "fold_enrichment": fold, "p_value": p,
            "genes_in_top5": ", ".join(sorted(in_top)) if in_top else "",
        })

        if p < 0.05:
            print(f"  ** {target} x {set_name}: "
                  f"{k}/{n_set} in top 5%, {fold:.1f}-fold, p={p:.2e}")
            if in_top:
                print(f"     Genes: {', '.join(sorted(in_top))}")

custom_df = pd.DataFrame(custom_results)

# Bonferroni correction for all custom tests
custom_df["p_bonferroni"] = np.minimum(custom_df["p_value"] * n_custom_tests, 1.0)
print(f"\n  Total custom tests: {n_custom_tests}")
print(f"  Bonferroni-significant results:")
sig_custom = custom_df[custom_df["p_bonferroni"] < 0.05]
for _, row in sig_custom.iterrows():
    print(f"    {row['target']} x {row['gene_set']}: p_raw={row['p_value']:.2e}, "
          f"p_bonf={row['p_bonferroni']:.2e}")

# Multi-region custom enrichment for NEMF epigenetic regulators
print(f"\n  Multi-region epigenetic enrichment for NEMF:")
epi_genes_all = CUSTOM_SETS["Epigenetic regulators"]

for abbr in REGIONS:
    rc = region_correlations[abbr].get("NEMF")
    if rc is None:
        continue
    other_r = [g for g in region_data[abbr]["expr"].columns if g not in TARGETS]
    n_top_r = int(len(other_r) * TOP_PCT)
    top_r = set(rc.head(n_top_r).index)
    expressed_r = [g for g in epi_genes_all if g in other_r]
    in_top_r = [g for g in expressed_r if g in top_r]
    k_r, n_r = len(in_top_r), len(expressed_r)
    if n_r > 0:
        _, p_r = stats.fisher_exact(
            [[k_r, len(top_r)-k_r],
             [n_r-k_r, max(len(other_r)-len(top_r)-(n_r-k_r), 0)]],
            alternative='greater')
        fold_r = (k_r/n_r) / (n_top_r/len(other_r))
        sig_str = "**SIG**" if p_r < 0.05 else ""
        print(f"    {abbr}: {k_r}/{n_r}, {fold_r:.1f}-fold, p={p_r:.4f} {sig_str}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 8 — FIGURE 1: Two-module architecture
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("GENERATING FIGURES")
print("="*70)

fig1, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# ── Fig 1a: Pathway schematic ──
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('a', fontsize=14, fontweight='bold', loc='left')

# Module A (PELO)
rect_a = mpatches.FancyBboxPatch((0.5, 6.5), 3.5, 2.5, boxstyle="round,pad=0.2",
                                  facecolor='#4ECDC4', alpha=0.3, edgecolor='#2C7A75', lw=2)
ax.add_patch(rect_a)
ax.text(2.25, 8.5, 'Module A', fontsize=10, fontweight='bold', ha='center', color='#2C7A75')
ax.text(2.25, 7.7, 'Surveillance–\nmRNA Decay', fontsize=8, ha='center', color='#2C7A75')
# PELO box
pelo_box = mpatches.FancyBboxPatch((1.2, 6.8), 2.1, 0.7, boxstyle="round,pad=0.1",
                                    facecolor='#4ECDC4', edgecolor='#2C7A75', lw=1.5)
ax.add_patch(pelo_box)
ax.text(2.25, 7.15, 'PELO', fontsize=10, fontweight='bold', ha='center', va='center')

# Module B (LTN1-NEMF)
rect_b = mpatches.FancyBboxPatch((5.5, 6.5), 4.0, 2.5, boxstyle="round,pad=0.2",
                                  facecolor='#FF6B6B', alpha=0.3, edgecolor='#C0392B', lw=2)
ax.add_patch(rect_b)
ax.text(7.5, 8.5, 'Module B', fontsize=10, fontweight='bold', ha='center', color='#C0392B')
ax.text(7.5, 7.7, 'Nascent Chain\nProcessing', fontsize=8, ha='center', color='#C0392B')
# LTN1 box
ltn1_box = mpatches.FancyBboxPatch((5.8, 6.8), 1.5, 0.7, boxstyle="round,pad=0.1",
                                    facecolor='#FF6B6B', edgecolor='#C0392B', lw=1.5)
ax.add_patch(ltn1_box)
ax.text(6.55, 7.15, 'LTN1', fontsize=10, fontweight='bold', ha='center', va='center')
# NEMF box
nemf_box = mpatches.FancyBboxPatch((7.8, 6.8), 1.5, 0.7, boxstyle="round,pad=0.1",
                                    facecolor='#FF6B6B', edgecolor='#C0392B', lw=1.5)
ax.add_patch(nemf_box)
ax.text(8.55, 7.15, 'NEMF', fontsize=10, fontweight='bold', ha='center', va='center')

# Arrow: PELO → LTN1-NEMF
ax.annotate('', xy=(5.5, 7.5), xytext=(4.0, 7.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax.text(4.75, 7.8, 'Ribosome\nsplitting', fontsize=7, ha='center', color='gray')

# Labels below
ax.text(2.25, 5.8, 'J = 0.165\n(PELO–LTN1)', fontsize=7, ha='center', color='gray')
ax.text(7.5, 5.8, 'J = 0.461\n(LTN1–NEMF)', fontsize=7, ha='center', color='#C0392B',
        fontweight='bold')

# Stalled ribosome at top
ax.text(5.0, 9.5, '80S Stalled Ribosome', fontsize=9, ha='center',
        fontweight='bold', style='italic', color='#555')

# ── Fig 1b: Three-way Venn ──
ax = axes[1]
ax.set_title('b', fontsize=14, fontweight='bold', loc='left')
v = venn3(subsets=(len(pelo_only), len(ltn1_only), len(pelo_ltn1_only),
                   len(nemf_only), len(pelo_nemf_only), len(ltn1_nemf_only),
                   len(all_three)),
          set_labels=('PELO', 'LTN1', 'NEMF'),
          set_colors=('#4ECDC4', '#FF6B6B', '#FFB347'), alpha=0.6, ax=ax)
# Bold the LTN1-NEMF shared count
if v.get_label_by_id('110'):
    v.get_label_by_id('110').set_fontweight('bold')
    v.get_label_by_id('110').set_fontsize(11)

# ── Fig 1c: Jaccard heatmap ──
ax = axes[2]
ax.set_title('c', fontsize=14, fontweight='bold', loc='left')
j_matrix = pd.DataFrame(
    [[1.0, pairwise[("PELO","LTN1")]["jaccard"], pairwise[("PELO","NEMF")]["jaccard"]],
     [pairwise[("PELO","LTN1")]["jaccard"], 1.0, pairwise[("LTN1","NEMF")]["jaccard"]],
     [pairwise[("PELO","NEMF")]["jaccard"], pairwise[("LTN1","NEMF")]["jaccard"], 1.0]],
    index=TARGETS, columns=TARGETS
)
sns.heatmap(j_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            vmin=0, vmax=0.5, square=True, linewidths=1,
            cbar_kws={'label': 'Jaccard Index', 'shrink': 0.8}, ax=ax)
ax.set_title('c', fontsize=14, fontweight='bold', loc='left')
# Add baseline annotation
ax.text(1.5, 3.3, 'Random baseline: J ≈ 0.02–0.05', fontsize=7,
        ha='center', style='italic', color='gray')

plt.tight_layout()
fig1.savefig(f'{OUT}/figures/Figure1_two_module_architecture.pdf')
fig1.savefig(f'{OUT}/figures/Figure1_two_module_architecture.png')
plt.close(fig1)
print("  Figure 1 saved.")


# ═══════════════════════════════════════════════════════════════════════
# CELL 9 — FIGURE 2: GO enrichment of unique gene sets
# ═══════════════════════════════════════════════════════════════════════
if all_enrichment:
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    panels = [
        ("PELO_unique", "PELO unique (528 genes)", axes2[0, 0], '#4ECDC4'),
        ("LTN1_unique", "LTN1 unique (235 genes)", axes2[0, 1], '#FF6B6B'),
        ("NEMF_unique", "NEMF unique (246 genes)", axes2[1, 0], '#FFB347'),
        ("Shared_all_three", "Shared RQC core (166 genes)", axes2[1, 1], '#9B59B6'),
    ]

    panel_labels = ['a', 'b', 'c', 'd']

    for idx, (set_name, title, ax, color) in enumerate(panels):
        ax.set_title(panel_labels[idx], fontsize=14, fontweight='bold', loc='left')

        subset = enrichment_df[enrichment_df['gene_set'] == set_name].copy()
        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No enrichment data', transform=ax.transAxes,
                    ha='center', va='center')
            continue

        # Top 5 GO:BP terms by p-value
        bp = subset[subset['source'] == 'GO:BP'].sort_values('p_value').head(5)
        if len(bp) == 0:
            bp = subset.sort_values('p_value').head(5)

        bp = bp.sort_values('p_value', ascending=True)
        y_pos = range(len(bp))
        bars = ax.barh(y_pos, -np.log10(bp['p_value'].values), color=color, alpha=0.8)
        ax.set_yticks(list(y_pos))
        # Truncate long names
        labels = [n[:45] + '...' if len(n) > 45 else n for n in bp['term_name'].values]
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('-log₁₀(p-value)', fontsize=9)
        ax.set_title(f'{panel_labels[idx]}   {title}', fontsize=10,
                     fontweight='bold', loc='left')

    plt.tight_layout()
    fig2.savefig(f'{OUT}/figures/Figure2_GO_enrichment.pdf')
    fig2.savefig(f'{OUT}/figures/Figure2_GO_enrichment.png')
    plt.close(fig2)
    print("  Figure 2 saved.")
else:
    print("  Figure 2 SKIPPED — no enrichment data (gProfiler may have been unreachable)")


# ═══════════════════════════════════════════════════════════════════════
# CELL 10 — FIGURE 3: Multi-region replication heatmaps
# ═══════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4.5))

region_labels = list(REGIONS.keys())

for idx, target in enumerate(TARGETS):
    ax = axes3[idx]
    mat = cross_region_rho[target].astype(float)
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)  # mask upper triangle
    sns.heatmap(mat, annot=True, fmt='.3f', cmap='YlGnBu',
                vmin=0.75, vmax=1.0, square=True, linewidths=1,
                mask=mask, cbar_kws={'shrink': 0.7}, ax=ax)
    ax.set_title(f'{"abc"[idx]}   {target}', fontsize=12, fontweight='bold', loc='left')

plt.tight_layout()
fig3.savefig(f'{OUT}/figures/Figure3_multi_region_replication.pdf')
fig3.savefig(f'{OUT}/figures/Figure3_multi_region_replication.png')
plt.close(fig3)
print("  Figure 3 saved.")


# ═══════════════════════════════════════════════════════════════════════
# CELL 11 — Tables (manuscript body)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("GENERATING TABLES")
print("="*70)

# Table 1: Pairwise comparison
table1_rows = []
for g1, g2 in pairs:
    d = pairwise[(g1, g2)]
    p_str = f"{d['fisher_p']:.2e}" if d['fisher_p'] > 0 else "< 10⁻³²⁴"
    table1_rows.append({
        "Gene A": g1, "Gene B": g2,
        "Jaccard": f"{d['jaccard']:.3f}",
        "Shared genes": d['shared'],
        "Fisher OR": f"{d['or']:.2f}",
        "Fisher p": p_str,
        "Rank ρ": f"{d['spearman']:.3f}",
    })
table1 = pd.DataFrame(table1_rows)
table1.to_csv(f'{OUT}/tables/Table1_pairwise_comparison.csv', index=False)
print("  Table 1 saved.")

# Table 2: Three-way overlap
table2_rows = []
for name, (count, _) in venn_regions.items():
    table2_rows.append({
        "Category": name,
        "Gene count": count,
        "% of total": f"{count/total_unique*100:.1f}",
    })
table2 = pd.DataFrame(table2_rows)
table2.to_csv(f'{OUT}/tables/Table2_three_way_overlap.csv', index=False)
print("  Table 2 saved.")

# Table 3: Cross-region ranges
table3_rows = []
for target in TARGETS:
    vals = []
    for i, r1 in enumerate(region_labels):
        for j, r2 in enumerate(region_labels):
            if i < j:
                v = cross_region_rho[target].loc[r1, r2]
                if not np.isnan(v):
                    vals.append(v)
    table3_rows.append({
        "Gene": target,
        "Min ρ": f"{min(vals):.2f}",
        "Max ρ": f"{max(vals):.2f}",
        "Median ρ": f"{np.median(vals):.2f}",
        "All > 0.80": "Yes" if all(v > 0.80 for v in vals) else "No",
    })
table3 = pd.DataFrame(table3_rows)
table3.to_csv(f'{OUT}/tables/Table3_cross_region_ranges.csv', index=False)
print("  Table 3 saved.")


# ═══════════════════════════════════════════════════════════════════════
# CELL 12 — Supplementary tables
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("GENERATING SUPPLEMENTARY MATERIALS")
print("="*70)

# Table S1: Complete genome-wide co-expression rankings
s1_rows = []
for g in other_genes:
    row = {"Gene": g}
    for target in TARGETS:
        row[f"{target}_r"] = correlations[target].get(g, np.nan)
        row[f"{target}_rank"] = correlations[target].rank(ascending=False).get(g, np.nan)
    # p-values for each correlation
    for target in TARGETS:
        r_val = correlations[target].get(g, np.nan)
        if not np.isnan(r_val):
            n = len(expr_v)
            t_stat = r_val * np.sqrt((n - 2) / (1 - r_val**2))
            p_val = 2 * stats.t.sf(abs(t_stat), df=n-2)
            row[f"{target}_p"] = p_val
        else:
            row[f"{target}_p"] = np.nan
    s1_rows.append(row)

table_s1 = pd.DataFrame(s1_rows)
# Sort by PELO rank
table_s1 = table_s1.sort_values("PELO_rank")
table_s1.to_csv(f'{OUT}/supplementary/TableS1_genome_wide_rankings.csv', index=False)
print(f"  Table S1: {len(table_s1)} genes saved.")

# Table S2: Already saved above (gProfiler enrichment)
# (saved in Cell 6)

# Table S3: Custom gene set enrichment + sensitivity analyses
custom_df.to_csv(f'{OUT}/supplementary/TableS3_custom_enrichment.csv', index=False)

# Add sensitivity summary to S3
sensitivity_rows = []
for r in rp_results:
    sensitivity_rows.append(r)
sensitivity_df = pd.DataFrame(sensitivity_rows)
sensitivity_df.to_csv(f'{OUT}/supplementary/TableS3_sensitivity_rank_preservation.csv',
                       index=False)

# Agonal Jaccard table
agonal_rows = []
for model_name in ["Unadjusted"] + list(models.keys()):
    row = {"Model": model_name}
    for g1, g2 in pairs:
        key = f"{g1}-{g2}"
        if model_name == "Unadjusted":
            row[key] = pairwise[(g1, g2)]["jaccard"]
        else:
            row[key] = adj_jaccards.get((model_name, g1, g2), np.nan)
    agonal_rows.append(row)
agonal_df = pd.DataFrame(agonal_rows)
agonal_df.to_csv(f'{OUT}/supplementary/TableS3_agonal_jaccard.csv', index=False)
print("  Table S3 (custom enrichment + sensitivity) saved.")

# Table S4: Complete list of 166 shared genes
s4_rows = []
for g in sorted(all_three):
    row = {"Gene": g}
    for target in TARGETS:
        row[f"{target}_r"] = correlations[target].get(g, np.nan)
    row["Mean_r"] = np.mean([correlations[t].get(g, np.nan) for t in TARGETS])
    s4_rows.append(row)
table_s4 = pd.DataFrame(s4_rows).sort_values("Mean_r", ascending=False)
table_s4.to_csv(f'{OUT}/supplementary/TableS4_shared_166_genes.csv', index=False)
print(f"  Table S4: {len(table_s4)} shared genes saved.")


# ═══════════════════════════════════════════════════════════════════════
# CELL 13 — Random baseline validation (empirical J for random pairs)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("RANDOM BASELINE VALIDATION")
print("="*70)

np.random.seed(42)
n_random = 1000
gene_pool = [g for g in expr_v.columns if g not in TARGETS]

# Precompute z-scored expression matrix for fast correlation via dot product
expr_mat = expr_v[gene_pool].values  # (n_samples, n_genes)
# Z-score each gene (column): (x - mean) / std
expr_z = (expr_mat - expr_mat.mean(axis=0)) / (expr_mat.std(axis=0, ddof=1) + 1e-12)
n_samp = expr_z.shape[0]
print(f"  Expression matrix: {expr_z.shape[0]} samples × {expr_z.shape[1]} genes (z-scored)")

random_jaccards = []
top_n = int(len(gene_pool) * 0.05)

for i in range(n_random):
    g1_idx = np.random.randint(0, len(gene_pool))
    g2_idx = np.random.randint(0, len(gene_pool))
    while g2_idx == g1_idx:
        g2_idx = np.random.randint(0, len(gene_pool))

    # Vectorized correlation: dot product of z-scored vectors / (n-1)
    r1 = expr_z[:, g1_idx] @ expr_z / (n_samp - 1)  # shape: (n_genes,)
    r2 = expr_z[:, g2_idx] @ expr_z / (n_samp - 1)

    # Zero out self-correlations
    r1[g1_idx] = -999
    r1[g2_idx] = -999
    r2[g1_idx] = -999
    r2[g2_idx] = -999

    # Top 5% by correlation
    t1 = set(np.argsort(r1)[-top_n:])
    t2 = set(np.argsort(r2)[-top_n:])

    inter = len(t1 & t2)
    union = len(t1 | t2)
    j = inter / union if union > 0 else 0
    random_jaccards.append(j)

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{n_random} random pairs computed...")

rj = np.array(random_jaccards)
print(f"\n  Random baseline Jaccard (n={n_random}):")
print(f"    Mean = {rj.mean():.4f}")
print(f"    Median = {np.median(rj):.4f}")
print(f"    SD = {rj.std():.4f}")
print(f"    Range = {rj.min():.4f} – {rj.max():.4f}")
print(f"    2.5th–97.5th percentile = {np.percentile(rj,2.5):.4f} – {np.percentile(rj,97.5):.4f}")

# Z-scores for observed Jaccards
for (g1, g2), d in pairwise.items():
    z = (d['jaccard'] - rj.mean()) / rj.std()
    print(f"    {g1}-{g2}: J={d['jaccard']:.4f}, z={z:.1f} SD above random mean")


# ═══════════════════════════════════════════════════════════════════════
# CELL 14 — Final summary output
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("═══ COMPLETE MANUSCRIPT NUMBER VERIFICATION ═══")
print("=" * 70)

print(f"\nDATASET:")
print(f"  Primary region: BA9, n = {N_SAMPLES} samples")
print(f"  Expressed genes: {N_GENES}")
print(f"  Non-target genes: {n_other}")
print(f"  Top 5% = {n_top} genes per target")

print(f"\nMEDIAN EXPRESSION:")
for t in TARGETS:
    print(f"  {t}: {region_data['BA9']['tpm'][t].median():.2f} TPM")

print(f"\nTHRESHOLDS:")
for t in TARGETS:
    print(f"  {t}: r >= {thresholds[t]:.3f}")

print(f"\n--- TABLE 1 ---")
for (g1, g2), d in pairwise.items():
    p_str = f"{d['fisher_p']:.2e}" if d['fisher_p'] > 0 else "< 10^-324"
    print(f"  {g1}-{g2}: J={d['jaccard']:.3f}, shared={d['shared']}, "
          f"OR={d['or']:.2f}, p={p_str}, rho={d['spearman']:.3f}")

print(f"\n--- TABLE 2 ---")
for name, (count, _) in venn_regions.items():
    print(f"  {name}: {count} ({count/total_unique*100:.1f}%)")
print(f"  Total: {total_unique}")

print(f"\n--- TABLE 3 (cross-region ranges) ---")
for target in TARGETS:
    vals = []
    for i, r1 in enumerate(region_labels):
        for j, r2 in enumerate(region_labels):
            if i < j:
                v = cross_region_rho[target].loc[r1, r2]
                if not np.isnan(v):
                    vals.append(v)
    print(f"  {target}: {min(vals):.2f}–{max(vals):.2f}, "
          f"median={np.median(vals):.2f}")

print(f"\n--- AGONAL SENSITIVITY ---")
print(f"  Rank preservation (full model):")
for r in rp_results:
    if r['model'] == 'Full model':
        print(f"    {r['gene']}: rho = {r['rho']:.4f}")

print(f"\n  Partition ratios:")
for model_name in ["Unadjusted"] + list(models.keys()):
    if model_name == "Unadjusted":
        pl = pairwise[("PELO","LTN1")]["jaccard"]
        pn = pairwise[("PELO","NEMF")]["jaccard"]
        ln = pairwise[("LTN1","NEMF")]["jaccard"]
    else:
        pl = adj_jaccards.get((model_name, "PELO", "LTN1"), 0)
        pn = adj_jaccards.get((model_name, "PELO", "NEMF"), 0)
        ln = adj_jaccards.get((model_name, "LTN1", "NEMF"), 0)
    ratio = ln / max(pl, pn)
    print(f"    {model_name}: {ratio:.2f}x")

print(f"\n--- RANDOM BASELINE ---")
print(f"  J = {rj.mean():.3f} ± {rj.std():.3f} (n={n_random})")

print(f"\n--- OUTPUT FILES ---")
print(f"  Figures:  {OUT}/figures/")
print(f"  Tables:   {OUT}/tables/")
print(f"  Suppl:    {OUT}/supplementary/")

print(f"\n{'='*70}")
print("PIPELINE COMPLETE")
print("="*70)
