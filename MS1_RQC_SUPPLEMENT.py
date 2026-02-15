"""
MS1_RQC_SUPPLEMENT.py
=====================
Run AFTER MS1_RQC_COMPLETE.py (all variables in memory).

Four reviewer-proofing analyses:
  CELL S1 — Permutation null model (10,000 iterations)
  CELL S2 — Specific region pairs driving sub-0.80 correlations
  CELL S3 — Bootstrap confidence intervals on Jaccard indices (1,000 resamples)
  CELL S4 — Threshold sensitivity analysis (1%, 3%, 5%, 7%, 10%)

Generates: Figures S1–S3, Table S5, all console statistics.
Environment: Google Colab, ≤12 GB RAM, appended to existing session.
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL S1 — Permutation null model for Jaccard indices
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("CELL S1 — PERMUTATION NULL MODEL (10,000 iterations)")
print("=" * 70)

import numpy as np
import pandas as pd
from scipy import stats
import time, gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Precompute standardized expression matrix for fast Pearson via dot product
# Pearson(x, y) = dot(z_x, z_y) / (n - 1) where z = (x - mean) / std
print("  Precomputing standardized expression matrix...")
expr_mat = expr_v[other_genes].values.T  # (genes × samples)
expr_means = expr_mat.mean(axis=1, keepdims=True)
expr_stds = expr_mat.std(axis=1, ddof=1, keepdims=True)
expr_stds[expr_stds == 0] = 1.0  # avoid division by zero
Z = (expr_mat - expr_means) / expr_stds  # (genes × samples)
n_samp = Z.shape[1]
gene_index = {g: i for i, g in enumerate(other_genes)}

print(f"  Z matrix: {Z.shape[0]} genes × {Z.shape[1]} samples")
print(f"  RAM for Z: ~{Z.nbytes / 1e9:.2f} GB")

def fast_jaccard_for_gene_pair(idx_a, idx_b, Z, n_top, n_samp):
    """Compute Jaccard between top-5% networks of two genes using vectorized Pearson."""
    # Correlation of gene A with all others
    corr_a = Z @ Z[idx_a] / (n_samp - 1)
    corr_b = Z @ Z[idx_b] / (n_samp - 1)

    # Zero out self-correlations
    corr_a[idx_a] = -999
    corr_b[idx_b] = -999
    # Zero out cross-correlation (gene B in A's ranking and vice versa)
    corr_a[idx_b] = -999
    corr_b[idx_a] = -999

    # Top n_top by descending correlation
    top_a = set(np.argpartition(corr_a, -n_top)[-n_top:])
    top_b = set(np.argpartition(corr_b, -n_top)[-n_top:])

    inter = len(top_a & top_b)
    union = len(top_a | top_b)
    return inter / union if union > 0 else 0.0

N_PERM = 10000
null_jaccards = np.zeros(N_PERM)

t0 = time.time()
print(f"\n  Running {N_PERM} permutations...")
for i in range(N_PERM):
    # Pick 2 random genes (without replacement)
    pair = np.random.choice(len(other_genes), size=2, replace=False)
    null_jaccards[i] = fast_jaccard_for_gene_pair(pair[0], pair[1], Z, n_top, n_samp)
    if (i + 1) % 2000 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (N_PERM - i - 1) / rate
        print(f"    {i+1}/{N_PERM} done ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

elapsed = time.time() - t0
print(f"  Completed {N_PERM} permutations in {elapsed:.1f}s")

# Null distribution statistics
null_mean = np.mean(null_jaccards)
null_std = np.std(null_jaccards)
null_median = np.median(null_jaccards)
null_p95 = np.percentile(null_jaccards, 95)
null_p99 = np.percentile(null_jaccards, 99)
null_max = np.max(null_jaccards)

print(f"\n  NULL DISTRIBUTION (n={N_PERM}):")
print(f"    Mean:   {null_mean:.4f}")
print(f"    Std:    {null_std:.4f}")
print(f"    Median: {null_median:.4f}")
print(f"    95th:   {null_p95:.4f}")
print(f"    99th:   {null_p99:.4f}")
print(f"    Max:    {null_max:.4f}")

# Observed values & empirical p-values / z-scores
print(f"\n  OBSERVED vs NULL:")
for g1, g2 in pairs:
    obs_j = pairwise[(g1, g2)]["jaccard"]
    z_score = (obs_j - null_mean) / null_std
    emp_p = np.sum(null_jaccards >= obs_j) / N_PERM
    fold_above = obs_j / null_mean if null_mean > 0 else float('inf')
    print(f"    {g1}-{g2}: J={obs_j:.4f}, z={z_score:.1f}, "
          f"empirical p={emp_p:.4f} ({emp_p * N_PERM:.0f}/{N_PERM}), "
          f"fold-above-mean={fold_above:.1f}×")

# ── Figure S1: Null distribution with observed values ──
fig_s1, ax = plt.subplots(figsize=(8, 5))
ax.hist(null_jaccards, bins=80, color='#B0C4DE', edgecolor='white',
        linewidth=0.3, density=True, alpha=0.85, label='Null distribution')

# Observed lines
colors = {'PELO-LTN1': '#E74C3C', 'PELO-NEMF': '#F39C12', 'LTN1-NEMF': '#2E86C1'}
for g1, g2 in pairs:
    obs_j = pairwise[(g1, g2)]["jaccard"]
    label = f"{g1}–{g2}"
    ax.axvline(obs_j, color=colors[f"{g1}-{g2}"], linewidth=2.5,
               linestyle='--', label=f'{label} (J={obs_j:.3f})')

ax.set_xlabel('Jaccard Index', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Permutation Null Distribution of Jaccard Indices\n'
             f'(n={N_PERM:,} random gene pairs, top 5% networks)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(-0.005, max(0.12, null_max * 1.2))

# Inset text box with statistics
textstr = (f'Null: μ={null_mean:.4f}, σ={null_std:.4f}\n'
           f'95th pctl: {null_p95:.4f}\n'
           f'99th pctl: {null_p99:.4f}')
props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray')
ax.text(0.62, 0.75, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

plt.tight_layout()
fig_s1.savefig(f'{OUT}/figures/FigureS1_permutation_null.pdf')
fig_s1.savefig(f'{OUT}/figures/FigureS1_permutation_null.png')
plt.close(fig_s1)
print("\n  Figure S1 saved.")

# Save null distribution
null_df = pd.DataFrame({'jaccard': null_jaccards})
null_df.to_csv(f'{OUT}/supplementary/TableS5_null_distribution.csv', index=False)
print("  Table S5 (null distribution) saved.")

# Clean up Z matrix to free RAM
del Z, expr_mat, expr_means, expr_stds
gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# CELL S2 — Specific region pairs driving sub-0.80 correlations
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("CELL S2 — CROSS-REGION PAIR DECOMPOSITION")
print("=" * 70)

# Extract all 10 pairwise ρ values per target from cross_region_rho matrices
print(f"\n  Regions: {region_abbrs}")
print(f"  Region pairs: {len(region_abbrs) * (len(region_abbrs) - 1) // 2}")

# Also get sample sizes per region for context
region_n = {}
for abbr in region_abbrs:
    region_n[abbr] = region_data[abbr]["n_samples"]
    print(f"  {abbr}: n={region_n[abbr]}")

all_pairs_data = []
for target in TARGETS:
    mat = cross_region_rho[target]
    print(f"\n  {target}:")
    print(f"  {'Pair':<12} {'ρ':>8}  {'n₁':>5}  {'n₂':>5}  {'> 0.80':>7}")
    print(f"  {'─'*45}")
    for i, r1 in enumerate(region_abbrs):
        for j, r2 in enumerate(region_abbrs):
            if i < j:
                rho = mat.loc[r1, r2]
                n1 = region_n[r1]
                n2 = region_n[r2]
                flag = "  ✓" if rho > 0.80 else "  *** LOW"
                print(f"  {r1}–{r2:<7} {rho:8.4f}  {n1:5d}  {n2:5d}  {flag}")
                all_pairs_data.append({
                    'Gene': target,
                    'Region 1': r1,
                    'Region 2': r2,
                    'Spearman ρ': round(rho, 4),
                    'n (Region 1)': n1,
                    'n (Region 2)': n2,
                    'Above 0.80': 'Yes' if rho > 0.80 else 'No'
                })

# Save as Table S6
pairs_df = pd.DataFrame(all_pairs_data)
pairs_df.to_csv(f'{OUT}/supplementary/TableS6_cross_region_all_pairs.csv', index=False)
print(f"\n  Table S6 saved ({len(all_pairs_data)} region-pair entries).")

# Summary: which pairs are sub-0.80?
print(f"\n  SUB-0.80 PAIRS:")
sub80 = pairs_df[pairs_df['Above 0.80'] == 'No']
if len(sub80) == 0:
    print("    None")
else:
    for _, row in sub80.iterrows():
        print(f"    {row['Gene']}: {row['Region 1']}–{row['Region 2']} "
              f"ρ={row['Spearman ρ']:.4f} "
              f"(n={row['n (Region 1)']}, n={row['n (Region 2)']})")

# PELO vs LTN1/NEMF comparison: count how many pairs > 0.80 for each
print(f"\n  PAIRS ABOVE 0.80 THRESHOLD:")
for target in TARGETS:
    t_df = pairs_df[pairs_df['Gene'] == target]
    above = t_df['Above 0.80'].value_counts().get('Yes', 0)
    total = len(t_df)
    print(f"    {target}: {above}/{total} pairs above 0.80 "
          f"({above/total*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════════════
# CELL S3 — Bootstrap confidence intervals on Jaccard indices
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("CELL S3 — BOOTSTRAP CONFIDENCE INTERVALS (1,000 resamples)")
print("=" * 70)

np.random.seed(123)
N_BOOT = 1000

# Precompute full expression array for fast resampling
expr_array = expr_v[other_genes].values  # (samples × genes)
target_arrays = {t: expr_v[t].values for t in TARGETS}
n_samples_boot = expr_array.shape[0]
n_genes_boot = expr_array.shape[1]

print(f"  Samples: {n_samples_boot}, Genes: {n_genes_boot}, Top 5%: {n_top}")

boot_jaccards = {p: np.zeros(N_BOOT) for p in pairs}
boot_spearman = {p: np.zeros(N_BOOT) for p in pairs}

t0 = time.time()
for b in range(N_BOOT):
    # Resample sample indices with replacement
    idx = np.random.choice(n_samples_boot, size=n_samples_boot, replace=True)

    # Resampled expression
    X_boot = expr_array[idx]  # (n_samples × n_genes)

    # Compute correlations for each target via vectorized Pearson
    boot_top5 = {}
    boot_rankings = {}
    for target in TARGETS:
        t_boot = target_arrays[target][idx]

        # Vectorized Pearson: corr(t, each column of X)
        t_mean = t_boot.mean()
        t_std = t_boot.std(ddof=1)
        if t_std == 0:
            continue
        t_z = (t_boot - t_mean) / t_std

        x_means = X_boot.mean(axis=0)
        x_stds = X_boot.std(axis=0, ddof=1)
        x_stds[x_stds == 0] = 1.0
        X_z = (X_boot - x_means) / x_stds  # (samples × genes)

        corrs = (t_z @ X_z) / (n_samples_boot - 1)  # (genes,)

        # Get top n_top indices
        top_idx = set(np.argpartition(corrs, -n_top)[-n_top:])
        boot_top5[target] = top_idx
        boot_rankings[target] = corrs

    # Compute pairwise Jaccards and Spearman
    for g1, g2 in pairs:
        if g1 in boot_top5 and g2 in boot_top5:
            s1, s2 = boot_top5[g1], boot_top5[g2]
            inter = len(s1 & s2)
            union = len(s1 | s2)
            boot_jaccards[(g1, g2)][b] = inter / union if union > 0 else 0.0

            rho, _ = stats.spearmanr(boot_rankings[g1], boot_rankings[g2])
            boot_spearman[(g1, g2)][b] = rho

    if (b + 1) % 200 == 0:
        elapsed = time.time() - t0
        rate = (b + 1) / elapsed
        eta = (N_BOOT - b - 1) / rate
        print(f"    {b+1}/{N_BOOT} done ({elapsed:.0f}s, ~{eta:.0f}s remaining)")

elapsed = time.time() - t0
print(f"  Completed {N_BOOT} bootstrap iterations in {elapsed:.1f}s")

# Report CIs
print(f"\n  BOOTSTRAP 95% CONFIDENCE INTERVALS:")
print(f"  {'Pair':<14} {'Observed J':>11} {'95% CI (J)':>22} "
      f"{'Observed ρ':>11} {'95% CI (ρ)':>22}")
print(f"  {'─'*82}")

ci_data = []
for g1, g2 in pairs:
    obs_j = pairwise[(g1, g2)]["jaccard"]
    obs_rho = pairwise[(g1, g2)]["spearman"]
    j_lo, j_hi = np.percentile(boot_jaccards[(g1, g2)], [2.5, 97.5])
    r_lo, r_hi = np.percentile(boot_spearman[(g1, g2)], [2.5, 97.5])
    print(f"  {g1}–{g2:<8} {obs_j:11.4f} [{j_lo:.4f}, {j_hi:.4f}]"
          f" {obs_rho:11.4f} [{r_lo:.4f}, {r_hi:.4f}]")
    ci_data.append({
        'Pair': f'{g1}–{g2}', 'Observed Jaccard': obs_j,
        'Jaccard 2.5%': round(j_lo, 4), 'Jaccard 97.5%': round(j_hi, 4),
        'Observed Spearman': obs_rho,
        'Spearman 2.5%': round(r_lo, 4), 'Spearman 97.5%': round(r_hi, 4),
    })

# Check non-overlap of CIs
pelo_ltn1_hi = np.percentile(boot_jaccards[("PELO","LTN1")], 97.5)
pelo_nemf_hi = np.percentile(boot_jaccards[("PELO","NEMF")], 97.5)
ltn1_nemf_lo = np.percentile(boot_jaccards[("LTN1","NEMF")], 2.5)

print(f"\n  CI NON-OVERLAP TEST (two-module separation):")
print(f"    Max upper CI of PELO pairs: {max(pelo_ltn1_hi, pelo_nemf_hi):.4f}")
print(f"    Lower CI of LTN1–NEMF:      {ltn1_nemf_lo:.4f}")
if ltn1_nemf_lo > max(pelo_ltn1_hi, pelo_nemf_hi):
    print(f"    → CIs DO NOT OVERLAP: Two-module separation is robust.")
else:
    gap = ltn1_nemf_lo - max(pelo_ltn1_hi, pelo_nemf_hi)
    print(f"    → Gap: {gap:.4f} (overlap exists if negative)")

# ── Figure S2: Bootstrap distributions ──
fig_s2, axes = plt.subplots(1, 3, figsize=(14, 4.5))
colors_list = ['#E74C3C', '#F39C12', '#2E86C1']
labels_list = ['PELO–LTN1', 'PELO–NEMF', 'LTN1–NEMF']

for idx, (g1, g2) in enumerate(pairs):
    ax = axes[idx]
    data = boot_jaccards[(g1, g2)]
    obs_j = pairwise[(g1, g2)]["jaccard"]
    lo, hi = np.percentile(data, [2.5, 97.5])

    ax.hist(data, bins=50, color=colors_list[idx], alpha=0.7,
            edgecolor='white', linewidth=0.3, density=True)
    ax.axvline(obs_j, color='black', linewidth=2, linestyle='-',
               label=f'Observed: {obs_j:.3f}')
    ax.axvline(lo, color='gray', linewidth=1.5, linestyle='--',
               label=f'95% CI: [{lo:.3f}, {hi:.3f}]')
    ax.axvline(hi, color='gray', linewidth=1.5, linestyle='--')

    # Shade CI region
    ax.axvspan(lo, hi, alpha=0.15, color='gray')

    ax.set_xlabel('Jaccard Index', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{labels_list[idx]}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')

plt.suptitle(f'Bootstrap Distributions of Jaccard Indices (n={N_BOOT:,} resamples)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig_s2.savefig(f'{OUT}/figures/FigureS2_bootstrap_jaccard.pdf')
fig_s2.savefig(f'{OUT}/figures/FigureS2_bootstrap_jaccard.png')
plt.close(fig_s2)
print("\n  Figure S2 saved.")

# Save bootstrap CIs
ci_df = pd.DataFrame(ci_data)
ci_df.to_csv(f'{OUT}/supplementary/TableS7_bootstrap_CIs.csv', index=False)
print("  Table S7 saved.")

del X_boot, boot_rankings, boot_top5
gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# CELL S4 — Threshold sensitivity analysis (1%, 3%, 5%, 7%, 10%)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("CELL S4 — THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 70)

THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

# Use the pre-computed genome-wide correlations from main pipeline
sensitivity_rows = []
partition_ratios = []

print(f"\n  {'Threshold':>10} {'PELO-LTN1':>11} {'PELO-NEMF':>11} {'LTN1-NEMF':>11} "
      f"{'Ratio':>8} {'Separation':>12}")
print(f"  {'─'*67}")

for pct in THRESHOLDS:
    n_t = int(n_other * pct)
    if n_t < 10:
        print(f"  {pct*100:9.0f}%  (skipped: n_top={n_t} too small)")
        continue

    # Get top sets at this threshold
    top_sets = {}
    for target in TARGETS:
        top_sets[target] = set(correlations[target].head(n_t).index)

    # Pairwise Jaccards
    jaccards = {}
    for g1, g2 in pairs:
        s1, s2 = top_sets[g1], top_sets[g2]
        inter = len(s1 & s2)
        union = len(s1 | s2)
        jaccards[(g1, g2)] = inter / union if union > 0 else 0.0

    # Partition ratio = LTN1-NEMF Jaccard / max(PELO-LTN1, PELO-NEMF)
    pelo_max = max(jaccards[("PELO","LTN1")], jaccards[("PELO","NEMF")])
    ratio = jaccards[("LTN1","NEMF")] / pelo_max if pelo_max > 0 else float('inf')

    # Is the partition maintained? (ratio > 1.5 = clear separation)
    sep = "STRONG" if ratio > 2.0 else ("MODERATE" if ratio > 1.5 else "WEAK")

    print(f"  {pct*100:9.0f}% {jaccards[('PELO','LTN1')]:11.4f} "
          f"{jaccards[('PELO','NEMF')]:11.4f} {jaccards[('LTN1','NEMF')]:11.4f} "
          f"{ratio:8.2f}× {sep:>12}")

    sensitivity_rows.append({
        'Threshold (%)': pct * 100,
        'n_top': n_t,
        'PELO-LTN1 Jaccard': round(jaccards[("PELO","LTN1")], 4),
        'PELO-NEMF Jaccard': round(jaccards[("PELO","NEMF")], 4),
        'LTN1-NEMF Jaccard': round(jaccards[("LTN1","NEMF")], 4),
        'Partition ratio': round(ratio, 2),
        'Separation': sep,
    })
    partition_ratios.append({
        'pct': pct * 100,
        'ratio': ratio,
        'j_pl': jaccards[("PELO","LTN1")],
        'j_pn': jaccards[("PELO","NEMF")],
        'j_ln': jaccards[("LTN1","NEMF")],
    })

# Also compute threshold-independent Spearman (already have this)
print(f"\n  THRESHOLD-INDEPENDENT SPEARMAN RANK CORRELATIONS (full genome):")
for g1, g2 in pairs:
    print(f"    {g1}–{g2}: ρ = {pairwise[(g1,g2)]['spearman']:.4f}")

# ── Figure S3: Threshold sensitivity plot (dual panel) ──
fig_s3, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

pcts = [r['pct'] for r in partition_ratios]
j_pl = [r['j_pl'] for r in partition_ratios]
j_pn = [r['j_pn'] for r in partition_ratios]
j_ln = [r['j_ln'] for r in partition_ratios]
ratios = [r['ratio'] for r in partition_ratios]

# Panel A: Jaccard indices across thresholds
ax1.plot(pcts, j_pl, 'o-', color='#E74C3C', linewidth=2, markersize=7,
         label='PELO–LTN1', zorder=3)
ax1.plot(pcts, j_pn, 's-', color='#F39C12', linewidth=2, markersize=7,
         label='PELO–NEMF', zorder=3)
ax1.plot(pcts, j_ln, '^-', color='#2E86C1', linewidth=2, markersize=7,
         label='LTN1–NEMF', zorder=3)

# Mark the primary threshold (5%)
ax1.axvline(5, color='gray', linewidth=1, linestyle=':', alpha=0.7)
ax1.text(5.3, ax1.get_ylim()[0] + 0.01, '5% (primary)', fontsize=8,
         color='gray', rotation=90, va='bottom')

ax1.set_xlabel('Network Threshold (%)', fontsize=12)
ax1.set_ylabel('Jaccard Index', fontsize=12)
ax1.set_title('a   Pairwise Jaccard Indices', fontsize=12,
              fontweight='bold', loc='left')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel B: Partition ratio across thresholds
ax2.plot(pcts, ratios, 'D-', color='#8E44AD', linewidth=2.5, markersize=8)
ax2.axhline(1.0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax2.axhline(2.0, color='green', linewidth=1, linestyle='--', alpha=0.5,
            label='2.0× (strong separation)')
ax2.axvline(5, color='gray', linewidth=1, linestyle=':', alpha=0.7)

# Shade strong separation zone
ax2.axhspan(2.0, max(ratios) * 1.15, alpha=0.08, color='green')

ax2.set_xlabel('Network Threshold (%)', fontsize=12)
ax2.set_ylabel('Partition Ratio\n(LTN1–NEMF J / max PELO J)', fontsize=11)
ax2.set_title('b   Two-Module Partition Ratio', fontsize=12,
              fontweight='bold', loc='left')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle('Threshold Sensitivity Analysis of Two-Module Architecture',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig_s3.savefig(f'{OUT}/figures/FigureS3_threshold_sensitivity.pdf')
fig_s3.savefig(f'{OUT}/figures/FigureS3_threshold_sensitivity.png')
plt.close(fig_s3)
print("\n  Figure S3 saved.")

# Save threshold sensitivity table
sens_df = pd.DataFrame(sensitivity_rows)
sens_df.to_csv(f'{OUT}/supplementary/TableS8_threshold_sensitivity.csv', index=False)
print("  Table S8 saved.")


# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("SUPPLEMENTARY ANALYSES COMPLETE")
print("=" * 70)
print(f"\n  OUTPUTS:")
print(f"    Figures:")
print(f"      {OUT}/figures/FigureS1_permutation_null.pdf/png")
print(f"      {OUT}/figures/FigureS2_bootstrap_jaccard.pdf/png")
print(f"      {OUT}/figures/FigureS3_threshold_sensitivity.pdf/png")
print(f"    Tables:")
print(f"      {OUT}/supplementary/TableS5_null_distribution.csv")
print(f"      {OUT}/supplementary/TableS6_cross_region_all_pairs.csv")
print(f"      {OUT}/supplementary/TableS7_bootstrap_CIs.csv")
print(f"      {OUT}/supplementary/TableS8_threshold_sensitivity.csv")

print(f"\n  KEY RESULTS FOR MANUSCRIPT:")
print(f"    Permutation null: μ={null_mean:.4f}, σ={null_std:.4f}")
for g1, g2 in pairs:
    obs_j = pairwise[(g1, g2)]["jaccard"]
    z_score = (obs_j - null_mean) / null_std
    emp_p = np.sum(null_jaccards >= obs_j) / N_PERM
    print(f"    {g1}–{g2}: J={obs_j:.4f}, z={z_score:.1f}, p_emp<{max(emp_p, 1/N_PERM):.4f}")

print(f"\n    Bootstrap 95% CIs:")
for g1, g2 in pairs:
    j_lo, j_hi = np.percentile(boot_jaccards[(g1, g2)], [2.5, 97.5])
    print(f"    {g1}–{g2}: [{j_lo:.4f}, {j_hi:.4f}]")

print(f"\n    Threshold sensitivity (partition ratio range):")
print(f"    {min(ratios):.2f}× – {max(ratios):.2f}× across {THRESHOLDS[0]*100:.0f}%–{THRESHOLDS[-1]*100:.0f}% thresholds")

print(f"\n    Sub-0.80 region pairs: {len(sub80)} total")
for _, row in sub80.iterrows():
    print(f"      {row['Gene']}: {row['Region 1']}–{row['Region 2']} ρ={row['Spearman ρ']:.4f}")

print(f"\n{'='*70}")
print("PASTE ALL OUTPUT ABOVE BACK TO CLAUDE")
print("=" * 70)
