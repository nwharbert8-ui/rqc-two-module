# Two-Module Architecture of Ribosome-Associated Quality Control in Human Brain

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18616697.svg)](https://doi.org/10.5281/zenodo.18616697)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Google Colab](https://img.shields.io/badge/platform-Google%20Colab-orange.svg)](https://colab.research.google.com/)

**Divergent co-expression networks of PELO, LTN1, and NEMF**

Drake H. Harbert  
Inner Architecture LLC, Canton, OH 44721  
ORCID: [0009-0007-7740-3616](https://orcid.org/0009-0007-7740-3616)

---

## Abstract

Ribosome-associated quality control (RQC) resolves stalled translational complexes through sequential ribosome rescue (PELO), nascent chain ubiquitination (LTN1), and C-terminal alanine–threonine tail modification (NEMF). Using genome-wide Pearson co-expression analysis across 16,225 genes in the GTEx v8 Brain–Frontal Cortex (BA9; n = 209), we show that RQC operates as a **two-module transcriptional system** in human brain:

- **Module A (Surveillance–mRNA decay):** PELO coordinates with proteostasis machinery and RNA metabolism pathways
- **Module B (Nascent chain processing):** LTN1–NEMF share near-identical co-expression networks (Jaccard = 0.461; ρ = 0.986; permutation p = 0.006)

The two-module partition is confirmed by non-overlapping bootstrap 95% confidence intervals, robust across network thresholds from 1–20% (partition ratio 1.64×–8.78×), and replicates across five brain regions. Hippocampus-specific divergence of the processing module suggests region-dependent transcriptional tuning of nascent chain quality control.

## Key Findings

| Metric | PELO–LTN1 | PELO–NEMF | LTN1–NEMF |
|--------|-----------|-----------|-----------|
| Jaccard Index | 0.165 | 0.156 | **0.461** |
| Spearman ρ | 0.913 | 0.900 | **0.986** |
| Permutation p | 0.110 | 0.119 | **0.006** |
| Bootstrap 95% CI | [0.095, 0.242] | [0.082, 0.227] | **[0.357, 0.546]** |

## Repository Structure

```
rqc-two-module/
├── README.md                  ← This file
├── LICENSE                    ← MIT License
├── CITATION.cff               ← Citation metadata
├── requirements.txt           ← Python dependencies
├── .gitignore                 ← Git ignore patterns
│
├── code/
│   ├── MS1_RQC_COMPLETE.py    ← Main analysis pipeline (Figures 1–3, Tables 1–3, Tables S1–S4)
│   └── MS1_RQC_SUPPLEMENT.py  ← Supplementary analyses (Figures S1–S3, Tables S5–S8)
│
├── notebooks/
│   ├── MS1_RQC_COMPLETE.ipynb       ← Colab notebook (main pipeline)
│   └── MS1_RQC_SUPPLEMENT.ipynb     ← Colab notebook (supplementary analyses)
│
├── figures/                   ← Generated figures (PDF + PNG)
│   ├── Figure1_two_module_architecture.{pdf,png}
│   ├── Figure2_GO_enrichment.{pdf,png}
│   ├── Figure3_multi_region_replication.{pdf,png}
│   ├── FigureS1_permutation_null.{pdf,png}
│   ├── FigureS2_bootstrap_jaccard.{pdf,png}
│   └── FigureS3_threshold_sensitivity.{pdf,png}
│
├── tables/                    ← Main manuscript tables (CSV)
│   ├── Table1_pairwise_comparison.csv
│   ├── Table2_three_way_overlap.csv
│   └── Table3_cross_region_ranges.csv
│
├── supplementary/             ← Supplementary tables (CSV)
│   ├── TableS1_genome_wide_rankings.csv
│   ├── TableS2_GO_enrichment_results.csv
│   ├── TableS3_custom_gene_set_enrichment.csv
│   ├── TableS4_shared_166_genes.csv
│   ├── TableS5_null_distribution.csv
│   ├── TableS6_cross_region_all_pairs.csv
│   ├── TableS7_bootstrap_CIs.csv
│   └── TableS8_threshold_sensitivity.csv
│
└── docs/
    └── METHODS_DETAIL.md      ← Extended methods documentation
```

## Quick Start

### Option 1: Google Colab (Recommended)

1. Open the notebook in Google Colab:
   - [![Open Main Pipeline](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/innerarchitecturellc/rqc-two-module/blob/main/notebooks/MS1_RQC_COMPLETE.ipynb)
   - [![Open Supplement](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/innerarchitecturellc/rqc-two-module/blob/main/notebooks/MS1_RQC_SUPPLEMENT.ipynb)

2. Mount your Google Drive when prompted

3. Ensure the GTEx v8 gene TPM file is at:
   ```
   /content/drive/MyDrive/GenomicAnalysis/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz
   ```

4. Run all cells sequentially. Main pipeline produces all primary results; supplement runs in the same session afterward.

### Option 2: Run as Python Scripts

```bash
# Clone the repository
git clone https://github.com/innerarchitecturellc/rqc-two-module.git
cd rqc-two-module

# Install dependencies
pip install -r requirements.txt

# Run main pipeline (requires GTEx data and Google Drive paths adjusted)
python code/MS1_RQC_COMPLETE.py

# Run supplementary analyses (same session, all variables in memory)
python code/MS1_RQC_SUPPLEMENT.py
```

## Data Requirements

### GTEx v8 (Required)

| File | Source | Size |
|------|--------|------|
| `GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz` | [GTEx Portal](https://gtexportal.org/) | ~1.3 GB |
| `GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt` | [GTEx Portal](https://gtexportal.org/) | ~50 KB |
| `GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt` | Auto-downloaded by pipeline | ~15 MB |

**Access:** GTEx v8 data is available through the GTEx Portal under dbGaP accession [phs000424.v8.p2](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000424.v8.p2). The gene TPM file requires dbGaP authorized access or can be downloaded directly from the GTEx Portal after agreeing to the data use terms.

### Brain Regions Analyzed

| Abbreviation | Full GTEx Name | n (samples) |
|-------------|----------------|-------------|
| BA9 | Brain – Frontal Cortex (BA9) | 209 |
| PUT | Brain – Putamen (basal ganglia) | 205 |
| HIP | Brain – Hippocampus | 197 |
| NAC | Brain – Nucleus accumbens (basal ganglia) | 246 |
| BA24 | Brain – Anterior cingulate cortex (BA24) | 176 |

## Computational Environment

- **Platform:** Google Colaboratory (free tier, ≤12 GB RAM)
- **Python:** 3.10+
- **Key dependencies:** pandas 1.5.3, scipy 1.11.4, numpy 1.24.3, matplotlib 3.7.1, seaborn, matplotlib-venn
- **Runtime:** ~15 minutes (main pipeline) + ~5 minutes (supplementary analyses)
- **Peak RAM:** ~4 GB (well within Colab free tier)

## Methods Overview

### Primary Analysis
1. Load GTEx v8 TPM data for five brain regions
2. Filter genes (median TPM ≥ 1.0), log₂(TPM + 1) transform
3. Compute genome-wide Pearson correlations for PELO, LTN1, NEMF vs. all 16,222 non-target genes
4. Define top 5% co-expression networks (811 genes per target)
5. Pairwise comparison: Jaccard index, Fisher's exact test, Spearman rank correlation
6. Three-way overlap analysis
7. Gene Ontology enrichment via gProfiler REST API
8. Multi-region replication across four additional brain regions
9. Cell-type deconvolution (53 marker genes, 6 cell types)
10. Covariate adjustment (age, sex) and agonal stress sensitivity (Hardy Scale, ischemic time)

### Supplementary Analyses
1. **Permutation null model** — 10,000 random gene pairs establish genomic baseline Jaccard distribution
2. **Cross-region pair decomposition** — All 30 pairwise ρ values identifying hippocampus as the divergent region
3. **Bootstrap confidence intervals** — 1,000 resamples with non-overlap test for module separation
4. **Threshold sensitivity** — Partition ratios across 1%, 2%, 3%, 5%, 7%, 10%, 15%, 20% thresholds

## Output Summary

### Main Pipeline Outputs

| Output | Description |
|--------|-------------|
| Figure 1 | Two-module architecture schematic, Venn diagram, Jaccard heatmap |
| Figure 2 | GO enrichment bar charts (4 panels: PELO, LTN1, NEMF unique + shared core) |
| Figure 3 | Cross-region replication heatmaps (3 panels: PELO, LTN1, NEMF) |
| Table 1 | Pairwise network comparison statistics |
| Table 2 | Three-way overlap gene counts |
| Table 3 | Cross-region correlation ranges |
| Tables S1–S4 | Genome-wide rankings, GO results, gene set enrichment, shared 166 genes |

### Supplementary Pipeline Outputs

| Output | Description |
|--------|-------------|
| Figure S1 | Permutation null distribution with observed Jaccard values |
| Figure S2 | Bootstrap distributions with 95% CIs for all three pairs |
| Figure S3 | Threshold sensitivity (Jaccard indices + partition ratios) |
| Table S5 | Full null distribution (10,000 values) |
| Table S6 | All 30 cross-region pairwise ρ values with sample sizes |
| Table S7 | Bootstrap 95% CIs for Jaccard and Spearman |
| Table S8 | Threshold sensitivity results (8 thresholds) |

## Citation

If you use this code or data in your work, please cite:

```bibtex
@article{harbert2026rqc,
  title={Two-module architecture of ribosome-associated quality control in human brain: 
         divergent co-expression networks of {PELO}, {LTN1}, and {NEMF}},
  author={Harbert, Drake H.},
  journal={BMC Genomics},
  year={2026},
  note={Manuscript submitted},
  doi={10.5281/zenodo.18616697}
}
```

See also [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## Related Work

- **EIF2S1–PELO convergence:** Harbert, D.H. (manuscript in preparation). EIF2S1 as a transcriptional hub linking translation initiation surveillance to ribosome rescue.
- **SIGMAR1 divergence:** Harbert, D.H. (manuscript in preparation). Sigma-1 receptor co-expression divergence across neurological conditions.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

Drake H. Harbert  
Inner Architecture LLC  
Email: drake@innerarchitecturellc.com  
ORCID: [0009-0007-7740-3616](https://orcid.org/0009-0007-7740-3616)  
GitHub: [@nwharbert8-ui](https://github.com/nwharbert8-ui)
