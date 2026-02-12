# Data Access and Preparation

## GTEx v8 (Genotype-Tissue Expression Project)

This analysis requires two files from the GTEx v8 release. Both are available from
the GTEx Portal without dbGaP application (median-level TPM data are open access).

### Required Files

| File | Description | Size |
|:-----|:------------|:-----|
| `GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz` | Gene-level TPM expression matrix | ~1.5 GB |
| `GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt` | Sample metadata (tissue labels, etc.) | ~15 MB |

For covariate adjustment (script 06), also download:

| File | Description | Size |
|:-----|:------------|:-----|
| `GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt` | Donor phenotypes (age, sex) | ~200 KB |

### Download Instructions

1. Visit https://gtexportal.org/
2. Navigate to **Datasets** → **Download** (or directly: https://gtexportal.org/home/datasets)
3. Accept the data use agreement
4. Download the files listed above
5. Place them in this `data/` directory

### Data Format

The gene TPM matrix is a GCT file (tab-delimited with header rows). The analysis
scripts can read either the raw `.gct.gz` or a pre-converted `.parquet` file.

To convert to Parquet for faster loading (recommended):

```python
import pandas as pd

# Read GCT (skip first 2 header lines)
tpm = pd.read_csv('GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz',
                   sep='\t', skiprows=2, index_col=1, compression='gzip')
tpm = tpm.drop(columns=['Name'])  # Drop Ensembl IDs, keep gene symbols as index
tpm.to_parquet('gtex_v8_tpm.parquet')
```

### Brain Regions Used

| Analysis Role | GTEx Tissue Label | SMTSD Value |
|:-------------|:------------------|:------------|
| Primary | Frontal Cortex (BA9) | `Brain - Frontal Cortex (BA9)` |
| Replication | Putamen | `Brain - Putamen (basal ganglia)` |
| Replication | Hippocampus | `Brain - Hippocampus` |
| Replication | Nucleus Accumbens | `Brain - Nucleus accumbens (basal ganglia)` |
| Replication | Anterior Cingulate (BA24) | `Brain - Anterior cingulate cortex (BA24)` |

### Citation

GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across
human tissues. *Science*. 2020;369:1318–30.

dbGaP accession: phs000424.v8.p2
