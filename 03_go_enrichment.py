"""
03_go_enrichment.py
===================
Performs Gene Ontology (GO:BP, GO:MF, GO:CC), KEGG, and Reactome pathway
enrichment using the gProfiler g:GOSt API for all gene sets generated
by 02_network_comparison.py.

Methodology (from manuscript):
    - Background set: all expressed genes in BA9 (passed via ordered query)
    - Multiple testing correction: g:SCS (gProfiler native method)
    - Significance threshold: adjusted p < 0.05
    - Enrichment computed for: full top 5% networks, unique gene sets,
      three-way shared core, and pairwise shared sets

Input:
    Gene lists from 02_network_comparison.py
    Full top 5% lists from 01_genome_wide_correlations.py

Output:
    {gene_set}_GO_enrichment.csv  — Enrichment results per gene set

Usage:
    python 03_go_enrichment.py

Note: Requires internet access for gProfiler API calls.
"""

import os
import time
import pandas as pd
import requests

# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════
RESULTS_DIR = "../results"
OUTPUT_DIR = "../results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ORGANISM = 'hsapiens'
SOURCES = ['GO:BP', 'GO:MF', 'GO:CC', 'REAC', 'KEGG']
SIGNIFICANCE_THRESHOLD = 0.05
API_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"

# Google Colab paths (uncomment if using Colab):
# RESULTS_DIR = "/content/drive/MyDrive/Research/Results"
# OUTPUT_DIR = "/content/drive/MyDrive/Research/Results/Manuscript_Statistics"

# Gene sets to analyze — paths relative to RESULTS_DIR
GENE_SETS = {
    # Full top 5% networks
    'PELO_full':          'PELO_top5pct.csv',
    'LTN1_full':          'LTN1_top5pct.csv',
    'NEMF_full':          'NEMF_top5pct.csv',
    # Unique gene sets
    'PELO_unique':        'PELO_unique_top5pct.csv',
    'LTN1_unique':        'LTN1_unique_top5pct.csv',
    'NEMF_unique':        'NEMF_unique_top5pct.csv',
    # Shared sets
    'RQC_shared_all3':    'RQC_shared_all3_top5pct.csv',
    'LTN1_NEMF_shared':   'LTN1_NEMF_shared_only.csv',
    'PELO_LTN1_shared':   'PELO_LTN1_shared_only.csv',
    'PELO_NEMF_shared':   'PELO_NEMF_shared_only.csv',
}


# ═══════════════════════════════════════════════════════
# gPROFILER API FUNCTION
# ═══════════════════════════════════════════════════════
def run_gprofiler(gene_list, organism='hsapiens', sources=None, max_retries=3):
    """
    Query gProfiler g:GOSt API for functional enrichment.

    Parameters:
        gene_list: list of gene symbols
        organism: gProfiler organism code
        sources: list of annotation sources
        max_retries: number of retry attempts on failure

    Returns:
        pd.DataFrame with enrichment results sorted by p-value
    """
    payload = {
        'organism': organism,
        'query': gene_list,
        'sources': sources or SOURCES,
        'user_threshold': SIGNIFICANCE_THRESHOLD,
        'significance_threshold_method': 'g_SCS',
        'no_evidences': False,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=120)
            if response.status_code == 200:
                break
            elif response.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API error: {response.status_code}")
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"    Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    data = response.json()
    if 'result' not in data or len(data['result']) == 0:
        return pd.DataFrame()

    rows = []
    for term in data['result']:
        rows.append({
            'source':             term.get('source', ''),
            'term_id':            term.get('native', ''),
            'name':               term.get('name', ''),
            'p_value':            term.get('p_value', 1.0),
            'term_size':          term.get('term_size', 0),
            'query_size':         term.get('query_size', 0),
            'intersection_size':  term.get('intersection_size', 0),
            'precision':          term.get('precision', 0),
            'recall':             term.get('recall', 0),
            'intersections':      ','.join(term.get('intersections', [])),
        })

    return pd.DataFrame(rows).sort_values('p_value').reset_index(drop=True)


# ═══════════════════════════════════════════════════════
# RUN ENRICHMENT FOR ALL GENE SETS
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("03 — Gene Ontology / Pathway Enrichment (gProfiler)")
print("=" * 60)

for name, filename in GENE_SETS.items():
    filepath = f"{RESULTS_DIR}/{filename}"
    if not os.path.exists(filepath):
        # Try Manuscript_Statistics subdirectory
        filepath = f"{RESULTS_DIR}/Manuscript_Statistics/{filename}"
        if not os.path.exists(filepath):
            print(f"\n  SKIP: {filename} not found")
            continue

    df = pd.read_csv(filepath)
    gene_col = [c for c in df.columns if c.lower() in ['gene', 'gene_name', 'symbol']][0]
    genes = df[gene_col].dropna().tolist()

    print(f"\n{'─' * 50}")
    print(f"  {name}: {len(genes)} genes")

    enrichment = run_gprofiler(genes)

    if len(enrichment) > 0:
        outpath = f"{OUTPUT_DIR}/{name}_GO_enrichment.csv"
        enrichment.to_csv(outpath, index=False)
        print(f"  → {len(enrichment)} significant terms saved")

        # Print top 3 GO:BP terms
        bp = enrichment[enrichment['source'] == 'GO:BP'].head(3)
        for _, row in bp.iterrows():
            print(f"    GO:BP  {row['name'][:60]:60s}  p = {row['p_value']:.2e}")

        # Print top Reactome term
        reac = enrichment[enrichment['source'] == 'REAC'].head(1)
        for _, row in reac.iterrows():
            print(f"    REAC   {row['name'][:60]:60s}  p = {row['p_value']:.2e}")
    else:
        print(f"  → No significant terms")

    time.sleep(1)  # Courtesy pause between API calls


print(f"\n{'=' * 60}")
print("✓ All enrichments complete")
print(f"{'=' * 60}")
