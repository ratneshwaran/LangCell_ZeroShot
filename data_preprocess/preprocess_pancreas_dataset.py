#!/usr/bin/env python3
import os
import scanpy as sc
import pandas as pd
import pickle
from pathlib import Path
from datasets import load_from_disk
from utils import LangCellTranscriptomeTokenizer

# 0) (Optional) local caches
os.environ["PIP_CACHE_DIR"] = "/cs/student/projects1/aibh/2024/rmaheswa/cache"
os.environ["HF_HOME"]       = "/cs/student/projects1/aibh/2024/rmaheswa/cache/huggingface"

print("LangCell Transcriptome Tokenization Pipeline")
print("===========================================")

# 1) Load AnnData
adata_path = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/data/pancreas_cancer_with_celltypes.h5ad"
print("Loading AnnData from:", adata_path)

# Check if file exists
if not os.path.exists(adata_path):
    raise FileNotFoundError(f"AnnData file not found: {adata_path}")

adata = sc.read_h5ad(adata_path)
print(f"-> Loaded AnnData with shape: {adata.shape}")

# 2) Load gene name to ID mapping and convert gene names to Ensembl IDs
gene_name_id_path = Path(__file__).parent.parent / "geneformer_001" / "geneformer" / "gene_name_id_dict.pkl"
print("Loading gene name to ID mapping from:", gene_name_id_path)

# Check if gene mapping file exists
if not gene_name_id_path.exists():
    raise FileNotFoundError(f"Gene mapping file not found: {gene_name_id_path}")

with open(gene_name_id_path, "rb") as f:
    gene_name_id_dict = pickle.load(f)
print(f"-> Loaded gene mapping with {len(gene_name_id_dict)} entries")

# Convert gene names to Ensembl IDs
if "feature_id" in adata.var.columns:
    adata.var["ensembl_id"] = adata.var["feature_id"]
else:
    adata.var["ensembl_id"] = adata.var.index

# Convert gene names to Ensembl IDs where possible
converted_count = 0
for i, gene_name in enumerate(adata.var["ensembl_id"]):
    if gene_name in gene_name_id_dict:
        adata.var.iloc[i, adata.var.columns.get_loc("ensembl_id")] = gene_name_id_dict[gene_name]
        converted_count += 1

print(f"-> {adata.var.shape[0]} genes in adata.var with 'ensembl_id'.")
print(f"-> Converted {converted_count} gene names to Ensembl IDs.")

# 3) Add required obs columns
adata.obs["n_counts"]    = adata.X.sum(axis=1)
adata.obs["filter_pass"] = True
print(f"-> Added 'n_counts' and 'filter_pass' to adata.obs ({adata.obs.shape[1]} columns).")

# 4) Build maps - updated for pancreas_cancer_with_celltypes.h5ad
# Map the available columns to appropriate names for LangCell
obs2col = {}
for col in adata.obs.columns:
    if col in ['group', 'sample_id', 'sample', 'cell_type', 'cell_subtype', 'disease', 'source']:
        obs2col[col] = col
    elif col == 'total_counts':
        obs2col[col] = 'n_counts'  # Map total_counts to n_counts
    elif col == 'n_genes_by_counts':
        obs2col[col] = 'n_genes'   # Map n_genes_by_counts to n_genes
    elif col == 'mp_assignment':
        obs2col[col] = 'cell_type_detailed'  # Map mp_assignment to cell_type_detailed
    elif col in ['complexity', 'umap1', 'umap2', 'g1s_score', 'g2m_score', 'cell_cycle_phase', 
                 'mp_top_score', 'mp_top']:
        obs2col[col] = col  # Keep other relevant columns
    # Skip columns with too many missing values or less relevant ones

print(f"-> obs2col: {len(obs2col)} metadata cols.")
print(f"-> Available columns: {list(obs2col.keys())}")

# 5) Tokenize
print("Tokenizing... (this may take a while)")
try:
    tk = LangCellTranscriptomeTokenizer(
        obs2col,
        nproc=4
    )
    tokenized_cells, cell_metadata = tk.tokenize_anndata(adata)
    print(f"-> Tokenized {len(tokenized_cells)} cells.")
except Exception as e:
    print(f"Error during tokenization: {e}")
    raise

# 6) Sanity check
print("Example token count:", len(tokenized_cells[0]), "first tokens:", tokenized_cells[0][:10])

# 7) Clean metadata
df_meta  = pd.DataFrame(cell_metadata)
cleaned  = {}
for col in df_meta.columns:
    s = df_meta[col]
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        cleaned[col] = num
        print(f"  - '{col}' kept as numeric")
    else:
        cleaned[col] = s.astype(str)
        print(f"  - '{col}' cast to string")
df_meta_clean = pd.DataFrame(cleaned)

# 8) Build and save HF dataset
print("Creating HF dataset…")
hf_ds = tk.create_dataset(tokenized_cells, df_meta_clean)

# Create unique dataset name based on input file and timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dataset_name = f"tokenized_dataset_pancreas_{timestamp}"
out_dir = f"/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/LangCell/your_data/{dataset_name}"

# Create output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

print("Saving to:", out_dir)
hf_ds.save_to_disk(out_dir)

# 9) Verify
print("Verifying saved dataset…")
try:
    ds = load_from_disk(out_dir)
    print("Loaded", len(ds), "cells; example input_ids length:", len(ds[0]["input_ids"]))
    print("✅ Dataset saved and verified successfully!")
except Exception as e:
    print(f"❌ Error verifying dataset: {e}")
    raise
