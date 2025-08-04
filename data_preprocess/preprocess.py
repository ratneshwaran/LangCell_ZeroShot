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
adata_path = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/data/integrated_with_quiescence.h5ad"
print("Loading AnnData from:", adata_path)
adata = sc.read_h5ad(adata_path)

# 2) Load gene name to ID mapping and convert gene names to Ensembl IDs
gene_name_id_path = Path(__file__).parent.parent / "geneformer_001" / "geneformer" / "gene_name_id_dict.pkl"
print("Loading gene name to ID mapping from:", gene_name_id_path)
with open(gene_name_id_path, "rb") as f:
    gene_name_id_dict = pickle.load(f)

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

# 4) Build maps
obs2col   = {c: c for c in adata.obs.columns}            # which obs cols to carry
print(f"-> obs2col: {len(obs2col)} metadata cols.")

# 5) Tokenize
print("Tokenizing... (this may take a while)")
tk = LangCellTranscriptomeTokenizer(
    obs2col,
    nproc=4
)
tokenized_cells, cell_metadata = tk.tokenize_anndata(adata)
print(f"-> Tokenized {len(tokenized_cells)} cells.")

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
out_dir = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/LangCell/your_data/tokenized_dataset"
print("Saving to:", out_dir)
hf_ds.save_to_disk(out_dir)

# 9) Verify
print("Verifying saved dataset…")
ds = load_from_disk(out_dir)
print("Loaded", len(ds), "cells; example input_ids length:", len(ds[0]["input_ids"]))
