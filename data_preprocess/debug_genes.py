#!/usr/bin/env python3
import os
import scanpy as sc
import pickle
from pathlib import Path

# 0) (Optional) local caches
os.environ["PIP_CACHE_DIR"] = "/cs/student/projects1/aibh/2024/rmaheswa/cache"
os.environ["HF_HOME"]       = "/cs/student/projects1/aibh/2024/rmaheswa/cache/huggingface"

print("Debug: Gene Overlap Analysis")
print("============================")

# 1) Load AnnData
adata_path = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/data/integrated_with_quiescence.h5ad"
print("Loading AnnData from:", adata_path)
adata = sc.read_h5ad(adata_path)

# 2) Ensure ensembl_id
if "feature_id" in adata.var.columns:
    adata.var["ensembl_id"] = adata.var["feature_id"]
else:
    adata.var["ensembl_id"] = adata.var.index

# 3) Load token dictionary
token_dict_path = Path(__file__).parent.parent / "geneformer_001" / "geneformer" / "token_dictionary.pkl"
print("Loading token dictionary from:", token_dict_path)
with open(token_dict_path, "rb") as f:
    token_dict = pickle.load(f)

# 4) Load gene median dictionary
gene_median_path = Path(__file__).parent.parent / "geneformer_001" / "geneformer" / "gene_median_dictionary.pkl"
print("Loading gene median dictionary from:", gene_median_path)
with open(gene_median_path, "rb") as f:
    gene_median_dict = pickle.load(f)

# 5) Load gene name to ID mapping
gene_name_id_path = Path(__file__).parent.parent / "geneformer_001" / "geneformer" / "gene_name_id_dict.pkl"
print("Loading gene name to ID mapping from:", gene_name_id_path)
with open(gene_name_id_path, "rb") as f:
    gene_name_id_dict = pickle.load(f)

# 6) Analyze overlap
dataset_genes = set(adata.var["ensembl_id"])
token_dict_genes = set(token_dict.keys())
gene_median_genes = set(gene_median_dict.keys())
gene_name_id_genes = set(gene_name_id_dict.keys())

print(f"\nDataset genes: {len(dataset_genes)}")
print(f"Token dictionary genes: {len(token_dict_genes)}")
print(f"Gene median dictionary genes: {len(gene_median_genes)}")
print(f"Gene name to ID mapping genes: {len(gene_name_id_genes)}")

# Check overlap with gene name mapping
overlap_with_name_mapping = dataset_genes.intersection(gene_name_id_genes)
print(f"\nOverlap with gene name mapping: {len(overlap_with_name_mapping)} ({len(overlap_with_name_mapping)/len(dataset_genes)*100:.1f}%)")

# Convert dataset genes to Ensembl IDs
converted_genes = set()
for gene in dataset_genes:
    if gene in gene_name_id_dict:
        converted_genes.add(gene_name_id_dict[gene])

print(f"Converted to Ensembl IDs: {len(converted_genes)}")

# Check overlap after conversion
overlap_with_token_converted = converted_genes.intersection(token_dict_genes)
overlap_with_median_converted = converted_genes.intersection(gene_median_genes)

print(f"\nOverlap with token dictionary (after conversion): {len(overlap_with_token_converted)} ({len(overlap_with_token_converted)/len(dataset_genes)*100:.1f}%)")
print(f"Overlap with gene median dictionary (after conversion): {len(overlap_with_median_converted)} ({len(overlap_with_median_converted)/len(dataset_genes)*100:.1f}%)")

# Show some examples
print(f"\nFirst 10 dataset genes: {list(dataset_genes)[:10]}")
print(f"First 10 token dict genes: {list(token_dict_genes)[:10]}")
print(f"First 10 converted genes: {list(converted_genes)[:10]}")
print(f"First 10 overlapping genes (after conversion): {list(overlap_with_token_converted)[:10]}")

# Check if any genes have non-zero expression
print(f"\nChecking gene expression...")
nonzero_genes = set()
for gene in list(dataset_genes)[:100]:  # Check first 100 genes
    if gene in adata.var.index:
        gene_idx = adata.var.index.get_loc(gene)
        if adata.X[:, gene_idx].sum() > 0:
            nonzero_genes.add(gene)

print(f"Genes with non-zero expression (first 100 checked): {len(nonzero_genes)}")
if nonzero_genes:
    print(f"Example non-zero genes: {list(nonzero_genes)[:5]}")
    # Check if any of these are in the token dictionary after conversion
    nonzero_converted = set()
    for gene in nonzero_genes:
        if gene in gene_name_id_dict:
            nonzero_converted.add(gene_name_id_dict[gene])
    
    nonzero_in_token = nonzero_converted.intersection(token_dict_genes)
    print(f"Non-zero genes in token dictionary (after conversion): {len(nonzero_in_token)}")
    if nonzero_in_token:
        print(f"Example: {list(nonzero_in_token)[:5]}") 