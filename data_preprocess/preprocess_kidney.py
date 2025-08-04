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

print("LangCell Transcriptome Tokenization Pipeline - Kidney Cancer")
print("============================================================")

# 1) Load AnnData
adata_path = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/data/kidney_cancer_with_celltypes.h5ad"
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
    # Remove quotes from gene names if present
    clean_gene_name = gene_name.strip('"') if isinstance(gene_name, str) else gene_name
    
    if clean_gene_name in gene_name_id_dict:
        adata.var.iloc[i, adata.var.columns.get_loc("ensembl_id")] = gene_name_id_dict[clean_gene_name]
        converted_count += 1

print(f"-> {adata.var.shape[0]} genes in adata.var with 'ensembl_id'.")
print(f"-> Converted {converted_count} gene names to Ensembl IDs.")

# Debug: Check if we have any valid genes for tokenization
if converted_count == 0:
    print("🚨 WARNING: No genes were converted to Ensembl IDs!")
    print("This will cause empty tokenization.")
    print("Sample gene names from kidney data (cleaned):")
    print([g.strip('"') for g in adata.var["ensembl_id"].head(10).tolist()])
    print("Sample gene names from mapping dict:")
    print(list(gene_name_id_dict.keys())[:10])
    
    # Try alternative approach - use gene names directly
    print("-> Trying to use gene names directly...")
    adata.var["ensembl_id"] = adata.var.index  # Use gene names as Ensembl IDs

# 3) Add required obs columns
adata.obs["n_counts"]    = adata.X.sum(axis=1)
adata.obs["filter_pass"] = True
print(f"-> Added 'n_counts' and 'filter_pass' to adata.obs ({adata.obs.shape[1]} columns).")

# 4) Build maps - comprehensive for kidney_cancer_with_celltypes.h5ad
# Map the available columns to appropriate names for LangCell
obs2col = {}
for col in adata.obs.columns:
    if col in ['sample', 'cell_type', 'source']:
        obs2col[col] = col
    elif col == 'total_counts':
        obs2col[col] = 'n_counts'  # Map total_counts to n_counts
    elif col == 'n_genes_by_counts':
        obs2col[col] = 'n_genes'   # Map n_genes_by_counts to n_genes
    elif col == 'mp_assignment':
        obs2col[col] = 'cell_type_detailed'  # Map mp_assignment to cell_type_detailed
    elif col in ['complexity', 'umap1', 'umap2', 'g1s_score', 'g2m_score', 'cell_cycle_phase', 
                 'mp_top_score', 'mp_top', 'clusters_by_authors', 'cell_name']:
        obs2col[col] = col  # Keep other relevant columns
    # Skip columns with too many missing values or less relevant ones

# Handle missing cell_type values
print(f"-> Found {adata.obs['cell_type'].isna().sum()} cells with missing cell_type")
print(f"-> Cell type distribution before filtering:")
print(adata.obs['cell_type'].value_counts(dropna=False))

# For cells with missing cell_type, try to use clusters_by_authors as fallback
if 'clusters_by_authors' in adata.obs.columns:
    missing_cell_type = adata.obs['cell_type'].isna()
    if missing_cell_type.sum() > 0:
        print(f"-> Using clusters_by_authors as fallback for {missing_cell_type.sum()} cells")
        # Map clusters_by_authors to cell_type where cell_type is missing
        cluster_to_celltype = {
            'Epithelial 1': 'Epithelial',
            'Epithelial 2': 'Epithelial', 
            'Epithelial 3': 'Epithelial',
            'Epithelial 4': 'Epithelial',
            'Fibroblast': 'Fibroblast',
            'Endothelial': 'Endothelial',
            'M2 Macrophage': 'Macrophage'
        }
        
        for cluster, celltype in cluster_to_celltype.items():
            mask = (adata.obs['clusters_by_authors'] == cluster) & missing_cell_type
            adata.obs.loc[mask, 'cell_type'] = celltype
        
        print(f"-> After fallback: {adata.obs['cell_type'].isna().sum()} cells still missing cell_type")

print(f"-> obs2col: {len(obs2col)} metadata cols.")
print(f"-> Available columns: {list(obs2col.keys())}")

# Show final cell type distribution
print(f"\n-> Final cell type distribution:")
final_cell_types = adata.obs['cell_type'].value_counts()
for cell_type, count in final_cell_types.items():
    percentage = (count / len(adata.obs)) * 100
    print(f"  {cell_type}: {count} cells ({percentage:.1f}%)")

# Show sample distribution
print(f"\n-> Sample distribution:")
sample_dist = adata.obs['sample'].value_counts()
for sample, count in sample_dist.items():
    percentage = (count / len(adata.obs)) * 100
    print(f"  {sample}: {count} cells ({percentage:.1f}%)")

# 5) Tokenize
print("Tokenizing... (this may take a while)")

# Debug: Check data before tokenization
print(f"-> AnnData shape: {adata.shape}")
print(f"-> Number of genes with valid Ensembl IDs: {(adata.var['ensembl_id'].isin(gene_name_id_dict.keys())).sum()}")
print(f"-> Sample gene names: {adata.var['ensembl_id'].head(5).tolist()}")

try:
    tk = LangCellTranscriptomeTokenizer(
        obs2col,
        nproc=4
    )
    tokenized_cells, cell_metadata = tk.tokenize_anndata(adata)
    print(f"-> Tokenized {len(tokenized_cells)} cells.")
    
    # Debug: Check first few tokenized cells
    if len(tokenized_cells) > 0:
        print(f"-> First cell token count: {len(tokenized_cells[0])}")
        print(f"-> First cell tokens: {tokenized_cells[0][:10]}")
    else:
        print("🚨 ERROR: No cells were tokenized!")
        
except Exception as e:
    print(f"Error during tokenization: {e}")
    import traceback
    traceback.print_exc()
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
dataset_name = f"tokenized_dataset_kidney_{timestamp}"
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

# 10) Create cell type descriptions for kidney data
print("\nCreating cell type descriptions...")
cell_types = {
    'Malignant': 'Malignant epithelial cells from kidney cancer, showing abnormal proliferation and cancer-specific gene expression patterns',
    'Fibroblast': 'Fibroblast cells in kidney tissue, involved in extracellular matrix production and tissue remodeling',
    'Endothelial': 'Endothelial cells lining blood vessels in kidney tissue, involved in vascular function and angiogenesis',
    'Macrophage': 'Macrophage immune cells in kidney tissue, involved in immune response and tissue homeostasis',
    'Epithelial': 'Normal epithelial cells in kidney tissue, involved in kidney function and tissue structure'
}

# Save cell type descriptions
import json
type2text_path = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/LangCell/your_data/type2text_kidney.json"
with open(type2text_path, 'w') as f:
    json.dump(cell_types, f, indent=2)
print(f"✅ Cell type descriptions saved to: {type2text_path}")

print("\n" + "="*60)
print("KIDNEY CANCER PREPROCESSING COMPLETE!")
print("="*60)
print(f"Tokenized dataset: {out_dir}")
print(f"Cell type descriptions: {type2text_path}")
print("\nYou can now use these files in your LangCell analysis.") 