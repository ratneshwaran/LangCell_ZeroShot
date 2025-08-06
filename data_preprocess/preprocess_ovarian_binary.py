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

print("LangCell Transcriptome Tokenization Pipeline - Ovarian Cancer Binary Classification")
print("==================================================================================")

# 1) Load AnnData
adata_path = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/data/ovarian_cancer_with_celltypes_v1.h5ad"
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
    print("Sample gene names from ovarian data (cleaned):")
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

# 4) Build maps - comprehensive for ovarian_cancer_with_celltypes_v1.h5ad
# Map the available columns to appropriate names for LangCell
obs2col = {}
for col in adata.obs.columns:
    if col in ['sample', 'cell_type', 'cell_subtype', 'source', 'disease', 'histology', 'stage']:
        obs2col[col] = col
    elif col == 'total_counts':
        obs2col[col] = 'n_counts'  # Map total_counts to n_counts
    elif col == 'n_genes_by_counts':
        obs2col[col] = 'n_genes'   # Map n_genes_by_counts to n_genes
    elif col == 'mp_assignment':
        obs2col[col] = 'cell_type_detailed'  # Map mp_assignment to cell_type_detailed
    elif col in ['complexity', 'umap1', 'umap2', 'g1s_score', 'g2m_score', 'cell_cycle_phase', 
                 'mp_top_score', 'mp_top', 'cell_name']:
        obs2col[col] = col  # Keep other relevant columns
    # Include clinical metadata columns
    elif col.startswith('sample_'):
        obs2col[col] = col  # Include all sample metadata
    # Skip columns with too many missing values or less relevant ones

# Handle missing cell_type values and Unassigned cells
print(f"-> Found {adata.obs['cell_type'].isna().sum()} cells with missing cell_type")
print(f"-> Cell type distribution before filtering:")
print(adata.obs['cell_type'].value_counts(dropna=False))

# Handle Unassigned cells - try to assign them based on available information
unassigned_mask = adata.obs['cell_type'] == 'Unassigned'
if unassigned_mask.sum() > 0:
    print(f"-> Found {unassigned_mask.sum()} Unassigned cells")
    
    # Try to assign Unassigned cells based on mp_assignment if available
    if 'mp_assignment' in adata.obs.columns:
        print("-> Attempting to assign Unassigned cells based on mp_assignment...")
        
        # Map MP assignments to cell types for ovarian cancer based on actual data
        mp_to_celltype = {
            'Stress': 'Malignant',
            'Cilia': 'Epithelial',
            'Cell Cycle - G1/S': 'Malignant',
            'Monocyte/Secreted': 'Macrophage',
            'CAF1': 'Fibroblast',
            'EMT I': 'Malignant',
            'Cell Cycle - G2/M': 'Malignant',
            'Secreted I': 'Epithelial',
            'CAF9': 'Fibroblast',
            'Complement': 'Macrophage',
            'Androgen-prostate': 'Epithelial',
            'Endo1': 'Endothelial',
            'HEV1': 'Endothelial',
            'Lipid-associated': 'Macrophage',
            'Protein maturation': 'Epithelial',
            'Notch signaling': 'Epithelial'
        }
        
        assigned_count = 0
        for mp_assignment, celltype in mp_to_celltype.items():
            mask = (adata.obs['mp_assignment'] == mp_assignment) & unassigned_mask
            if mask.sum() > 0:
                adata.obs.loc[mask, 'cell_type'] = celltype
                assigned_count += mask.sum()
                print(f"  - Assigned {mask.sum()} cells from '{mp_assignment}' to '{celltype}'")
        
        print(f"-> Successfully assigned {assigned_count} Unassigned cells")
        print(f"-> Remaining Unassigned cells: {(adata.obs['cell_type'] == 'Unassigned').sum()}")
    
    # For remaining Unassigned cells, try to assign based on gene expression patterns
    # This is a simplified approach - in practice, you might want more sophisticated methods
    remaining_unassigned = adata.obs['cell_type'] == 'Unassigned'
    if remaining_unassigned.sum() > 0:
        print(f"-> {remaining_unassigned.sum()} cells remain Unassigned - keeping as is")

print(f"-> obs2col: {len(obs2col)} metadata cols.")
print(f"-> Available columns: {list(obs2col.keys())}")

# Show final cell type distribution
print(f"\n-> Final cell type distribution:")
final_cell_types = adata.obs['cell_type'].value_counts()
for cell_type, count in final_cell_types.items():
    percentage = (count / len(adata.obs)) * 100
    print(f"  {cell_type}: {count} cells ({percentage:.1f}%)")

# 5) Create binary classification labels
print("\n-> Creating binary classification labels...")
print("-> Binary classification: Malignant vs Non-malignant")

# Create binary labels
def create_binary_labels(row):
    if row['cell_type'] == 'Malignant':
        return 1  # Malignant
    else:
        return 0  # Non-malignant (all other cell types)

adata.obs['binary_label'] = adata.obs.apply(create_binary_labels, axis=1)

# Count binary classes
malignant_count = (adata.obs['binary_label'] == 1).sum()
non_malignant_count = (adata.obs['binary_label'] == 0).sum()
total_cells = len(adata.obs)

print(f"-> Malignant cells: {malignant_count}")
print(f"-> Non-malignant cells: {non_malignant_count}")
print(f"-> Total cells: {total_cells}")
print(f"-> Malignant ratio: {malignant_count/total_cells:.3f}")
print(f"-> Non-malignant ratio: {non_malignant_count/total_cells:.3f}")

# Add binary label to obs2col mapping
obs2col['binary_label'] = 'binary_label'

# Show sample distribution
print(f"\n-> Sample distribution:")
sample_dist = adata.obs['sample'].value_counts()
for sample, count in sample_dist.items():
    percentage = (count / len(adata.obs)) * 100
    print(f"  {sample}: {count} cells ({percentage:.1f}%)")

# Show clinical information distribution
print(f"\n-> Clinical information distribution:")
if 'sample_cancer_type' in adata.obs.columns:
    cancer_type_dist = adata.obs['sample_cancer_type'].value_counts()
    for cancer_type, count in cancer_type_dist.items():
        percentage = (count / len(adata.obs)) * 100
        print(f"  Cancer type: {cancer_type}: {count} cells ({percentage:.1f}%)")

if 'sample_grade' in adata.obs.columns:
    grade_dist = adata.obs['sample_grade'].value_counts()
    for grade, count in grade_dist.items():
        percentage = (count / len(adata.obs)) * 100
        print(f"  Grade: {grade}: {count} cells ({percentage:.1f}%)")

if 'sample_age_at_diagnosis' in adata.obs.columns:
    age_dist = adata.obs['sample_age_at_diagnosis'].value_counts()
    print(f"  Age at diagnosis distribution:")
    for age, count in age_dist.items():
        percentage = (count / len(adata.obs)) * 100
        print(f"    Age {age}: {count} cells ({percentage:.1f}%)")

if 'sample_race' in adata.obs.columns:
    race_dist = adata.obs['sample_race'].value_counts()
    print(f"  Race distribution:")
    for race, count in race_dist.items():
        percentage = (count / len(adata.obs)) * 100
        print(f"    {race}: {count} cells ({percentage:.1f}%)")

if 'sample_bmi' in adata.obs.columns:
    bmi_dist = adata.obs['sample_bmi'].value_counts()
    print(f"  BMI distribution:")
    for bmi, count in bmi_dist.items():
        percentage = (count / len(adata.obs)) * 100
        print(f"    BMI {bmi}: {count} cells ({percentage:.1f}%)")

if 'disease' in adata.obs.columns:
    disease_dist = adata.obs['disease'].value_counts()
    print(f"  Disease distribution:")
    for disease, count in disease_dist.items():
        percentage = (count / len(adata.obs)) * 100
        print(f"    {disease}: {count} cells ({percentage:.1f}%)")

if 'histology' in adata.obs.columns:
    histology_dist = adata.obs['histology'].value_counts()
    print(f"  Histology distribution:")
    for histology, count in histology_dist.items():
        percentage = (count / len(adata.obs)) * 100
        print(f"    {histology}: {count} cells ({percentage:.1f}%)")

if 'stage' in adata.obs.columns:
    stage_dist = adata.obs['stage'].value_counts()
    print(f"  Stage distribution:")
    for stage, count in stage_dist.items():
        percentage = (count / len(adata.obs)) * 100
        print(f"    {stage}: {count} cells ({percentage:.1f}%)")

# 6) Tokenize
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

# 7) Sanity check
print("Example token count:", len(tokenized_cells[0]), "first tokens:", tokenized_cells[0][:10])

# 8) Clean metadata
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

# 9) Build and save HF dataset
print("Creating HF dataset…")
hf_ds = tk.create_dataset(tokenized_cells, df_meta_clean)

# Create unique dataset name based on input file and timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dataset_name = f"tokenized_dataset_ovarian_binary_{timestamp}"
out_dir = f"/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/LangCell/your_data/{dataset_name}"

# Create output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

print("Saving to:", out_dir)
hf_ds.save_to_disk(out_dir)

# 10) Verify
print("Verifying saved dataset…")
try:
    ds = load_from_disk(out_dir)
    print("Loaded", len(ds), "cells; example input_ids length:", len(ds[0]["input_ids"]))
    print("✅ Dataset saved and verified successfully!")
except Exception as e:
    print(f"❌ Error verifying dataset: {e}")
    raise

# 11) Create binary classification descriptions
print("\nCreating binary classification descriptions...")
binary_types = {
    'Malignant': 'Malignant epithelial cells from ovarian cancer, showing abnormal proliferation and cancer-specific gene expression patterns',
    'Non-malignant': 'Non-malignant cells including normal epithelial, immune cells, stromal cells, and other non-cancerous cell types in ovarian tissue'
}

# Save binary classification descriptions
import json
binary_type2text_path = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/LangCell/your_data/binary_type2text_ovarian.json"
with open(binary_type2text_path, 'w') as f:
    json.dump(binary_types, f, indent=2)
print(f"✅ Binary classification descriptions saved to: {binary_type2text_path}")

# 12) Create cell type descriptions for reference (keeping original cell types)
print("\nCreating cell type descriptions for reference...")
cell_types = {
    'Malignant': 'Malignant epithelial cells from ovarian cancer, showing abnormal proliferation and cancer-specific gene expression patterns',
    'Unassigned': 'Cells with unclear cell type assignment, may represent rare cell types or cells in transition states',
    'Epithelial': 'Normal epithelial cells in ovarian tissue, involved in ovarian function and tissue structure',
    'T_cell': 'T lymphocytes in ovarian tissue, involved in adaptive immune response and immune surveillance',
    'Endothelial': 'Endothelial cells lining blood vessels in ovarian tissue, involved in vascular function and angiogenesis',
    'Fibroblast': 'Fibroblast cells in ovarian tissue, involved in extracellular matrix production and tissue remodeling',
    'Macrophage': 'Macrophage immune cells in ovarian tissue, involved in immune response and tissue homeostasis',
    'Mast': 'Mast cells in ovarian tissue, involved in inflammatory responses and immune regulation',
    'B_cell': 'B lymphocytes in ovarian tissue, involved in antibody production and humoral immune response',
    'Myocyte': 'Muscle cells in ovarian tissue, involved in tissue structure and contractility'
}

# Save cell type descriptions
type2text_path = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/LangCell/your_data/type2text_ovarian.json"
with open(type2text_path, 'w') as f:
    json.dump(cell_types, f, indent=2)
print(f"✅ Cell type descriptions saved to: {type2text_path}")

print("\n" + "="*70)
print("OVARIAN CANCER BINARY CLASSIFICATION PREPROCESSING COMPLETE!")
print("="*70)
print(f"Tokenized dataset: {out_dir}")
print(f"Binary classification descriptions: {binary_type2text_path}")
print(f"Cell type descriptions: {type2text_path}")
print(f"\nBinary classification summary:")
print(f"  - Malignant cells: {malignant_count} ({malignant_count/total_cells:.1%})")
print(f"  - Non-malignant cells: {non_malignant_count} ({non_malignant_count/total_cells:.1%})")
print(f"  - Total cells: {total_cells}")
print("\nYou can now use these files in your LangCell binary classification analysis.") 