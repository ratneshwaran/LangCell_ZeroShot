import pandas as pd
import json

# Load your h5ad (if not already in memory)
import scanpy as sc
adata = sc.read_h5ad("/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/data/integrated_with_quiescence.h5ad")

# Get unique cell types
celltypes = adata.obs["celltype"].dropna().unique()

# Simple rules to convert to text descriptions
def describe(celltype):
    name = celltype.lower()
    if "pip" in name or "luminal" in name:
        return f"cell type: {celltype}. a luminal epithelial cell subtype from mammary tissue."
    elif "cycling" in name or "proliferating" in name:
        return f"cell type: {celltype}. a proliferating cell likely involved in active cell division."
    elif "endothelial" in name:
        return f"cell type: {celltype}. a vascular endothelial cell involved in angiogenesis."
    elif "pericyte" in name:
        return f"cell type: {celltype}. a perivascular cell that stabilizes blood vessels."
    elif "basal" in name:
        return f"cell type: {celltype}. a basal epithelial cell from the mammary gland."
    elif "immune" in name or "t cell" in name or "b cell" in name:
        return f"cell type: {celltype}. an immune lineage cell."
    else:
        return f"cell type: {celltype}. a mammary tissue-derived or stromal cell."

# Create mapping
type2text = {ct: describe(ct) for ct in celltypes}

# Save
with open("type2text.json", "w") as f:
    json.dump(type2text, f, indent=2)

print("✅ Saved type2text.json with", len(type2text), "entries.")
