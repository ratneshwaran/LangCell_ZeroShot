#!/usr/bin/env python3
"""
Update Kidney Notebook Paths

This script updates the kidney notebook with the correct paths for kidney data.
"""

import json
import re

def update_kidney_notebook():
    """Update the kidney notebook with correct paths."""
    
    notebook_path = "zero-shot-kidney.ipynb"
    
    # Read the notebook
    with open(notebook_path, 'r') as f:
        content = f.read()
    
    # Update dataset path
    content = re.sub(
        r'tokenized_dataset_pancreas_20250802_001220',
        r'tokenized_dataset_kidney_20250802_001220',
        content
    )
    
    # Update type2text path
    content = re.sub(
        r'type2text_pancreas\.json',
        r'type2text_kidney.json',
        content
    )
    
    # Update results path
    content = re.sub(
        r'results_pancreas\.pt',
        r'results_kidney.pt',
        content
    )
    
    # Write updated notebook
    with open(notebook_path, 'w') as f:
        f.write(content)
    
    print("✅ Updated kidney notebook paths:")
    print("  - Dataset: tokenized_dataset_kidney_20250802_001220")
    print("  - Type2text: type2text_kidney.json")
    print("  - Results: results_kidney.pt")

if __name__ == "__main__":
    update_kidney_notebook() 