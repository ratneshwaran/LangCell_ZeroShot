#!/usr/bin/env python3
"""
Ovarian Cancer Binary Classification Pipeline

This script:
1. Runs the binary preprocessing if needed
2. Sets up the binary classification dataset
3. Provides instructions for the zero-shot classification
"""

import os
import subprocess
import sys
from pathlib import Path

def run_preprocessing():
    """Run the ovarian binary preprocessing script"""
    print("="*60)
    print("STEP 1: Running Ovarian Binary Preprocessing")
    print("="*60)
    
    preprocess_script = Path(__file__).parent / "data_preprocess" / "preprocess_ovarian_binary.py"
    
    if not preprocess_script.exists():
        print(f"❌ Preprocessing script not found: {preprocess_script}")
        return False
    
    print(f"Running: {preprocess_script}")
    try:
        result = subprocess.run([sys.executable, str(preprocess_script)], 
                              capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ Preprocessing completed successfully!")
            print(result.stdout)
            return True
        else:
            print("❌ Preprocessing failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running preprocessing: {e}")
        return False

def check_datasets():
    """Check what datasets are available"""
    print("\n" + "="*60)
    print("STEP 2: Checking Available Datasets")
    print("="*60)
    
    data_dir = Path(__file__).parent / "your_data"
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return None
    
    # Look for ovarian datasets
    ovarian_datasets = list(data_dir.glob("tokenized_dataset_ovarian*"))
    binary_datasets = list(data_dir.glob("tokenized_dataset_ovarian_binary*"))
    
    print(f"Found {len(ovarian_datasets)} ovarian datasets:")
    for ds in ovarian_datasets:
        print(f"  - {ds.name}")
    
    print(f"Found {len(binary_datasets)} binary datasets:")
    for ds in binary_datasets:
        print(f"  - {ds.name}")
    
    # Return the most recent binary dataset
    if binary_datasets:
        latest_binary = max(binary_datasets, key=lambda x: x.stat().st_mtime)
        print(f"\n✅ Latest binary dataset: {latest_binary.name}")
        return latest_binary
    elif ovarian_datasets:
        latest_ovarian = max(ovarian_datasets, key=lambda x: x.stat().st_mtime)
        print(f"\n⚠️  No binary datasets found, but found ovarian dataset: {latest_ovarian.name}")
        print("You can create binary labels from this dataset in the notebook.")
        return latest_ovarian
    else:
        print("\n❌ No ovarian datasets found!")
        return None

def create_binary_notebook():
    """Create a binary classification notebook"""
    print("\n" + "="*60)
    print("STEP 3: Creating Binary Classification Notebook")
    print("="*60)
    
    notebook_path = Path(__file__).parent / "LangCell-annotation-zeroshot" / "zero-shot-ovarian_binary_fixed.ipynb"
    
    if notebook_path.exists():
        print(f"✅ Binary classification notebook already exists: {notebook_path}")
    else:
        print(f"❌ Binary classification notebook not found: {notebook_path}")
        print("Please create the notebook manually or copy from the template.")

def main():
    """Main pipeline"""
    print("OVARIAN CANCER BINARY CLASSIFICATION PIPELINE")
    print("="*60)
    
    # Step 1: Run preprocessing
    preprocessing_success = run_preprocessing()
    
    # Step 2: Check datasets
    dataset_path = check_datasets()
    
    # Step 3: Create notebook
    create_binary_notebook()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if preprocessing_success:
        print("✅ Preprocessing completed successfully")
    else:
        print("❌ Preprocessing failed or was skipped")
    
    if dataset_path:
        print(f"✅ Dataset available: {dataset_path.name}")
    else:
        print("❌ No suitable dataset found")
    
    print("\nNEXT STEPS:")
    print("1. Open the binary classification notebook:")
    print("   LangCell/LangCell-annotation-zeroshot/zero-shot-ovarian_binary_fixed.ipynb")
    print("2. Update the dataset path in the notebook if needed")
    print("3. Run the notebook cells to perform binary classification")
    print("\nThe key difference from zero-shot_malignant.ipynb is:")
    print("- Uses 'binary_label' column instead of 'malignancy'")
    print("- Specifically designed for ovarian cancer data")
    print("- Handles the different column structure")

if __name__ == "__main__":
    main() 