# Zero-Shot Cancer Classification Notebook: Step-by-Step Explanation

## 1. Environment Setup
- The notebook sets up environment variables for cache directories to ensure reproducibility and efficient storage of model/data caches.

## 2. Model Components
- **Cell Encoder:**  
  - Uses a BERT-based model (`BertModel`) pre-trained on cell data, loaded from a checkpoint (`cell_bert`).
  - A custom `Pooler` projects the BERT output to a 256-dimensional embedding space.
- **Text Encoder:**  
  - Uses a BERT-based model (`MedBertModel`) pre-trained on biomedical text, loaded from a checkpoint (`text_bert`).
  - Also uses a custom `Pooler` for 256-dimensional embeddings.
- **CTM Head:**  
  - A linear layer (`ctm_head`) for cell-type matching, loaded from a checkpoint.

All models are moved to the GPU for efficient computation.

## 3. Data Preparation for Cancer Classification
- **Original Dataset:**  
  - The original dataset is loaded from disk using HuggingFace's `datasets` library. The path is `/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/LangCell/your_data/tokenized_dataset`.
  - The dataset contains tokenized gene expression data for single cells, with columns like `input_ids`, `celltype`, `malignancy`, and other metadata.
- **Cancer Label Mapping:**  
  - The `malignancy` column is mapped to binary labels: `"malignant"` → 1, everything else → 0.
  - A new `cancer_status` column is added with string labels: `"malignant"` or `"non-malignant"`.
- **New Dataset Creation:**  
  - The processed dataset is saved as `tokenized_dataset_cancer` for cancer classification.

## 4. type2text_cancer.json: Cancer Class Descriptions
- The file `type2text_cancer.json` maps each cancer status to a descriptive string:
  - `"malignant"`: "cancer status: malignant. A cell identified as cancerous or tumor-derived."
  - `"non-malignant"`: "cancer status: non-malignant. A cell identified as normal or not cancerous."
- These descriptions are used as prompts for the text encoder, enabling zero-shot cancer classification by providing the model with semantic information about each class.

## 5. Encoding Cancer Class Descriptions
- For each cancer status, its description from `type2text_cancer.json` is encoded using the text encoder to produce a 256-dimensional embedding.
- These embeddings are stacked to form a matrix (`text_embs`) representing both cancer classes in the embedding space.

## 6. Preparing the DataLoader for Cancer Classification
- The cancer dataset is used directly since the `label` field is already 0/1 (non-malignant/malignant).
- Only the necessary columns (`input_ids`, `attention_mask`, `label`) are kept.
- A PyTorch `DataLoader` is created for efficient batch processing.

## 7. Zero-Shot Cancer Classification Inference Loop
- For each batch of cells:
  - The cell encoder produces an embedding for each cell.
  - **Similarity Branch:**  
    - Computes the similarity between each cell embedding and both cancer class (text) embeddings.
    - Softmax is applied to get class probabilities.
  - **CTM Branch:**  
    - For each cancer class, the text encoder processes the class description together with the cell embedding (multimodal input).
    - The CTM head produces logits for each class.
    - Softmax is applied to get probabilities.
  - **Ensembling:**  
    - The outputs of the similarity and CTM branches are averaged to produce the final prediction.
  - The predicted class (argmax) and all logits are stored.

## 8. Saving Results
- The following are saved to `results_cancer.pt`:
  - `cell_embs`: All cell embeddings.
  - `sim_logits`, `ctm_logits`, `logits`: Logits from each branch and the ensemble.
  - `preds`: Predicted class indices (0 for non-malignant, 1 for malignant).
  - `labels`: True class indices.

## 9. Evaluation
- The predictions and true labels are loaded from `results_cancer.pt`.
- A confusion matrix and classification report (precision, recall, F1-score) are computed and printed.
- ROC AUC and average precision are computed using the positive class (malignant) scores.

## 10. Visualization
- A confusion matrix heatmap is plotted using seaborn, showing the 2x2 matrix for binary cancer classification.
- UMAP visualization is performed on the cell embeddings using Scanpy:
  - The AnnData object is created from the cell embeddings.
  - Cancer status, predicted label, and batch (if available) are added as metadata.
  - UMAP is computed and plotted, colored by cancer status and predictions.

## 11. Interpretation
- The notebook demonstrates that the LangCell model, without any fine-tuning, can classify cells as malignant or non-malignant in a zero-shot setting.
- The results show the model's ability to distinguish cancer status based on semantic class descriptions alone.
- The UMAP visualization shows good separation between malignant and non-malignant cells in the embedding space.

---

## Key Files and Their Roles

- **zero-shot_malignant.ipynb:**  
  The main notebook orchestrating the zero-shot cancer classification workflow.
- **type2text_cancer.json:**  
  Maps each cancer status to a detailed, human-readable description for use in zero-shot classification.
- **tokenized_dataset_cancer:**  
  The processed dataset with binary cancer labels (0/1) and cancer status strings.
- **results_cancer.pt:**  
  Stores all outputs from the inference loop for later evaluation and visualization.
- **utils.py:**  
  Contains custom model and data collator classes used in the workflow.

---

## Summary Table

| Step                | What Happens                                                                 |
|---------------------|------------------------------------------------------------------------------|
| Model loading       | Loads pre-trained cell and text encoders, and CTM head                       |
| Data preparation    | Maps malignancy column to binary labels (0/1)                                |
| type2text_cancer.json | Provides semantic descriptions for cancer classes                          |
| Embedding           | Encodes cancer class descriptions and cells into a shared space              |
| Inference           | Computes similarity and CTM scores, averages them, predicts cancer status    |
| Saving              | Stores embeddings, logits, predictions, and labels to disk                  |
| Evaluation          | Computes confusion matrix, classification report, ROC AUC, and AP           |
| Visualization       | Plots confusion matrix and UMAP of cell embeddings                          |

---

## What You Are Doing

- You are evaluating the zero-shot cancer classification ability of the LangCell model.
- You use semantic cancer class descriptions (from `type2text_cancer.json`) to enable the model to predict cancer status it was not explicitly trained on.
- You analyze the results quantitatively (metrics) and qualitatively (visualization) to assess model performance for binary cancer classification.

---

## Key Differences from Cell Type Classification

| Aspect                    | Cell Type Classification           | Cancer Classification              |
|---------------------------|-----------------------------------|-----------------------------------|
| Number of classes         | 21 cell types                     | 2 classes (malignant/non-malignant) |
| Label mapping             | String to numeric mapping         | Binary mapping (0/1)              |
| Class descriptions        | Cell type descriptions            | Cancer status descriptions        |
| Evaluation metrics        | Multi-class metrics               | Binary classification metrics     |
| Visualization             | 21x21 confusion matrix           | 2x2 confusion matrix             |
| UMAP coloring            | By cell type                     | By cancer status                  |

---

## Performance Interpretation

- **Binary Classification**: The model performs cancer vs. non-cancer classification.
- **Imbalanced Data**: Typically, non-malignant cells outnumber malignant cells.
- **Key Metrics**: Focus on precision, recall, and F1-score for the malignant class (positive class).
- **ROC AUC**: Measures the model's ability to rank malignant cells higher than non-malignant cells.
- **Visual Assessment**: UMAP plots show how well the model separates malignant from non-malignant cells in the embedding space. 