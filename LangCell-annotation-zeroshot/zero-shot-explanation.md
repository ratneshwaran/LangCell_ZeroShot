# Zero-Shot Cell Type Annotation Notebook: Step-by-Step Explanation

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

## 3. Data Loading and Preprocessing
- **Dataset:**  
  - The main dataset is loaded from disk using HuggingFace's `datasets` library. The path is `/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/LangCell/your_data/tokenized_dataset`.
  - The dataset contains tokenized gene expression data for single cells, with columns like `input_ids`, `celltype`, and other metadata.
- **Label Normalization:**  
  - The code ensures the cell type label column is named `celltype` for consistency.
- **Class List:**  
  - The set of unique cell types is extracted for downstream processing.

## 4. type2text.json: Class Descriptions
- The file `type2text.json` maps each cell type (e.g., `"Plasma cell"`) to a descriptive string (e.g., `"cell type: Plasma cell. a B-cell-derived immune cell specialized in antibody secretion."`).
- These descriptions are used as prompts for the text encoder, enabling zero-shot classification by providing the model with semantic information about each class.

## 5. Encoding Class Descriptions
- For each cell type, its description from `type2text.json` is encoded using the text encoder to produce a 256-dimensional embedding.
- These embeddings are stacked to form a matrix (`text_embs`) representing all possible cell types in the embedding space.

## 6. Preparing the DataLoader
- The dataset is mapped so that each cell's `celltype` string is converted to a numeric label.
- Only the necessary columns (`input_ids`, `attention_mask`, `label`) are kept.
- A PyTorch `DataLoader` is created for efficient batch processing.

## 7. Zero-Shot Inference Loop
- For each batch of cells:
  - The cell encoder produces an embedding for each cell.
  - **Similarity Branch:**  
    - Computes the similarity between each cell embedding and all class (text) embeddings.
    - Softmax is applied to get class probabilities.
  - **CTM Branch:**  
    - For each class, the text encoder processes the class description together with the cell embedding (multimodal input).
    - The CTM head produces logits for each class.
    - Softmax is applied to get probabilities.
  - **Ensembling:**  
    - The outputs of the similarity and CTM branches are averaged to produce the final prediction.
  - The predicted class (argmax) and all logits are stored.

## 8. Saving Results
- The following are saved to `results.pt`:
  - `cell_embs`: All cell embeddings.
  - `sim_logits`, `ctm_logits`, `logits`: Logits from each branch and the ensemble.
  - `preds`: Predicted class indices.
  - `labels`: True class indices.

## 9. Evaluation
- The predictions and true labels are loaded from `results.pt`.
- A confusion matrix and classification report (precision, recall, F1-score) are computed and printed.
- Macro-averaged ROC AUC and average precision are also computed.

## 10. Visualization
- A confusion matrix heatmap is plotted using seaborn.
- UMAP visualization is performed on the cell embeddings using Scanpy:
  - The AnnData object is created from the cell embeddings.
  - Cell type, predicted label, and batch (if available) are added as metadata.
  - UMAP is run and plotted, colored by cell type, prediction, and batch.

## 11. Interpretation
- The notebook demonstrates that the LangCell model, without any fine-tuning, can cluster cells by type and reduce batch effects in a zero-shot setting.
- The results show the model's ability to annotate cell types based on semantic class descriptions alone.

---

## Key Files and Their Roles

- **zero-shot.ipynb:**  
  The main notebook orchestrating the zero-shot annotation workflow.
- **type2text.json:**  
  Maps each cell type to a detailed, human-readable description for use in zero-shot classification.
- **results.pt:**  
  Stores all outputs from the inference loop for later evaluation and visualization.
- **utils.py:**  
  Contains custom model and data collator classes used in the workflow.

---

## Summary Table

| Step                | What Happens                                                                 |
|---------------------|------------------------------------------------------------------------------|
| Model loading       | Loads pre-trained cell and text encoders, and CTM head                       |
| Data loading        | Loads tokenized single-cell data                                             |
| type2text.json      | Provides semantic descriptions for each cell type                            |
| Embedding           | Encodes class descriptions and cells into a shared space                     |
| Inference           | Computes similarity and CTM scores, averages them, predicts class            |
| Saving              | Stores embeddings, logits, predictions, and labels to disk                   |
| Evaluation          | Computes confusion matrix, classification report, ROC AUC, and AP            |
| Visualization       | Plots confusion matrix and UMAP of cell embeddings                           |

---

## What You Are Doing

- You are evaluating the zero-shot cell type annotation ability of the LangCell model.
- You use semantic class descriptions (from `type2text.json`) to enable the model to predict cell types it was not explicitly trained on.
- You analyze the results quantitatively (metrics) and qualitatively (visualization) to assess model performance.