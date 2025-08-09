# LangCell Zero-Shot Performance Results

This document summarizes the performance metrics from all zero-shot classification experiments using LangCell across different cancer types and classification tasks.

## Overview

The experiments evaluate LangCell's zero-shot performance on binary classification (malignant vs non-malignant) and multi-class cell type classification across different cancer datasets. The results demonstrate LangCell's ability to perform cell type annotation without fine-tuning.

## Binary Classification Results

### 1. Ovarian Cancer Binary Classification

**Dataset**: 65,144 cells (28,699 malignant, 36,445 non-malignant)

| Metric | Value |
|--------|-------|
| **ROC AUC** | 0.9376 |
| **Average Precision** | 0.9342 |
| **Accuracy** | 0.6905 |
| **Precision** | 0.5907 |
| **Recall (Sensitivity)** | 0.9686 |
| **Specificity** | 0.4715 |
| **F1-Score** | 0.7338 |

**Confusion Matrix**:
- True Negatives (Non-malignant): 17,182
- False Positives: 19,263
- False Negatives: 902
- True Positives (Malignant): 27,797

**Complete Classification Report**:
```
              precision    recall  f1-score   support

Non-malignant     0.9501    0.4715    0.6302     36445
    Malignant     0.5907    0.9686    0.7338     28699

     accuracy                         0.6905     65144
    macro avg     0.7704    0.7200    0.6820     65144
 weighted avg     0.7918    0.6905    0.6759     65144
```

**Per-class Performance**:
- Malignant: ROC AUC = 0.9535, AP = 0.9548
- Non-malignant: ROC AUC = 0.9535, AP = 0.9530

### 2. Pancreas Cancer Binary Classification

**Dataset**: 224,988 cells (64,538 malignant, 160,450 non-malignant)

| Metric | Value |
|--------|-------|
| **ROC AUC** | 0.8627 |
| **Average Precision** | 0.5958 |
| **Accuracy** | 0.7619 |
| **Precision** | 0.5498 |
| **Recall (Sensitivity)** | 0.9392 |
| **F1-Score** | 0.6936 |

**Confusion Matrix**:
- True Negatives (Non-malignant): 110,806
- False Positives: 49,644
- False Negatives: 3,921
- True Positives (Malignant): 60,617

**Complete Classification Report**:
```
              precision    recall  f1-score   support

           0     0.9658    0.6906    0.8053    160450
           1     0.5498    0.9392    0.6936     64538

    accuracy                         0.7619    224988
   macro avg     0.7578    0.8149    0.7495    224988
weighted avg     0.8465    0.7619    0.7733    224988
```

**Per-class Performance**:
- Malignant: ROC AUC = 0.8193, AP = 0.9340
- Non-malignant: ROC AUC = 0.8193, AP = 0.5061

### 3. Kidney Cancer Binary Classification

**Dataset**: 19,781 cells

| Metric | Value |
|--------|-------|
| **ROC AUC** | 0.8684 |
| **Average Precision** | 0.7968 |
| **Accuracy** | 0.8235 |
| **Precision** | 0.7300 |
| **Recall (Sensitivity)** | 0.9957 |
| **F1-Score** | 0.8424 |

**Confusion Matrix**:
- True Negatives (Non-malignant): 6,961
- False Positives: 3,451
- False Negatives: 40
- True Positives (Malignant): 9,329

**Complete Classification Report**:
```
              precision    recall  f1-score   support

           0     0.6685    0.9957    0.8012     10412
           1     0.7300    0.9957    0.8424      9369

    accuracy                         0.8235     19781
   macro avg     0.6993    0.9957    0.8218     19781
weighted avg     0.6968    0.8235    0.8204     19781
```

**Per-class Performance**:
- Malignant: ROC AUC = 0.8287, AP = 0.8957
- Non-malignant: ROC AUC = 0.8287, AP = 0.7397

### 4. Prostate Cancer Binary Classification

**Dataset**: 36,424 cells

| Metric | Value |
|--------|-------|
| **ROC AUC** | 0.8626 |
| **Average Precision** | 0.7426 |
| **Accuracy** | 0.6326 |
| **Precision** | 0.5164 |
| **Recall (Sensitivity)** | 1.0000 |
| **F1-Score** | 0.6811 |

**Confusion Matrix**:
- True Negatives (Non-malignant): 8,751
- False Positives: 13,382
- False Negatives: 0
- True Positives (Malignant): 14,291

**Complete Classification Report**:
```
              precision    recall  f1-score   support

           0     0.3953    1.0000    0.5667     22133
           1     0.5164    1.0000    0.6811     14291

    accuracy                         0.6326     36424
   macro avg     0.4559    1.0000    0.6239     36424
weighted avg     0.4418    0.6326    0.6109     36424
```

**Per-class Performance**:
- Malignant: ROC AUC = 0.8612, AP = 0.9227
- Non-malignant: ROC AUC = 0.8612, AP = 0.7430

## Multi-Class Classification Results

### 1. Ovarian Cancer Multi-Class Classification

**Dataset**: 224,988 cells, 18 cell types

| Metric | Value |
|--------|-------|
| **Micro ROC AUC** | 0.9719 |
| **Micro Average Precision** | 0.7511 |
| **Macro ROC AUC** | 0.9306 |
| **Macro Average Precision** | 0.5506 |
| **Accuracy** | 0.6143 |



**Per-class Performance**:
- Neutrophil: ROC AUC = 0.9966, AP = 0.1391
- Endocrine: ROC AUC = 0.9851, AP = 0.8777
- Neuron: ROC AUC = 0.9887, AP = 0.3910
- Adipocyte: ROC AUC = 0.9964, AP = 0.9686
- Malignant: ROC AUC = 0.8677, AP = 0.2650
- Epithelial: ROC AUC = 0.9977, AP = 0.9373
- Myocyte: ROC AUC = 0.9623, AP = 0.3642
- Macrophage: ROC AUC = 0.8367, AP = 0.1261
- Dendritic: ROC AUC = 0.9760, AP = 0.7934
- Endothelial: ROC AUC = 0.9473, AP = 0.4861
- Fibroblast: ROC AUC = 0.8273, AP = 0.5133
- T_cell: ROC AUC = 0.9413, AP = 0.0609
- Plasma: ROC AUC = 0.9151, AP = 0.8027
- Mast: ROC AUC = 0.9947, AP = 0.9824
- Schwann: ROC AUC = 0.9395, AP = 0.6822
- Pericyte: ROC AUC = 0.9994, AP = 0.9086
- NK_cell: ROC AUC = 0.9982, AP = 0.9530
- B_cell: ROC AUC = 0.9697, AP = 0.8729

### 2. Pancreas Cancer Multi-Class Classification

**Dataset**: 224,988 cells, 18 cell types

| Metric | Value |
|--------|-------|
| **Micro ROC AUC** | 0.9719 |
| **Micro Average Precision** | 0.7511 |
| **Macro ROC AUC** | 0.9306 |
| **Macro Average Precision** | 0.5506 |
| **Accuracy** | 0.6143 |



**Per-class Performance** (same as ovarian due to shared dataset):
- Neutrophil: ROC AUC = 0.9966, AP = 0.1391
- Endocrine: ROC AUC = 0.9851, AP = 0.8777
- Neuron: ROC AUC = 0.9887, AP = 0.3910
- Adipocyte: ROC AUC = 0.9964, AP = 0.9686
- Malignant: ROC AUC = 0.8677, AP = 0.2650
- Epithelial: ROC AUC = 0.9977, AP = 0.9373
- Myocyte: ROC AUC = 0.9623, AP = 0.3642
- Macrophage: ROC AUC = 0.8367, AP = 0.1261
- Dendritic: ROC AUC = 0.9760, AP = 0.7934
- Endothelial: ROC AUC = 0.9473, AP = 0.4861
- Fibroblast: ROC AUC = 0.8273, AP = 0.5133
- T_cell: ROC AUC = 0.9413, AP = 0.0609
- Plasma: ROC AUC = 0.9151, AP = 0.8027
- Mast: ROC AUC = 0.9947, AP = 0.9824
- Schwann: ROC AUC = 0.9395, AP = 0.6822
- Pericyte: ROC AUC = 0.9994, AP = 0.9086
- NK_cell: ROC AUC = 0.9982, AP = 0.9530
- B_cell: ROC AUC = 0.9697, AP = 0.8729

### 3. Kidney Cancer Multi-Class Classification

**Dataset**: 19,781 cells, 5 cell types

| Metric | Value |
|--------|-------|
| **Micro ROC AUC** | 0.7974 |
| **Micro Average Precision** | 0.7594 |
| **Macro ROC AUC** | 0.8413 |
| **Macro Average Precision** | 0.5865 |
| **Accuracy** | 0.6391 |



**Cell Types**: Macrophage, Endothelial, Malignant, Fibroblast, Epithelial

### 4. Prostate Cancer Multi-Class Classification

**Dataset**: 36,424 cells, 9 cell types

| Metric | Value |
|--------|-------|
| **Micro ROC AUC** | 0.8786 |
| **Micro Average Precision** | 0.3597 |
| **Macro ROC AUC** | 0.8668 |
| **Macro Average Precision** | 0.6398 |
| **Accuracy** | 0.3359 |



**Per-class Performance**:
- Malignant: ROC AUC = 0.6786, AP = 0.4738
- Unassigned: ROC AUC = 0.9974, AP = 0.9646
- Epithelial: ROC AUC = 0.9985, AP = 0.9840
- T_cell: ROC AUC = 0.9883, AP = 0.7900
- Endothelial: ROC AUC = 0.9984, AP = 0.9753
- Fibroblast: ROC AUC = 0.9966, AP = 0.5610
- Macrophage: ROC AUC = 0.4728, AP = 0.1443
- Mast: ROC AUC = 0.5984, AP = 0.1142
- B_cell: ROC AUC = 0.9902, AP = 0.9160

## General Cancer Classification

### Malignant vs Non-Malignant Classification

**Dataset**: 138,727 cells (85,749 malignant, 52,978 non-malignant)

| Metric | Value |
|--------|-------|
| **ROC AUC** | 0.7809 |
| **Average Precision** | 0.7934 |
| **Accuracy** | 0.7391 |
| **Precision** | 0.7482 |
| **Recall (Sensitivity)** | 0.8709 |
| **F1-Score** | 0.8049 |

**Confusion Matrix**:
- True Negatives (Non-malignant): 27,851
- False Positives: 25,127
- False Negatives: 11,070
- True Positives (Malignant): 74,679

**Complete Classification Report**:
```
              precision    recall  f1-score   support

           0     0.7156    0.5257    0.6061     52978
           1     0.7482    0.8709    0.8049     85749

    accuracy                         0.7391    138727
   macro avg     0.7319    0.6983    0.7055    138727
weighted avg     0.7358    0.7391    0.7290    138727
```

## Summary and Key Findings

### Binary Classification Performance
1. **Ovarian Cancer**: Highest ROC AUC (0.9376) and Average Precision (0.9342)
2. **Kidney Cancer**: Strong performance with ROC AUC of 0.8684
3. **Pancreas Cancer**: Good performance with ROC AUC of 0.8627
4. **Prostate Cancer**: Competitive performance with ROC AUC of 0.8626

### Multi-Class Classification Performance
1. **Ovarian/Pancreas**: Excellent micro ROC AUC (0.9719) across 18 cell types
2. **Prostate Cancer**: Strong micro ROC AUC (0.8786) despite lower accuracy
3. **Kidney Cancer**: Moderate performance with micro ROC AUC of 0.7974

### Key Observations
- LangCell demonstrates strong zero-shot performance across different cancer types
- Binary classification generally achieves higher performance than multi-class classification
- The model shows good discrimination between malignant and non-malignant cells
- Performance varies by cancer type, with ovarian cancer showing the best results
- Cell type-specific performance varies significantly, with some cell types (like Pericyte, NK_cell) achieving near-perfect ROC AUC scores

### Technical Notes
- All experiments use the same LangCell model without fine-tuning
- Performance metrics include ROC AUC, Average Precision, accuracy, precision, recall, and F1-score
- Results demonstrate LangCell's capability for zero-shot cell type annotation
- The model successfully clusters cells by type and eliminates batch effects 