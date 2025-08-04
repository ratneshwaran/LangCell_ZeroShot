# LangCell Zero-Shot Implementation

This repository contains an implementation and adaptation of the **LangCell** framework for zero-shot cell type annotation. The code and methodology are based on the original LangCell paper and repository.

## Original LangCell Framework

**LangCell: Language-Cell Pre-training for Cell Identity Understanding**

Cell identity encompasses various semantic aspects of a cell, including cell type, pathway information, disease information, and more, which are essential for biologists to gain insights into its biological characteristics. Understanding cell identity from the transcriptomic data, such as annotating cell types, have become an important task in bioinformatics. 

As these semantic aspects are determined by human experts, it is impossible for AI models to effectively carry out cell identity understanding tasks without the supervision signals provided by single-cell and label pairs. 

The single-cell pre-trained language models (PLMs) currently used for this task are trained only on a single modality, transcriptomics data, lack an understanding of cell identity knowledge. As a result, they have to be fine-tuned for downstream tasks and struggle when lacking labeled data with the desired semantic labels.

To address this issue, the original LangCell authors propose an innovative solution by constructing a unified representation of single-cell data and natural language during the pre-training phase, allowing the model to directly incorporate insights related to cell identity.

More specifically, they introduce **LangCell**, the first **Lang**uage-**Cell** pre-training framework. 
LangCell utilizes texts enriched with cell identity information to gain a profound comprehension of cross-modal knowledge.
Results from experiments conducted on different benchmarks show that LangCell is the only single-cell PLM that can work effectively in zero-shot cell identity understanding scenarios, and also significantly outperforms existing models in few-shot and fine-tuning cell identity understanding scenarios.

More information can be found at [https://arxiv.org/abs/2405.06708](https://arxiv.org/abs/2405.06708).

LangCell will soon be added to the OpenBioMed toolkit: [https://github.com/PharMolix/OpenBioMed](https://github.com/PharMolix/OpenBioMed).

![LangCell](assets/image.png)

## This Implementation

This repository provides:
- **Zero-shot cell type annotation** implementations for multiple cancer datasets
- **Binary classification** (malignant vs normal) capabilities
- **Data preprocessing** scripts for various single-cell datasets
- **Evaluation and visualization** tools for model performance

### Key Features:
- Multi-dataset support (Ovarian, Prostate, Kidney, Pancreas cancer)
- Binary and multi-class classification approaches
- Comprehensive evaluation metrics and visualizations
- Preprocessing pipelines for different data formats

## Installation

[![python >3.9.18](https://img.shields.io/badge/python-3.9.18-brightgreen)](https://www.python.org/) 
```
pip install -r requirements.txt
```

## Model Checkpoints

The model's checkpoint is divided into five modules: text_bert, cell_bert, text_proj, cell_proj, and ctm_head. Users can select and load the necessary modules according to the downstream task requirements. Among them, cell_bert is the standard Huggingface BertModel; text_bert is a multifunctional encoder provided in utils.py; cell_proj and text_proj are linear layers that map the model outputs corresponding to the [CLS] position in cells and text to a unified feature space; and ctm_head is a linear layer that maps the output of text_bert to matching scores when performing Cell-Text Matching.

[Download checkpoint](https://drive.google.com/drive/folders/1cuhVG9v0YoAnjW-t_WMpQQguajumCBTp)

## Usage

### Data Preprocessing
Similar to the example in `data_preprocess/preprocess.py`, you can use `scanpy` to read any single-cell data and process it into a format accepted by the model. The processing method is similar to `Geneformer`. For more detailed instructions, please refer to [Geneformer's tokenizing scRNAseq data example](https://huggingface.co/ctheodoris/Geneformer/blob/main/examples/tokenizing_scRNAseq_data.ipynb).

### Zero-Shot Cell Type Annotation
We strongly recommend that users unfamiliar with LangCell start by experiencing this core task to quickly understand the features and usage of LangCell. We have prepared a [demo dataset](https://drive.google.com/drive/folders/1cuhVG9v0YoAnjW-t_WMpQQguajumCBTp?usp=sharing) for this task; you just need to download the dataset and run `LangCell-annotation-zeroshot/zero-shot.ipynb`.

### Binary Classification (Malignant vs Normal)
For binary classification tasks, use the modified notebooks in `LangCell-annotation-zeroshot/` that implement malignant vs normal cell classification.

### Textual Descriptions
We have uploaded the OBO Foundry file "obo.json" [here](https://drive.google.com/drive/folders/1cuhVG9v0YoAnjW-t_WMpQQguajumCBTp), which contains textual descriptions of common cell identities. You can use these as examples to write textual descriptions for new cell types.

## Citation

**Original LangCell Paper:**
```
@misc{zhao2024langcell,
      title={LangCell: Language-Cell Pre-training for Cell Identity Understanding}, 
      author={Suyuan Zhao and Jiahuan Zhang and Yizhen Luo and Yushuai Wu and Zaiqing Nie},
      year={2024},
      eprint={2405.06708},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
```

**If you use this implementation in your research, please cite both the original LangCell paper and this repository.**

## Acknowledgments

- **Original LangCell Authors**: Suyuan Zhao, Jiahuan Zhang, Yizhen Luo, Yushuai Wu, Zaiqing Nie
- **Original Repository**: The code and methodology in this repository are based on the original LangCell implementation
- **Geneformer**: The data preprocessing approach is inspired by the Geneformer framework

## License

This project is licensed under the same license as the original LangCell repository. Please refer to the LICENSE file for details.
