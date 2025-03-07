# Towards Robust Multimodal Representation: A Unified Approach with Adaptive Experts and Alignment

![](/data/Method.png)

## Preprocessing

Tissue segmentation, patching, and patch embedding extraction can be done using the [CLAM toolbox](https://github.com/mahmoodlab/CLAM) and should be saved in the `feats_pt_wsi` directory.

Preprocessing the corresponding gene expression profile can be done using the [TANGLE](https://github.com/mahmoodlab/TANGLE) and should be saved as `rna_all.csv`.

Preprocessing the corresponding clinical reports is done by using fine-tuned sentence transformer from ModernBERT to encode textual data at sentence level and entire report level. The entire clinical report ebbeddings per WSI is saved in `global_text`

The GMM model is fit over patch embeddings per WSI and sentence embeddings per report using [PANTHER](https://github.com/mahmoodlab/PANTHER) and should be saved as `proto_save_wsi` and `proto_save_text`.

```bash
MoSARe
├── ...
├── data
│   ├── BRCA
│   │   ├── feats_pt_wsi
│   │   ├── rna_all.csv
│   │   ├── proto_save_wsi
│   │   ├── proto_save_text
│   │   ├── global_text
│   ├── NSCLC
│   │   ├── feats_pt_wsi
│   │   ├── rna_all.csv
│   │   ├── proto_save_wsi
│   │   ├── proto_save_text
│   │   ├── global_text
│   ├── RCC
│   │   ├── feats_pt_wsi
│   │   ├── rna_all.csv
├── ...
```



## Train the model

To train the model please run the below scripts:


train_tangle.py --fold 0 --mask_percentage 0.1

## Text pre-processing
Download the TCGA Reports data from here: [TCGA Reports](https://data.mendeley.com/datasets/hyg5xkznpx/1). Refer to [this paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10935496/) for details about cleaning the reports dataset.

### Fine-tune sentence transformer
GitHub Repo: https://github.com/AnswerDotAI/ModernBERT

Specificially used [train_st.py](https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/train_st.py) to fine tune a sentence transformer on Nvidia A100 GPU. We used this fine-tuned model to encode sentences and documents. The 8192 context length is useful for encoding large documents.

### Regex matching to filter cancer subtype keywords
Use this notebook: [text_preprocessing.ipynb](./text_preprocessing.ipynb) to filter TCGA reports data.

## Comparison code
Find the comparison code in [comparison_code](./comparison_code). This directory contains the code we used for generating 5-fold CV results for [MCAT](https://github.com/mahmoodlab/MCAT), [ABMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL) and [PORPOISE](https://github.com/mahmoodlab/PORPOISE). We utilize the preprocessed datasets for genomic data helpfully provided to us by their respective authors (specifically for MCAT and PORPOISE) and modify the train/val splits based on what we used for MoSARe.

# Acknowledgments

This project is based on the following open-source projects. We thank their authors for making the source code publicly available.

https://github.com/mahmoodlab/TANGLE

https://github.com/MrPetrichor/MECOM

https://github.com/NazaninMn/GenGMM


