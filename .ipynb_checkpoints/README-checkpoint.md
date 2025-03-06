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


# Acknowledgments

This project is based on the following open-source projects. We thank their authors for making the source code publicly available.

https://github.com/mahmoodlab/TANGLE

https://github.com/MrPetrichor/MECOM

https://github.com/NazaninMn/GenGMM


