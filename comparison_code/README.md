# Commands to run comparison codes

## ABMIL - MCAT - Porpoise
We modified code from [PORPOISE](https://github.com/mahmoodlab/PORPOISE) to run these experiments.

Extract WSI reps using UNI into directories named lung/kidney/breast. Use the slide names as the names for the .pt files.

```bash
REPS_DIR
├── lung
│   ├── <TCGA_slide_name.pt>
│   ├── <TCGA_slide_name.pt>
│   ├── ...
├── kidney
│   ├── <TCGA_slide_name.pt>
│   ├── <TCGA_slide_name.pt>
│   ├── ...
├── breast
│   ├── <TCGA_slide_name.pt>
│   ├── <TCGA_slide_name.pt>
├── ...
```

## ABMIL
### Lung
```
python main.py --which_splits 5foldcv --split_dir tcga_lung_ --model_type amil  --study_dir lung --mode path --data_root_dir <REPS_DIR>
```
### Kidney
```
python main.py --which_splits 5foldcv --split_dir tcga_kidney_ --model_type amil  --study_dir kidney --mode path --data_root_dir <REPS_DIR>
```
### BRCA
```
python main.py --which_splits 5foldcv --split_dir tcga_breast_ --model_type amil  --study_dir breast --mode path --data_root_dir <REPS_DIR>
```


## MCAT
### Lung
```
python main.py --which_splits 5foldcv --split_dir tcga_lung_ --mode coattn --model_type mcat --fusion concat --study_dir lung --apply_sig --data_root_dir <REPS_DIR>
```
### Kidney
```
python main.py --which_splits 5foldcv --split_dir tcga_kidney_ --mode coattn --model_type mcat --fusion concat --study_dir kidney --apply_sig --data_root_dir <REPS_DIR>
```
### BRCA
```
python main.py --which_splits 5foldcv --split_dir tcga_breast_ --mode coattn --model_type mcat --fusion concat --study_dir breast --apply_sig --data_root_dir <REPS_DIR>
```


## PORPOISE
### Lung
```
python main.py --which_splits 5foldcv --split_dir tcga_lung_ --mode pathomic --reg_type pathomic --model_type porpoise_mmf --fusion concat --study_dir lung --data_root_dir <REPS_DIR>
```
### Kidney
```
python main.py --which_splits 5foldcv --split_dir tcga_kidney_ --mode pathomic --reg_type pathomic --model_type porpoise_mmf --fusion concat --study_dir kidney --data_root_dir <REPS_DIR>
```
### BRCA
```
python main.py --which_splits 5foldcv --split_dir tcga_breast_ --mode pathomic --reg_type pathomic --model_type porpoise_mmf --fusion concat --study_dir breast --data_root_dir <REPS_DIR>
```

## SNN
We modified code from [SNN](https://github.com/bioinf-jku/SNNs) to run this experiments.

```bash
project_root/
├── main.py             
├── requirements.txt    
├── README.md           
└── data/               # data example
    ├── 5foldsplits/
    ├── rna1.csv
    └── rna2.csv
```
```
python main.py \
  --data_root ./data/splits \
  --rna_path ./data/rna_data1.csv \
  --lusc_path ./data/rna_data2.csv \
  --batch_size 128 \
  --max_epochs 100 \
  --dropout 0.2
```

## GSCNN
Please follow the instructions from [SCNN](https://github.com/PathologyDataScience/SCNN) to run the model on Docker.

## BP
We modified code from [PathomicFusion](https://github.com/mahmoodlab/PathomicFusion) to run this experiments.
```
python main.py --which_splits 5foldcv --split_dir tcga_breast_ --mode  --reg_type  --model_type porpoise_mmf --fusion concat --study_dir breast --data_root_dir <REPS_DIR>
```
