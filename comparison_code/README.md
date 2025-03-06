# Commands to run comparison codes

Extract WSI reps using UNI into directories named lung/kidney/breast. Use the slide names as the names for the .pt files.

## ABMIL
### Lung
```python main.py --which_splits 5foldcv --split_dir tcga_lung_ --model_type amil  --study_dir lung --mode path --data_root_dir <REPS_DIR>```
### Kidney
```python main.py --which_splits 5foldcv --split_dir tcga_kidney_ --model_type amil  --study_dir kidney --mode path --data_root_dir <REPS_DIR>```
### BRCA
```python main.py --which_splits 5foldcv --split_dir tcga_breast_ --model_type amil  --study_dir breast --mode path --data_root_dir <REPS_DIR>```


## MCAT
### Lung
```python main.py --which_splits 5foldcv --split_dir tcga_lung_ --mode coattn --model_type mcat --fusion concat --study_dir lung --apply_sig --data_root_dir <REPS_DIR>```
### Kidney
```python main.py --which_splits 5foldcv --split_dir tcga_kidney_ --mode coattn --model_type mcat --fusion concat --study_dir kidney --apply_sig --data_root_dir <REPS_DIR>```
### BRCA
```python main.py --which_splits 5foldcv --split_dir tcga_breast_ --mode coattn --model_type mcat --fusion concat --study_dir breast --apply_sig --data_root_dir <REPS_DIR>```


## PORPOISE
### Lung
```python main.py --which_splits 5foldcv --split_dir tcga_lung_ --mode pathomic --reg_type pathomic --model_type porpoise_mmf --fusion concat --study_dir lung --data_root_dir <REPS_DIR>```
### Kidney
```python main.py --which_splits 5foldcv --split_dir tcga_kidney_ --mode pathomic --reg_type pathomic --model_type porpoise_mmf --fusion concat --study_dir kidney --data_root_dir <REPS_DIR>```
### BRCA
```python main.py --which_splits 5foldcv --split_dir tcga_breast_ --mode pathomic --reg_type pathomic --model_type porpoise_mmf --fusion concat --study_dir breast --data_root_dir <REPS_DIR>```