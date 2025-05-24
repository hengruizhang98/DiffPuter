# DiffPuter
Official Implementation of DiffPuter: Empowering Diffusion Models for Missing Data Imputation, at ICLR 2025


## Installing Dependencies
To run experiments of all the baselines, we have to create three different environments.


```
conda create -n diffputer python=3.12       
conda activate diffputer
pip install -r requirements/diffputer.txt
```

## Preparing Datasets
Run the following command to prepare all the datasets, splits and masks.

```
python download_and_process.py
```

## Reproducing the results


To run DiffPuter on a single dataset under a single mask, use the following command
```
conda activate diffputer
python main.py --dataname [NAME_OF_DATASET] --split_idx [MASK_IDX] 
```

[NAME_OF_DATASET]: california magic bean gesture letter adult default shoppers news
[MASK_IDX]: 0 1 2 3 4 5 6 7 8 9


