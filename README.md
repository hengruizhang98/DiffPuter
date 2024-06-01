# DiffPuter

This repository is the official implementation of DiffPuter.

## Installing Dependencies

Python version: 3.12

Create environment

```
conda create -n diffputer python=3.12       
conda activate diffputer
```

Install required packages
```
pip3 install torch torchvision torchaudio
pip install pandas
pip install xlrd
pip install openpyxl
pip install scipy
pip install tqdm
pip install scikit-learn
```

## Preparing Datasets
Download and process datasets from UCI Machine Learning Repository:

```
python download_and_process.py
```

## Train and Evaluate DiffPuter
```
python main.py --dataname [NAME_OF_DATASET]
```
