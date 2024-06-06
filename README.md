# DiffPuter

<p align="center">
  <!-- <a href="https://github.com/hengruizhang98/tabsyn/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/hengruizhang98/tabsyn">
  </a> -->
  <a href="https://arxiv.org/abs/2405.20690">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2405.20690-blue">
  </a>
</p>

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
