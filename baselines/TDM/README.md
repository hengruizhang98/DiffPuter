# Transformed Distribution Matching (TDM) for Missing Value Imputation, ICML 2023

This repository contains the implementation of the ICML 2023 paper [1].

## Dependencies
The dependencies are specified in the file of ```requirements.txt```. 

## Data

* The input data is an N by D matrix, the missing values of which are indicated by numpy.nan (N and D are the number of data samples and the feature dimensions respectively).

* The completed data can also be provided (for evaluation only, not necessary), i.e., another an N by D matrix with the missing values filled.

* We provide a dataset used in our paper, named ```seeds``` under the folder of ```datasets```, which is preprocessed from [UCI Datasets](https://archive.ics.uci.edu/datasets).

## Run TDM

Simply run ```demo.py```. 

## Acknowledgment
We gratefully thank the authors for the following software and datasets
  * [UCI Datasets](https://archive.ics.uci.edu/datasets) (We used the datasets for evaluation)
  * [MissingDataOT](https://github.com/BorisMuzellec/MissingDataOT) (We used the functions of data preparation, mask generation, and evaluation)
  * [hyperimpute](https://github.com/vanderschaarlab/hyperimpute) (We used the implementations of several baselines)
  * [FrEIA](https://github.com/vislearn/FrEIA) (We used the implementations of the invertible neural networks)

## Reference
  
[1] [He Zhao](https://hezgit.github.io/), [Ke Sun](https://courbure.com/), [Amir Dezfouli](https://adezfouli.github.io/), [Edwin V. Bonilla](https://ebonilla.github.io/), Transformed Distribution Matching for Missing Value Imputation, [ICML 2023](https://proceedings.mlr.press/v202/zhao23h.html).

```
@inproceedings{zhao2023transformed,
  title={Transformed Distribution Matching for Missing Value Imputation},
  author={Zhao, He and Sun, Ke and Dezfouli, Amir and Bonilla, Edwin V},
  booktitle={International Conference on Machine Learning},
  pages={42159--42186},
  year={2023},
  organization={PMLR}
}
```

All the authors of the paper are with [CSIRO's Data61](https://www.csiro.au/en/about/people/business-units/data61).

The code comes without support.
