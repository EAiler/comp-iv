# A causal view on compositional data

This is the code for reconstructing the experiments in [A Causal view on Compositional Data](https://arxiv.org/abs/2106.11234), E. Ailer, C.L. Müller and N. Kilbertus.

## Setup

First, clone this repository.

```sh
git clone git@github.com:eailer/comp-iv.git
```

To run the code, please first create a new Python3 environment (Python version >= 3.7 should work).
Then install the required packages into your newly created environment via

```sh
python -m pip install -r requirements.txt
```

## Experiments
The experiments are explained step by step in individual jupyter notebooks.

```sh
cd notebooks
```

The experiments are separated in three different notebooks. 

### One Dimensional IV Estimation
The notebook `1_RealData_Diversity`is centered around the motivational experiments. For input, it takes an instrumental variable setup with parameters `Z, X, Y`, whereas `X` is a diversity estimate. 


### Compositional IV Estimation

The other notebooks `2_MicrobiomeAnalysis_SimulatedData` and `2_MicrobiomeAnalysis_RealData` include the higher dimensional setup with compositional `X`. They include the exact parameter settings used in the paper. Moreover, they provide the possibility to exchange parameters or use own data to apply the proposed methods.


