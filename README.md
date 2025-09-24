# MolGa

# Introduction

We provide the code (in pytorch) and datasets for our paper 
**"MolGa: Molecular Graph Adaptation with Pre-trained 2D Graph Encoder"**

## Description

The repository is organised as follows:

- **MolGa_classification/**: implements pre-training and downstream classification tasks .
- **MolGa_regression/**: implements pre-training and downstream molecular property tasks.

## Package Dependencies

* python 3.8
* torch 1.11.0
* cuda 12.4
* torch_geometric 2.1.0

## Running experiments

### Regression for MoleculeNet dataset

Default dataset is lipophilicity. You need to change the corresponding parameters in *execute.py* to train and evaluate on other datasets. 

Pretrain and downstream test are in the same file, You can find a pretrain flag to control whether pretrain or not.

```python
IF_PRETRAIN = 0
```

In the folder "MolGa_regression/MolGA_reg", run:

- python execute.py

### Regression for QM9 and MD17

Default dataset is MD17. You need to change the corresponding parameters in *execute.py* to train and evaluate on other datasets. 

Pretrain and downstream test are in the same file, You can find a pretrain flag to control whether pretrain or not.

```python
IF_PRETRAIN = 0
```

In the folder "MolGa_regression/MolGa_reg_Moleculenet", run:

- python execute.py


### Classification

Default dataset is BBBP. You need to change the corresponding parameters in *execute.py* to train and evaluate on other datasets.

Pretrain and downstream test are in the same file, You can find a pretrain flag to control whether pretrain or not.

```python
IF_PRETRAIN = 0
```

In the folder "MolGa_classification", run:

- python execute.py