# MolGa

# Introduction

We provide the code (in pytorch) and datasets for our paper 
**"MolGA: Molecular Graph Adaptation with Pre-trained 2D Graph Encoder"**

## Description

The repository is organised as follows:

- **MolGA_classification/**: implements pre-training and downstream molecular classification.
- **MolGA_regression/**: implements pre-training and downstream molecular property prediction.
- **data.zip**: contains *Lipophilicity* and *BBBP*  in the quick-start examples. Due to storage limitations on Anonymous GitHub, other datasets are not included in this release. We will make the remaining datasets publicly available upon publication.

## Datasets

We use the following datasets in our experiments (direct download links):

### MoleculeNet (DeepChem)
- BBBP: http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/BBBP.csv  
- SIDER: http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/sider.csv.gz  
- ClinTox: http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/clintox.csv.gz  
- MUV: https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/muv.csv.gz  
- BACE: http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/bace.csv  
- ESOL: http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/delaney-processed.csv  
- Lipophilicity: http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/Lipophilicity.csv  
- FreeSolv: https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/FreeSolv.zip  

### Quantum chemistry / dynamics
- QM8: http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb8.tar.gz  
- QM9: http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz  
- MD17-aspirin: http://quantum-machine.org/gdml/data/npz/md17_aspirin.npz

## Package Dependencies

* python 3.8
* torch 1.11.0
* cuda 12.4
* torch_geometric 2.1.0

## Running experiments

### Molecular property prediction for MoleculeNet dataset

Default dataset is *lipophilicity*. You need to change the corresponding parameters in *execute.py* to train and evaluate on other datasets. 

Pretrain and downstream test are in the same file, You can find a pretrain flag to control whether pretrain or not.

```python
IF_PRETRAIN = 0
```

In the folder "MolGA_regression/MolGA_reg", run:

- python execute.py

### Molecular property prediction for QM9 and MD17

Default dataset is *MD17*. You need to change the corresponding parameters in *execute.py* to train and evaluate on other datasets. 

Pretrain and downstream test are in the same file, You can find a pretrain flag to control whether pretrain or not.

```python
IF_PRETRAIN = 0
```

In the folder "MolGA_regression/MolGa_reg_Moleculenet", run:

- python execute.py


### Molecular classification

Default dataset is *BBBP*. You need to change the corresponding parameters in *execute.py* to train and evaluate on other datasets.

Pretrain and downstream test are in the same file, You can find a pretrain flag to control whether pretrain or not.

```python
IF_PRETRAIN = 0
```

In the folder "MolGA_classification", run:

- python execute.py


