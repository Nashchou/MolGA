
import argparse
parser = argparse.ArgumentParser("MolGa")
parser.add_argument('--dataset', type=str, default="qm9", help='data')
parser.add_argument('--downstreamdataset', type=str, default="md17", help='Dataset name')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seeds', nargs='+', type=int, default=[38, 39, 40, 41, 42], help='List of random seeds')
parser.add_argument('--label_num', type=int, default=1, help='') 
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--val_name', type=str, default='noval_graphcl.pkl', help='save val')
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining') 
parser.add_argument('--pretrainmodel', type=str, default='joaov2', help='pretrainmodel like joaov2 or graphcl')   
parser.add_argument('--test3Dinfo', type=int, default=1, help='')  
parser.add_argument('--hid_units', type=int, default=64, help='')  
parser.add_argument('--targetlist', type=list, default=['homo','lumo','gap'], help='')   
parser.add_argument('--lr_decay_factor', type=float, default= 0.1, help='')   
parser.add_argument('--lr_decay_step_size', type=int, default=150, help='')   
args = parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 


from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random

from torch_cluster import radius_graph
from dig.threedgraph.dataset import QM93D,MD17
from dig.threedgraph.evaluation import ThreeDEvaluator
from torch_geometric.datasets import QM9, TUDataset
from downstream import continueweight
from models import LogReg
from preprompt import PrePrompt,pca_compression
import preprompt
from utils import process
import pdb
import aug


import csv
from tqdm import tqdm

import torch.nn.functional as F


print('-' * 100)
print(args)
print('-' * 100)

aug_type = args.aug_type
drop_percent = args.drop_percent

import torch
import torch.nn as nn
# torch.cuda.set_device(int(local_rank))

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB, WikipediaNetwork ,QM9
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import GCL.losses as L
import GCL.augmentors as A
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
batch_size = 21000
nb_epochs = 20000
patience = 50
lr = 0.00001
l2_coef = 0.0
drop_prob = 0.0
hid_units = args.hid_units
sparse = True
useMLP = False
class_num = 2
LP = False
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
nonlinearity = 'prelu'  # special name to separate parameters
dataset = args.dataset
device = torch.device("cuda")
best = 1e9
firstbest = 0

threeD_dataset=['qm9']

def load_dataset(name):
    if name == 'qm9':
        dataset = QM93D(root='dataset/')
        target = 'mu'
        dataset.data.y = dataset.data[target]
        split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=21000, valid_size=1000, seed=42)
        train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

    else:
        raise ValueError(f"Unknown dataset name: {name}")
    return train_dataset


cnt_wait = 0
num_layers_num = 3
pretrain_datasets = ['qm9']
print(pretrain_datasets)
num_pretrain_dataset_num = len(pretrain_datasets)
pretrain_loaders = DataLoader(load_dataset(dataset), batch_size, shuffle=True)

features = []
adjs = []
aug_adjs = []
aug_features = []
lbls = []
aug1 = A.Identity()
aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                                A.NodeDropping(pn=0.1),
                                A.FeatureMasking(pf=0.1),
                                A.EdgeRemoving(pe=0.1)], 1)
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
pretrain_dataset_str = ''
for str in pretrain_datasets:
    pretrain_dataset_str += '_'+str
save_dir = os.path.join(current_dir, 'checkpoints')
os.makedirs(save_dir, exist_ok=True)
save_name = os.path.join(save_dir, f'model_node_{args.pretrainmodel}_{pretrain_dataset_str}_{hid_units}.pkl')
model = PrePrompt(hid_units, aug1,aug2, num_layers_num, 0.1,args.pretrainmodel)
if torch.cuda.is_available():
        print('Using CUDA')
model = model.cuda()
optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_coef)
IF_PRETRAIN = 0
if IF_PRETRAIN:
    for step, data in enumerate(pretrain_loaders):

        loss=0
        data = data.cuda()
        model = model.cuda()
        batch = data.batch
        z = data.z.long().cuda()
        pos = data.pos
        num_nodes = z.size(0)
        edge_index = radius_graph(pos, r=8.0, batch=batch).cuda()
        totalloss = 0
        for epoch in tqdm(range(nb_epochs)):
            model.train()
            optimiser.zero_grad()
            loss = model(z,edge_index,batch)
            loss.backward()
            optimiser.step()

        if totalloss < best:
            firstbest = 1
            best = totalloss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping!')
            break


print('#'*50)
print('PreTrain datasets are ', pretrain_datasets)
print('Downastream dataset is ', args.dataset)




print(f'loading model from {save_name}')

if args.downstreamdataset == 'qm9':
    dataset = QM93D(root='dataset/')
    target_list = ['homo','lumo','gap']
    y_list = []
    for target in target_list:
        y = dataset.data[target]
        y_list.append(y)
    torch_y = torch.stack(y_list, dim=0)
    dataset.data.y = torch_y.t()
elif args.downstreamdataset == 'md17':
    dataset = MD17(root='dataset/',name='aspirin')

loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()
device = torch.device("cuda")
# Train and evaluate

lr_list  = [0.001]
lr_decay_factor_list = [1]
all_rmse = []
for seed in args.seeds:
    for lr in lr_list:
        random.seed(seed)
        np.random.seed(seed)
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print(f'loading model from {save_name}')

        if args.downstreamdataset == 'qm9':
            split_idx = process.get_idx_split(len(dataset.data.y), train_size=100 , valid_size=100, seed=seed,start_idx=21000)
        elif args.downstreamdataset == 'md17':
            split_idx = process.get_idx_split(len(dataset.data.y), train_size=100 , valid_size=100, seed=seed,start_idx=0,test_size=10000)

        train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
        train_loader = DataLoader(train_dataset, 100, shuffle=True)
        valid_loader = DataLoader(valid_dataset, 1000, shuffle=False)
        test_loader = DataLoader(test_dataset, 1000, shuffle=False)
        down = continueweight(hid_units, aug1,aug2, num_layers_num, 0.1,save_name,test3Dinfo=args.test3Dinfo,pretrainmodel=args.pretrainmodel,label_num=args.label_num)
        bestmae = down.run(device = device, train_dataset = train_dataset, valid_dataset = valid_dataset,test_dataset = test_dataset,loss_func = loss_func,evaluation = evaluation,
                epochs=300, batch_size=100, vt_batch_size=1000, lr=lr, lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size)
        all_rmse.append(bestmae)
print('-' * 100)
print('-' * 100)
rmse_array = np.array(all_rmse)
avg = rmse_array.mean()
std = rmse_array.std()
row = [args.downstreamdataset,lr,avg,std]
out = open("data/threeDgcn.csv".format(args.dataset.lower()), "a", newline="")
csv_writer = csv.writer(out, dialect="excel")
csv_writer.writerow(row)