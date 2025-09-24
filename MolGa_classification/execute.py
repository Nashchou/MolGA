
import argparse
parser = argparse.ArgumentParser("MolGa")
parser.add_argument('--pretraindataset', type=str, default='qm9' ,help='data')
parser.add_argument('--dataset', type=str, default="bbbp", help='data')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--val_name', type=str, default='noval_graphcl.pkl', help='save val')
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining') 
parser.add_argument('--pretrainmodel', type=str, default='joaov2', help='pretrainmodel like joaov2 or graphcl')   
parser.add_argument('--downstreamtype', type=str, default='prompt', help='prompt or finetune') 
parser.add_argument('--mode', type=str, default='G2G', help='L2L or G2G')   
parser.add_argument('--backbone', type=str, default='gcn', help='gcn or gat')   
parser.add_argument('--backbonelayernum', type=int, default=3, help='')
parser.add_argument('--test3Dinfo', type=int, default=1, help='')  
parser.add_argument('--hid_units', type=int, default=64, help='')  
parser.add_argument('--targetlist', type=list, default=['homo','lumo','gap'], help='')   
parser.add_argument('--lr_decay_factor', type=float, default= 0.1, help='')   
parser.add_argument('--lr_decay_step_size', type=int, default=100, help='')   
parser.add_argument('--kshot', type=int, default=5, help='')   
parser.add_argument('--useembeddingas2Dinfo', type=int, default=1, help='')   
parser.add_argument('--condition_emb', type=int, default=32, help='')  
args = parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 


import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random

from torch_cluster import radius_graph
from dig.threedgraph.evaluation import ThreeDEvaluator
from downstream import continueweight
from preprompt import PrePrompt,pca_compression
from utils import process
from genfewshot import get_fewshot_indices
from torch.utils.data import Subset

import csv
from tqdm import tqdm

import torch.nn.functional as F


print('-' * 100)
print(args)
print('-' * 100)

aug_type = args.aug_type
drop_percent = args.drop_percent

seed = args.seed
random.seed(seed)
np.random.seed(seed)

import torch
import torch.nn as nn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.set_device(int(local_rank))

from torch_geometric.data import DataLoader
import GCL.augmentors as A


import matplotlib.pyplot as plt

batch_size = 25000
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


from molecule_datasets import MoleculeDataset

cnt_wait = 0
num_layers_num = args.backbonelayernum
pretrain_datasets = args.pretraindataset

dataset_folder = 'data/'
dataset = MoleculeDataset(dataset_folder+pretrain_datasets, dataset=pretrain_datasets)


print(pretrain_datasets)
num_pretrain_dataset_num = len(pretrain_datasets)
pretrain_loaders = DataLoader(dataset, batch_size, shuffle=True)

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
save_name = os.path.join(save_dir, f'model_node_{args.pretrainmodel}_{pretrain_dataset_str}_{hid_units}_{args.backbone}_{args.backbonelayernum}.pkl')


model = PrePrompt(hid_units, aug1,aug2, num_layers_num, 0.1,args.pretrainmodel,args.mode)
if torch.cuda.is_available():
        print('Using CUDA')
model = model.cuda()
optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_coef)
IF_PRETRAIN = 0
if IF_PRETRAIN:
    for step, data in enumerate(pretrain_loaders):
        pretrain_losslist = []
        loss=0
        data = data.cuda()
        model = model.cuda()
        batch = data.batch
        z = data.x.long().cuda()
        # pos = data.pos
        num_nodes = z.size(0)
        edge_index = data.edge_index.cuda()
        totalloss = 0
        for epoch in tqdm(range(nb_epochs)):
            model.train()
            optimiser.zero_grad()
            loss = model(z,edge_index,batch)
            pretrain_losslist.append(loss.item())
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
        epochs_list = list(range(1, nb_epochs + 1))
        plt.plot(epochs_list, pretrain_losslist, label='Pretrain Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Pretrain Training Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig('pretrain_loss.png', dpi=300) 
        plt.close() 
        break

print('#'*50)
print('PreTrain datasets are ', pretrain_datasets)
print('Downastream dataset is ', args.dataset)




print(f'loading model from {save_name}')
dataset_folder = 'data/'
dataset = MoleculeDataset(dataset_folder + args.dataset, dataset=args.dataset)
smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()

num_graph = dataset.data.id.shape[0]
print('num_graph', num_graph)

graph_view_y = dataset.data.y.view(num_graph,-1)
labelnum = 1
print('labelnum', labelnum)


if args.downstreamtype == 'prompt':
    loss_func = nn.CrossEntropyLoss()
elif args.downstreamtype == 'finetune':
    loss_func = nn.BCEWithLogitsLoss(reduction='none')
evaluation = ThreeDEvaluator()
# Train and evaluate
lr_list = [0.001]

# 先做划分

train_idx_path = os.path.join(save_dir, f'train_idx_{args.dataset.lower()}.pt')
valid_idx_path = os.path.join(save_dir, f'valid_idx_{args.dataset.lower()}.pt')
test_idx_path  = os.path.join(save_dir, f'test_idx_{args.dataset.lower()}.pt')


seed_try = seed
while True:
    if not os.path.exists(train_idx_path):
        split_idx = process.get_idx_split(len(dataset), smiles_list,train_size=int(len(dataset)*0.2), valid_size=int(len(dataset)*0.2), seed=42)

        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx  = split_idx['test']

        print('dataset.data.y.shape', dataset.data.y.shape)
        if process.has_enough_shot(dataset, train_idx, args.kshot):
            torch.save(train_idx, train_idx_path)
            torch.save(valid_idx, valid_idx_path)
            torch.save(test_idx, test_idx_path)
            print(f"Indices saved to {train_idx_path}, {valid_idx_path}, {test_idx_path}")
            break
        else:
            seed_try += 1
            print(f"Not enough samples for k-shot ({args.kshot}) in train set, trying with seed {seed_try}...")

    else:
        train_idx = torch.load(train_idx_path)
        valid_idx = torch.load(valid_idx_path)
        test_idx  = torch.load(test_idx_path)
        break

fewshot_train_path = os.path.join(save_dir, f'fewshot_train_{args.dataset.lower()}_{args.kshot}shot.pt')
fewshot_valid_path = os.path.join(save_dir, f'fewshot_valid_{args.dataset.lower()}_{args.kshot}shot.pt')

fewshot_train_idx_list = get_fewshot_indices(
    global_dataset=dataset,
    indices=train_idx,
    k=args.kshot,
    num_tasks=100,
    save_path=fewshot_train_path
)

fewshot_valid_idx_list = get_fewshot_indices(
    global_dataset=dataset,
    indices=valid_idx,
    k=args.kshot,
    num_tasks=100,
    save_path=fewshot_valid_path
)

test_dataset = Subset(dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

for lr in lr_list:
    all_roc = []
    for task_id, (train_indices, valid_indices) in enumerate(zip(fewshot_train_idx_list, fewshot_valid_idx_list)):
        print(f"\n===== Few-shot Task {task_id} =====")

        train_dataset = Subset(dataset, train_indices)
        valid_dataset = Subset(dataset, valid_indices)

        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1000, shuffle=False)


        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Train indices: {train_indices}")
        # print(dataset.data.y[train_dataset.indices])
        print(train_dataset.dataset.data.y[train_dataset.indices])

        down = continueweight(
            hid_units, aug1, aug2, num_layers_num, 0.1, save_name,
            test3Dinfo=args.test3Dinfo, pretrainmodel=args.pretrainmodel,
            mode=args.mode, labelnum=labelnum,useembeddingas2Dinfo=args.useembeddingas2Dinfo,
            downtreamtype=args.downstreamtype, condition_emb=args.condition_emb
        )

        best_roc = down.run(
            device=device,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            loss_func=loss_func,
            evaluation=evaluation,
            epochs=500,
            batch_size=100,
            vt_batch_size=1000,
            lr=lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size
        )
        all_roc.append(best_roc)
    
    print(f"Average ROC AUC across all tasks: {np.mean(all_roc):.4f}")
    mean_roc_auc = np.mean(all_roc)
    std_roc_auc = np.std(all_roc)
    print('-' * 100)
    csv_path = f"data/threeDgcn_{args.dataset.lower()}.csv"
    row = [args.pretraindataset, args.downstreamtype, args.condition_emb , args.dataset, args.kshot, args.useembeddingas2Dinfo ,lr, mean_roc_auc, std_roc_auc]
    with open(csv_path, "a", newline="") as out:
        csv_writer = csv.writer(out, dialect="excel")
        csv_writer.writerow(row)
