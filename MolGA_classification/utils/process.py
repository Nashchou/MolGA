import os

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import torch
import torch.nn as nn
import pandas as pd
from sklearn.utils import shuffle
from torch_geometric.data import DataLoader, Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from torch_sparse import SparseTensor

from torch_scatter import scatter
from math import sqrt, pi as PI
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

from torch_cluster import radius_graph


def has_enough_shot(dataset, indices, k):
    """检查指定索引中的每个类别是否至少有 k 个样本"""
    raw_labels = dataset.data.y[indices]
    is_valid = raw_labels ** 2 > 0
    labels = (raw_labels[is_valid] + 1) / 2 
    if labels.dim() > 1:
        labels = labels.argmax(dim=-1)  
    label_count = torch.bincount(labels.long())  
    return (label_count >= k).all()

def generate_scaffold(smiles):
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=True)
    return scaffold

def get_idx_split(data_size, smiles_list, train_size, valid_size, seed,test_size = 0):
    scaffolds = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles)
        scaffolds[scaffold].append(idx)

    # 排序：从大到小处理 scaffold 簇
    scaffold_sets = sorted(scaffolds.values(), key=lambda x: (len(x), x), reverse=True)

    train_idx, valid_idx, test_idx = [], [], []
    train_cutoff = train_size
    valid_cutoff = valid_size

    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_sets)
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) <= train_cutoff:
            train_idx += scaffold_set
        elif len(valid_idx) + len(scaffold_set) <= valid_cutoff:
            valid_idx += scaffold_set
        else:
            test_idx += scaffold_set

    return {
        'train': torch.tensor(train_idx, dtype=torch.long),
        'valid': torch.tensor(valid_idx, dtype=torch.long),
        'test': torch.tensor(test_idx, dtype=torch.long)
    }


def find_2hop_neighbors(adj, node):
    neighbors = []
    # print(type(adj))
    for i in range(len(adj[node])):
        if len(neighbors) >= 10:
            break
        # print('i',i)
        # print('node',node)
        # print('adj[node][i]',adj[node,i])
        if adj[node][i] != 0 and node != i:
            neighbors.append(i)
    neighbors_2hop = []
    for i in neighbors:
        cnt = 0
        for j in range(len(adj[i])):
            if cnt >= 4:
                break
            if adj[i][j] != 0 and j != i:
                neighbors_2hop.append(j)
                cnt += 1
    return neighbors, neighbors_2hop
def plotlabels(feature, Trure_labels, name):
 # maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    # 设置散点颜色

    S_lowDWeights = visual(feature)
    colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab','hotpink']
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels)) # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape) # [num, 3]
    for index in range(4): # 假设总共有三个类别，类别的表示为0,1,2
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=20, marker='.', c=colors[index], edgecolors=colors[index])
        plt.xticks([]) # 去掉横坐标值
        plt.yticks([]) # 去掉纵坐标值
    plt.title(name, fontsize=32, fontweight='normal', pad=20)
    
    plt.savefig('plt_graph/exceptcomputers/{}.png'.format(name),dpi=500)
    plt.show()
    plt.clf()

def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_ts = ts.fit_transform(feat)
    print(x_ts.shape) # [num, 2]
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)
    return x_final

def combine_dataset(*args):
    # print(feature1.shape)
    # print(feature2.shape)
    for step,adj in enumerate(args):
        if step == 0:
            adj1 = adj.todense()
        else:
            adj2 = adj.todense()
            zeroadj = np.zeros((adj1.shape[0], adj2.shape[0]))
            tmpadj1 = np.column_stack((adj1, zeroadj))
            tmpadj2 = np.column_stack((zeroadj.T, adj2))
            adj1 = np.row_stack((tmpadj1, tmpadj2))
            
    adj = sp.csr_matrix(adj1)
    
    return adj

def combine_dataset_list(args):
    # print(feature1.shape)
    # print(feature2.shape)
    for step,adj in enumerate(args):
        if step == 0:
            adj1 = adj.todense()
        else:
            adj2 = adj.todense()
            zeroadj = np.zeros((adj1.shape[0], adj2.shape[0]))
            tmpadj1 = np.column_stack((adj1, zeroadj))
            tmpadj2 = np.column_stack((zeroadj.T, adj2))
            adj1 = np.row_stack((tmpadj1, tmpadj2))
            
    adj = sp.csr_matrix(adj1)
    
    return adj

def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret




def process_3Dgraph(data):
    batch = data.batch
    # print(batch)
    z = data.z.long()
    pos = data.pos
    num_nodes = z.size(0)
    
    edge_index = radius_graph(pos, r=8.0, batch=batch).cpu()
    coo = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0, :], edge_index[1, :])),
                                shape=(num_nodes, num_nodes))
    adj = sp.csr_matrix(coo)

    # graphlabels = labels

    return z, adj



def xyztodat(pos, edge_index, num_nodes):
    # print(pos.shape)
    # print(edge_index.shape)
    # print(num_nodes)
    j, i = edge_index  # j->i
    # print(j.shape)
    # Calculate distances. # number of edges
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

    value = torch.arange(j.size(0), device=j.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    # Calculate angles. 0 to pi
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)
            
    idx_batch = torch.arange(len(idx_i))
    idx_k_n = adj_t[idx_j].storage.col()
    repeat = num_triplets - 1
    num_triplets_t = num_triplets.repeat_interleave(repeat)
    idx_i_t = idx_i.repeat_interleave(num_triplets_t)
    idx_j_t = idx_j.repeat_interleave(num_triplets_t)
    idx_k_t = idx_k.repeat_interleave(num_triplets_t)
    idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
    mask = idx_i_t != idx_k_n       
    idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], idx_k_n[mask], idx_batch_t[mask]

    # Calculate torsions.
    pos_j0 = pos[idx_k_t] - pos[idx_j_t]
    pos_ji = pos[idx_i_t] - pos[idx_j_t]
    pos_jk = pos[idx_k_n] - pos[idx_j_t]
    dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(pos_ji, pos_j0)
    plane2 = torch.cross(pos_ji, pos_jk)
    a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    torsion1 = torch.atan2(b, a) # -pi to pi
    torsion1[torsion1<=0]+=2*PI # 0 to 2pi
    torsion = scatter(torsion1,idx_batch_t,reduce='min')

    return dist, angle, torsion, i, j, idx_kj, idx_ji

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))
    
    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    current_path = os.path.dirname(__file__)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




