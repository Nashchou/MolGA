import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp,GcnLayers
from layers import GCN, AvgReadout 
import tqdm
import numpy as np
from sklearn.decomposition import PCA
from torch_cluster import radius_graph
from torch_scatter import scatter, scatter_min
from torch_sparse import SparseTensor
import torch_sparse
import torch_scatter
from utils import process
import scipy.sparse as sp
from dig.threedgraph.method.comenet.features import torsion_emb,angle_emb
import math
from math import sqrt
from torch_geometric.nn import GraphConv, GraphNorm ,global_mean_pool
from torch_geometric.utils import k_hop_subgraph
import GCL.losses as L
import GCL.augmentors as A
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast,SingleBranchContrast
from GCL.augmentors import Compose, RandomChoice, EdgeRemoving, FeatureMasking,Augmentor,Graph
from torch import Tensor


def pca_compression(seq,k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    
    print(pca.explained_variance_ratio_.sum())
    return seq

def svd_compression(seq, k):
    res = np.zeros_like(seq)
    U, Sigma, VT = np.linalg.svd(seq)
    print(U[:,:k].shape)
    print(VT[:k,:].shape)
    res = U[:,:k].dot(np.diag(Sigma[:k]))
 
    return res

class combineprompt(nn.Module):
    def __init__(self):
        super(combineprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(0)
        self.weight[0][1].data.fill_(1)

    def forward(self, graph_embedding1, graph_embedding2):
        
        # weight = F.softmax(self.weight, dim=1)
        # print("weight",weight)
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)


    
class PrePrompt(nn.Module):
    def __init__(self, n_h, aug1,aug2, num_layers_num, dropout,pretrainmodel,mode):
        super(PrePrompt, self).__init__()
        self.n_h = n_h
        aug_list = [
            EdgeRemoving(pe=0.2),
            FeatureMasking(pf=0.2),
        ]
        
        self.joaov2_aug_list = [
            EdgeRemoving(pe=0.2),
            FeatureMasking(pf=0.2),
        ]
        self.temperature = 0.5
        self.augmentor = Compose([RandomChoice(aug_list, num_choices=1) for _ in (0, 1)])
        self.pretrainmodel = pretrainmodel
        self.gcn = GcnLayers(n_h, n_h, num_layers_num, dropout)
        self.read = AvgReadout()
        self.prompttype = type
        self.emb1 = nn.Embedding(120,n_h)
        self.emb2 = nn.Embedding(3,n_h)
        self.feature1 = torsion_emb(num_radial=3, num_spherical=2, cutoff=8.0)
        self.feature2 = angle_emb(num_radial=3, num_spherical=2, cutoff=8.0)
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G')
        #self.negative_sample = torch.tensor(sample, dtype=torch.int64).cuda()
        self.fc = nn.Linear(n_h,1)
        self.loss = nn.BCEWithLogitsLoss()
        self.cutoff =8.0
        self.project = torch.nn.Linear(n_h, n_h)
        self.reset_parameters()
        self.DGIloss = SingleBranchContrast(loss=L.JSD(), mode='G2L')

        self.mode = mode
        if self.mode == 'G2G':
            self.dual_contrast = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G')
        elif self.mode == 'L2L':
            self.dual_contrast = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L')
        if pretrainmodel == 'joaov2':
            self.logits = torch.nn.Parameter(torch.zeros(len(self.joaov2_aug_list)))        
        
    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index
    def reset_parameters(self):
        self.emb1.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.emb2.weight.data.uniform_(-sqrt(3), sqrt(3))


    def forward(self, x,edge_index,batch):
        # Check that the lengths of seq_list and adj_list match
        x = self.emb1(x[:, 0]) + self.emb2(x[:, 1])
        if self.pretrainmodel == 'graphcl':
            x1, edge1 ,_ = self.augmentor(x, edge_index,None)
            x2, edge2 ,_= self.augmentor(x, edge_index,None)
        elif self.pretrainmodel == 'joaov2':
            gumbel_noise = -torch.empty_like(self.logits).exponential_().log()
            probs = torch.softmax((self.logits + gumbel_noise) / self.temperature, dim=0)

            idx = torch.multinomial(probs, 1).item()
            selected_aug = self.joaov2_aug_list[idx]
            x1, edge1, _ = selected_aug(x, edge_index, None)
            x2, edge2, _ = selected_aug(x, edge_index, None)


        h1 = self.gcn(x1, edge1)
        h2 = self.gcn(x2, edge2)       
        
        if self.mode == 'G2G':
            g1 = global_mean_pool(h1, batch)
            g2 = global_mean_pool(h2, batch)

            # print(g1.shape)
            # print(g2.shape)
            z1, z2 = self.project(g1), self.project(g2)
            loss = self.dual_contrast(g1=z1, g2=z2)
        elif self.mode == 'L2L':
            z1, z2 = self.project(h1), self.project(h2)
            loss = self.dual_contrast(h1=z1, h2=z2)            

        return loss
    

    def embed(self, data):
        usecondition_net = 'true'
        # print(data)
        batch = data.batch
        # print(batch)
        z = data.z.long()
        pos = data.pos
        num_nodes = z.size(0)
        
        edge_index = radius_graph(pos, r=8.0, batch=batch).cuda()
        j, i = edge_index
        seq = self.emb(z)


        if usecondition_net == 'true':
            pos1,pos2 = self.posfeature(data)
            pos = torch.cat((pos1,pos2,seq),dim=1).cuda()

            conditionprompt = self.condition(pos)

        
            h_1 = self.gcn(seq,edge_index,conditionprompt)

        elif usecondition_net == 'false':
            h_1 = self.gcn(seq,edge_index)


        h = h_1
        # print(h_1.shape)

        h = torch_scatter.scatter(h.squeeze(),batch,dim=0,reduce='mean')
        return self.fc(h)



def get_neighbors_features_and_similarity(x, edge_index, v_list):
    print(v_list.shape)
    print(edge_index.shape)
    num_hops = 1
    subset, edge_index_sub, mapping, _ = k_hop_subgraph(
        v_list, num_hops, edge_index, relabel_nodes=True)

    x_sub = x[subset]  
    node_to_center = {}
    for i, node_idx in enumerate(v_list):

        
        neighbors, _, _, _ = k_hop_subgraph(
            node_idx.unsqueeze(0), num_hops, edge_index, relabel_nodes=False)
        
        for neighbor in neighbors.tolist():
            if neighbor not in node_to_center:
                node_to_center[neighbor] = []
            node_to_center[neighbor].append(i) 
    mask = torch.zeros((x_sub.size(0), v_list.size(0)), dtype=torch.bool).cuda()
    for i, node_idx in enumerate(subset.tolist()):
        centers = node_to_center[node_idx]
        mask[i, centers] = True  

    x_centers = x[v_list]  
    cos_sim = F.cosine_similarity(
        x_sub.unsqueeze(1),  
        x_centers.unsqueeze(0),  
        dim=2
    ) 
    cos_sim = cos_sim * mask.float()

    weighted_features = cos_sim.t() @ x_sub  

    return weighted_features


class textprompt(nn.Module):
    def __init__(self,hid_units,type_):
        super(textprompt, self).__init__()
        self.act = nn.ELU()
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.prompttype = type_
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        # self.weight[0][0].data.fill_(0.3)
        # self.weight[0][1].data.fill_(0.3)
        # self.weight[0][2].data.fill_(0.3)
    def forward(self, graph_embedding):
        # print("weight",self.weight)
        if self.prompttype == 'add':
            weight = self.weight.repeat(graph_embedding.shape[0],1)
            graph_embedding = weight + graph_embedding
        if self.prompttype == 'mul':
            graph_embedding=self.weight * graph_embedding

        return graph_embedding


def mygather(feature, index):
    # print("index",index)
    # print("indexsize",index.shape)  
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    # print(tuples)

    # print("feature",feature)
    # print("featuresize",feature.shape)
    # print("index",index)
    # print("indexsize",index.shape)
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))


def compareloss(feature,tuples,temperature):
    # print("feature",feature)
    # print("tuple",tuples)
    # feature=feature.cpu()
    # tuples = tuples.cpu()
    h_tuples=mygather(feature,tuples)
    # print("tuples",h_tuples)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    # temp = m(temp)
    temp=temp.cuda()
    h_i = mygather(feature, temp)
    # print("h_i",h_i)
    # print("h_tuple",h_tuples)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    # print("sim",sim)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)

    # print("numerator",numerator)
    # print("denominator",denominator)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()


def prompt_pretrain_sample(adj,n):
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))
    # print("#############")
    # print("start sampling disconnected tuples")
    for i in range(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    return res.astype(int)


