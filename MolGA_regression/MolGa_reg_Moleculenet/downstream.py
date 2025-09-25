import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL
from layers import GCN, AvgReadout
import torch_scatter
from torch_sparse import SparseTensor
from dig.threedgraph.dataset import QM93D
from dig.threedgraph.method import ComENet
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method.comenet.features import torsion_emb,angle_emb
from torch_scatter import scatter
from math import sqrt, pi as PI
import math
import sympy as sym
from torch_scatter import scatter, scatter_min
from torch_cluster import radius_graph
from torch.optim import Adam,AdamW
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import DataLoader
from tqdm import tqdm
import os
from models import DGI, GraphCL, Lp,GcnLayers
import torch_sparse
import numpy as np
import matplotlib.pyplot as plt

from preprompt import PrePrompt
from tqdm import trange 

import torch
import torch.nn.functional as F



class GradNormLoss(nn.Module):
    def __init__(self, model, num_losses, alpha=1.5):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.num_losses = num_losses
        self.w = nn.Parameter(torch.ones(num_losses))  

    def forward(self, losses, t=1):
        weights = F.softmax(self.w, dim=0)  # [n]

        total_loss = sum(w * l for w, l in zip(weights, losses))

        total_loss.backward(retain_graph=True)

        G = []
        for i, loss in enumerate(losses):
            self.model.zero_grad()
            (weights[i] * loss).backward(retain_graph=True)
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += (p.grad.detach() ** 2).sum()
            G.append(grad_norm.sqrt())

        G = torch.stack(G)  # [n]
        G_avg = G.mean().detach()

        with torch.no_grad():
            L_hat = torch.tensor([loss.item() / t for loss in losses], device=G.device)
            r = L_hat / L_hat.mean()

        grad_loss = (G - G_avg * (r ** self.alpha)).abs().sum()

        return total_loss + grad_loss

def npair_loss(h_2d, h_3d, temperature=1):


    h_2d = F.normalize(h_2d, p=2, dim=1)
    h_3d = F.normalize(h_3d, p=2, dim=1)

    logits_2d_to_3d = torch.matmul(h_2d, h_3d.T)  / temperature
    labels = torch.arange(h_2d.size(0), device=h_2d.device)

    loss_2d_to_3d = F.cross_entropy(logits_2d_to_3d, labels)
    logits_3d_to_2d = torch.matmul(h_3d, h_2d.T) / temperature


    loss_3d_to_2d = F.cross_entropy(logits_3d_to_2d, labels)

    loss = 0.5 * (loss_2d_to_3d + loss_3d_to_2d)
    return loss




class ConditionNet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):

        super(ConditionNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_fc = nn.Linear(node_dim, hidden_dim)
        self.edge_fc = nn.Linear(edge_dim, hidden_dim)

        self.hidden_fc = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])

        self.fusion_fc = nn.Linear(hidden_dim, output_dim)

        self.output_fc = nn.Linear(hidden_dim, output_dim)

        self.dropout_layer = nn.Dropout(p=dropout)

        self.reset_parameters()  

    def reset_parameters(self):
        self.node_fc.reset_parameters()
        self.edge_fc.reset_parameters()
        for layer in self.hidden_fc:
            layer.reset_parameters()
        self.fusion_fc.reset_parameters()
        self.output_fc.reset_parameters()

    def forward(self, node_feat, edge_feat):

        node_out = F.elu(self.node_fc(node_feat))  
        edge_out = F.elu(self.edge_fc(edge_feat))  

        fusion_out = node_out + edge_out
        fusion_out = F.elu(self.dropout_layer(self.fusion_fc(fusion_out)))  

        return fusion_out

class Downstream(nn.Module):
    def __init__(self, emb1, emb2, gcn, n_h,labelnum=1,condition_emb=32):
        super(Downstream, self).__init__()
        self.emb1 = emb1
        self.emb2 = emb2       
        self.gcn = gcn
        self.fc = nn.Linear(n_h,labelnum)



        self.token2D =  nn.Parameter(torch.ones(n_h))
        self.generator3D = nn.Linear(n_h * 2, 18)



        self.condition_prompt = ConditionNet(n_h*2,n_h*2,condition_emb, n_h)


        self.feature1 = torsion_emb(num_radial=3, num_spherical=2, cutoff=8.0)
        self.feature2 = angle_emb(num_radial=3, num_spherical=2, cutoff=8.0)
        self.cutoff = 8.0
        self.threeDtotwoDMLP = nn.Linear(18,n_h*2)    

        self.reset_parameters()


    def reset_parameters(self):
        self.fc.reset_parameters()
        self.generator3D.reset_parameters()
        self.threeDtotwoDMLP.reset_parameters()
        nn.init.ones_(self.token2D)

    def embed(self, data,train=0,test_3D_info=0):
        usecondition_net = 'true'
        batch = data.batch
        z = data.x.long().cuda()
        pos = data.pos
        
        edge_index = data.edge_index
        j, i = edge_index
        # print(edge_index.shape)
        seq = self.emb1(z[:, 0]) + self.emb2(z[:, 1])
        # seq = self.gcn(seq, edge_index)
        node_feat_expanded = torch.cat((seq[i], seq[j]), dim=1)
        # tokenseq  = seq * self.token2D  
        tokenseq  = seq   
        token_edge_seq = torch.cat((tokenseq[i], tokenseq[j]), dim=1)

        if train == 1:
            pos1, pos2 = self.posfeature(data)
            pos = torch.cat((pos1, pos2), dim=1).cuda()

            generate_3D_edge = self.generator3D(token_edge_seq)
            # print(pos.shape)
            # print(generate_3D_edge.shape)
            generate_loss = npair_loss(generate_3D_edge,pos)
            transfering2D = self.threeDtotwoDMLP(pos)
            npairloss = npair_loss(node_feat_expanded,transfering2D)
            
            N = node_feat_expanded.shape[0]

            npairloss = npairloss / max(1,math.log(N + 1e-8))

            conditionprompt = self.condition_prompt(node_feat_expanded,transfering2D)   
            h_1 = self.gcn(seq, edge_index,conditionprompt)
            h = torch_scatter.scatter(h_1.squeeze(), batch, dim=0, reduce='mean')
            return self.fc(h) , npairloss ,generate_loss
        else:
            if test_3D_info == 0:
                generate_3D_edge = self.generator3D(token_edge_seq)
                transfering2D = self.threeDtotwoDMLP(generate_3D_edge)
                # transfering2D = F.normalize(transfering2D, p=2, dim=1)
            else:
                pos1, pos2 = self.posfeature(data)
                pos = torch.cat((pos1, pos2), dim=1).cuda() 
                transfering2D = self.threeDtotwoDMLP(pos)  
                # transfering2D = F.normalize(transfering2D, p=2, dim=1)

            conditionprompt = self.condition_prompt(node_feat_expanded,transfering2D)
            h_1 = self.gcn(seq, edge_index, conditionprompt)
            h = torch_scatter.scatter(h_1.squeeze(), batch, dim=0, reduce='mean')
            return self.fc(h) 
    
    def posfeature(self, data):
        batch = data.batch
        z = data.x.long()
        pos = data.pos
        num_nodes = z.size(0)
        
        j, i = data.edge_index
        
        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)
        
        # x = self.emb1(z)
        
        _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
        argmin0[argmin0 >= len(i)] = 0
        n0 = j[argmin0]
        
        add = torch.zeros_like(dist).to(dist.device)
        add[argmin0] = self.cutoff
        dist1 = dist + add
        
        _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
        argmin1[argmin1 >= len(i)] = 0
        n1 = j[argmin1]
        
        _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(j)] = 0
        n0_j = i[argmin0_j]
        
        add_j = torch.zeros_like(dist).to(dist.device)
        add_j[argmin0_j] = self.cutoff
        dist1_j = dist + add_j
        
        _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(j)] = 0
        n1_j = i[argmin1_j]
        
        n0 = n0[i]
        n1 = n1[i]
        n0_j = n0_j[j]
        n1_j = n1_j[j]
        
        mask_iref = n0 == j
        iref = torch.clone(n0)
        iref[mask_iref] = n1[mask_iref]
        idx_iref = argmin0[i]
        idx_iref[mask_iref] = argmin1[i][mask_iref]
        
        mask_jref = n0_j == i
        jref = torch.clone(n0_j)
        jref[mask_jref] = n1_j[mask_jref]
        idx_jref = argmin0_j[j]
        idx_jref[mask_jref] = argmin1_j[j][mask_jref]
        
        pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
            vecs,
            vecs[argmin0][i],
            vecs[argmin1][i],
            vecs[idx_iref],
            vecs[idx_jref]
        )
        
        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] += math.pi
        
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(-pos_ji, pos_in0)
        plane2 = torch.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi[phi < 0] += math.pi
        
        plane1 = torch.cross(pos_ji, pos_jref_j)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] += math.pi
        
        feature1 = self.feature1(dist, theta, phi)
        feature2 = self.feature2(dist, tau)
        
        return feature1, feature2


class continueweight():
    def __init__(self, n_h, aug1,aug2, num_layers_num, dropout,save_name,test3Dinfo,pretrainmodel,mode,labelnum, condition_emb=32):
        self.labelnum = labelnum
        self.model = PrePrompt(n_h,  aug1,aug2, num_layers_num, dropout,pretrainmodel,mode)
        print(f'loading model from {save_name}')
        self.model.load_state_dict(torch.load(save_name))
        self.downstream = Downstream(self.model.emb1,self.model.emb2,self.model.gcn,self.model.n_h,labelnum,condition_emb=condition_emb)
        if test3Dinfo == 0:
            self.gradnorm = GradNormLoss(self.downstream,num_losses=3)
        else:
            self.gradnorm = GradNormLoss(self.downstream,num_losses=2)
        self.test3Dinfo = test3Dinfo
        self.downstream.cuda()
        self.gradnorm.cuda()

    def run(self, device, train_dataset, valid_dataset, test_dataset, loss_func, evaluation, epochs=500, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=1e-5, 
        energy_and_force=False, p=100, save_dir='modelsave'):
        # print(self.downstream.parameters())
        for names,param in self.downstream.named_parameters():
            print(names)
            param.requires_grad = True
        for names,param in self.downstream.gcn.named_parameters():
            print(names)
            param.requires_grad = False
        for names,param in self.downstream.emb1.named_parameters():
            print(names)
            param.requires_grad = False
        for names,param in self.downstream.emb2.named_parameters():
            print(names)
            param.requires_grad = False
        optimizer = AdamW(list(self.downstream.parameters()) + list(self.gradnorm.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
        best_valid = float('inf')
        best_test = float('inf')
        train_losses = []
        valid_maes = []
        patience = 50
        progress_bar = trange(1, epochs + 1, desc="Training", ncols=100)

        for epoch in progress_bar:
            # Training
            train_loss = self.train(optimizer, train_loader, energy_and_force, p, loss_func, device, epoch)
            train_losses.append(train_loss)

            # Validation
            valid_mae,_ = self.val(valid_loader, energy_and_force, p, evaluation, device)
            valid_maes.append(valid_mae)

            # Update tqdm bar description
            progress_bar.set_description(f"Epoch {epoch:03d}")
            progress_bar.set_postfix(train_mae=f"{train_loss:.4f}", val_mae=f"{valid_mae:.4f}")

            # Save best model
            if valid_mae < best_valid:
                best_valid = valid_mae
                patience_counter = 0 
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.downstream.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_valid_mae': best_valid
                }
                valid_checkpointpath = os.path.join(save_dir, 'valid_checkpoint.pt')
                torch.save(checkpoint, valid_checkpointpath)
            else:
                patience_counter += 1
            # Step LR scheduler
            scheduler.step()
            # Early stopping condition
            if patience_counter >= patience:
                print(f"\n[EarlyStopping] No improvement in {patience} epochs. Stopping at epoch {epoch}.")
                break
        epochs_list = list(range(1, len(train_losses) + 1))
        plt.plot(epochs_list, train_losses, label='Train Loss / MAE')
        plt.plot(epochs_list, valid_maes, label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Loss / MAE')
        plt.title('Training & Validation Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig('loss_curve1.png', dpi=300) 
        plt.close() 
        downstreammodel = torch.load(valid_checkpointpath)
        # print(downstreammodel)
        best_epoch = downstreammodel['epoch']
        print(f'best epoch:{ best_epoch }')
        self.downstream.load_state_dict(downstreammodel['model_state_dict'])
        print('\n\nTesting...', flush=True)
        best_test,best_std = self.val( test_loader, energy_and_force, p, evaluation, device)
        print(f'Best validation MAE so far: {best_valid}')
        print(f'Test MAE when got best validation result: {best_test}')
        return best_test, best_std

    def train(self, optimizer, train_loader, energy_and_force, p, loss_func, device,epoch):
        # self.fc.cuda()
        loss_accum = 0
        torch.autograd.set_detect_anomaly(True)

        for step, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            batch_data = batch_data.cuda()
            # print(batch_data)
            out , npairloss ,generate_loss = self.downstream.embed(batch_data,1,self.test3Dinfo)


            # print('out.shape',out.shape)
            # print('true label',batch_data.y.shape)

            label_loss =loss_func(out, batch_data.y.reshape(-1,self.labelnum).unsqueeze(1)) 
            # loss = label_loss

            if self.test3Dinfo == 0:
                losses = [label_loss,npairloss,generate_loss]
            else:
                losses = [label_loss,npairloss]
            loss = self.gradnorm(losses,torch.tensor(epoch).cuda())
            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
        return loss_accum / (step + 1)


    def val(self, data_loader, energy_and_force, p, evaluation, device):
        self.downstream.eval()

        preds = torch.Tensor([]).to(device)
        targets = torch.Tensor([]).to(device)


        for step, batch_data in enumerate(data_loader):
            batch_data = batch_data.to(device)

            out = self.downstream.embed(batch_data,0,self.test3Dinfo)
            preds = torch.cat([preds, out.detach_()], dim=0)
            targets = torch.cat([targets, batch_data.y.reshape(-1,self.labelnum)], dim=0)

        input_dict = {"y_true": targets, "y_pred": preds}
        # print('targets',targets)
        # print('y_pred',preds)


        rmse = float(np.sqrt(np.mean((np.array(preds.cpu()) - np.array(targets.cpu())) ** 2)))
        std_rmse = float(np.std(np.sqrt(np.mean((np.array(preds.cpu()) - np.array(targets.cpu())) ** 2))))



        mae = float(np.mean(np.absolute(np.array(preds.cpu()) - np.array(targets.cpu()))))
        return rmse,std_rmse
 
    

    def posfeature(self,data):
        batch = data.batch
        z = data.z.long()
        pos = data.pos
        num_nodes = z.size(0)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        j, i = edge_index

        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)

        # Embedding block.
        x = self.emb(z)

        # Calculate distances.
        _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
        argmin0[argmin0 >= len(i)] = 0
        n0 = j[argmin0]
        add = torch.zeros_like(dist).to(dist.device)
        add[argmin0] = self.cutoff
        dist1 = dist + add

        _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
        argmin1[argmin1 >= len(i)] = 0
        n1 = j[argmin1]
        # --------------------------------------------------------

        _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(j)] = 0
        n0_j = i[argmin0_j]

        add_j = torch.zeros_like(dist).to(dist.device)
        add_j[argmin0_j] = self.cutoff
        dist1_j = dist + add_j

        # i[argmin] = range(0, num_nodes)
        _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(j)] = 0
        n1_j = i[argmin1_j]

        # ----------------------------------------------------------

        # n0, n1 for i
        n0 = n0[i]
        n1 = n1[i]

        # n0, n1 for j
        n0_j = n0_j[j]
        n1_j = n1_j[j]

        # tau: (iref, i, j, jref)
        # when compute tau, do not use n0, n0_j as ref for i and j,
        # because if n0 = j, or n0_j = i, the computed tau is zero
        # so if n0 = j, we choose iref = n1
        # if n0_j = i, we choose jref = n1_j
        mask_iref = n0 == j
        iref = torch.clone(n0)
        iref[mask_iref] = n1[mask_iref]
        idx_iref = argmin0[i]
        idx_iref[mask_iref] = argmin1[i][mask_iref]

        mask_jref = n0_j == i
        jref = torch.clone(n0_j)
        jref[mask_jref] = n1_j[mask_jref]
        idx_jref = argmin0_j[j]
        idx_jref[mask_jref] = argmin1_j[j][mask_jref]

        pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
            vecs,
            vecs[argmin0][i],
            vecs[argmin1][i],
            vecs[idx_iref],
            vecs[idx_jref]
        )

        # Calculate angles.
        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + math.pi

        # Calculate torsions.
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(-pos_ji, pos_in0)
        plane2 = torch.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + math.pi

        # Calculate right torsions.
        plane1 = torch.cross(pos_ji, pos_jref_j)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + math.pi

        feature1 = self.feature1(dist, theta, phi)
        feature2 = self.feature2(dist, tau)
        
        return feature1,feature2
    
    def embed(self, data):
        print(data)
        batch = data.batch
        # print(batch)
        z = data.z.long()
        pos = data.pos
        num_nodes = z.size(0)
        
        edge_index = radius_graph(pos, r=8.0, batch=batch).cpu()
        j, i = edge_index
        print(edge_index.shape)
        seq = self.emb(z)
        adj = SparseTensor(row=i, col=j, value=torch.ones(j.size(0)), sparse_sizes=(num_nodes, num_nodes)).cuda()
        identity = SparseTensor.eye(num_nodes).cuda()
        adj = adj + identity
        # print(adj)
        deg = torch_sparse.sum(adj, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj = torch_sparse.mul(adj, deg_inv_sqrt.view(-1, 1))
        normalized_adj = torch_sparse.mul(adj, deg_inv_sqrt.view(1, -1))
        h_1 = self.gcn(seq, normalized_adj.to_torch_sparse_coo_tensor(), True)


        print(h_1.shape)

        h_1 = torch_scatter.scatter(h_1.squeeze(),batch,dim=0,reduce='mean')
        return h_1.detach()