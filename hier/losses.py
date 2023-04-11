import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import torch.distributed as dist

import hyptorch.pmath as pmath
import hyptorch.nn as hypnn
from hyptorch.pmath import dist_matrix

    
class HIERLoss(nn.Module):
    def __init__(self, nb_proxies, sz_embed, mrg=0.1, tau=0.1, hyp_c=0.1, clip_r=2.3):
        super().__init__()
        self.nb_proxies = nb_proxies
        self.sz_embed = sz_embed
        self.tau = tau
        self.hyp_c = hyp_c
        self.mrg = mrg
        self.clip_r = clip_r
        
        self.lcas = torch.randn(self.nb_proxies, self.sz_embed).cuda()
        self.lcas = self.lcas / math.sqrt(self.sz_embed) * clip_r * 0.9
        self.lcas = torch.nn.Parameter(self.lcas)
        self.to_hyperbolic = hypnn.ToPoincare(c=hyp_c, ball_dim=sz_embed, riemannian=True, clip_r=clip_r, train_c=False)
                
        if hyp_c > 0:
            self.dist_f = lambda x, y: dist_matrix(x, y, c=hyp_c)
        else:
            self.dist_f = lambda x, y: 2 - 2 * F.linear(x,y)
    
    def compute_gHHC(self, z_s, lcas, dist_matrix, indices_tuple, sim_matrix):
        i, j, k = indices_tuple
        bs = len(z_s)
        
        cp_dist = dist_matrix
        
        max_dists_ij = torch.maximum(cp_dist[i], cp_dist[j])
        lca_ij_prob = F.gumbel_softmax(-max_dists_ij / self.tau, dim=1, hard=True)
        lca_ij_idx = lca_ij_prob.argmax(-1)
        
        max_dists_ijk = torch.maximum(cp_dist[k], max_dists_ij)
        lca_ijk_prob = F.gumbel_softmax(-max_dists_ijk / self.tau, dim=1, hard=True)
        lca_ijk_idx = lca_ijk_prob.argmax(-1)
        
        dist_i_lca_ij, dist_i_lca_ijk = (cp_dist[i] * lca_ij_prob).sum(1), (cp_dist[i] * lca_ijk_prob).sum(1)
        dist_j_lca_ij, dist_j_lca_ijk = (cp_dist[j] * lca_ij_prob).sum(1), (cp_dist[j] * lca_ijk_prob).sum(1)
        dist_k_lca_ij, dist_k_lca_ijk = (cp_dist[k] * lca_ij_prob).sum(1), (cp_dist[k] * lca_ijk_prob).sum(1)
                    
        hc_loss = torch.relu(dist_i_lca_ij - dist_i_lca_ijk + self.mrg) \
                    + torch.relu(dist_j_lca_ij - dist_j_lca_ijk + self.mrg) \
                    + torch.relu(dist_k_lca_ijk - dist_k_lca_ij + self.mrg)
                                        
        hc_loss = hc_loss * (lca_ij_idx!=lca_ijk_idx).float()
        loss = hc_loss.mean()
                
        return loss
        
    def get_reciprocal_triplets(self, sim_matrix, topk=20, t_per_anchor = 100):
        anchor_idx, positive_idx, negative_idx = [], [], []
        
        topk_index = torch.topk(sim_matrix, topk)[1]
        nn_matrix = torch.zeros_like(sim_matrix).scatter_(1, topk_index, torch.ones_like(sim_matrix))
        sim_matrix = ((nn_matrix + nn_matrix.t())/2).float()         
        sim_matrix = sim_matrix.fill_diagonal_(-1)
                
        for i in range(len(sim_matrix)):
            if len(torch.nonzero(sim_matrix[i]==1)) <= 1:
                continue
            pair_idxs1 = np.random.choice(torch.nonzero(sim_matrix[i]==1).squeeze().cpu().numpy(), t_per_anchor, replace=True)
            pair_idxs2 = np.random.choice(torch.nonzero(sim_matrix[i]<1).squeeze().cpu().numpy(), t_per_anchor, replace=True)              
            positive_idx.append(pair_idxs1)
            negative_idx.append(pair_idxs2)
            anchor_idx.append(np.ones(t_per_anchor) * i)
        anchor_idx = np.concatenate(anchor_idx)
        positive_idx = np.concatenate(positive_idx)
        negative_idx = np.concatenate(negative_idx)
        return anchor_idx, positive_idx, negative_idx
    
    def forward(self, z_s, y, topk=30):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """        
        bs = len(z_s)
        lcas = self.to_hyperbolic(self.lcas)
        all_nodes = torch.cat([z_s, lcas])
        all_dist_matrix = self.dist_f(all_nodes, all_nodes)
          
        sim_matrix = torch.exp(-all_dist_matrix[:bs,:bs]).detach()
        one_hot_mat = (y.unsqueeze(1) == y.unsqueeze(0))
        sim_matrix[one_hot_mat] += 1
        sim_matrix2 = torch.exp(-all_dist_matrix[bs:,bs:]).detach()
        
        indices_tuple = self.get_reciprocal_triplets(sim_matrix, topk=topk, t_per_anchor = 50)
        loss = self.compute_gHHC(z_s, lcas, all_dist_matrix[:bs, bs:], indices_tuple, sim_matrix)
        
        indices_tuple2 = self.get_reciprocal_triplets(sim_matrix2, topk=topk, t_per_anchor = 50)
        loss += self.compute_gHHC(lcas, lcas, all_dist_matrix[bs:, bs:], indices_tuple2, sim_matrix2)
        return loss
    
class MSLoss(nn.Module):
    def __init__(self, tau=0.2, hyp_c=0.1):
        super().__init__()
        self.tau = tau
        self.hyp_c = hyp_c
        self.mrg = 0.5
        self.alpha, self.beta = 1, 5
        
        if hyp_c == 0:
            self.sim_f = lambda x, y: x @ y.t()
        else:
            self.sim_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
            
    def forward(self, X, y):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """        
        batch_size = X.shape[0]
        device = X.device
                
        labels = y.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        class_eq_mask = torch.eq(labels, labels.T).float().to(device)
        
        # mask-out self-contrast cases
        self_mask = torch.scatter(torch.ones_like(class_eq_mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        
        pos_mask = class_eq_mask * self_mask
        neg_mask = (1-class_eq_mask)
        
        # compute logits
        logits =  self.sim_f(X, X)
        
        mean_logit = logits[~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)].mean()        
        pos_exp = torch.exp(-self.alpha * (logits - mean_logit)) * pos_mask
        neg_exp = torch.exp(self.beta * (logits - mean_logit)) * neg_mask
        
        pos_loss = 1.0 / self.alpha * torch.log(1 + torch.sum(pos_exp, dim=1))
        neg_loss = 1.0 / self.beta * torch.log(1 + torch.sum(neg_exp, dim=1))
        
        # loss
        loss = (pos_loss + neg_loss).mean()
                
        return loss
    
class Contrastive_Angle(nn.Module):
    def __init__(self, IPC, tau=0.2):
        torch.nn.Module.__init__(self)
        self.tau = 0.2
        self.sim_f = lambda x, y: F.linear(F.normalize(x), F.normalize(y))
    
    def contrastive_loss(x, y):
        # x0 and x1 - positive pair
        # tau - temperature
        bsize = x0.shape[0]
        target = torch.arange(bsize).cuda()
        eye_mask = torch.eye(bsize).cuda() * 1e9
        logits00 = self.sim_f(x0, x0) / self.tau - eye_mask
        logits01 = self.sim_f(x0, x1) / self.tau
        logits = torch.cat([logits01, logits00], dim=1)
        logits -= logits.max(1, keepdim=True)[0].detach()
        loss = F.cross_entropy(logits, target)
        return loss
    
class MSLoss_Angle(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = 0.5
        self.alpha, self.beta = 2, 50
            
    def forward(self, X, y):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """        
        batch_size = X.shape[0]
        device = X.device
                
        labels = y.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        class_eq_mask = torch.eq(labels, labels.T).float().to(device)
        
        # mask-out self-contrast cases
        self_mask = torch.scatter(torch.ones_like(class_eq_mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        
        pos_mask = class_eq_mask * self_mask
        neg_mask = (1-class_eq_mask)
        
        # compute logits
        X = F.normalize(X)
        logits =  F.linear(X, X) 
               
        pos_exp = torch.exp(-self.alpha * (logits - self.base)) * pos_mask
        neg_exp = torch.exp(self.beta * (logits - self.base)) * neg_mask
        
        pos_loss = 1.0 / self.alpha * torch.log(1 + torch.sum(pos_exp, dim=1))
        neg_loss = 1.0 / self.beta * torch.log(1 + torch.sum(neg_exp, dim=1))
        
        # loss
        loss = (pos_loss + neg_loss).mean()
                
        return loss
    
class PALoss_Angle(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
        self.proxies = torch.nn.Parameter(torch.randn(self.nb_classes, self.sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
    def forward(self, X, T, P=None):
        if P is None:
            P = self.proxies
        else:
            P = P[:self.nb_classes]
                
        cos = F.linear(F.normalize(X), F.normalize(P))  # Calcluate cosine similarity
        P_one_hot = F.one_hot(T, num_classes = self.nb_classes).float()        
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
                
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        
        loss = (pos_term + neg_term)
        return loss
    
class PNCALoss_Angle(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32, normalize=True):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.normalize = normalize
        
        self.proxies = torch.nn.Parameter(torch.randn(self.nb_classes, self.sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
    def forward(self, X, T, P=None):
        if P is None:
            P = self.proxies
        else:
            P = P[:self.nb_classes]
                
        cos = F.linear(F.normalize(X), F.normalize(P))  # Calcluate cosine similarity
        P_one_hot = F.one_hot(T, num_classes = self.nb_classes).float()        
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 1) != 0).squeeze(dim = 0)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=1) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=1)
                
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / len(cos)
        
        loss = (pos_term + neg_term)
        
        return loss
    
class SoftTripleLoss_Angle(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, centers_per_class=10, la=20, gamma=0.1, margin=0.01):
        torch.nn.Module.__init__(self)
        self.loss_func = losses.SoftTripleLoss(nb_classes, sz_embed, centers_per_class, la, gamma, margin)
    
    def forward(self, X, T):
        X = F.normalize(X)
        loss = self.loss_func(X, T)
        return loss
    
class SupCon(torch.nn.Module):
    def __init__(self, tau=0.2, hyp_c=0.1, IPC=1):
        torch.nn.Module.__init__(self)
        self.tau = tau
        self.hyp_c = hyp_c
        self.IPC = IPC
        
        if hyp_c == 0:
            self.dist_f = lambda x, y: x @ y.t()
        else:
            self.dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
            
    def compute_loss(self, x0, x1):
        bsize = x0.shape[0]
        target = torch.arange(bsize).cuda()
        eye_mask = torch.eye(bsize).cuda() * 1e9
        logits00 = self.dist_f(x0, x0) / self.tau - eye_mask
        logits01 = self.dist_f(x0, x1) / self.tau
        logits = torch.cat([logits01, logits00], dim=1)
        logits -= logits.max(1, keepdim=True)[0].detach()
        loss = F.cross_entropy(logits, target)
        return loss
    
    def forward(self, X, T):
        # x0 and x1 - positive pair
        # tau - temperature
        # hyp_c - hyperbolic curvature, "0" enables sphere mode
        loss = 0
        step = 0
        for i in range(self.IPC):
            for j in range(self.IPC):
                if i != j:
                    loss += self.compute_loss(X[:, i], X[:, j])
                step += 1
        loss /= step
        return loss