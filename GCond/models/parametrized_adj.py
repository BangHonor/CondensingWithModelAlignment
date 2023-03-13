import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from itertools import product
import numpy as np

class PGE(nn.Module):

    def __init__(self, nfeat, nnodes, nhid=128, nlayers=3, device=None, args=None):
        super(PGE, self).__init__()
        if args.dataset in ['ogbn-arxiv', 'arxiv', 'flickr','ogbn-products','ogbn-papers100M']:
           nhid = 256
        if args.dataset in ['reddit']:
           nhid = 256
           if args.reduction_rate==0.01:
               nhid = 128
           nlayers = 3
           # nhid = 128

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat*2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))

        edge_index = np.array(list(product(range(nnodes), range(nnodes))))#内存占用蛮大 shape:(20511841, 2)
        self.edge_index = edge_index.T
        self.nnodes = nnodes
        self.device = device
        self.reset_parameters()
        self.cnt = 0
        self.args = args
        self.nnodes = nnodes

    def forward(self, x, inference=False):
        if self.args.dataset in ['reddit','ogbn-products','ogbn-papers100M']:
            edge_index = self.edge_index
            n_part = 10
            splits = np.array_split(np.arange(edge_index.shape[1]), n_part)#divide into sevelral parts/lists
            edge_embed = []
            for idx in splits:
                tmp_edge_embed = torch.cat([x[edge_index[0][idx]],#edge_index[0]是product中元组的第一个元素组成的list，0*2577,1*2577，edge_index[0]对应(0-2576)*2577
x[edge_index[1][idx]]], axis=1)#列合并
                for ix, layer in enumerate(self.layers):
                    tmp_edge_embed = layer(tmp_edge_embed)
                    if ix != len(self.layers) - 1:
                        tmp_edge_embed = self.bns[ix](tmp_edge_embed)
                        tmp_edge_embed = F.relu(tmp_edge_embed)
                edge_embed.append(tmp_edge_embed)#不论切多少分 这里都会比较大
            edge_embed = torch.cat(edge_embed)#每一次得到一些边的embedding 最后组合reshape
        else:
            edge_index = self.edge_index#小图140*140=19600条边,不变
            edge_embed = torch.cat([x[edge_index[0]],#由于这一步edge_index是2*edge_num,edgenum在节点为千时就已经上万，再加上x本身的维度可能上百，该数量级可能很大
                    x[edge_index[1]]], axis=1)
            for ix, layer in enumerate(self.layers):#PGE里面的每一层
                edge_embed = layer(edge_embed)#X'->edge_embed->邻接矩阵A'
                if ix != len(self.layers) - 1:
                    edge_embed = self.bns[ix](edge_embed)
                    edge_embed = F.relu(edge_embed)

        adj = edge_embed.reshape(self.nnodes, self.nnodes)

        adj = (adj + adj.T)/2
        adj = torch.sigmoid(adj)
        adj = adj - torch.diag(torch.diag(adj, 0))
        return adj

    @torch.no_grad()#loss不对此项进行梯度下降
    def inference(self, x):
        # self.eval()
        adj_syn = self.forward(x, inference=True)
        return adj_syn

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)

