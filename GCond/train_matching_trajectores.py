from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from gcond_agent_transduct import GCond
import scipy.sparse as sp
import json
import time
import scipy
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import match_loss, regularization, row_normalize_tensor,loss_fn_kd
import deeprobust.graph.utils as utils
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
import os

from utils import match_loss, regularization, row_normalize_tensor
import deeprobust.graph.utils as utils
from copy import deepcopy
from tqdm import tqdm
from models.gcn import GCN
from models.sgc import SGC
from models.sgc_multi import SGC as SGC1
from models.parametrized_adj import PGE
from models.deepgcn import DeeperGCN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='ogbn-products')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=500)#one epoch means update condensed graph once and gcn model for multiple times
parser.add_argument('--nlayers', type=int, default=0)
parser.add_argument('--hidden', type=int, default=256)#columns of w matrix
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)#L2
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)#
parser.add_argument('--reduction_rate', type=float, default=0.005)
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=10)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--model', type=str, default='GCN')


args = parser.parse_args()
print(args)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
torch.cuda.set_device(args.gpu_id)
device='cuda'
# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def generate_labels_syn(data):
    from collections import Counter
    counter = Counter(data.labels_train)#每个class进行数数量统计 字典
    num_class_dict = {}
    n = len(data.labels_train)

    sorted_counter = sorted(counter.items(), key=lambda x:x[1])#对次数进行排序,每一个元素为{class,n}
    sum_ = 0
    labels_syn = []
    syn_class_indices = {}

    for ix, (c, num) in enumerate(sorted_counter):#to make num of labels_syn=counter*redcution_rate 
        if num==0:
            num_class_dict[c]=0
            continue
        num_class_dict[c] = max(int(num * args.reduction_rate), 1)
        syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
        labels_syn += [c] * num_class_dict[c]

    num_class_dict = num_class_dict
    return labels_syn


def get_sub_adj_feat(features, data, args):#获取相邻接点的特征
    idx_selected = []

    from collections import Counter;
    counter = Counter(labels_syn.cpu().numpy())#self.labels_syn类似于{分类1，分类1，分类1，分类2，分类2，分类3}，统计了小图每个分类的数量

    for c in range(data.nclass):
        tmp = data.retrieve_class(c, num=counter[c])#检索小图对应class的节点index
        tmp = list(tmp)
        idx_selected = idx_selected + tmp#每一行都是各自分类选择的节点index
    idx_selected = np.array(idx_selected).reshape(-1)#化为一行
    features = features[data.idx_train][idx_selected]#在train中选部分，因为idx_train是index中最前的一批，所以不会选择到test的index

    # adj_knn = torch.zeros((data.nclass*args.nsamples, data.nclass*args.nsamples)).to(self.device)
    # for i in range(data.nclass):
    #     idx = np.arange(i*args.nsamples, i*args.nsamples+args.nsamples)
    #     adj_knn[np.ix_(idx, idx)] = 1

    from sklearn.metrics.pairwise import cosine_similarity
    # features[features!=0] = 1
    k = 10
    sims = cosine_similarity(features.cpu().numpy())#计算小图中feature的两两距离
    sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0#对角线初始化为0
    for i in range(len(sims)):
        indices_argsort = np.argsort(sims[i])#argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
        sims[i, indices_argsort[: -k]] = 0#除了k个相邻的（余弦最大的） 其他设置为0
    adj_knn = torch.FloatTensor(sims).to(device)
    return features, adj_knn


#训练大图
def train_main_trajector():
    print("训练大图！")
    model = GCN(nfeat=data.feat_train.shape[1], nhid=256, nclass=data.nclass, dropout=0.5, nlayers=3, with_bn=True, device=device).to(device)#把模型加载到CUDA上
    optimizer_origin=torch.optim.Adam(model.parameters(), lr=args.lr_feat)

    best_val=0
    for it in range(501):
        model.train()
        optimizer_origin.zero_grad()
        # output = model.inference(feat,adj)
        output = model.forward(feat,adj)
        loss = F.nll_loss(output[data.idx_train], labels_train)
        loss.backward()
        optimizer_origin.step()
        acc_train = utils.accuracy(output[data.idx_train], labels_train)
        print('Epoch {}'.format(it),"Train set results:","loss= {:.4f}".format(loss.item()),"accuracy= {:.4f}".format(acc_train.item()))
        if(it%100==0):
            model.eval()
            loss_val = F.nll_loss(output[data.idx_train], labels_train)
            acc_val = utils.accuracy(output[data.idx_val], labels_val)
            print('Epoch {}'.format(it),"Val set results:","loss= {:.4f}".format(loss_val.item()),"accuracy= {:.4f}".format(acc_val.item()))
            if(acc_val.item()>best_val):
                best_val=acc_val.item()
                torch.save(model.state_dict(), f'/home/disk3/xzb/GCond/saved_model/{args.model}_{args.dataset}_{args.seed}.pt') 

    model = GCN(nfeat=data.feat_train.shape[1], nhid=256, nclass=data.nclass, dropout=0.5, nlayers=3, with_bn=True, device=device).to(device)#把模型加载到CUDA上
    model.load_state_dict(torch.load(f'/home/disk3/xzb/GCond/saved_model/{args.model}_{args.dataset}_{args.seed}.pt'))
    model.eval()
    output = model.forward(feat,adj)
    loss_test = F.nll_loss(output[data.idx_test], labels_test)
    acc_test = utils.accuracy(output[data.idx_test], labels_test)
    print("Test set results:","loss= {:.4f}".format(loss_test.item()),"accuracy= {:.4f}".format(acc_test.item()))
    #Epoch 500 Val set results: loss= 0.6977 accuracy= 0.7083   Test set results: loss= 0.9219 accuracy= 0.7136
    return


if __name__ == '__main__':
    data_full = get_dataset(args.dataset, args.normalize_features)#get a Pyg2Dpr class, contains all index, adj, labels, features
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)#transductive to inductive 同时实现了neighborsampler
    feat=data.feat_full
    feat=torch.FloatTensor(feat).to(device)
    adj,_,labels=utils.to_tensor(data.adj_full,data.feat_full,data.labels_full,device=device)

    if utils.is_sparse_tensor(adj):
        adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
    else:
        adj_norm = utils.normalize_adj_tensor(adj)
    adj = adj_norm
    adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
            value=adj._values(), sparse_sizes=adj.size()).t()
    labels_train=labels[data.idx_train]
    labels_val=labels[data.idx_val]
    labels_test=labels[data.idx_test]

    #训练大图
    train_main_trajector()

    #训练小图
    print("训练小图！")
    labels_syn = torch.LongTensor(generate_labels_syn(data)).to(device)#得到小图的所有label
    nnodes_syn = len(labels_syn)
    n = nnodes_syn
    d = data.feat_train.shape[1]
    feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))#feat_syn是小图X'，随机生成可训练参数
    feat_syn.data.copy_(torch.randn(feat_syn.size()))
    # feat_sub, adj_sub = get_sub_adj_feat(feat, data, args)#从大图X中选出对应的小图的X'
    # feat_syn.data.copy_(feat_sub)#小图的X'
    pge = PGE(nfeat=d, nnodes=n, device=device,args=args).to(device)#X'得到A'的算法,参数是φ
    model = GCN(nfeat=data.feat_train.shape[1], nhid=256, nclass=data.nclass, dropout=0.5, nlayers=3, with_bn=True, device=device).to(device)#把模型加载到CUDA上
    model.initialize()#对参数进行初始化
    model_parameters = list(model.parameters())
    teacher_model=GCN(nfeat=data.feat_train.shape[1], nhid=256, nclass=data.nclass, dropout=0.5, nlayers=3, with_bn=True, device=device).to(device)#把模型加载到CUDA上
    teacher_model.load_state_dict(torch.load(f'/home/disk3/xzb/GCond/saved_model/{args.model}_{args.dataset}_{args.seed}.pt'))
    parameters=list(teacher_model.parameters())

    optimizer_feat = optim.Adam([feat_syn], lr=args.lr_feat)
    optimizer_pge = optim.Adam(pge.parameters(), lr=args.lr_adj)
    optimizer=optim.Adam(model.parameters(), lr=args.lr_model)
    outer_loop=2000
    inner_loop1=1000
    inner_loop2=1
    best_val=0

    for i in range(outer_loop):#训练x pge
        torch.cuda.empty_cache()  # 释放显存
        # adj_syn=pge(feat_syn)
        # adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
        adj_syn_norm=torch.eye(n,dtype=torch.float32).to(device)
        model.initialize()#对参数进行初始化
        model.load_state_dict(torch.load(f'/home/disk3/xzb/GCond/saved_model/{args.model}_{args.dataset}_{args.seed}.pt'))
        student_params_trajector=[]

        for j in range(inner_loop1+1):#训练模型
            model.train()
            output = model.forward(feat_syn,adj_syn_norm)
            loss = F.nll_loss(output, labels_syn)

            gw=torch.autograd.grad(loss, model_parameters, retain_graph=True, create_graph=True)
            gw=list((_ for _ in gw))
            gw_feat_syn=[]
            for gww in gw:
                gw_feat_syn.append(torch.autograd.grad(gww, [feat_syn], grad_outputs=torch.ones_like(gww), retain_graph=True, create_graph=True))
            model_parameters=[a-args.lr_model*b for a,b in zip(model_parameters,gw)]
            student_params_trajector.append(model_parameters)#记录模型更新的轨迹，backward()才能追溯

            k=0
            for para in model.parameters():
                para.data=model_parameters[k]
                k=k+1
            model_parameters = list(model.parameters())

            # optimizer.zero_grad()#X'梯度设置为0
            # loss=F.nll_loss(output, labels_syn)
            # loss.backward(retain_graph=True, create_graph=True)
            # optimizer.step()
            # student_params.append(model.parameters())

            if(j%100==0):
                output = model.predict(feat_syn,adj_syn_norm)
                loss_train = F.nll_loss(output, labels_syn)
                acc_train = utils.accuracy(output, labels_syn)
                print('Epoch {}'.format(j),"Syn set results:","loss= {:.4f}".format(loss_train.item()),"accuracy= {:.4f}".format(acc_train.item()))

            if(j==inner_loop1):
                output = model.predict(feat,adj)
                acc_val = utils.accuracy(output[data.idx_val], labels_val)
                print("Validation set results:","accuracy= {:.4f}".format(acc_val.item()))
                if(acc_val>best_val):
                    best_val=acc_val
                    torch.save(feat_syn, f'/home/disk3/xzb/GCond/saved_ours/feat_{args.dataset}_{args.model}_{args.reduction_rate}_{args.seed}.pt')
        
        for j in range(1):
            optimizer_feat.zero_grad()#X'梯度设置为0
            optimizer_pge.zero_grad()#φ梯度设置为0
            loss_fn=nn.MSELoss()
            loss=torch.tensor(0.0).to(device)
            for k in range(len(parameters)):
                loss+=loss_fn(parameters[k].requires_grad_(False),student_params_trajector[-1][k])
            gw_feat=torch.autograd.grad(loss, [feat_syn], retain_graph=True, create_graph=True)
            # gw_pge=torch.autograd.grad(loss, list(pge.parameters()), retain_graph=True, create_graph=True)
            loss.backward()
            optimizer_feat.step()

    print("Best Validation Acc:",best_val)
    