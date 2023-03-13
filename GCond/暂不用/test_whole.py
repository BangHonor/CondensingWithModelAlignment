from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from GCond.暂不用.gcond_agent_induct_partition_Identity import GCond
from utils_graphsaint_partition import DataGraphSAINT
import pymetis
import scipy.sparse as sp
import json
import time
import scipy
from models.gcn import GCN
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import match_loss, regularization, row_normalize_tensor,loss_fn_kd
import deeprobust.graph.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.005)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--option', type=int, default=0)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--label_rate', type=float, default=1)
args = parser.parse_args()
args.dataset='reddit'
args.sgc=1
args.nlayers=2
args.lr_feat=0.1
args.lr_adj=0.1
args.r=0.01
args.reduction_rate=args.r
args.seed=1
args.gpu_id=3
args.epochs=1000
args.inner=1
args.outer=10
args.save=1

torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print(args)

dataset_str='/home/xzb/GCond/data/'+args.dataset+'/'

loaded=np.load(dataset_str+'adj_full.npz')
#loaded is a csr matrix, contains 3 rows(indptr,indices,data), we can directly turn it into adj list

#get sparse format
try:
    matrix_format = loaded['format']
except KeyError as e:
    raise ValueError('The file {} does not contain a sparse matrix.'.format(file)) from e

matrix_format = matrix_format.item()
if not isinstance(matrix_format, str):
    # Play safe with Python 2 vs 3 backward compatibility;
    # files saved with SciPy < 1.0.0 may contain unicode or bytes.
    matrix_format = matrix_format.decode('ascii')
try:
    cls = getattr(scipy.sparse, '{}_matrix'.format(matrix_format))
except AttributeError as e:
    raise ValueError('Unknown matrix format "{}"'.format(matrix_format)) from e

parts=[1]
role = json.load(open(dataset_str+'role.json','r'))
idx_train = role['tr']
idx_test = role['te']
idx_val = role['va']

for part in parts:
    nclass=0
    feat = np.load(dataset_str+'feats.npy')
    feat_trains=[]
    adj_trains=[]
    label_trains=[]
    t_models=[]
    dropout = 0.5 if args.dataset in ['reddit'] else 0
    best_it=[]

    for i in range(part):
        print("第",i,"个part：")
        print("training/val/test的size为",len(idx_train),len(idx_val),len(idx_test))
        data = DataGraphSAINT(args.dataset,idx_train,idx_test,idx_val,label_rate=args.label_rate)
        nclass=data.nclass

        adj_train,feat_train=utils.to_tensor(data.adj_train,data.feat_train,labels=None,device='cuda')
        feat_trains.append(feat_train.detach())
        label_trains.append(torch.LongTensor(data.labels_train).to('cuda'))
        if utils.is_sparse_tensor(adj_train):
            adj_train = utils.normalize_adj_tensor(adj_train, sparse=True)
        else:
            adj_train = utils.normalize_adj_tensor(adj_train)
        adj_trains.append(adj_train.detach())
        start = time.perf_counter()
        agent = GCond(data, args, device='cuda')
        best_it.append(agent.train(i,part))
        end = time.perf_counter()
        print("第i个小图凝聚用时:",round(end-start), '秒')
        t_models.append(GCN(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=2,nclass=nclass, device='cuda').to('cuda'))
        t_models[i].load_state_dict(torch.load(f'/home/xzb/GCond/saved_distillation/GCN_{args.dataset}_{args.reduction_rate}_{args.seed}_{i}_{part}_{best_it[i]}.pt'))
        print("读取GCN Teacher模型成功！")

    #now we get all teachers and student model, let's tam!
    stu_model = GCN(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=2,nclass=nclass, device='cuda').to('cuda')
    stu_model.initialize()
    optimizer = optim.Adam(stu_model.parameters(), lr=0.01, weight_decay=5e-4)
        
    print("开始执行TAM算法！")
    eval_epoch=[20,40,60,80,120,160,200,300,400,500,600,700,800,900,1000]
    data = DataGraphSAINT(args.dataset,idx_train,idx_test,idx_val,label_rate=args.label_rate)
    feat_test = data.feat_test
    labels_test =torch.LongTensor(data.labels_test).to('cuda')
    adj_test=data.adj_test
    start = time.perf_counter()
    for epoch in range(0,eval_epoch[-1]+1):
        loss_soft=torch.tensor(0.0).to('cuda')
        loss_hard=torch.tensor(0.0).to('cuda')
        loss_total=torch.tensor(0.0).to('cuda')
        T=1
        alpha=100
        for i in range(0,part):
            optimizer.zero_grad()
            t_T=t_models[i].forward_T(feat_trains[i],adj_trains[i],T)#所有数据需要在同一个设备上，所以先确保都是cuda上的tensor！
            stu_T=stu_model.forward_T(feat_trains[i],adj_trains[i],T)
            hard_labels=label_trains[i]
            stu_softmax=stu_model.forward(feat_trains[i],adj_trains[i])

            loss_fn=torch.nn.MSELoss(reduction='mean')
            loss=loss_fn(stu_T,t_T)
            loss_soft=loss_soft+loss

            loss_fn=torch.nn.NLLLoss()
            loss=loss_fn(stu_softmax,hard_labels)
            loss_hard=loss_hard+loss

            loss_total=loss_soft+alpha*loss_hard

        loss_total.backward()
        optimizer.step()
        if epoch in eval_epoch:
            print("epoch:",epoch,"test model")
            output=stu_model.predict(feat_test,adj_test)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = utils.accuracy(output, labels_test)
            res = []
            res.append(acc_test.item())
            print("Test set results of amalgamated student model:",
                    "loss= {:.4f}".format(loss_test.item()),
                    "accuracy= {:.4f}".format(acc_test.item()))
    end = time.perf_counter()
    print("知识蒸馏时长:",round(end-start), '秒')
    print("TAM算法执行结束！")
    torch.save(stu_model.state_dict(), f'saved_distillation/GCN_Student_{args.dataset}_{args.reduction_rate}_{args.seed}_{part}.pt') 

    print("蒸馏前：")
    output=t_models[0].predict(feat_test,adj_test)
    loss_test = F.nll_loss(output, labels_test)
    acc_test = utils.accuracy(output, labels_test)
    res = []
    res.append(acc_test.item())
    print("Test set results of teacher model:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))