import numpy as np
import pandas as pd
import random
import time
import argparse
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import time
import scipy
import torch.nn as nn
import math
import torch.optim as optim
import deeprobust.graph.utils as utils
import torch_sparse
import torch_geometric.transforms as T
import os 
import gc
import matplotlib.pyplot as plt

from deeprobust.graph.data import Dataset
from gcond_agent_transduct import GCond
from utils import *
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import match_loss, regularization, row_normalize_tensor, loss_fn_kd
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_sparse import SparseTensor
from sklearn.manifold import TSNE
from copy import deepcopy
from tqdm import tqdm
from torch_scatter import scatter_add

from models.basicgnn import GCN as GCN_PYG, GIN as GIN_PYG, SGC as SGC_PYG, GraphSAGE as SAGE_PYG, JKNet as JKNet_PYG
from models.mlp import MLP
from models.gcn import GCN
from models.gin import GIN
from models.monet import MoNet
from models.sgc_multi import SGC as SGC1
from models.myappnp1 import APPNP1 as APPNP
from models.mygraphsage import GraphSage as GraphSage
from models.parametrized_adj import PGE
from models.deepgcn import DeeperGCN


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=0.05)
parser.add_argument('--lr_feat', type=float, default=0.05)
parser.add_argument('--lr_model', type=float, default=0.01)#GCN:0.01 SGC:0.01 GIN:0.02 SAGE:0.01
parser.add_argument('--weight_decay', type=float, default=0.0)#L2
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)#
parser.add_argument('--reduction_rate', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=36000)
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--teacher_model_loop', type=int, default=500)
parser.add_argument('--syn_loop', type=int, default=1500)
parser.add_argument('--student_model_loop', type=int, default=500)#GCN:2000 SGC:500 GIN:10000 SAGE:5000 JKNET:500
parser.add_argument('--threshold', type=float, default=0.1, help='adj threshold.')
parser.add_argument('--kl_t', type=float, default=2, help='distillation temperature.')#GCN:2 SGC:0.5(2差0.14) GIN:2 SAGE:2
parser.add_argument('--alpha', type=float, default=0.05, help='distillation term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--teacher_model', type=str, default='GCN')
parser.add_argument('--model', type=str, default='SGC')
parser.add_argument('--kernel', type=str, default='linear')
parser.add_argument('--pyg', type=bool, default=True)

args = parser.parse_args()
print(args)

device='cuda'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
torch.cuda.set_device(args.gpu_id)

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

    return labels_syn, num_class_dict


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


def get_cos_sim(feature1,feature2):
    num = torch.dot(feature1, feature2)  # 向量点乘 0.04s
    denom = torch.linalg.norm(feature1) * torch.linalg.norm(feature2)  # 求模长的乘积 0.008s
    return num / denom if denom != 0 else 0


#训练大图
def train_main_trajector():
    start = time.perf_counter()
    if args.teacher_model=='GCN':#每个模型有自己的一套参数设置，teacher和student一样
        model = GCN_PYG(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=0.5, nlayers=args.nlayers, norm='BatchNorm').to(device)
    elif args.teacher_model=='SGC':
        model = SGC_PYG(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=0, nlayers=1, norm='BatchNorm').to(device)
    else:
        model=SAGE_PYG(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=0.5, nlayers=args.nlayers, norm='BatchNorm').to(device)   
    optimizer_origin=torch.optim.Adam(model.parameters(), lr=args.lr_feat)

    from collections import Counter
    counter = Counter(data.labels_train)#每个class进行数数量统计 字典
    if args.nlayers == 1:
        sizes = [15]
    elif args.nlayers == 2:
        sizes = [10, 5]
    elif args.nlayers == 3:
        sizes = [15, 10, 5]
    elif args.nlayers == 4:
        sizes = [15, 10, 5, 5]
    else:
        sizes = [15, 10, 5, 5, 5]

    train_loader=NeighborSampler(adj,#返回的是一个batch的loader，里面可能有很多个batch
            node_idx=torch.LongTensor(data.idx_train),
            sizes=sizes, 
            batch_size=args.batch_size,#越小越久
            num_workers=12, 
            return_e_id=False,
            num_nodes=len(data.labels_full),
            shuffle=False
        )
    
    best_val=0
    best_test=0
    for it in range(args.teacher_model_loop+1):
        #whole graph
        model.train()
        optimizer_origin.zero_grad()
        output = model.forward(feat,edge_index)#选择返回gpu or cpu
        loss = F.nll_loss(output[data.idx_train], labels_train)
        loss.backward()
        optimizer_origin.step()

        #subgraph 
        # model.train()
        # loss=torch.tensor(0.0).to(device)
        # start = time.perf_counter()
        # for batch_size, n_id, adjs in train_loader:
        #     if args.nlayers == 1:
        #         adjs = [adjs]
        #     adjs = [adj.to(device) for adj in adjs]

        #     optimizer_origin.zero_grad()
        #     output = model.forward_sampler(feat[n_id].to(device), adjs)
        #     loss = F.nll_loss(output, labels[n_id[:batch_size]])
        #     loss.backward()
        #     optimizer_origin.step()
        # end = time.perf_counter()
        # print('Epoch',it,'用时:',end-start, '秒')

        if(it%50==0):
            output = model.predict(feat,edge_index)
            loss_val = F.nll_loss(output[data.idx_val], labels_val)
            acc_val = utils.accuracy(output[data.idx_val], labels_val)
            print('Epoch {}'.format(it),"Val set results:","loss= {:.4f}".format(loss_val.item()),"accuracy= {:.4f}".format(acc_val.item()))
            loss_test = F.nll_loss(output[data.idx_test], labels_test)
            acc_test = utils.accuracy(output[data.idx_test], labels_test)
            print('Epoch {}'.format(it),"Test set results:","loss= {:.4f}".format(loss_test.item()),"accuracy= {:.4f}".format(acc_test.item()))

            if(acc_val>best_val):
                best_val=acc_val
                best_test=acc_test
                torch.save(model.state_dict(), f'/home/disk3/xzb/GCond/saved_model/pyg_{args.teacher_model}_{args.dataset}_{args.seed}_best.pt')

    end = time.perf_counter()
    print("Best Test Acc:",best_test)
    print('大图训练用时:',round(end-start), '秒')
    return


def train_syn():
    start = time.perf_counter()
    output = teacher_model.predict(feat,adj,syn=False)
    acc_test = utils.accuracy(output[data.idx_test], labels_test)
    print("Teacher model test set results:","accuracy= {:.4f}".format(acc_test.item()))

    optimizer_feat = optim.Adam([feat_syn], lr=args.lr_feat)
    optimizer_pge = optim.Adam(pge.parameters(), lr=args.lr_adj)
    best_train_val=0
    best_test=0

    # edge = pd.read_csv('/home/disk3/xzb/GCond/dataset/'+'_'.join(args.dataset.split('-'))+'/raw/edge.csv.gz')
    # src = np.array(edge.iloc[:, 0])
    # dst = np.array(edge.iloc[:, 1])
    # feat_src=feat[src]
    # feat_dist=feat[dst]

    #alignment
    feat_mean=[]
    feat_std=[]
    coeff=[]
    for c in range(data.nclass):
        index=torch.where(labels==c)
        coeff.append(num_class_dict[c] / max(num_class_dict.values()))
        feat_mean.append(feat[index].mean(dim=0))
        feat_std.append(feat[c].std(dim=0))

    #edge similarity
    #gussian
    # loss_fn=nn.MSELoss()
    # euclidean=torch.sum(torch.pow(feat_src-feat_dist,2),dim=1)
    # kernel=torch.exp(-0.5*euclidean)
    # kernel_np=edge_kernel.detach().cpu().numpy()
    # torch.save(gussian_kernel, f'/home/disk3/xzb/GCond/saved_ours/kernel_{args.dataset}.pt')
    #linear
    # kernel=torch.empty(len(feat_src))
    # for i in range(len(feat_src)):
    #     kernel[i]=torch.dot(feat_src[i],feat_dist[i])
    # torch.save(linear_kernel, f'/home/disk3/xzb/GCond/saved_ours/kernel_{args.dataset}.pt')
    #poly
    # kernel=torch.empty(len(feat_src))
    # for i in range(len(feat_src)):
    #     kernel[i]=torch.dot(feat_src[i],feat_dist[i])
    # torch.save(poly_kernel, f'/home/disk3/xzb/GCond/saved_ours/kernel_{args.dataset}.pt')
    # del kernel
    # gc.collect()
    # kernel=torch.load(f'/home/disk3/xzb/GCond/saved_ours/kernel_{args.dataset}.pt')
    # kernel_mean=torch.mean().to(device)
    kernel_mean=torch.tensor(32.7496).to(device)#linear32.7496 poly2270.93

    #归一矩阵
    # edge_index_norm, edge_weight_norm=utils.gcn_norm(edge_index, edge_weight, len(labels))
    # deg = scatter_add(edge_weight_norm, edge_index_norm[0], dim=0, dim_size=len(labels))
    deg_mean=torch.tensor(0.8162).to(device)

    for i in range(args.syn_loop+1):#训练adj  通过相似度、pge等
        teacher_model.train()#必须要train eval是没有bn+dropout的，eval只适用于输入值和训练该模型的输入值分布相似时，但该情况的feat_syn和adj明显不是
        adj_syn=pge(feat_syn).to(device)
        edge_index_syn=torch.nonzero(adj_syn).T
        edge_weight_syn= adj_syn[edge_index_syn[0], edge_index_syn[1]]
        
        optimizer_pge.zero_grad()
        optimizer_feat.zero_grad()
        output_syn = teacher_model.forward(feat_syn, edge_index_syn, edge_weight=edge_weight_syn, syn=True)
        hard_loss = F.nll_loss(output_syn, labels_syn)

        #gradient loss
        gw=torch.autograd.grad(hard_loss, list(teacher_model.parameters()), retain_graph=True, create_graph=True)
        gw_loss=torch.tensor(0.0).to(device)
        for para in gw:
            loss_fn=nn.MSELoss()
            gw_loss+=loss_fn(para,torch.zeros_like(para))

        #align_loss
        aligning_loss=torch.tensor(0.0).to(device)#由于是单位矩阵，所以假设相同class的feat_syn会趋向于这个class的平均值
        loss_fn=nn.MSELoss()
        for c in range(data.nclass):
            index=torch.where(labels_syn==c)
            feat_mean_loss=coeff[c]*loss_fn(feat_mean[c],feat_syn[index].mean(dim=0))
            feat_std_loss=coeff[c]*loss_fn(feat_std[c],feat_syn[index].std(dim=0))
            if feat_syn[index].shape[0]!=1:
                aligning_loss+=(feat_mean_loss+2*feat_std_loss)
            else:
                aligning_loss+=(feat_mean_loss)

        #struct loss
        kernel_syn=torch.mm(feat_syn,feat_syn.T)
        kernel_mean_syn=torch.sum(torch.mul(adj_syn,kernel_syn))/n
        edge_simlarity_loss=torch.abs(kernel_mean_syn-kernel_mean)
        
        #matrix norm loss
        edge_index_norm_syn, edge_weight_norm_syn=utils.gcn_norm(edge_index_syn, edge_weight_syn, n)
        deg_syn = scatter_add(edge_weight_norm_syn, edge_index_norm_syn[0], dim=0, dim_size=n)
        deg_mean_syn=torch.mean(deg_syn).to(device)
        matrix_norm_loss=torch.abs(deg_mean_syn-deg_mean)

        loss=hard_loss+0.5*aligning_loss+0.05*edge_simlarity_loss+matrix_norm_loss+10*gw_loss 
        loss.backward()
        if i%50<10:
            optimizer_pge.step()
        else:
            optimizer_feat.step()

        if i>=500 and i%100==0:#不同i下acc的不同可能一大部分是初始化的原因，图可能是差不多的
            #用此时小图训练模型测试结果
            model.initialize()
            adj_syn=pge(feat_syn).to(device)
            adj_syn-=torch.tensor(args.threshold).to(device)
            adj_syn=F.relu(adj_syn).detach()
            adj_syn.requires_grad=False
            edge_index_syn=torch.nonzero(adj_syn).T
            edge_weight_syn= adj_syn[edge_index_syn[0], edge_index_syn[1]]

            teacher_output_syn = teacher_model.predict(feat_syn, edge_index_syn, edge_weight=edge_weight_syn, syn=True)
            acc = utils.accuracy(teacher_output_syn, labels_syn)
            print('Epoch {}'.format(i),"Teacher on syn results:","accuracy= {:.4f}".format(acc.item()))

            for j in range(args.student_model_loop+1):
                model.train()
                optimizer.zero_grad()
                output_syn = model.forward(feat_syn, edge_index_syn, edge_weight=edge_weight_syn, syn=True)
                loss=torch.tensor(0.0).to(device)
                soft_loss = kl_div(output_syn,teacher_output_syn)
                hard_loss=F.nll_loss(output_syn, labels_syn)
                loss=soft_loss+args.alpha*hard_loss#GCN+GCN/SGC:0.05 10 0.05 GCN/APPNP:
                loss.backward()
                optimizer.step()

                if j%100==0:
                    output = model.predict(feat,edge_index,edge_weight=edge_weight, syn=False)
                    acc_train = utils.accuracy(output[data.idx_train], labels_train)
                    print("Train set results:","accuracy= {:.4f}".format(acc_train.item()))
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)
                    print('Epoch {}'.format(j),"Validation set results:","accuracy= {:.4f}".format(acc_val.item()))
                    acc_test = utils.accuracy(output[data.idx_test], labels_test)
                    print('Epoch {}'.format(j),"Test set results:","accuracy= {:.4f}".format(acc_test.item()))

                    if(acc_train+acc_val>best_train_val):
                        best_train_val=acc_train+acc_val
                        best_test=acc_test
                        torch.save(feat_syn, f'/home/disk3/xzb/GCond/saved_ours/feat_pyg_{args.dataset}_{args.teacher_model}_{args.model}_{args.reduction_rate}_{args.seed}.pt')
                        torch.save(pge.state_dict(), f'/home/disk3/xzb/GCond/saved_ours/pge_pyg_{args.dataset}_{args.teacher_model}_{args.model}_{args.reduction_rate}_{args.seed}.pt')
                        torch.save(model.state_dict(), f'/home/disk3/xzb/GCond/saved_model/model_pyg_{args.dataset}_{args.teacher_model}_{args.model}_{args.reduction_rate}_{args.seed}.pt')

    end = time.perf_counter()
    print('训练小图用时:',round(end-start), '秒')
    print("最优test acc:",best_test)


if __name__ == '__main__':
    data_full = get_dataset(args.dataset, args.normalize_features)#get a Pyg2Dpr class, contains all index, adj, labels, features
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)#transductive to inductive 同时实现了neighborsampler
    feat=data.feat_full
    #先不放在gpu 数据可能很大
    feat=torch.FloatTensor(feat).to(device)#全图训练用cuda subgraph用cpu
    adj,_,labels=utils.to_tensor(data.adj_full,data.feat_full,data.labels_full,device=device)
    labels_train=labels[data.idx_train]
    labels_val=labels[data.idx_val]
    labels_test=labels[data.idx_test]
    edge_index=adj._indices()
    edge_weight=adj._values()
    # adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
    #         value=adj._values(), sparse_sizes=adj.size()).t()

    #训练大图
    # print("训练大图！")
    # train_main_trajector()

    labels_syn, num_class_dict = generate_labels_syn(data)
    labels_syn=torch.LongTensor(labels_syn).to(device)
    nnodes_syn = len(labels_syn)
    n = nnodes_syn
    d = data.feat_train.shape[1]
    feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))#feat_syn是小图X'，随机生成可训练参数
    feat_syn.data.copy_(torch.randn(feat_syn.size()))
    pge = PGE(nfeat=d, nnodes=n, device=device,args=args).to(device)#X'得到A'的算法,参数是φ
    kl_div = DistillKL(args.kl_t).to(device)

    if args.teacher_model=='GCN':#每个模型有自己的一套参数设置，teacher和student一样
        teacher_model = GCN_PYG(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=0.5, nlayers=args.nlayers, norm='BatchNorm', cached=False).to(device)
    elif args.teacher_model=='SGC':
        teacher_model = SGC_PYG(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=0, nlayers=1, norm='BatchNorm', cached=False).to(device)
    else:
        teacher_model=SAGE_PYG(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=0.5, nlayers=args.nlayers, norm='BatchNorm', cached=False).to(device)   
    teacher_model.load_state_dict(torch.load(f'/home/disk3/xzb/GCond/saved_model/pyg_{args.teacher_model}_{args.dataset}_{args.seed}_best.pt'))

    if args.model=='GCN':
        model = GCN_PYG(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=0, nlayers=args.nlayers, norm='BatchNorm', cached=True).to(device)
    elif args.model=='SGC':
        model = SGC_PYG(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=0, nlayers=1, norm='BatchNorm', cached=True).to(device)
    elif args.model=='SAGE':
        model = SAGE_PYG(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=0, nlayers=args.nlayers, norm='BatchNorm', cached=True).to(device)   
    elif args.model=='GIN':
        model = GIN_PYG(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=0, nlayers=args.nlayers, norm='BatchNorm', cached=True).to(device)
    elif args.model=='MoNet':
        model=MoNet(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, nlayers=args.nlayers, dropout=0, kernel=3, dim=1, with_bias=True, norm=True, device=device).to(device)
    else:
        model=JKNet_PYG(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=0, nlayers=args.nlayers, norm='BatchNorm', cached=True, jk='cat').to(device)
    model.initialize()#对参数进行初始化
    optimizer=optim.Adam(model.parameters(), lr=args.lr_model)
    
    #训练小图
    # print("训练小图！")
    # train_syn()

    #小图训练模型
    print("小图训练模型！")
    start = time.perf_counter()
    best_train_val=0
    best_test=0
    feat_syn=torch.load(f'/home/disk3/xzb/GCond/saved_ours/feat_pyg_{args.dataset}_{args.teacher_model}_GCN_{args.reduction_rate}_{args.seed}.pt').to(device)
    pge.load_state_dict(torch.load(f'/home/disk3/xzb/GCond/saved_ours/pge_pyg_{args.dataset}_{args.teacher_model}_GCN_{args.reduction_rate}_{args.seed}.pt'))
    
    adj_syn=pge(feat_syn).to(device)
    adj_syn-=torch.tensor(args.threshold).to(device)
    adj_syn=F.relu(adj_syn).detach()
    adj_syn.requires_grad=False
    edge_index_syn=torch.nonzero(adj_syn).T#为什么不是先行后列 因为propagate的时候信息是从第一行到第二行的，比如a[0][100]的权重是1，第一行是0，第二行是100，那么代表着0传递到100的信息的权重是1，但按照行归一的话事实应该反过来
    edge_weight_syn= adj_syn[edge_index_syn[0], edge_index_syn[1]]
    teacher_output_syn=teacher_model.predict(feat_syn, edge_index_syn, edge_weight=edge_weight_syn, syn=True)

    # del adj,teacher_model,pge
    # gc.collect()

    for j in range(args.student_model_loop+1):
        model.train()
        optimizer.zero_grad()
        output_syn = model.forward(feat_syn, edge_index_syn, edge_weight=edge_weight_syn, syn=True)
        loss=torch.tensor(0.0).to(device)
        soft_loss = kl_div(output_syn,teacher_output_syn)
        hard_loss=F.nll_loss(output_syn, labels_syn)
        loss=soft_loss+args.alpha*hard_loss#GCN+GCN/SGC/APPNP:0.05 GCN/APPNP:
        loss.backward()
        optimizer.step()
        
        if j%10==0:
            output = model.predict(feat,edge_index,edge_weight=edge_weight, syn=False)
            print("epoch",j)
            acc_train = utils.accuracy(output[data.idx_train], labels_train)
            print("Train set results:","accuracy= {:.4f}".format(acc_train.item()))
            acc_val = utils.accuracy(output[data.idx_val], labels_val)
            print("Validation set results:","accuracy= {:.4f}".format(acc_val.item()))
            acc_test = utils.accuracy(output[data.idx_test], labels_test)
            print("Test set results:","accuracy= {:.4f}".format(acc_test.item()))
            if(acc_train+acc_val>best_train_val):
                best_train_val=acc_train+acc_val
                best_test=acc_test
                torch.save(model.state_dict(), f'/home/disk3/xzb/GCond/saved_model/pyg_model_{args.dataset}_{args.teacher_model}_{args.model}_{args.reduction_rate}_{args.seed}.pt')

    end = time.perf_counter()
    print('小图模型训练用时:',round(end-start), '秒')
    print("Best Test Acc:",best_test)
    