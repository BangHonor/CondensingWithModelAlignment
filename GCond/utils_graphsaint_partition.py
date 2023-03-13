import scipy.sparse as sp
import numpy as np
import sys
import json
import os
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import InMemoryDataset, Data
import torch
from itertools import repeat
from torch_geometric.data import NeighborSampler
import pymetis

class DataGraphSAINT:#input a dataset and return full data
    '''datasets used in GraphSAINT paper'''

    def __init__(self, dataset,idx_train,idx_test,idx_val, **kwargs):#**kwargs 允许你将不定长度的键值对, 作为参数传递给一个函数。
        dataset_str='data/'+dataset+'/'
        adj_full = sp.load_npz('/home/xzb/GCond/'+dataset_str+'adj_full.npz')#easier to partition
        self.nnodes = adj_full.shape[0]
        if dataset == 'ogbn-arxiv':
            adj_full = adj_full + adj_full.T
            adj_full[adj_full > 1] = 1
        
        if 'label_rate' in kwargs:#when input is uncertain
            label_rate = kwargs['label_rate']
            if label_rate < 1:
                idx_train = idx_train[:int(label_rate*len(idx_train))]

        self.adj_train = adj_full[np.ix_(idx_train, idx_train)]
        self.adj_val = adj_full[np.ix_(idx_val, idx_val)]
        self.adj_test = adj_full[np.ix_(idx_test, idx_test)]

        idx_full=idx_train+idx_val+idx_test#按照train val test的顺序来
        #adj_full=adj_full[np.ix_(idx_full, idx_full)]#adj of one partition

        feat = np.load('/home/xzb/GCond/'+dataset_str+'feats.npy')
        # ---- normalize feat ----
        feat_train = feat[idx_train]
        # scaler = StandardScaler()
        # scaler.fit(feat_train)
        # feat = scaler.transform(feat)

        self.feat_train = feat[idx_train]
        self.feat_val = feat[idx_val]
        self.feat_test = feat[idx_test]

        self.adj_full = adj_full[np.ix_(idx_full, idx_full)]#adj of one partition
        self.feat_full = feat[idx_full]

        #change whole graph index into partitioned graph index
        self.idx_train = np.arange(len(idx_train))
        self.idx_val = np.arange(len(idx_train)+len(idx_val))
        self.idx_test = np.arange(len(idx_train)+len(idx_val),len(idx_full))
        self.samplers = None
        feat=feat[idx_full]
        
        class_map_full = json.load(open('/home/xzb/GCond/'+dataset_str + 'class_map.json','r'))#dict 每一个节点的分类
        labels = self.process_labels(class_map_full)#这里的labels应该是全部的数据 没有分割
        self.nnodes = self.adj_full.shape[0]
        self.labels_train = labels[idx_train]
        self.labels_val = labels[idx_val]
        self.labels_test = labels[idx_test]
        self.labels_full = []
        self.labels_full.extend(self.labels_train)
        self.labels_full.extend(self.labels_val)
        self.labels_full.extend(self.labels_test)
        #顺序应该按照train val test来

        #统计一共有几类
        class_part=[]
        for i in self.labels_train:
            if i in class_part:
                continue;
            else:
                class_part.append(i)
        class_part.sort()
        #print("现有的分类:",class_part)

        #发现一个part并不包括所有的class可能,所以要记录有哪些class，对应的小图也只能有这些class
        self.class_part=class_part

        self.data_full = GraphData(self.adj_full, self.feat_full, self.labels_full, self.idx_train,self.idx_val, self.idx_test)
        self.class_dict = None
        self.class_dict2 = None

    def process_labels(self, class_map):
        """
        setup vertex property map for output classests
        """
        num_vertices = self.nnodes
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            self.nclass = num_classes
            class_arr = np.zeros((num_vertices, num_classes))
            for k,v in class_map.items():
                class_arr[int(k)] = v
        else:
            class_arr = np.zeros(num_vertices, dtype=np.int)
            for k, v in class_map.items():
                class_arr[int(k)] = v
            class_arr = class_arr - class_arr.min()
            self.nclass = max(class_arr) + 1
        #print("整幅图有的分类数为:",self.nclass)
        return class_arr

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.labels_train == i)#record each class's node index
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]#mess idx's order

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        if args.nlayers == 1:
            sizes = [30]
        if args.nlayers == 2:
            if args.dataset in ['reddit', 'flickr']:
                if args.option == 0:
                    sizes = [15, 8]
                if args.option == 1:
                    sizes = [20, 10]
                if args.option == 2:
                    sizes = [25, 10]
            else:
                sizes = [10, 5]

        if self.class_dict2 is None:
            #print(sizes)
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx_train = np.array(self.idx_train)
                    idx = idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx#each class's index

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                if len(node_idx) == 0:
                    continue

                self.samplers.append(NeighborSampler(adj,
                                    node_idx=node_idx,
                                    sizes=sizes, batch_size=num,
                                    num_workers=8, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[-1].sample(batch)
        return out


class GraphData:

    def __init__(self, adj, features, labels, idx_train, idx_val, idx_test):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test


class Data2Pyg:

    def __init__(self, data, device='cuda', transform=None, **kwargs):
        self.data_train = Dpr2Pyg(data.data_train, transform=transform)[0].to(device)
        self.data_val = Dpr2Pyg(data.data_val, transform=transform)[0].to(device)
        self.data_test = Dpr2Pyg(data.data_test, transform=transform)[0].to(device)
        self.nclass = data.nclass
        self.nfeat = data.nfeat
        self.class_dict = None

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.data_train.y == i).cpu().numpy()
        idx = np.arange(len(self.data_train.y))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]


class Dpr2Pyg(InMemoryDataset):

    def __init__(self, dpr_data, transform=None, **kwargs):
        root = 'data/' # dummy root; does not mean anything
        self.dpr_data = dpr_data
        super(Dpr2Pyg, self).__init__(root, transform)
        pyg_data = self.process()
        self.data, self.slices = self.collate([pyg_data])
        self.transform = transform

    def process(self):
        dpr_data = self.dpr_data
        edge_index = torch.LongTensor(dpr_data.adj.nonzero())
        # by default, the features in pyg data is dense
        if sp.issparse(dpr_data.features):
            x = torch.FloatTensor(dpr_data.features.todense()).float()
        else:
            x = torch.FloatTensor(dpr_data.features).float()
        y = torch.LongTensor(dpr_data.labels)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None
        return data


    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        pass


