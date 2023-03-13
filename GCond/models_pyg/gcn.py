import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score
from torch.nn import init
import torch_sparse
from torch_geometric.nn import GCNConv

# class GraphConvolution(Module):
#     """Simple GCN layer, similar to https://github.com/tkipf/pygcn
#     """

#     def __init__(self, in_features, out_features, with_bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))#确定了这些是GCN的参数，在back pro的时候会对齐求导
#         self.bias = Parameter(torch.FloatTensor(out_features))
#         self.weight.requires_grad=True
#         self.bias.requires_grad=True

#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.T.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input, adj):#整体的表达式为：A(HW) out grad=true
#         """ Graph Convolutional Layer forward function
#         """
#         if input.data.is_sparse:
#             support = torch.spmm(input, self.weight)
#         else:
#             support = torch.mm(input, self.weight)#张量相乘
#         if isinstance(adj, torch_sparse.SparseTensor):
#             output = torch_sparse.matmul(adj, support)
#         else:
#             output = torch.sparse.mm(adj, support)
#             # output = torch_sparse.matmul(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output

#     def forward_sparse(self, input, a, u , v):#整体的表达式为：A(HW) out grad=true
#         """ Graph Convolutional Layer forward function
#         """
#         if input.data.is_sparse:
#             support = torch.spmm(input, self.weight)
#         else:
#             support = torch.mm(input, self.weight)#张量相乘
#         add=torch.zeros_like(support,requires_grad=True)
#         output=torch.zeros_like(support,requires_grad=True)
#         for i in range(1000):
#             for j in range(support.shape[1]):
#                 add.data[u[i]][j]=a.data[0][i]*support.data[v[i]][j]
#         output=output+add
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

    def forward_logits(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer.forward(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x
        
    def forward_sparse(self, x, a, u, v):
        for ix, layer in enumerate(self.layers):
            x = layer.forward_sparse(x, a, u, v)#layer是一个GraphConvolution layer
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler(self, x, adjs):#子图forward
        # for ix, layer in enumerate(self.layers):
        for ix, (adj, _, size) in enumerate(adjs):
            x = self.layers[ix](x, adj)#每一层采样的节点不一样，邻接矩阵也不一样
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler_syn(self, x, adjs):
        for ix, (adj) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)


    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def fit_with_val(self, features, adj, labels, data, train_iters=200, initialize=True, verbose=False, normalize=True, patience=None, noval=False, **kwargs):
        #features, adj, labels are from syn!!!!!!!!!!!!!!!!!
        '''data: full data class'''
        if initialize:
            self.initialize()#renew gcn

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)#from syn
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm
        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        if noval:
            self._train_with_val(labels, data, train_iters, verbose, adj_val=True)#pass syn labels
        else:
            self._train_with_val(labels, data, train_iters, verbose)

    def _train_with_val(self, labels, data, train_iters, verbose, adj_val=False):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val#inductive
        else:
            feat_full, adj_full = data.feat_full, data.adj_full#transductive 
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)#trans: whole big graph
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)#use the whole data, but only the validation labels

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0

        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)#gcn parameters!!!!!!!!!!!

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)#use condensed graph and gcn to fit gcn
            loss_train = self.loss(output, labels)
            loss_train.backward()
            optimizer.step()#fit gcn para

            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            with torch.no_grad():#with torch.no_grad()或者@torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
                self.eval()
                output = self.forward(feat_full, adj_full_norm)#use all validation data into new gcn to get new label

                if adj_val:#induct
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:#transduct
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())#state_dict是在定义了model或optimizer之后pytorch自动生成的,即model此时的参数

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        


    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    #@torch.no_grad()
    def predict(self, features=None, adj=None):#align format and forward
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        # if features is None and adj is None:
        #     return self.forward(self.features, self.adj_norm)
        # else:
        #     if type(adj) is not torch.Tensor:
        #         adj,features = utils.to_tensor(adj,features, device=self.device)

        #     self.features = features
        #     if utils.is_sparse_tensor(adj):
        #         self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        #     else:
        #         self.adj_norm = utils.normalize_adj_tensor(adj)

        return self.forward(features, adj)

    @torch.no_grad()
    def predict_unnorm(self, features=None, adj=None):
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            self.adj_norm = adj
            return self.forward(self.features, self.adj_norm)


    def _train_with_val2(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

def train(model, data, train_idx, optimizer):#transductive
    model.train()

    optimizer.zero_grad()
    out = model(data.x[train_idx], data.adj_t[train_idx,train_idx])
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc