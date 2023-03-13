import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv import (
    MessagePassing,
    GatedGraphConv
)
from torch_geometric.nn.models import MLP
from models.gated_gcn_layer import ResGatedGCNLayer
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)

from torch_scatter import scatter, scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    

# class GatedGNN(torch.nn.Module):
#     def __init__(
#         self,
#         nfeat: int,
#         nhid: int,
#         nlayers: int,
#         nclass: Optional[int] = None,
#         dropout: float = 0.0,
#         act: Union[str, Callable, None] = "relu",
#         act_first: bool = False,
#         act_kwargs: Optional[Dict[str, Any]] = None,
#         norm: Union[str, Callable, None] = None,
#         norm_kwargs: Optional[Dict[str, Any]] = None,
#         jk: Optional[str] = None,
#         cached: bool = True,
#         **kwargs,
#     ):
#         super().__init__()

#         self.nfeat = nfeat
#         self.nhid = nhid
#         self.nlayers = nlayers

#         self.dropout = dropout
#         self.act = activation_resolver(act, **(act_kwargs or {}))
#         self.jk_mode = jk
#         self.act_first = act_first
#         self.norm = norm if isinstance(norm, str) else None
#         self.norm_kwargs = norm_kwargs
#         self.cached=cached
#         if nclass is not None:
#             self.nclass = nclass
#         else:
#             self.nclass = nhid

#         self.embedding = Linear(nfeat, nhid, bias=True)
#         self.convs = ModuleList()
#         self.convs.append(self.init_conv(nhid, nhid, **kwargs))
#         self.MLP_layer = Linear(nhid, nclass, bias=True)#输出

#         self.norms = None
#         if norm is not None:
#             norm_layer = normalization_resolver(
#                 norm,
#                 nhid,
#                 **(norm_kwargs or {}),
#             )
#             self.norms = ModuleList()
#             self.norms.append(copy.deepcopy(norm_layer))
#             self.norms.append(copy.deepcopy(norm_layer))

#     supports_edge_weight = True
#     supports_edge_attr = False

#     def init_conv(self, nhid: Union[int, Tuple[int, int]],
#                   nclass: int, **kwargs) -> MessagePassing:
#         return GatedGraphConv(out_channels=nhid, num_layers=self.nlayers, normalize=True, add_self_loops=True, cached=self.cached, **kwargs)

#     def initialize(self):
#         r"""Resets all learnable parameters of the module."""
#         self.convs[0].reset_parameters()
#         self.norms[0].reset_parameters()
#         self.norms[1].reset_parameters()
#         self.embedding.reset_parameters()
#         self.MLP_layer.reset_parameters()

#     def forward(
#         self,
#         x: Tensor,
#         edge_index: Adj,
#         *,
#         edge_weight: OptTensor = None,
#         edge_attr: OptTensor = None,
#         syn: bool = False
#     ) -> Tensor:
#         r"""
#         Args:
#             x (torch.Tensor): The input node features.
#             edge_index (torch.Tensor): The edge indices.
#             edge_weight (torch.Tensor, optional): The edge weights (if
#                 supported by the underlying GNN layer). (default: :obj:`None`)
#             edge_attr (torch.Tensor, optional): The edge features (if supported
#                 by the underlying GNN layer). (default: :obj:`None`)
#         """
#         xs: List[Tensor] = []
#         # Tracing the module is not allowed with *args and **kwargs :(
#         # As such, we rely on a static solution to pass optional edge
#         # weights and edge attributes to the module.

#         x=self.embedding(x)
#         x=self.norms[0](x)
#         if self.supports_edge_weight and self.supports_edge_attr:
#             x = self.convs[0](x, edge_index, edge_weight=edge_weight,
#                                 edge_attr=edge_attr, syn=syn)
#         elif self.supports_edge_weight:
#             x = self.convs[0](x, edge_index, edge_weight=edge_weight, syn=syn)
#         elif self.supports_edge_attr:
#             x = self.convs[0](x, edge_index, edge_attr=edge_attr, syn=syn)
#         else:
#             x = self.convs[0](x, edge_index)
#         x=self.norms[1](x)
#         x=self.MLP_layer(x)
#         return F.log_softmax(x,dim=1)
    
#     @torch.no_grad()
#     def predict(
#         self,
#         x: Tensor,
#         edge_index: Adj,
#         *,
#         edge_weight: OptTensor = None,
#         edge_attr: OptTensor = None,
#         syn: bool=False
#     ) -> Tensor:

#         self.eval()
#         return self.forward(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr, syn=syn)

#     @torch.no_grad()
#     def inference(self, loader: NeighborLoader,
#                   device: Optional[torch.device] = None,
#                   progress_bar: bool = False) -> Tensor:
#         r"""Performs layer-wise inference on large-graphs using a
#         :class:`~torch_geometric.loader.NeighborLoader`, where
#         :class:`~torch_geometric.loader.NeighborLoader` should sample the
#         full neighborhood for only one layer.
#         This is an efficient way to compute the output embeddings for all
#         nodes in the graph.
#         Only applicable in case :obj:`jk=None` or `jk='last'`.
#         """
#         assert self.jk_mode is None or self.jk_mode == 'last'
#         assert isinstance(loader, NeighborLoader)
#         assert len(loader.dataset) == loader.data.num_nodes
#         assert len(loader.node_sampler.num_neighbors) == 1
#         assert not self.training
#         # assert not loader.shuffle  # TODO (matthias) does not work :(
#         if progress_bar:
#             pbar = tqdm(total=len(self.convs) * len(loader))
#             pbar.set_description('Inference')

#         x_all = loader.data.x.cpu()
#         loader.data.n_id = torch.arange(x_all.size(0))

#         for i in range(self.nlayers):
#             xs: List[Tensor] = []
#             for batch in loader:
#                 x = x_all[batch.n_id].to(device)
#                 if hasattr(batch, 'adj_t'):
#                     edge_index = batch.adj_t.to(device)
#                 else:
#                     edge_index = batch.edge_index.to(device)
#                 x = self.convs[i](x, edge_index)[:batch.batch_size]
#                 if i == self.nlayers - 1 and self.jk_mode is None:
#                     xs.append(x.cpu())
#                     if progress_bar:
#                         pbar.update(1)
#                     continue
#                 if self.act is not None and self.act_first:
#                     x = self.act(x)
#                 if self.norms is not None:
#                     x = self.norms[i](x)
#                 if self.act is not None and not self.act_first:
#                     x = self.act(x)
#                 if i == self.nlayers - 1 and hasattr(self, 'lin'):
#                     x = self.lin(x)
#                 xs.append(x.cpu())
#                 if progress_bar:
#                     pbar.update(1)
#             x_all = torch.cat(xs, dim=0)
#         if progress_bar:
#             pbar.close()
#         del loader.data.n_id

#         return x_all

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.nfeat}, '
#                 f'{self.nclass}, nlayers={self.nlayers})')
    

"""
    Gated Graph Sequence Neural Networks
    An Experimental Study of Neural Networks for Variable Graphs 
    Li Y, Tarlow D, Brockschmidt M, et al. Gated graph sequence neural networks[J]. arXiv preprint arXiv:1511.05493, 2015.
    https://arxiv.org/abs/1511.05493
    Note that the pyg and dgl of the gatedGCN are different models.
"""
class GatedGNN(nn.Module):

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nlayers: int,
        nclass: Optional[int] = None,
        dropout: float = 0.0,
        norm: Union[str, Callable, None] = None,
        cached: bool = True,
        **kwargs,
    ):
        super().__init__()

        in_dim_node = nfeat
        hidden_dim = nhid
        n_classes = nclass
        self.dropout = dropout
        self.n_layers = nlayers
        self.batch_norm = norm
        self.residual = True
        self.n_classes = n_classes

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)      
        self.layers = nn.ModuleList([GatedGraphConv(hidden_dim, nlayers, aggr = 'add')])
        if self.batch_norm:
            self.normlayers = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
        self.MLP_layer = nn.Linear(hidden_dim, n_classes, bias=True)

    def initialize(self):
        r"""Resets all learnable parameters of the module."""
        if self.batch_norm:
            for norm in self.normlayers:
                norm.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.embedding_h.reset_parameters()
        self.MLP_layer.reset_parameters()

    def forward(self, h, edge_index, edge_weight, syn=False):
        h = self.embedding_h(h)
        for conv in self.layers:
            h_in = h
            h = conv(h, edge_index, edge_weight)
        if self.batch_norm:
            h = self.normlayers[0](h)
        if self.residual:
            h = h_in + h  # residual connection
        h = F.dropout(h, self.dropout, training=self.training)
        # output
        h_out = self.MLP_layer(h)

        return h_out

    @torch.no_grad()
    def predict(self, h, edge_index, edge_weight, syn=False):
        self.eval()
        return self.forward(h, edge_index, edge_weight, syn)

    
"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""

class ResGatedGCN(nn.Module):

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nlayers: int,
        nclass: Optional[int] = None,
        dropout: float = 0.0,
        norm: Union[str, Callable, None] = None,
        cached: bool = True,
        **kwargs,
    ):
        super().__init__()

        in_dim_node = nfeat
        in_dim_edge= 1
        hidden_dim = nhid
        n_classes = nclass
        num_bond_type=3
        self.dropout = dropout
        self.n_layers = nlayers
        self.batch_norm = norm
        self.residual = True
        self.n_classes = n_classes
        self.pos_enc = False
        self.edge_feat = False

        if self.pos_enc:
            pos_enc_dim = 1
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)       # node feat is an integer
        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)  # edge feat is a float

        self.layers = nn.ModuleList([ResGatedGCNLayer(hidden_dim, hidden_dim, self.dropout,
                                                   self.batch_norm, self.residual) for _ in range(self.n_layers)])
        if self.batch_norm:
            self.normlayers_h = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(self.n_layers)])
            self.normlayers_e = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(self.n_layers)])
        # self.MLP_layer = MLPReadout(hidden_dim, n_classes)
        self.MLP_layer = nn.Linear(hidden_dim, n_classes, bias=True)

    def initialize(self):
        r"""Resets all learnable parameters of the module."""
        if self.batch_norm:
            for norm in self.normlayers_h:
                norm.reset_parameters()
            for norm in self.normlayers_e:
                norm.reset_parameters()
        self.embedding_h.reset_parameters()
        self.embedding_e.reset_parameters()
        self.MLP_layer.reset_parameters()

    def forward(self, h, edge_index, edge_weight, h_pos_enc=None, syn=True):
        #TODO:还没有归一化
        e=edge_weight.reshape(-1,1)
        
        # input embedding
        h = self.embedding_h(h)
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
            h = h + h_pos_enc
        e = self.embedding_e(e)
        # res gated convnets
        for i in range(self.n_layers):
            h_in = h
            e_in = e
            h, e = self.layers[i](h, edge_index, e)
            if self.batch_norm:
                h = self.normlayers_h[i](h)
                e = self.normlayers_e[i](e)  # batch normalization
            if self.residual:
                h = h_in + h  # residual connection
                e = e_in + e
            h = F.dropout(h, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)
        # output
        h_out = self.MLP_layer(h)

        return h_out

    @torch.no_grad()
    def predict(self, h, edge_index, edge_weight, h_pos_enc=None, syn=False):
        self.eval()
        return self.forward(h, edge_index, edge_weight, h_pos_enc, syn)