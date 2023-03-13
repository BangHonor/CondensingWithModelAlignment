import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dgl.nn.pytorch.softmax import edge_softmax
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F


def loss_fn_kd(logits, logits_t):
    """This is the function of computing the soft target loss by using soft labels

    Args:
        logits (torch.Tensor): predictions of the student
        logits_t (torch.Tensor): logits generated by the teacher

    Returns:
        tuple: a tuple containing the soft target loss and the soft labels
    """

    loss_fn = nn.BCEWithLogitsLoss()

    # generate soft labels from logits
    labels_t = torch.where(logits_t > 0.0, 
                        torch.ones(logits_t.shape).to(logits_t.device), 
                        torch.zeros(logits_t.shape).to(logits_t.device)) 
    loss = loss_fn(logits, labels_t)

    return loss, labels_t


def gen_attrib_norm(graph, attrib):
    """This is the function that performs topological-aware edge gradient normalization, 
       described in the last paragraph of Sect. 4.3 of the paper

    Args:
        graph (DGLGraph): the input graphs containing the topological information
        attrib (torch.Tensor): obtained topological attributions from Eq. 1 of the paper

    Returns:
        torch.Tensor: topological-aware normalized attributions
    """

    device = attrib.device

    nnode = graph.number_of_nodes()
    graph.edata.update({'attrib': attrib})
    graph.ndata.update({'unit_node': torch.ones(nnode,1).to(device)})

    # compute the mean of the topological attributions around each center node
    graph.update_all(fn.u_mul_e('unit_node', 'attrib', 'node_attrib'), fn.mean('node_attrib', 'attrib_mean'))

    # subtract the mean topological attribution
    graph.apply_edges(fn.e_sub_v('attrib', 'attrib_mean', 'attrib_sub_mean'))

    # obtain the squared subtracted attributions
    squared_attrib_sub = graph.edata['attrib_sub_mean']**2
    graph.edata.update({'squared_attrib_sub': squared_attrib_sub})

    # divided by the number of neighboring nodes
    graph.update_all(fn.u_mul_e('unit_node', 'squared_attrib_sub', 'node_squared_attrib_sub'), fn.mean('node_squared_attrib_sub', 'mean_node_squared_attrib_sub'))

    # compute the standard deviation of the attributions
    attrib_sd = torch.sqrt(graph.ndata['mean_node_squared_attrib_sub'] + 1e-5)
    graph.ndata.update({'attrib_sd': attrib_sd})

    # normalize the topological attributions
    graph.apply_edges(fn.e_div_v('attrib_sub_mean', 'attrib_sd', 'attrib_norm'))
    e = graph.edata.pop('attrib_norm')

    return e


def gen_mi_attrib_loss(graph, attrib_t1, attrib_t2, attrib_st1, attrib_st2):
    """This is the function that computes the topological attribution loss

    Args:
        graph (DGLGraph): the input graphs containing the topological information
        attrib_t1 (torch.Tensor): target topological attributions of teacher #1
        attrib_t2 (torch.Tensor): target topological attributions of teacher #2
        attrib_st1 (torch.Tensor): derived topological attributions of the student for the task of teacher #1
        attrib_st2 (torch.Tensor): derived topological attributions of the student for the task of teacher #2

    Returns:
        torch.Tensor: topological attribution loss
    """

    loss_fcn = nn.MSELoss()

    # perform topological-aware edge gradient normalization to address the scale issue
    attrib_t1 = gen_attrib_norm(graph, attrib_t1)
    attrib_t2 = gen_attrib_norm(graph, attrib_t2)
    attrib_st1 = gen_attrib_norm(graph, attrib_st1)
    attrib_st2 = gen_attrib_norm(graph, attrib_st2)

    # compute the topological attribution loss with the normalized attributions
    loss = loss_fcn(attrib_st1, attrib_t1.detach()) + loss_fcn(attrib_st2, attrib_t2.detach())

    return loss


def optimizing(auxiliary_model, loss, model_list):
    """This is the function that performs model optimizations

    Args:
        auxiliary_model (dict): model dictionary ([model_name][model/optimizer])
        loss (torch.Tensor): the total loss defined in Eq. 3 of the paper
        model_list (list): the list containing the names of the models for optimizations
    """
    
    for model in model_list:
        auxiliary_model[model]['optimizer'].zero_grad()
    
    loss.backward()
    
    for model in model_list:
        auxiliary_model[model]['optimizer'].step()
        