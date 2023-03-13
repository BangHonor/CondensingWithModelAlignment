import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import torch.autograd as autograd


def get_attrib(net, output, labels, A):
    """This is the function that computes the topological attributions for the student

    Args:
        net (nn.Module): the student GNN model
        graph (DGLGraph): the input graphs containing the topological information
        features (torch.Tensor): the input node features
        labels (torch.Tensor): the soft labels
        mode (string): ('t1', 't2')

    Returns:
        torch.Tensor: topological attributions
    """

    labels = torch.where(labels > 0.0, 
                            torch.ones(labels.shape).to(labels.device), 
                            torch.zeros(labels.shape).to(labels.device)).type(torch.bool)

    # set the gradients of the corresponding output activations to one 
    output_grad = torch.zeros_like(output)
    output_grad[labels] = 1

    # compute the gradients
    attrib = autograd.grad(outputs=output, inputs=A, grad_outputs=output_grad, create_graph=True, retain_graph=True, only_inputs=True)[0]

    return attrib
    

class ATTNET_s(nn.Module):#any dgl model can use
    """This is the class that returns the topological attribution maps of the student GNN

    Args:
        nn.Module: torch module
    """
    
    def __init__(self,
                 model,
                 args):

        super(ATTNET_s, self).__init__()
        # set up the network
        self.net = model
        self.args = args

    def observe(self, output, labels,u):
        """This is the function that returns the topological attribution maps

        Args:
            graph (DGLGraph): the input graphs containing the topological information
            features (torch.Tensor): the input node features
            labels (torch.Tensor): the soft labels

        Returns:
            torch.Tensor: topological attributions
        """

        A = torch.cuda.FloatTensor( [1.0] * len(u)).view((-1, 1, 1))
        A.requires_grad = True

        attrib = get_attrib(self.net, output, labels, A)

        return attrib
