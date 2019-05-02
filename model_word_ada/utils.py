"""
.. module:: utils
    :synopsis: utils
 
.. moduleauthor:: Liyuan Liu
"""
import numpy as np
import torch
import json

import torch
import torch.nn as nn
import torch.nn.init

from torch.autograd import Variable

def repackage_hidden(h):
    """
    Wraps hidden states in new Variables, to detach them from their history

    Parameters
    ----------
    h : ``Tuple`` or ``Tensors``, required.
        Tuple or Tensors, hidden states.

    Returns
    -------
    hidden: ``Tuple`` or ``Tensors``.
        detached hidden states
    """
    if type(h) == torch.Tensor:
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def to_scalar(var):
    """
    convert a tensor to a scalar number
    """
    return var.view(-1).item()

def init_embedding(input_embedding):
    """
    random initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)

def init_linear(input_linear):
    """
    random initialize linear projection.
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def adjust_learning_rate(optimizer, lr):
    """
    adjust learning to the the new value.

    Parameters
    ----------
    optimizer : required.
        pytorch optimizer.
    float :  ``float``, required.
        the target learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_lstm(input_lstm):
    """
    random initialize lstms
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
    
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1