"""
.. module:: elmo
    :synopsis: deep contextualized representation
 
.. moduleauthor:: Liyuan Liu
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import model_seq.utils as utils

import torch
import torch.nn as nn
import torch.nn.functional as F

class EBUnit(nn.Module):
    """
    The basic recurrent unit for the ELMo RNNs wrapper.

    Parameters
    ----------
    ori_unit : ``torch.nn.Module``, required.
        The original module of rnn unit.
    droprate : ``float``, required.
        The dropout ratrio.
    fix_rate: ``bool``, required.
        Whether to fix the rqtio.
    """
    def __init__(self, ori_unit, droprate, fix_rate):
        super(EBUnit, self).__init__()

        self.layer = ori_unit.layer

        self.droprate = droprate

        self.output_dim = ori_unit.output_dim

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.FloatTensor``, required.
            The input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The output of RNNs.
        """
        out, _ = self.layer(x)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        return out

class ERNN(nn.Module):
    """
    The multi-layer recurrent networks for the ELMo RNNs wrapper.

    Parameters
    ----------
    ori_drnn : ``torch.nn.Module``, required.
        The original module of rnn networks.
    droprate : ``float``, required.
        The dropout ratrio.
    fix_rate: ``bool``, required.
        Whether to fix the rqtio.
    """
    def __init__(self, ori_drnn, droprate, fix_rate):
        super(ERNN, self).__init__()

        self.layer_list = [EBUnit(ori_unit, droprate, fix_rate) for ori_unit in ori_drnn.layer._modules.values()]

        self.gamma = nn.Parameter(torch.FloatTensor([1.0]))
        self.weight_list = nn.Parameter(torch.FloatTensor([0.0] * len(self.layer_list)))

        self.layer = nn.ModuleList(self.layer_list)

        for param in self.layer.parameters():
            param.requires_grad = False

        if fix_rate:
            self.gamma.requires_grad = False
            self.weight_list.requires_grad = False

        self.output_dim = self.layer_list[-1].output_dim

    def regularizer(self):
        """
        Calculate the regularization term.

        Returns
        ----------
        The regularization term.
        """
        srd_weight = self.weight_list - (1.0 / len(self.layer_list))
        return (srd_weight ** 2).sum()

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.FloatTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The ELMo outputs.
        """
        out = 0
        nw = self.gamma * F.softmax(self.weight_list, dim=0)
        for ind in range(len(self.layer_list)):
            x = self.layer[ind](x)
            out += x * nw[ind]
        return out

class ElmoLM(nn.Module):
    """
    The language model for the ELMo RNNs wrapper.

    Parameters
    ----------
    ori_lm : ``torch.nn.Module``, required.
        the original module of language model.
    backward : ``bool``, required.
        whether the language model is backward.
    droprate : ``float``, required.
        the dropout ratrio.
    fix_rate: ``bool``, required.
        whether to fix the rqtio.
    """

    def __init__(self, ori_lm, backward, droprate, fix_rate):
        super(ElmoLM, self).__init__()

        self.rnn = ERNN(ori_lm.rnn, droprate, fix_rate)

        self.w_num = ori_lm.w_num
        self.w_dim = ori_lm.w_dim
        self.word_embed = ori_lm.word_embed
        self.word_embed.weight.requires_grad = False

        self.output_dim = ori_lm.rnn_output

        self.backward = backward

    def init_hidden(self):
        """
        initialize hidden states.
        """
        return

    def regularizer(self):
        """
        Calculate the regularization term.

        Returns
        ----------
        reg: ``list``.
            The list of regularization terms.
        """
        return self.rnn.regularizer()

    def prox(self, lambda0):
        """
        the proximal calculator.
        """
        return 0.0

    def forward(self, w_in, ind=None):
        """
        Calculate the output.

        Parameters
        ----------
        w_in : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size).
        ind : ``torch.LongTensor``, optional, (default=None).
            the index tensor for the backward language model, of shape (seq_len, batch_size).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The ELMo outputs.
        """
        w_emb = self.word_embed(w_in)
        
        out = self.rnn(w_emb)

        if self.backward:
            out_size = out.size()
            out = out.view(out_size[0] * out_size[1], out_size[2]).index_select(0, ind).contiguous().view(out_size)

        return out