"""
.. module:: sparse_lm
    :synopsis: sparse language model for sequence labeling
 
.. moduleauthor:: Liyuan Liu
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import model_seq.utils as utils

class SBUnit(nn.Module):
    """
    The basic recurrent unit for the dense-RNNs wrapper.

    Parameters
    ----------
    ori_unit : ``torch.nn.Module``, required.
        the original module of rnn unit.
    droprate : ``float``, required.
        the dropout ratrio.
    fix_rate: ``bool``, required.
        whether to fix the rqtio.
    """
    def __init__(self, ori_unit, droprate, fix_rate):
        super(SBUnit, self).__init__()

        self.unit_type = ori_unit.unit_type

        self.layer = ori_unit.layer

        self.droprate = droprate

        self.input_dim = ori_unit.input_dim
        self.increase_rate = ori_unit.increase_rate
        self.output_dim = ori_unit.input_dim + ori_unit.increase_rate

    def prune_rnn(self, mask):
        """
        Prune dense rnn to be smaller by delecting layers.

        Parameters
        ----------
        mask : ``torch.ByteTensor``, required.
            The selection tensor for the input matrix.
        """
        mask_index = mask.nonzero().squeeze(1)
        self.layer.weight_ih_l0 = nn.Parameter(self.layer.weight_ih_l0.data.index_select(1, mask_index).contiguous())
        self.layer.input_size = self.layer.weight_ih_l0.size(1)

    def forward(self, x, weight=1):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.FloatTensor``, required.
            The input tensor, of shape (seq_len, batch_size, input_dim).
        weight : ``torch.FloatTensor``, required.
            The selection variable.

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The output of RNNs.
        """

        if self.droprate > 0:
            new_x = F.dropout(x, p=self.droprate, training=self.training)
        else:
            new_x = x

        out, _ = self.layer(new_x)

        out = weight * out

        return torch.cat([x, out], 2)

class SDRNN(nn.Module):
    """
    The multi-layer recurrent networks for the dense-RNNs wrapper.

    Parameters
    ----------
    ori_unit : ``torch.nn.Module``, required.
        the original module of rnn unit.
    droprate : ``float``, required.
        the dropout ratrio.
    fix_rate: ``bool``, required.
        whether to fix the rqtio.
    """
    def __init__(self, ori_drnn, droprate, fix_rate):
        super(SDRNN, self).__init__()

        if ori_drnn.layer:
            self.layer_list = [SBUnit(ori_unit, droprate, fix_rate) for ori_unit in ori_drnn.layer._modules.values()]

            self.weight_list = nn.Parameter(torch.FloatTensor([1.0] * len(self.layer_list)))
            self.weight_list.requires_grad = not fix_rate

            # self.layer = nn.Sequential(*self.layer_list)
            self.layer = nn.ModuleList(self.layer_list)

            for param in self.layer.parameters():
                param.requires_grad = False
        else:
            self.layer_list = list()
            self.weight_list = list()
            self.layer = None

        # self.output_dim = self.layer_list[-1].output_dim
        self.emb_dim = ori_drnn.emb_dim
        self.output_dim = ori_drnn.output_dim
        self.unit_type = ori_drnn.unit_type

    def to_params(self):
        """
        To parameters.
        """
        return {
            "rnn_type": "LDRNN",
            "unit_type": self.unit_type,
            "layer_num": 0 if not self.layer else len(self.layer),
            "emb_dim": self.emb_dim,
            "hid_dim": -1 if not self.layer else self.layer[0].increase_rate,
            "droprate": -1 if not self.layer else self.layer[0].droprate,
            "after_pruned": True
        }

    def prune_dense_rnn(self):
        """
        Prune dense rnn to be smaller by delecting layers.
        """
        prune_mask = torch.ones(self.layer_list[0].input_dim)
        increase_mask_one = torch.ones(self.layer_list[0].increase_rate)
        increase_mask_zero = torch.zeros(self.layer_list[0].increase_rate)

        new_layer_list = list()
        new_weight_list = list()
        for ind in range(0, len(self.layer_list)):
            if self.weight_list.data[ind] > 0:
                new_weight_list.append(self.weight_list.data[ind])

                self.layer_list[ind].prune_rnn(prune_mask)
                new_layer_list.append(self.layer_list[ind])

                prune_mask = torch.cat([prune_mask, increase_mask_one], dim = 0)
            else:
                prune_mask = torch.cat([prune_mask, increase_mask_zero], dim = 0)

        if not new_layer_list:
            self.output_dim = self.layer_list[0].input_dim
            self.layer = None
            self.weight_list = None
            self.layer_list = None
        else:
            self.layer_list = new_layer_list
            self.layer = nn.ModuleList(self.layer_list)
            self.weight_list = nn.Parameter(torch.FloatTensor(new_weight_list))
            self.weight_list.requires_grad = False


            for param in self.layer.parameters():
                param.requires_grad = False

        return prune_mask

    def prox(self):
        """
        the proximal calculator.
        """
        self.weight_list.data.masked_fill_(self.weight_list.data < 0, 0)
        self.weight_list.data.masked_fill_(self.weight_list.data > 1, 1)
        none_zero_count = (self.weight_list.data > 0).sum()
        return none_zero_count

    def regularizer(self):
        """
        Calculate the regularization term.

        Returns
        ----------
        reg0: ``torch.FloatTensor``.
            The value of reg0.
        reg1: ``torch.FloatTensor``.
            The value of reg1.
        reg2: ``torch.FloatTensor``.
            The value of reg2.
        """
        reg3 = (self.weight_list * (1 - self.weight_list)).sum()
        none_zero = self.weight_list.data > 0
        none_zero_count = none_zero.sum()
        reg0 = none_zero_count
        reg1 = self.weight_list[none_zero].sum()
        return reg0, reg1, reg3

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
        if self.layer_list is not None:
            for ind in range(len(self.layer_list)):
                x = self.layer[ind](x, self.weight_list[ind])
        return x
        # return self.layer(x)

class SparseSeqLM(nn.Module):
    """
    The language model for the dense rnns with layer-wise selection.

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
        super(SparseSeqLM, self).__init__()

        self.rnn = SDRNN(ori_lm.rnn, droprate, fix_rate)

        self.w_num = ori_lm.w_num
        self.w_dim = ori_lm.w_dim
        self.word_embed = ori_lm.word_embed
        self.word_embed.weight.requires_grad = False

        self.output_dim = ori_lm.rnn_output

        self.backward = backward

    def to_params(self):
        """
        To parameters.
        """
        return {
            "backward": self.backward,
            "rnn_params": self.rnn.to_params(),
            "word_embed_num": self.word_embed.num_embeddings,
            "word_embed_dim": self.word_embed.embedding_dim
        }

    def prune_dense_rnn(self):
        """
        Prune dense rnn to be smaller by delecting layers.
        """
        prune_mask = self.rnn.prune_dense_rnn()
        self.output_dim = self.rnn.output_dim
        return prune_mask

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

    def prox(self):
        """
        the proximal calculator.
        """
        return self.rnn.prox()

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
        