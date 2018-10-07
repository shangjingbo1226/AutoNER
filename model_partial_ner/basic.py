"""
.. module:: basic
    :synopsis: basic rnn
 
.. moduleauthor:: Liyuan Liu, Jingbo Shang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_partial_ner.utils as utils

class BasicUnit(nn.Module):
    """
    The basic recurrent unit for the vanilla stacked RNNs.

    Parameters
    ----------
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``int``, required.
        The input dimension fo the unit.
    hid_dim : ``int``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    batch_norm: ``bool``, required.
        Incorporate batch norm or not. 
    """
    def __init__(self, unit, input_dim, hid_dim, droprate, batch_norm):
        super(BasicUnit, self).__init__()

        self.unit_type = unit
        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        self.layer = rnnunit_map[unit](input_dim, hid_dim//2, 1, batch_first=True, bidirectional=True)
        
        self.droprate = droprate
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(hid_dim)
        self.output_dim = hid_dim

        self.init_hidden()

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        self.hidden_state = None

    def rand_ini(self):
        """
        Random Initialization.
        """
        utils.init_lstm(self.layer)

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.   
            The output of RNNs.
        """
        out, _ = self.layer(x)

        if self.batch_norm:
            output_size = out.size()
            out = self.bn(out.view(-1, self.output_dim)).view(output_size)
        
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        return out

class BasicRNN(nn.Module):
    """
    The multi-layer recurrent networks for the vanilla stacked RNNs.

    Parameters
    ----------
    layer_num: ``int``, required.
        The number of layers. 
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``int``, required.
        The input dimension fo the unit.
    hid_dim : ``int``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    batch_norm: ``bool``, required.
        Incorporate batch norm or not. 
    """
    def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate, batch_norm):
        super(BasicRNN, self).__init__()

        self.layer_list = [BasicUnit(unit, emb_dim, hid_dim, droprate, batch_norm)] + [BasicUnit(unit, hid_dim, hid_dim, droprate, batch_norm) for i in range(layer_num - 1)]
        self.layer = nn.Sequential(*self.layer_list)
        self.output_dim = self.layer_list[-1].output_dim

        self.init_hidden()

    def to_params(self):
        """
        To parameters.
        """
        return {
            "rnn_type": "Basic",
            "unit_type": self.layer_list[0].unit_type,
            "layer_num": len(self.layer_list),
            "emb_dim": self.layer_list[0].layer.input_size,
            "hid_dim": self.layer_list[0].layer.hidden_size,
            "droprate": self.layer_list[0].droprate,
            "batch_norm": self.layer_list[0].batch_norm
        }

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        for tup in self.layer_list:
            tup.init_hidden()

    def rand_ini(self):
        """
        Random Initialization.
        """
        for tup in self.layer_list:
            tup.rand_ini()

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The output of RNNs.
        """
        return self.layer(x)