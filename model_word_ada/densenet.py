"""
.. module:: densenet
    :synopsis: densernn
 
.. moduleauthor:: Liyuan Liu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_word_ada.utils as utils

class BasicUnit(nn.Module):
    """
    The basic recurrent unit for the densely connected RNNs.

    Parameters
    ----------
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``float``, required.
        The input dimension fo the unit.
    increase_rate : ``float``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    """
    def __init__(self, unit, input_dim, increase_rate, droprate):
        super(BasicUnit, self).__init__()

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

        self.unit_type = unit

        self.layer = rnnunit_map[unit](input_dim, increase_rate, 1)

        if 'lstm' == self.unit_type:
            utils.init_lstm(self.layer)

        self.droprate = droprate

        self.input_dim = input_dim
        self.increase_rate = increase_rate
        self.output_dim = input_dim + increase_rate

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
        return

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
        if self.droprate > 0:
            new_x = F.dropout(x, p=self.droprate, training=self.training)
        else:
            new_x = x

        out, new_hidden = self.layer(new_x, self.hidden_state)

        self.hidden_state = utils.repackage_hidden(new_hidden)

        out = out.contiguous()

        return torch.cat([x, out], 2)

class DenseRNN(nn.Module):
    """
    The multi-layer recurrent networks for the densely connected RNNs.

    Parameters
    ----------
    layer_num: ``float``, required.
        The number of layers. 
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``float``, required.
        The input dimension fo the unit.
    hid_dim : ``float``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    """
    def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate):
        super(DenseRNN, self).__init__()
        
        self.unit_type = unit
        self.layer_list = [BasicUnit(unit, emb_dim + i * hid_dim, hid_dim, droprate) for i in range(layer_num)]
        self.layer = nn.Sequential(*self.layer_list) if layer_num > 0 else None
        self.output_dim = self.layer_list[-1].output_dim if layer_num > 0 else emb_dim
        self.emb_dim = emb_dim

        self.init_hidden()

    def to_params(self):
        """
        To parameters.
        """
        return {
            "rnn_type": "DenseRNN",
            "unit_type": self.layer[0].unit_type,
            "layer_num": len(self.layer),
            "emb_dim": self.layer[0].input_dim,
            "hid_dim": self.layer[0].increase_rate,
            "droprate": self.layer[0].droprate
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