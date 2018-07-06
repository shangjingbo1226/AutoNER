"""
.. module:: resnet
    :synopsis: ResRNN
 
.. moduleauthor:: Liyuan Liu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_partial_ner.utils as utils

# from model_partial_ner.bnlstm import BNLSTM

class FirstUnit(nn.Module):
    def __init__(self, unit, input_dim, hid_dim, droprate, batch_norm):
        super(FirstUnit, self).__init__()

        # rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU, 'bnlstm': BNLSTM}
        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

        self.unit = unit

        self.layer = rnnunit_map[unit](input_dim, hid_dim, 1)

        self.droprate = droprate
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(hid_dim)
        self.output_dim = hid_dim

        self.init_hidden()

    def init_hidden(self):

        self.hidden_state = None

    def rand_ini(self):

        if 'lstm' == self.unit:
            utils.init_lstm(self.layer)

    def forward(self, x):

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate, training=self.training)

        out, _ = self.layer(x)#, self.hidden_state)

        # self.hidden_state = utils.repackage_hidden(new_hidden)

        if self.batch_norm:
            output_size = out.size()
            out = self.bn(out.view(-1, self.output_dim)).view(output_size)
        
        return out

class InnerUnit(nn.Module):
    def __init__(self, unit, hid_dim, droprate, batch_norm):
        super(InnerUnit, self).__init__()

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}#, 'bnlstm': BNLSTM}

        self.unit = unit

        self.layer = rnnunit_map[unit](hid_dim, hid_dim, 1)
        self.droprate = droprate
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(hid_dim)
        self.output_dim = hid_dim

        self.init_hidden()

    def init_hidden(self):

        self.hidden_state = None

    def rand_ini(self):

        if 'lstm' == self.unit:
            utils.init_lstm(self.layer)

    def forward(self, x):

        if self.droprate > 0:
            new_x = F.dropout(x, p=self.droprate, training=self.training)
            x = F.dropout(x, p=self.droprate, training=self.training)
        else:
            new_x = x

        out, _ = self.layer(new_x)#, self.hidden_state)

        # self.hidden_state = utils.repackage_hidden(new_hidden)

        if self.batch_norm:
            output_size = out.size()
            out = self.bn(out.view(-1, self.output_dim)).view(output_size)
        
        # if self.droprate > 0:
        #     out = F.dropout(out, p=self.droprate, training=self.training)

        out = out + x

        return out

class ResRNN(nn.Module):
    def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate, batch_norm):
        super(ResRNN, self).__init__()

        self.layer_list = [FirstUnit(unit, emb_dim, hid_dim, droprate, batch_norm)] + [InnerUnit(unit, hid_dim, droprate, batch_norm) for i in range(layer_num - 1)]
        self.layer = nn.Sequential(*self.layer_list)
        self.output_dim = self.layer_list[-1].output_dim

        self.init_hidden()

    def init_hidden(self):

        for tup in self.layer_list:
            tup.init_hidden()

    def rand_ini(self):

        for tup in self.layer_list:
            tup.rand_ini()

    def forward(self, x):
        return self.layer(x)