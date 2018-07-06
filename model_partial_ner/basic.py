import torch
import torch.nn as nn
import torch.nn.functional as F
import model_partial_ner.utils as utils

class BasicUnit(nn.Module):
    def __init__(self, unit, input_dim, hid_dim, droprate, batch_norm):
        super(BasicUnit, self).__init__()

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

        self.layer = rnnunit_map[unit](input_dim, hid_dim//2, 1, batch_first=True, bidirectional=True)
        
        self.droprate = droprate
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(hid_dim)
        self.output_dim = hid_dim

        self.init_hidden()

    def init_hidden(self):

        self.hidden_state = None

    def rand_ini(self):

        utils.init_lstm(self.layer)

    def forward(self, x):

        out, _ = self.layer(x)

        if self.batch_norm:
            output_size = out.size()
            out = self.bn(out.view(-1, self.output_dim)).view(output_size)
        
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        return out

class BasicRNN(nn.Module):
    def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate, batch_norm):
        super(BasicRNN, self).__init__()

        self.layer_list = [BasicUnit(unit, emb_dim, hid_dim, droprate, batch_norm)] + [BasicUnit(unit, hid_dim, hid_dim, droprate, batch_norm) for i in range(layer_num - 1)]
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