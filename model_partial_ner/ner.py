import torch
import torch.nn as nn
import torch.nn.functional as F
import model_partial_ner.utils as utils
from model_partial_ner.highway import highway

class NER(nn.Module):

    def __init__(self, rnn, w_num, w_dim, c_num, c_dim, y_dim, y_num, droprate, bi_type):
        super(NER, self).__init__()

        self.rnn = rnn
        self.rnn_outdim = self.rnn.output_dim
        self.one_direction_dim = self.rnn_outdim // 2

        self.word_embed = nn.Embedding(w_num, w_dim)
        self.char_embed = nn.Embedding(c_num, c_dim)

        self.drop = nn.Dropout(p=droprate)

        self.add_proj = y_dim > 0
        self.bi_type = bi_type

        self.to_chunk = highway(self.rnn_outdim)
        if bi_type:
            self.to_type = highway(self.rnn_outdim * 2)
        else:
            self.to_type = highway(self.rnn_outdim)

        if self.add_proj:

            self.to_chunk_proj = nn.Linear(self.rnn_outdim, y_dim)
            if bi_type:
                self.to_type_proj = nn.Linear(self.rnn_outdim * 2, y_dim)
            else:
                self.to_type_proj = nn.Linear(self.rnn_outdim, y_dim)

            self.chunk_weight = nn.Linear(y_dim, 1)
            self.type_weight = nn.Linear(y_dim, y_num)

            self.chunk_layer = nn.Sequential(self.to_chunk, self.drop, self.to_chunk_proj, self.drop, self.chunk_weight)

            self.type_layer = nn.Sequential(self.to_type, self.drop, self.to_type_proj, self.drop, self.type_weight)
        else:

            self.chunk_weight = nn.Linear(self.rnn_outdim, 1)

            if bi_type:
                self.type_weight = nn.Linear(self.rnn_outdim * 2, y_num)
            else:
                self.type_weight = nn.Linear(self.rnn_outdim, y_num)

            self.chunk_layer = nn.Sequential(self.to_chunk, self.drop, self.chunk_weight)

            self.type_layer = nn.Sequential(self.to_type, self.drop, self.type_weight)

    def load_pretrained_word_embedding(self, pre_word_embeddings):

        self.word_embed.weight = nn.Parameter(pre_word_embeddings)

    def rand_ini(self):
        
        self.rnn.rand_ini()
        self.to_chunk.rand_ini()
        self.to_type.rand_ini()

        utils.init_embedding(self.char_embed.weight)

        utils.init_linear(self.chunk_weight)
        utils.init_linear(self.type_weight)

        if self.add_proj:
            utils.init_linear(self.to_chunk_proj)
            utils.init_linear(self.to_type_proj)

    def forward(self, w_in, c_in, mask):

        w_emb = self.word_embed(w_in)

        c_emb = self.char_embed(c_in)

        emb = self.drop( torch.cat([w_emb, c_emb], 2) )

        out = self.rnn(emb)

        mask = mask.unsqueeze(2).expand_as(out)

        out = out.masked_select(mask).view(-1, self.rnn_outdim)

        return out

    def chunking(self, z_in):

        z_in = self.drop(z_in)

        out = self.chunk_layer(z_in).squeeze(1)

        return out

    def typing(self, z_in, mask):

        mask = mask.unsqueeze(1).expand_as(z_in)

        if self.bi_type:
            z_in = z_in.masked_select(mask).view(-1, self.rnn_outdim)

            z_in = torch.cat([z_in[0:-1], z_in[1:]], dim = 1)
        else:
            z_in = z_in.masked_select(mask).view(-1, 2, self.one_direction_dim)
            z_in = torch.cat([z_in[0:-1, 1, :].squeeze(1), z_in[1:, 0, :].squeeze(1)], dim = 1)

        z_in = self.drop(z_in)

        out = self.type_layer(z_in)

        return out
        
    def to_span(self, chunk_label, type_label, none_idx):

        span_list = list()

        pre_idx = -1
        cur_idx = 0
        type_idx = 0
        while cur_idx < len(chunk_label):
            if chunk_label[cur_idx].data[0] == 1:
                if pre_idx >= 0:
                    cur_type = type_label[type_idx].data[0]
                    if cur_type != none_idx:
                        span_list.append('('+str(pre_idx)+','+str(cur_idx)+')')
                    type_idx += 1
                pre_idx = cur_idx
            cur_idx += 1
            
        assert type_idx == len(type_label)

        return set(span_list)


    def to_typed_span(self, chunk_label, type_label, none_idx):

        span_list = list()

        pre_idx = -1
        cur_idx = 0
        type_idx = 0
        while cur_idx < len(chunk_label):
            if chunk_label[cur_idx].data[0] == 1:
                if pre_idx >= 0:
                    cur_type = type_label[type_idx].data[0]
                    if cur_type != none_idx:
                        span_list.append(str(cur_type)+'@('+str(pre_idx)+','+str(cur_idx)+')')
                    type_idx += 1
                pre_idx = cur_idx
            cur_idx += 1

        assert type_idx == len(type_label)

        return set(span_list)