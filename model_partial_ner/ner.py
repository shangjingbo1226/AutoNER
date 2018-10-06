"""
.. module:: NER module
    :synopsis: NER module
    
.. moduleauthor:: Liyuan Liu, Jingbo Shang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_partial_ner.utils as utils
from model_partial_ner.highway import highway

class NER(nn.Module):
    """
    Sequence Labeling model augumented with language model.

    Parameters
    ----------
    rnn : ``torch.nn.Module``, required.
        The RNN unit..
    w_num : ``int`` , required.
        The number of words.
    w_dim : ``int`` , required.
        The dimension of word embedding.
    c_num : ``int`` , required.
        The number of characters.
    c_dim : ``int`` , required.
        The dimension of character embedding.
    y_dim : ``int`` , required.
        The dimension of tags types.
    y_num : ``int`` , required.
        The number of tags types.
    droprate : ``float`` , required
        The dropout ratio.
    """
    def __init__(self, rnn, 
                w_num: int, 
                w_dim: int, 
                c_num: int, 
                c_dim: int, 
                y_dim: int, 
                y_num: int, 
                droprate: float):

        super(NER, self).__init__()

        self.rnn = rnn
        self.rnn_outdim = self.rnn.output_dim
        self.one_direction_dim = self.rnn_outdim // 2
        self.word_embed = nn.Embedding(w_num, w_dim)
        self.char_embed = nn.Embedding(c_num, c_dim)
        self.drop = nn.Dropout(p=droprate)
        self.add_proj = y_dim > 0
        self.to_chunk = highway(self.rnn_outdim)
        self.to_type = highway(self.rnn_outdim)

        if self.add_proj:
            self.to_chunk_proj = nn.Linear(self.rnn_outdim, y_dim)
            self.to_type_proj = nn.Linear(self.rnn_outdim, y_dim)
            self.chunk_weight = nn.Linear(y_dim, 1)
            self.type_weight = nn.Linear(y_dim, y_num)
            self.chunk_layer = nn.Sequential(self.to_chunk, self.drop, self.to_chunk_proj, self.drop, self.chunk_weight)
            self.type_layer = nn.Sequential(self.to_type, self.drop, self.to_type_proj, self.drop, self.type_weight)
        else:
            self.chunk_weight = nn.Linear(self.rnn_outdim, 1)
            self.type_weight = nn.Linear(self.rnn_outdim, y_num)
            self.chunk_layer = nn.Sequential(self.to_chunk, self.drop, self.chunk_weight)
            self.type_layer = nn.Sequential(self.to_type, self.drop, self.type_weight)

    def to_params(self):
        """
        To parameters.
        """
        return {
            "model_type": "char-lstm-two-level",
            "rnn_params": self.rnn.to_params(),
            "word_embed_num": self.word_embed.num_embeddings,
            "word_embed_dim": self.word_embed.embedding_dim,
            "char_embed_num": self.char_embed.num_embeddings,
            "char_embed_dim": self.char_embed.embedding_dim,
            "type_dim": self.type_weight.in_features if self.add_proj else -1,
            "type_num": self.type_weight.out_features,
            "droprate": self.drop.p,
            "label_schema": "tie-or-break"
        }

    def load_pretrained_word_embedding(self, pre_word_embeddings):
        """
        Load pre-trained word embedding.

        Parameters
        ----------
        pre_word_embeddings : ``torch.FloatTensor``, required.
            pre-trained word embedding
        """
        self.word_embed.weight = nn.Parameter(pre_word_embeddings)

    def rand_ini(self):
        """
        Random initialization.
        """
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
        """
        Sequence labeling model.

        Parameters
        ----------
        w_in : ``torch.LongTensor``, required.
            The RNN unit.
        c_in : ``torch.LongTensor`` , required.
            The number of characters.
        mask : ``torch.ByteTensor`` , required.
            The mask for character-level input.
        """
        w_emb = self.word_embed(w_in)

        c_emb = self.char_embed(c_in)

        emb = self.drop( torch.cat([w_emb, c_emb], 2) )

        out = self.rnn(emb)

        mask = mask.unsqueeze(2).expand_as(out)

        out = out.masked_select(mask).view(-1, self.rnn_outdim)

        return out

    def chunking(self, z_in):
        """
        Chunking.

        Parameters
        ----------
        z_in : ``torch.LongTensor``, required.
           The output of the character-level lstms.
        """
        z_in = self.drop(z_in)

        out = self.chunk_layer(z_in).squeeze(1)

        return out

    def typing(self, z_in, mask):
        """
        Typing

        Parameters
        ----------
        z_in : ``torch.LongTensor``, required.
           The output of the character-level lstms.
        mask : ``torch.ByteTensor`` , required.
            The mask for word-level input.
        """
        mask = mask.unsqueeze(1).expand_as(z_in)

        z_in = z_in.masked_select(mask).view(-1, 2, self.one_direction_dim)
        z_in = torch.cat([z_in[0:-1, 1, :].squeeze(1), z_in[1:, 0, :].squeeze(1)], dim = 1)

        z_in = self.drop(z_in)

        out = self.type_layer(z_in)

        return out
        
    def to_span(self, chunk_label, type_label, none_idx):
        """
        Convert word-level labels to entity spans.

        Parameters
        ----------
        chunk_label : ``torch.LongTensor``, required.
            The chunk label for one sequence.
        type_label : ``torch.LongTensor`` , required.
            The type label for one sequence.
        none_idx: ``int``, required.
            Label index fot the not-target-type entity.
        """
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


    def to_typed_span(self, chunk_label, type_label, none_idx, id2label):
        """
        Convert word-level labels to typed entity spans.

        Parameters
        ----------
        chunk_label : ``torch.LongTensor``, required.
           The output of the character-level lstms.
        mask : ``torch.ByteTensor`` , required.
            The mask for word-level input.
        none_idx: ``int``, required.
            Label index fot the not-target-type entity.
        """
        span_list = list()

        pre_idx = -1
        cur_idx = 0
        type_idx = 0
        while cur_idx < len(chunk_label):
            if chunk_label[cur_idx].item() == 1:
                if pre_idx >= 0:
                    cur_type_idx = type_label[type_idx].item()
                    if cur_type_idx != none_idx:
                        span_list.append(id2label[cur_type_idx]+'@('+str(pre_idx)+','+str(cur_idx)+')')
                    type_idx += 1
                pre_idx = cur_idx
            cur_idx += 1

        assert type_idx == len(type_label)

        return set(span_list)