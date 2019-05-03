"""
.. module:: seqlm
    :synopsis: language model for sequence labeling
 
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

class BasicSeqLM(nn.Module):
    """
    The language model for the dense rnns.

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
        super(BasicSeqLM, self).__init__()

        self.rnn = ori_lm.rnn

        for param in self.rnn.parameters():
            param.requires_grad = False

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
            "rnn_params": self.rnn.to_params(),
            "word_embed_num": self.word_embed.num_embeddings,
            "word_embed_dim": self.word_embed.embedding_dim
        }

    def init_hidden(self):
        """
        initialize hidden states.
        """
        self.rnn.init_hidden()
    
    def regularizer(self):
        """
        Calculate the regularization term.

        Returns
        ----------
        reg: ``list``.
            The list of regularization terms.
        """
        return self.rnn.regularizer()

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