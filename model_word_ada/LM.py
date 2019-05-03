"""
.. module:: LM
    :synopsis: language modeling
 
.. moduleauthor:: Liyuan Liu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_word_ada.utils as utils

class LM(nn.Module):
    """
    The language model model.
    
    Parameters
    ----------
    rnn : ``torch.nn.Module``, required.
        The RNNs network.
    soft_max : ``torch.nn.Module``, required.
        The softmax layer.
    w_num : ``int`` , required.
        The number of words.
    w_dim : ``int`` , required.
        The dimension of word embedding.
    droprate : ``float`` , required
        The dropout ratio.
    label_dim : ``int`` , required.
        The input dimension of softmax.    
    """

    def __init__(self, rnn, soft_max, w_num, w_dim, droprate, label_dim = -1, add_relu=False):
        super(LM, self).__init__()

        self.rnn = rnn
        self.soft_max = soft_max

        self.w_num = w_num
        self.w_dim = w_dim
        self.word_embed = nn.Embedding(w_num, w_dim)

        self.rnn_output = self.rnn.output_dim

        self.add_proj = label_dim > 0
        if self.add_proj:
            self.project = nn.Linear(self.rnn_output, label_dim)
            if add_relu:
                self.relu = nn.ReLU()
            else:
                self.relu = lambda x: x

        self.drop = nn.Dropout(p=droprate)

    def load_embed(self, origin_lm):
        """
        Load embedding from another language model.
        """
        self.word_embed = origin_lm.word_embed
        self.soft_max = origin_lm.soft_max

    def rand_ini(self):
        """
        Random initialization.
        """
        self.rnn.rand_ini()
        # utils.init_linear(self.project)
        self.soft_max.rand_ini()
        # if not self.tied_weight:
        utils.init_embedding(self.word_embed.weight)

        if self.add_proj:
            utils.init_linear(self.project)

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        self.rnn.init_hidden()

    def forward(self, w_in, target):
        """
        Calculate the loss.

        Parameters
        ----------
        w_in : ``torch.FloatTensor``, required.
            the input tensor, of shape (word_num, input_dim).
        target : ``torch.FloatTensor``, required.
            the target of the language model, of shape (word_num).
        
        Returns
        ----------
        loss: ``torch.FloatTensor``.
            The NLL loss.
        """

        w_emb = self.word_embed(w_in)
        
        w_emb = self.drop(w_emb)

        out = self.rnn(w_emb).contiguous().view(-1, self.rnn_output)

        if self.add_proj:
            out = self.drop(self.relu(self.project(out)))
            # out = self.drop(self.project(out))

        out = self.soft_max(out, target)

        return out

    def log_prob(self, w_in):
        """
        Calculate log-probability for the whole dictionary.
        
        Parameters
        ----------
        w_in : ``torch.FloatTensor``, required.
            the input tensor, of shape (word_num, input_dim).
        
        Returns
        ----------
        prob: ``torch.FloatTensor``.
            The full log-probability.
        """

        w_emb = self.word_embed(w_in)
        
        out = self.rnn(w_emb).contiguous().view(-1, self.rnn_output)

        if self.add_proj:
            out = self.relu(self.project(out))

        out = self.soft_max.log_prob(out, w_emb.device)

        return out