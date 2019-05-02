"""
.. module:: adaptive
    :synopsis: adaptive softmax
 
.. moduleauthor:: Liyuan Liu
"""
import torch
from torch import nn

from math import sqrt

class AdaptiveSoftmax(nn.Module):
    """
    The adaptive softmax layer.
    Modified from: https://github.com/rosinality/adaptive-softmax-pytorch/blob/master/adasoft.py

    Parameters
    ----------
    input_size : ``int``, required.
        The input dimension.
    cutoff : ``list``, required.
        The list of cutoff values.
    """
    def __init__(self, input_size, cutoff):
        super().__init__()

        self.input_size = input_size
        self.cutoff = cutoff
        self.output_size = cutoff[0] + len(cutoff) - 1

        self.head = nn.Linear(input_size, self.output_size)
        self.tail = nn.ModuleList()

        self.cross_entropy = nn.CrossEntropyLoss(size_average=False)

        for i in range(len(self.cutoff) - 1):
            seq = nn.Sequential(
                nn.Linear(input_size, input_size // 4 ** i, False),
                nn.Linear(input_size // 4 ** i, cutoff[i + 1] - cutoff[i], False)
            )

            self.tail.append(seq)

    def rand_ini(self):
        """
        Random Initialization.
        """
        nn.init.xavier_normal_(self.head.weight)

        for tail in self.tail:
            nn.init.xavier_normal_(tail[0].weight)
            nn.init.xavier_normal_(tail[1].weight)

    def log_prob(self, w_in, device):
        """
        Calculate log-probability for the whole dictionary.
        
        Parameters
        ----------
        w_in : ``torch.FloatTensor``, required.
            the input tensor, of shape (word_num, input_dim).
        device: ``torch.device``, required.
            the target device for calculation.

        Returns
        ----------
        prob: ``torch.FloatTensor``.
            The full log-probability.
        """
        lsm = nn.LogSoftmax(dim=1).to(device)

        head_out = self.head(w_in)

        batch_size = head_out.size(0)
        prob = torch.zeros(batch_size, self.cutoff[-1]).to(device)

        lsm_head = lsm(head_out) 
        prob.narrow(1, 0, self.output_size).add_(lsm_head.narrow(1, 0, self.output_size).data)

        for i in range(len(self.tail)):
            pos = self.cutoff[i]
            i_size = self.cutoff[i + 1] - pos
            buffer = lsm_head.narrow(1, self.cutoff[0] + i, 1)
            buffer = buffer.expand(batch_size, i_size)
            lsm_tail = lsm(self.tail[i](w_in)) 
            prob.narrow(1, pos, i_size).copy_(buffer.data).add_(lsm_tail.data)

        return prob

    def forward(self, w_in, target):
        """
        Calculate the log-likihood w.o. calculate the full distribution.

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
        batch_size = w_in.size(0)
        output = 0.0

        first_target = target.clone()

        for i in range(len(self.cutoff) - 1):
            
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))

            if mask.sum() > 0:

                first_target[mask] = self.cutoff[0] + i

                second_target = target[mask].add(-self.cutoff[i])
                second_input = w_in.index_select(0, mask.nonzero().squeeze())

                second_output = self.tail[i](second_input)

                output += self.cross_entropy(second_output, second_target)

        output += self.cross_entropy(self.head(w_in), first_target)
        output /= batch_size
        return output
