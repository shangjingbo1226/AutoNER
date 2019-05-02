"""
.. module:: crf
    :synopsis: conditional random field
 
.. moduleauthor:: Liyuan Liu
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.sparse as sparse
import model_seq.utils as utils

class CRF(nn.Module):
    """
    Conditional Random Field Module

    Parameters
    ----------
    hidden_dim : ``int``, required.
        the dimension of the input features.
    tagset_size : ``int``, required.
        the size of the target labels.
    if_bias: ``bool``, optional, (default=True).
        whether the linear transformation has the bias term.
    """
    def __init__(self, 
                hidden_dim: int, 
                tagset_size: int, 
                if_bias: bool = True):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)
        self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))

    def rand_init(self):
        """
        random initialization
        """
        utils.init_linear(self.hidden2tag)
        self.transitions.data.zero_()

    def forward(self, feats):
        """
        calculate the potential score for the conditional random field.

        Parameters
        ----------
        feats: ``torch.FloatTensor``, required.
            the input features for the conditional random field, of shape (*, hidden_dim).

        Returns
        -------
        output: ``torch.FloatTensor``.
            A float tensor of shape (ins_num, from_tag_size, to_tag_size)
        """
        scores = self.hidden2tag(feats).view(-1, 1, self.tagset_size)
        ins_num = scores.size(0)
        crf_scores = scores.expand(ins_num, self.tagset_size, self.tagset_size) + self.transitions.view(1, self.tagset_size, self.tagset_size).expand(ins_num, self.tagset_size, self.tagset_size)

        return crf_scores

class CRFLoss(nn.Module):
    """
    
    The negative loss for the Conditional Random Field Module

    Parameters
    ----------
    y_map : ``dict``, required.
        a ``dict`` maps from tag string to tag index.
    average_batch : ``bool``, optional, (default=True).
        whether the return score would be averaged per batch.
    """

    def __init__(self, 
                 y_map: dict, 
                 average_batch: bool = True):
        super(CRFLoss, self).__init__()
        self.tagset_size = len(y_map)
        self.start_tag = y_map['<s>']
        self.end_tag = y_map['<eof>']
        self.average_batch = average_batch

    def forward(self, scores, target, mask):
        """
        calculate the negative log likehood for the conditional random field.

        Parameters
        ----------
        scores: ``torch.FloatTensor``, required.
            the potential score for the conditional random field, of shape (seq_len, batch_size, from_tag_size, to_tag_size).
        target: ``torch.LongTensor``, required.
            the positive path for the conditional random field, of shape (seq_len, batch_size).
        mask: ``torch.ByteTensor``, required.
            the mask for the unpadded sentence parts, of shape (seq_len, batch_size).

        Returns
        -------
        loss: ``torch.FloatTensor``.
            The NLL loss.
        """
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target.unsqueeze(2)).view(seq_len, bat_size)
        tg_energy = tg_energy.masked_select(mask).sum()

        seq_iter = enumerate(scores)

        _, inivalues = seq_iter.__next__()
        partition = inivalues[:, self.start_tag, :].squeeze(1).clone()

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.unsqueeze(2).expand(bat_size, self.tagset_size, self.tagset_size)

            cur_partition = utils.log_sum_exp(cur_values)

            mask_idx = mask[idx, :].view(bat_size, 1).expand(bat_size, self.tagset_size)
            partition.masked_scatter_(mask_idx, cur_partition.masked_select(mask_idx))

        partition = partition[:, self.end_tag].sum()

        if self.average_batch:
            return (partition - tg_energy) / bat_size
        else:
            return (partition - tg_energy)

class CRFDecode():
    """
    
    The negative loss for the Conditional Random Field Module

    Parameters
    ----------
    y_map : ``dict``, required.
        a ``dict`` maps from tag string to tag index.
    """
    def __init__(self, y_map: dict):
        self.tagset_size = len(y_map)
        self.start_tag = y_map['<s>']
        self.end_tag = y_map['<eof>']
        self.y_map = y_map
        self.r_y_map = {v:k for k, v in self.y_map.items()}

    def decode(self, scores, mask):
        """
        find the best path from the potential scores by the viterbi decoding algorithm.

        Parameters
        ----------
        scores: ``torch.FloatTensor``, required.
            the potential score for the conditional random field, of shape (seq_len, batch_size, from_tag_size, to_tag_size).
        mask: ``torch.ByteTensor``, required.
            the mask for the unpadded sentence parts, of shape (seq_len, batch_size).

        Returns
        -------
        output: ``torch.LongTensor``.
            A LongTensor of shape (seq_len - 1, batch_size)
        """
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        mask = 1 - mask.data
        decode_idx = torch.LongTensor(seq_len-1, bat_size)

        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()
        forscores = inivalues[:, self.start_tag, :]
        back_points = list()

        for idx, cur_values in seq_iter:
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)

            forscores, cur_bp = torch.max(cur_values, 1)
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size), self.end_tag)
            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.end_tag]
        decode_idx[-1] = pointer
        for idx in range(len(back_points)-2, -1, -1):
            back_point = back_points[idx]
            index = pointer.contiguous().view(-1, 1)
            pointer = torch.gather(back_point, 1, index).view(-1)
            decode_idx[idx] = pointer
        return decode_idx

    def to_spans(self, sequence):
        """
        decode the best path to spans.

        Parameters
        ----------
        sequence: list, required.
            the list of best label indexes paths .

        Returns
        -------
        output: ``set``.
            A set of chunks contains the position and type of the entities.
        """
        chunks = []
        current = None

        for i, y in enumerate(sequence):
            label = self.r_y_map[y]

            if label.startswith('B-'):

                if current is not None:
                    chunks.append('@'.join(current))
                current = [label.replace('B-', ''), '%d' % i]

            elif label.startswith('S-'):

                if current is not None:
                    chunks.append('@'.join(current))
                    current = None
                base = label.replace('S-', '')
                chunks.append('@'.join([base, '%d' % i]))

            elif label.startswith('I-'):

                if current is not None:
                    base = label.replace('I-', '')
                    if base == current[0]:
                        current.append('%d' % i)
                    else:
                        chunks.append('@'.join(current))
                        current = [base, '%d' % i]

                else:
                    current = [label.replace('I-', ''), '%d' % i]

            elif label.startswith('E-'):

                if current is not None:
                    base = label.replace('E-', '')
                    if base == current[0]:
                        current.append('%d' % i)
                        chunks.append('@'.join(current))
                        current = None
                    else:
                        chunks.append('@'.join(current))
                        current = [base, '%d' % i]
                        chunks.append('@'.join(current))
                        current = None

                else:
                    current = [label.replace('E-', ''), '%d' % i]
                    chunks.append('@'.join(current))
                    current = None
            else:
                if current is not None:
                    chunks.append('@'.join(current))
                current = None

        if current is not None:
            chunks.append('@'.join(current))

        return set(chunks)