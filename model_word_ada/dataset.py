"""
.. module:: dataset
    :synopsis: dataset for language modeling
 
.. moduleauthor:: Liyuan Liu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import pickle
import random
from tqdm import tqdm

from torch.utils.data import Dataset

class EvalDataset(object):
    """    
    Dataset for Language Modeling

    Parameters
    ----------
    dataset : ``list``, required.
        The encoded dataset (outputs of preprocess scripts).
    sequence_length: ``int``, required.
        Sequence Length.
    """
    def __init__(self, dataset, sequence_length):
        super(EvalDataset, self).__init__()
        self.dataset = dataset

        self.sequence_length = sequence_length

        self.construct_index()

    def get_tqdm(self, device):
        """
        construct dataset reader and the corresponding tqdm.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        """
        return tqdm(self.reader(device), mininterval=2, total=self.index_length, leave=False, file=sys.stdout, ncols=80)

    def construct_index(self):
        """
        construct index for the dataset.
        """
        token_per_batch = self.sequence_length
        tot_num = len(self.dataset) - 1
        res_num = tot_num - tot_num % token_per_batch

        self.x = list(torch.unbind(torch.LongTensor(self.dataset[0:res_num]).view(-1, self.sequence_length), 0))
        self.y = list(torch.unbind(torch.LongTensor(self.dataset[1:res_num+1]).view(-1, self.sequence_length), 0))

        self.x.append(torch.LongTensor(self.dataset[res_num:tot_num]))
        self.y.append(torch.LongTensor(self.dataset[res_num+1:tot_num+1]))

        self.index_length = len(self.x)
        self.cur_idx = 0

    def reader(self, device):
        """
        construct dataset reader.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        Returns
        -------
        reader: ``iterator``.
            A lazy iterable object        
        """
        if self.cur_idx == self.index_length:
            self.cur_idx = 0
            raise StopIteration

        word_t = self.x[self.cur_idx].to(device).view(-1, 1)
        label_t = self.y[self.cur_idx].to(device).view(-1, 1)

        self.cur_idx += 1
        
        yield word_t, label_t

class LargeDataset(object):
    """    
    Lazy Dataset for Language Modeling

    Parameters
    ----------
    root : ``str``, required.
        The root folder for dataset files.
    range_idx : ``int``, required.
        The maximum file index for the input files (train_*.pk).
    batch_size : ``int``, required.
        Batch size.
    sequence_length: ``int``, required.
        Sequence Length.
    """
    def __init__(self, root, range_idx, batch_size, sequence_length):
        super(LargeDataset, self).__init__()
        self.root = root
        self.range_idx = range_idx
        self.shuffle_list = list(range(0, range_idx))
        self.shuffle()

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.token_per_batch = self.batch_size * self.sequence_length

        self.total_batch_num = -1

    def shuffle(self):
        """
        shuffle dataset
        """
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device):
        """
        construct dataset reader and the corresponding tqdm.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.        
        """
        self.batch_count = 0
        self.cur_idx = 0
        self.file_idx = 0
        self.index_length = 0

        if self.total_batch_num <= 0:
            return tqdm(self.reader(device), mininterval=2, leave=False, file=sys.stdout).__iter__()
        else:
            return tqdm(self.reader(device), mininterval=2, total=self.total_batch_num, leave=False, file=sys.stdout, ncols=80).__iter__()


    def reader(self, device):
        """
        construct dataset reader.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        Returns
        -------
        reader: ``iterator``.
            A lazy iterable object        
        """
        while self.file_idx < self.range_idx:

            self.open_next()
            while self.cur_idx < self.index_length:

                word_t = self.x[self.cur_idx].to(device)
                # label_t = self.y[self.cur_idx].to(device)
                label_t = self.y[self.cur_idx].to(device)

                self.cur_idx += 1

                yield word_t, label_t

        self.total_batch_num = self.batch_count
        self.shuffle()

    def open_next(self):
        """
        Open the next file.
        """
        self.dataset = pickle.load(open(self.root + 'train_' + str( self.shuffle_list[self.file_idx])+'.pk', 'rb'))

        res_num = len(self.dataset) - 1
        res_num = res_num - res_num % self.token_per_batch

        self.x = torch.LongTensor(self.dataset[0:res_num]).view(self.batch_size, -1, self.sequence_length).transpose_(0, 1).transpose_(1, 2).contiguous()
        self.y = torch.LongTensor(self.dataset[1:res_num+1]).view(self.batch_size, -1, self.sequence_length).transpose_(0, 1).transpose_(1, 2).contiguous()

        self.index_length = self.x.size(0)
        self.cur_idx = 0

        self.batch_count += self.index_length
        self.file_idx += 1