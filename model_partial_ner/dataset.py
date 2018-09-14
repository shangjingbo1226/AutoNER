"""
.. module:: dataset
    :synopsis: dataset for sequence labeling
 
.. moduleauthor:: Liyuan Liu, Jingbo Shang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import sys
import pickle
from tqdm import tqdm
import random

from torch.utils.data import Dataset

class RawDataset(object):
    """    
    Raw Dataset for Sequence Labeling

    Parameters
    ----------
    dataset : ``list``, required.
        The encoded dataset (outputs of preprocess scripts).
    w_pad : ``int``, required.
        The pad index for the word-level inputs.
    c_pad : ``int``, required.
        The pad index for the character-level inputs.
    token_per_batch: ``int``, required.
        Batch size.
    """
    def __init__(self, 
                dataset: list, 
                w_pad: int, 
                c_pad: int, 
                token_per_batch: int):
        super(RawDataset, self).__init__()
        self.dataset = dataset
        self.w_pad = w_pad
        self.c_pad = c_pad
        self.token_per_batch = token_per_batch

        self.construct_index()

    def get_tqdm(self, device):
        """
        construct dataset reader and the corresponding tqdm.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        """
        return tqdm(self.reader(device), mininterval=2, total=self.index_length, leave=False, file=sys.stdout)

    def construct_index(self):
        """
        construct index for the dataset.

        Parameters
        ----------
        dataset: ``list``, required.
            the encoded dataset (outputs of preprocess scripts).        
        """
        self.index_length =len(self.dataset)

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
        cur_idx = 0
        while cur_idx < self.index_length:

            batch = self.dataset[cur_idx]

            word_t = torch.LongTensor([batch[0]]).to(device)
            char_t = torch.LongTensor([batch[1]]).to(device)
            chunk_mask = torch.ByteTensor([batch[2]]).to(device)
            chunk_index = torch.LongTensor(batch[3]).to(device)
            chunk_surface = batch[4]
            cur_idx += 1

            yield word_t, char_t, chunk_mask, chunk_index, chunk_surface

class NERDataset(object):
    """
    Evaluation Dataset for Sequence Labeling

    Parameters
    ----------
    dataset : ``list``, required.
        The encoded dataset (outputs of preprocess scripts).
    w_pad : ``int``, required.
        The pad index for the word-level inputs.
    c_pad : ``int``, required.
        The pad index for the character-level inputs.
    token_per_batch: ``int``, required.
        Batch size.
    """
    def __init__(self, 
                dataset: list, 
                w_pad: int, 
                c_pad: int, 
                token_per_batch: int):
        super(NERDataset, self).__init__()
        self.dataset = dataset
        self.w_pad = w_pad
        self.c_pad = c_pad
        self.token_per_batch = token_per_batch

        self.construct_index()

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
        return tqdm(self.reader(device), mininterval=2, total=self.index_length, leave=False, file=sys.stdout)

    def construct_index(self):
        """
        construct index for the dataset.    
        """
        dataset_size = len(self.dataset)
        self.index_list = list()
        start_index = 0
        while start_index < dataset_size:
            self.index_list.append(start_index)
            cur_seq_length = len(self.dataset[start_index][0]) - 1
            cur_batch_size = max(int(self.token_per_batch / cur_seq_length), 1)
            start_index = start_index + cur_batch_size
        self.index_length =len(self.index_list)
        self.index_list.append(dataset_size)
        self.shuffle_list = list(range(self.index_length-1, -1, -1))

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
        cur_idx = 0
        while cur_idx < self.index_length:
            batch_idx = self.shuffle_list[cur_idx]
            batch = self.dataset[self.index_list[batch_idx]: self.index_list[batch_idx + 1]]
            cur_seq_length = len(batch[0][0])
            word_t = torch.LongTensor([tup[0] + [self.w_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            char_t = torch.LongTensor([tup[1] + [self.c_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            chunk_mask = torch.ByteTensor([tup[2] + [0] * (cur_seq_length - len(tup[2])) for tup in batch]).to(device)
            chunk_label = torch.FloatTensor([label for tup in batch for label in tup[3]]).to(device)
            type_mask = torch.ByteTensor([mask for tup in batch for mask in tup[4]]).to(device)
            label_list = [label for tup in batch for label in tup[5]]
            type_label = torch.FloatTensor(label_list[0:-1]).to(device)
            cur_idx += 1
            yield word_t, char_t, chunk_mask, chunk_label, type_mask, type_label
        self.shuffle()
            
class TrainDataset(object):
    """
    Training Dataset for Sequence Labeling

    Parameters
    ----------
    dataset_name : ``str``, required.
        The name of dataset (outputs of preprocess scripts).
    w_pad : ``int``, required.
        The pad index for the word-level inputs.
    c_pad : ``int``, required.
        The pad index for the character-level inputs.
    token_per_batch: ``int``, required.
        Batch size.
    sample_ratio: ``float``, optional (default = 1.0)
        The ratio for sampling.
    """
    def __init__(self, 
                dataset_name: str, 
                w_pad: int, 
                c_pad: int, 
                token_per_batch: int, 
                sample_ratio: float = 1.0):
        
        super(TrainDataset, self).__init__()
        self.sample_ratio = sample_ratio

        self.dataset_name = dataset_name

        self.w_pad = w_pad
        self.c_pad = c_pad
        self.token_per_batch = token_per_batch

        self.total_batch_num = -1

        self.open_file()

    def get_tqdm(self, device):
        """
        construct dataset reader and the corresponding tqdm.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        """
        return tqdm(self.reader(device), mininterval=2, total=self.total_batch_num, leave=False, file=sys.stdout).__iter__()

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
        cur_idx = 0

        while cur_idx < self.index_length:

            batch_idx = self.shuffle_list[cur_idx]
            batch = self.dataset[self.index_list[batch_idx]: self.index_list[batch_idx + 1]]

            cur_seq_length = len(batch[0][0])
            word_t = torch.LongTensor([tup[0] + [self.w_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            char_t = torch.LongTensor([tup[1] + [self.c_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            chunk_mask = torch.ByteTensor([tup[2] + [0] * (cur_seq_length - len(tup[2])) for tup in batch]).to(device)
            chunk_label = torch.FloatTensor([label for tup in batch for label in tup[3]]).to(device)
            type_mask = torch.ByteTensor([mask for tup in batch for mask in tup[4]]).to(device)
            label_list = [label for tup in batch for label in tup[5]]
            type_label = torch.FloatTensor(label_list[0:-1]).to(device)

            cur_idx += 1

            yield word_t, char_t, chunk_mask, chunk_label, type_mask, type_label

        random.shuffle(self.shuffle_list)

    def open_file(self):
        """
        Open the dataset by name.      
        """
        self.dataset = pickle.load(open(self.dataset_name, 'rb'))

        self.dataset = list(filter(lambda t: random.uniform(0, 1) <= self.sample_ratio, self.dataset))

        dataset_size = len(self.dataset)
        self.index_list = list()
        start_index = 0
        while start_index < dataset_size:
            self.index_list.append(start_index)
            cur_seq_length = len(self.dataset[start_index][0]) - 1
            cur_batch_size = max(int(self.token_per_batch / cur_seq_length), 1)
            start_index = start_index + cur_batch_size
        self.index_length =len(self.index_list)
        self.index_list.append(dataset_size)
        
        self.shuffle_list = list(range(self.index_length-1, -1, -1))

        self.total_batch_num = self.index_length

class DS_GOLD_MIXED_Dataset(object):
    """
    Training Dataset for Sequence Labeling

    Parameters
    ----------
    dataset_name : ``str``, required.
        The name of dataset (outputs of preprocess scripts).
    w_pad : ``int``, required.
        The pad index for the word-level inputs.
    c_pad : ``int``, required.
        The pad index for the character-level inputs.
    token_per_batch: ``int``, required.
        Batch size.
    sample_ratio: ``float``, optional (default = 1.0)
        The ratio for sampling.
    """    
    def __init__(self, 
                dataset_name: str, 
                w_pad: int, 
                c_pad: int, 
                token_per_batch: int, 
                sample_ratio: float = 1.0):
        
        super(DS_GOLD_MIXED_Dataset, self).__init__()
        self.sample_ratio = sample_ratio

        self.dataset_name = dataset_name

        self.w_pad = w_pad
        self.c_pad = c_pad
        self.token_per_batch = token_per_batch

        self.total_batch_num = -1

        self.open_file()

    def get_tqdm(self, device):
        """
        construct dataset reader and the corresponding tqdm.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        """
        return tqdm(self.reader(device), mininterval=2, total=self.total_batch_num, leave=False, file=sys.stdout).__iter__()

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
        cur_idx = 0

        while cur_idx < self.index_length:

            batch_idx = self.shuffle_list[cur_idx]
            batch = self.dataset[self.index_list[batch_idx]: self.index_list[batch_idx + 1]]

            cur_seq_length = len(batch[0][0])
            word_t = torch.LongTensor([tup[0] + [self.w_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            char_t = torch.LongTensor([tup[1] + [self.c_pad] * (cur_seq_length - len(tup[0])) for tup in batch]).to(device)
            chunk_mask = torch.ByteTensor([tup[2] + [0] * (cur_seq_length - len(tup[2])) for tup in batch]).to(device)
            chunk_label = torch.FloatTensor([label for tup in batch for label in tup[3]]).to(device)
            type_mask = torch.ByteTensor([mask for tup in batch for mask in tup[4]]).to(device)
            label_list = [label for tup in batch for label in tup[5]]
            type_label = torch.FloatTensor(label_list[0:-1]).to(device)

            cur_idx += 1

            yield word_t, char_t, chunk_mask, chunk_label, type_mask, type_label

        random.shuffle(self.shuffle_list)

    def open_file(self):
        """
        Open the dataset by name.      
        """
        self.dataset = pickle.load(open(self.dataset_name, 'rb'))
        self.dataset = list(filter(lambda t: t[6] or random.uniform(0, 1) <= self.sample_ratio, self.dataset))

        dataset_size = len(self.dataset)
        print(dataset_size)
        self.index_list = list()
        start_index = 0
        while start_index < dataset_size:
            self.index_list.append(start_index)
            cur_seq_length = len(self.dataset[start_index][0]) - 1
            cur_batch_size = max(int(self.token_per_batch / cur_seq_length), 1)
            start_index = start_index + cur_batch_size
        self.index_length =len(self.index_list)
        self.index_list.append(dataset_size)
        
        self.shuffle_list = list(range(self.index_length-1, -1, -1))

        self.total_batch_num = self.index_length
