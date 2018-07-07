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
#[tmp_w, tmp_c, tmp_mc, tmp_idx]

    def __init__(self, dataset, w_pad, c_pad, token_per_batch):
        super(RawDataset, self).__init__()
        self.dataset = dataset
        self.w_pad = w_pad
        self.c_pad = c_pad
        self.token_per_batch = token_per_batch

        self.construct_index()

    def get_tqdm(self):
        return tqdm(self, mininterval=2, total=self.index_length, leave=False, file=sys.stdout)

    def construct_index(self):

        self.index_length =len(self.dataset)

        self.cur_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_idx == self.index_length:
            self.cur_idx = 0
            raise StopIteration

        batch = self.dataset[self.cur_idx]

        word_t = autograd.Variable(torch.LongTensor([batch[0]])).cuda()
        char_t = autograd.Variable(torch.LongTensor([batch[1]])).cuda()
        chunk_mask = autograd.Variable(torch.ByteTensor([batch[2]])).cuda()
        chunk_index = autograd.Variable(torch.LongTensor(batch[3])).cuda()
        chunk_surface = batch[4]

        self.cur_idx += 1
        return word_t, char_t, chunk_mask, chunk_index, chunk_surface


class NERDataset(object):
#[tmp_w, tmp_c, tmp_mc, tmp_lc, tmp_mt, tmp_lt]

    def __init__(self, dataset, w_pad, c_pad, token_per_batch):
        super(NERDataset, self).__init__()
        self.dataset = dataset
        self.w_pad = w_pad
        self.c_pad = c_pad
        self.token_per_batch = token_per_batch

        self.construct_index()

    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self):
        return tqdm(self, mininterval=2, total=self.index_length, leave=False, file=sys.stdout)

    def construct_index(self):

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

        self.cur_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_idx == self.index_length:
            self.cur_idx = 0
            self.shuffle()
            raise StopIteration

        batch_idx = self.shuffle_list[self.cur_idx]
        batch = self.dataset[self.index_list[batch_idx]: self.index_list[batch_idx + 1]]


        cur_seq_length = len(batch[0][0])
        word_t = autograd.Variable(torch.LongTensor([tup[0] + [self.w_pad] * (cur_seq_length - len(tup[0])) for tup in batch])).cuda()
        char_t = autograd.Variable(torch.LongTensor([tup[1] + [self.c_pad] * (cur_seq_length - len(tup[0])) for tup in batch])).cuda()
        chunk_mask = autograd.Variable(torch.ByteTensor([tup[2] + [0] * (cur_seq_length - len(tup[2])) for tup in batch])).cuda()
        chunk_label = autograd.Variable(torch.FloatTensor([label for tup in batch for label in tup[3]])).cuda()
        type_mask = autograd.Variable(torch.ByteTensor([mask for tup in batch for mask in tup[4]])).cuda()
        label_list = [label for tup in batch for label in tup[5]]
        type_label = autograd.Variable(torch.FloatTensor(label_list[0:-1])).cuda()

        self.cur_idx += 1

        return word_t, char_t, chunk_mask, chunk_label, type_mask, type_label

class LargeDataset(object):

    def __init__(self, root, range_idx, w_pad, c_pad, token_per_batch, sample_ratio=1.0):
        
        super(LargeDataset, self).__init__()
        self.sample_ratio = sample_ratio

        self.root = root
        self.range_idx = range_idx

        self.w_pad = w_pad
        self.c_pad = c_pad
        self.token_per_batch = token_per_batch

        self.total_batch_num = -1

        self.open_file()

    def get_tqdm(self):

        return tqdm(self, mininterval=2, total=self.total_batch_num, leave=False, file=sys.stdout).__iter__()

    def __iter__(self):

        self.cur_idx = 0
        return self

    def __next__(self):

        if self.cur_idx >= self.index_length:
            self.open_next()

        batch_idx = self.shuffle_list[self.cur_idx]
        batch = self.dataset[self.index_list[batch_idx]: self.index_list[batch_idx + 1]]

        cur_seq_length = len(batch[0][0])
        word_t = autograd.Variable(torch.LongTensor([tup[0] + [self.w_pad] * (cur_seq_length - len(tup[0])) for tup in batch])).cuda()
        char_t = autograd.Variable(torch.LongTensor([tup[1] + [self.c_pad] * (cur_seq_length - len(tup[0])) for tup in batch])).cuda()
        chunk_mask = autograd.Variable(torch.ByteTensor([tup[2] + [0] * (cur_seq_length - len(tup[2])) for tup in batch])).cuda()
        chunk_label = autograd.Variable(torch.FloatTensor([label for tup in batch for label in tup[3]])).cuda()
        type_mask = autograd.Variable(torch.ByteTensor([mask for tup in batch for mask in tup[4]])).cuda()
        label_list = [label for tup in batch for label in tup[5]]
        type_label = autograd.Variable(torch.FloatTensor(label_list[0:-1])).cuda()

        self.cur_idx += 1

        return word_t, char_t, chunk_mask, chunk_label, type_mask, type_label

    def open_next(self):

        random.shuffle(self.shuffle_list)
        self.cur_idx = 0

        raise StopIteration

    def open_file(self):

        self.dataset = pickle.load(open(self.root + 'train_0.pk', 'rb'))

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

        self.cur_idx = 0

        self.total_batch_num = self.index_length

class DS_GOLD_MIXED_Dataset(object):

    def __init__(self, root, range_idx, w_pad, c_pad, token_per_batch, sample_ratio=1.0):
        
        super(DS_GOLD_MIXED_Dataset, self).__init__()
        self.sample_ratio = sample_ratio

        self.root = root
        self.range_idx = range_idx

        self.w_pad = w_pad
        self.c_pad = c_pad
        self.token_per_batch = token_per_batch

        self.total_batch_num = -1

        self.open_file()

    def get_tqdm(self):

        return tqdm(self, mininterval=2, total=self.total_batch_num, leave=False, file=sys.stdout).__iter__()

    def __iter__(self):

        self.cur_idx = 0
        return self

    def __next__(self):

        if self.cur_idx >= self.index_length:
            self.open_next()

        batch_idx = self.shuffle_list[self.cur_idx]
        batch = self.dataset[self.index_list[batch_idx]: self.index_list[batch_idx + 1]]

        cur_seq_length = len(batch[0][0])
        word_t = autograd.Variable(torch.LongTensor([tup[0] + [self.w_pad] * (cur_seq_length - len(tup[0])) for tup in batch])).cuda()
        char_t = autograd.Variable(torch.LongTensor([tup[1] + [self.c_pad] * (cur_seq_length - len(tup[0])) for tup in batch])).cuda()
        chunk_mask = autograd.Variable(torch.ByteTensor([tup[2] + [0] * (cur_seq_length - len(tup[2])) for tup in batch])).cuda()
        chunk_label = autograd.Variable(torch.FloatTensor([label for tup in batch for label in tup[3]])).cuda()
        type_mask = autograd.Variable(torch.ByteTensor([mask for tup in batch for mask in tup[4]])).cuda()
        label_list = [label for tup in batch for label in tup[5]]
        type_label = autograd.Variable(torch.FloatTensor(label_list[0:-1])).cuda()

        self.cur_idx += 1

        return word_t, char_t, chunk_mask, chunk_label, type_mask, type_label

    def open_next(self):

        random.shuffle(self.shuffle_list)
        self.cur_idx = 0

        raise StopIteration

    def open_file(self):

        self.dataset = pickle.load(open(self.root + 'train_0.pk', 'rb'))
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

        self.cur_idx = 0

        self.total_batch_num = self.index_length
