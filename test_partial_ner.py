from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
import pickle
import math

from model_partial_ner.ner import NER
from model_partial_ner.basic import BasicRNN
from model_partial_ner.dataset import RawDataset
import model_partial_ner.utils as utils

from tensorboardX import SummaryWriter

import argparse
import json
import os
import sys
import itertools
import functools

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_corpus', default='./data/target.pk')
    parser.add_argument('--output_text', default='./data/output_text.tsv')
    parser.add_argument('--load_checkpoint', default='./checkpoint/ner_basic.model')
    parser.add_argument('--batch_token_number', type=int, default=3000)
    parser.add_argument('--label_dim', type=int, default=50)
    parser.add_argument('--hid_dim', type=int, default=300)
    parser.add_argument('--word_dim', type=int, default=100)
    parser.add_argument('--char_dim', type=int, default=30)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--rnn_layer', choices=['Basic'], default='Basic')
    parser.add_argument('--rnn_unit', choices=['gru', 'lstm', 'rnn'], default='lstm')
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--bi_type', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.0)
    args = parser.parse_args()

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    print('loading dataset')
    checkpoint = torch.load(open(args.load_checkpoint, 'rb'), map_location=lambda storage, loc: storage)

    w_map, c_map, model, tl_map = checkpoint['w_map'], checkpoint['c_map'], checkpoint['model'], checkpoint['tl_map']

    raw_data = pickle.load(open(args.input_corpus, 'rb'))

    data_loader = RawDataset(raw_data, w_map['<\n>'], c_map['<\n>'], args.batch_token_number)

    print('building model')

    rnn_map = {'Basic': BasicRNN}#, 'DenseNet': DenseRNN}
    rnn_layer = rnn_map[args.rnn_layer](args.layer_num, args.rnn_unit, args.word_dim + args.char_dim, args.hid_dim, args.droprate, args.batch_norm)

    ner_model = NER(rnn_layer, len(w_map), args.word_dim, len(c_map), args.char_dim, args.label_dim, len(tl_map), args.droprate, args.bi_type)
    
    ner_model.load_state_dict(model)

    ner_model.cuda()

    ner_model.eval()

    output_list = list()

    fout = open(args.output_text, 'w')

    iterator = data_loader.get_tqdm()
    max_score = -1000000000000
    min_score = 1000000000000

    for word_t, char_t, chunk_mask, chunk_index in iterator:
        output = ner_model(word_t, char_t, chunk_mask)
        chunk_score = ner_model.chunking(output)

        tmp_min = utils.to_scalar(chunk_score.min())
        tmp_max = utils.to_scalar(chunk_score.max())
        max_score = max(max_score, tmp_max)
        min_score = min(min_score, tmp_min)

        pred_chunk = (chunk_score < args.threshold)

        chunk_index = chunk_index.masked_select(pred_chunk).data.cpu()

        output = ner_model.typing(output, pred_chunk)

        output = output.data.cpu()

        for ind in range(0, output.size(0)):
            fout.write(str(chunk_index[ind]) + '\t' + str(chunk_index[ind+1]) + '\t' + '\t'.join([str(v) for v in output[ind]]) +'\n')
        fout.write('\n')

    print('max: '+str(max_score))
    print('min: '+str(min_score))
    fout.close()