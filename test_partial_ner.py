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
import numpy as np

from model_partial_ner.ner import NER
from model_partial_ner.basic import BasicRNN
from model_partial_ner.dataset import RawDataset
import model_partial_ner.utils as utils

from torch_scope import basic_wrapper as bw

import argparse
import logging
import json
import os
import sys
import itertools
import functools

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="auto")
    parser.add_argument('--checkpoint_folder', default='models/BC5DR/checkpoint/autoner/')

    parser.add_argument('--input_corpus', default='./data/target.pk')
    parser.add_argument('--output_text', default='./data/output_text.tsv')
    parser.add_argument('--batch_token_number', type=int, default=3000)
    parser.add_argument('--label_dim', type=int, default=50)
    parser.add_argument('--hid_dim', type=int, default=300)
    parser.add_argument('--word_dim', type=int, default=100)
    parser.add_argument('--char_dim', type=int, default=30)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--rnn_layer', choices=['Basic'], default='Basic')
    parser.add_argument('--rnn_unit', choices=['gru', 'lstm', 'rnn'], default='lstm')
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.0)
    args = parser.parse_args()
    
    gpu_index = bw.auto_device() if 'auto' == args.gpu else int(args.gpu)
    device = torch.device("cuda:" + str(gpu_index) if gpu_index >= 0 else "cpu")
    if gpu_index >= 0:
        torch.cuda.set_device(gpu_index)

    logger.info('loading checkpoint')
    # dictionary = bw.restore_configue(args.checkpoint_folder, name = 'dict.json')
    # w_map, c_map, tl_map = dictionary['w_map'], dictionary['c_map'], dictionary['tl_map']
    checkpoint_file = bw.restore_best_checkpoint(args.checkpoint_folder)
    w_map, c_map, tl_map, model = [checkpoint_file[name] for name in ['w_map', 'c_map', 'tl_map', 'model']]
    id2label = {v: k for k, v in tl_map.items()}

    logger.info('loading dataset')
    raw_data = pickle.load(open(args.input_corpus, 'rb'))
    data_loader = RawDataset(raw_data, w_map['<\n>'], c_map['<\n>'], args.batch_token_number)

    logger.info('building model')
    rnn_map = {'Basic': BasicRNN}
    rnn_layer = rnn_map[args.rnn_layer](args.layer_num, args.rnn_unit, args.word_dim + args.char_dim, args.hid_dim, args.droprate, args.batch_norm)
    ner_model = NER(rnn_layer, len(w_map), args.word_dim, len(c_map), args.char_dim, args.label_dim, len(tl_map), args.droprate)
    ner_model.load_state_dict(model)
    ner_model.to(device)
    ner_model.eval()

    output_list = list()

    fout = open(args.output_text, 'w')

    iterator = data_loader.get_tqdm(device)
    max_score = -float('inf')
    min_score = float('inf')

    for word_t, char_t, chunk_mask, chunk_index, chunk_surface in iterator:
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
        offset = chunk_index[0]
        for ind in range(0, output.size(0)):
            st, ed = chunk_index[ind].item(), chunk_index[ind + 1].item()
            surface = ' '.join(chunk_surface[st - offset : ed - offset])
            ent_type_id = np.argmax(output[ind]).item()
            ent_type = id2label[ent_type_id]

            values = [st, ed, surface, ent_type_id, ent_type]
            str_values = [str(v) for v in values]
            fout.write('\t'.join(str_values) + '\n')
            # logger.info('\t'.join(str_values) + '\n')
        fout.write('\n')
        # logger.info('\n')

    logger.info('max: '+str(max_score))
    logger.info('min: '+str(min_score))
    fout.close()
