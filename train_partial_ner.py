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
import model_partial_ner.utils as utils
from model_partial_ner.object import softCE
from model_partial_ner.basic import BasicRNN
from model_partial_ner.resnet import ResRNN
from model_partial_ner.dataset import NERDataset, LargeDataset

from tensorboardX import SummaryWriter

import argparse
import json
import os
import sys
import itertools
import functools

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='ner_0')
    parser.add_argument('--dataset_folder', default='./data/hqner/')
    parser.add_argument('--checkpoint', default='./checkpoint/ner_basic_new.model')
    parser.add_argument('--batch_token_number', type=int, default=3000)
    parser.add_argument('--label_dim', type=int, default=50)
    parser.add_argument('--hid_dim', type=int, default=300)
    parser.add_argument('--word_dim', type=int, default=100)
    parser.add_argument('--char_dim', type=int, default=30)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--update', choices=['Adam', 'Adagrad', 'Adadelta', 'SGD'], default='Adam')
    parser.add_argument('--rnn_layer', choices=['Basic', 'ResNet'], default='Basic')
    parser.add_argument('--rnn_unit', choices=['gru', 'lstm', 'rnn'], default='lstm')
    parser.add_argument('--lr', type=float, default=-1)
    parser.add_argument('--tolerance', type=int, default=5)
    parser.add_argument('--lr_decay', type=float, default=0.05)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--check', type=int, default=1000)
    parser.add_argument('--bi_type', action='store_true')
    args = parser.parse_args()

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    print('loading dataset')
    key_list = ['emb_array', 'w_map', 'c_map', 'tl_map', 'cl_map', 'range', 'test_data', 'dev_data']
    dataset = pickle.load(open(args.dataset_folder+'test.pk', 'rb'))
    emb_array, w_map, c_map, tl_map, cl_map, range_idx, test_data, dev_data = [dataset[tup] for tup in key_list]

    train_loader = LargeDataset(args.dataset_folder, range_idx, w_map['<\n>'], c_map['<\n>'], args.batch_token_number, sample_ratio = args.sample_ratio)
    test_loader = NERDataset(test_data, w_map['<\n>'], c_map['<\n>'], args.batch_token_number)
    dev_loader = NERDataset(dev_data, w_map['<\n>'], c_map['<\n>'], args.batch_token_number)

    print('building model')

    rnn_map = {'Basic': BasicRNN, 'ResNet': ResRNN}
    rnn_layer = rnn_map[args.rnn_layer](args.layer_num, args.rnn_unit, args.word_dim + args.char_dim, args.hid_dim, args.droprate, args.batch_norm)

    ner_model = NER(rnn_layer, len(w_map), args.word_dim, len(c_map), args.char_dim, args.label_dim, len(tl_map), args.droprate, args.bi_type)

    ner_model.rand_ini()
    ner_model.load_pretrained_word_embedding(torch.FloatTensor(emb_array))
    ner_model.cuda()
    
    optim_map = {'Adam' : optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta, 'SGD': functools.partial(optim.SGD, momentum=0.9)}
    if args.lr > 0:
        optimizer=optim_map[args.update](ner_model.parameters(), lr=args.lr)
    else:
        optimizer=optim_map[args.update](ner_model.parameters())

    # crit_chunk = utils.hinge_loss
    crit_chunk = nn.BCEWithLogitsLoss()
    crit_type = softCE()

    writer = SummaryWriter(log_dir='./partial/'+args.log_dir)
    name_list = ['loss_chunk', 'loss_type', 'scores_chunking', 'scores_typing', 'scores_ner']
    c_loss, t_loss, chunk_sco, type_sco, ner_sco = [args.log_dir+'/'+tup for tup in name_list]

    batch_index = 0
    best_eval = float('-inf')
    best_f1, best_pre, best_rec = -1, -1, -1
    best_type2f1, best_type2pre, best_type2rec = {}, {}, {}

    patience = 0
    current_lr = args.lr
    tolerance = args.tolerance

    try:

        for indexs in range(args.epoch):

            updated = False

            iterator = train_loader.get_tqdm()

            ner_model.train()

            for word_t, char_t, chunk_mask, chunk_label, type_mask, type_label in iterator:
                ner_model.zero_grad()
                output = ner_model(word_t, char_t, chunk_mask)

                chunk_score = ner_model.chunking(output)
                chunk_loss = crit_chunk(chunk_score, chunk_label)

                type_score = ner_model.typing(output, type_mask)
                type_loss = crit_type(type_score, type_label)

                loss = type_loss + chunk_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm(ner_model.parameters(), args.clip)
                optimizer.step()

                batch_index += 1 

                if 0 == batch_index % args.interval:

                    writer.add_scalar(c_loss, utils.to_scalar(chunk_loss), batch_index)
                    writer.add_scalar(t_loss, utils.to_scalar(type_loss), batch_index)

                if 0 == batch_index % args.check:

                    # Chunking evaluation
                    # pre_dev, rec_dev, f1_dev = utils.evaluate_chunking(dev_loader.get_tqdm(), ner_model, tl_map['None'])
                    # pre_test, rec_test, f1_test = utils.evaluate_chunking(test_loader.get_tqdm(), ner_model, tl_map['None'])
                    # writer.add_scalars(chunk_sco, {'dev_f1': f1_dev, 'dev_pre': pre_dev, 'dev_rec': rec_dev, 'test_f1': f1_test, 'test_pre': pre_test, 'test_rec': rec_test}, batch_index)
                    # print ('\n[chunking, test] f1 = %.6f, pre = %.6f, rec = %.6f' % (f1_test, pre_test, rec_test))

                    # Typing evaluation
                    # pre_dev, rec_dev, f1_dev = utils.evaluate_typing(dev_loader.get_tqdm(), ner_model, tl_map['None'])
                    # pre_test, rec_test, f1_test = utils.evaluate_typing(test_loader.get_tqdm(), ner_model, tl_map['None'])
                    # writer.add_scalars(type_sco, {'dev_f1': f1_dev, 'dev_pre': pre_dev, 'dev_rec': rec_dev, 'test_f1': f1_test, 'test_pre': pre_test, 'test_rec': rec_test}, batch_index)
                    # print ('\n[typing, test] f1 = %.6f, pre = %.6f, rec = %.6f' % (f1_test, pre_test, rec_test))

                    # NER evaluation
                    pre_dev, rec_dev, f1_dev, type2pre_dev, type2rec_dev, type2f1_dev = utils.evaluate_ner(dev_loader.get_tqdm(), ner_model, tl_map['None'])
                    pre_test, rec_test, f1_test, type2pre_test, type2rec_test, type2f1_test = utils.evaluate_ner(test_loader.get_tqdm(), ner_model, tl_map['None'])
                    writer.add_scalars(ner_sco, {'dev_f1': f1_dev, 'dev_pre': pre_dev, 'dev_rec': rec_dev, 'test_f1': f1_test, 'test_pre': pre_test, 'test_rec': rec_test}, batch_index)
                    print ('\n[ner, test] f1 = %.6f, pre = %.6f, rec = %.6f' % (f1_test, pre_test, rec_test))

                    if f1_dev > best_eval:
                        torch.save({'model': ner_model.state_dict(), 'w_map': w_map, 'c_map': c_map, 'tl_map': tl_map, 'cl_map': cl_map}, args.checkpoint)
                        best_eval = f1_dev
                        best_f1 = f1_test
                        best_pre = pre_test
                        best_rec = rec_test
                        best_type2f1, best_type2pre, best_type2rec = type2f1_test, type2pre_test, type2rec_test
                        patience = 0
                    else:
                        patience += 1
                        if patience >= tolerance:
                            patience = 0
                            current_lr *= 0.9 ** tolerance
                            if args.update == 'SGD':
                                utils.adjust_learning_rate(optimizer, current_lr)
                                print ('current_lr = %.10f' % current_lr)
                    print ('\nbest dev f1: %.6f, corresponding test f1: %.6f, pre: %.6f, rec: %.6f' % (best_eval, best_f1, best_pre, best_rec))
                    for entity_type in best_type2f1:
                        print('\ttype: %s, f1: %.6f, pre: %.6f, rec: %.6f' % (entity_type, best_type2f1[entity_type], best_type2pre[entity_type], best_type2rec[entity_type]))

            # if args.update == 'SGD':
            #     current_lr = args.lr / (1 + (indexs + 1) * args.lr_decay)
            #     utils.adjust_learning_rate(optimizer, current_lr)
            #     print ('current_lr = %.10f' % current_lr)

    except KeyboardInterrupt:

        print('Exiting from training early')

        # NER evaluation
        pre_dev, rec_dev, f1_dev = utils.evaluate_ner(dev_loader.get_tqdm(), ner_model, tl_map['None'])
        pre_test, rec_test, f1_test = utils.evaluate_ner(test_loader.get_tqdm(), ner_model, tl_map['None'])
        writer.add_scalars(ner_sco, {'dev_f1': f1_dev, 'dev_pre': pre_dev, 'dev_rec': rec_dev, 'test_f1': f1_test, 'test_pre': pre_test, 'test_rec': rec_test}, batch_index)

        if f1_dev > best_eval:
            torch.save({'model': ner_model.state_dict(), 'w_map': w_map, 'c_map': c_map, 'tl_map': tl_map, 'cl_map': cl_map}, args.checkpoint)
            best_eval = f1_dev
            best_f1 = f1_test

    print ('\nbest dev f1: %.6f, corresponding test f1: %.6f' % (best_eval, best_f1))

    writer.close()