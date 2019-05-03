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

from model_partial_ner.ner import NER, ContextNER
import model_partial_ner.utils as utils
from model_partial_ner.object import softCE
from model_partial_ner.basic import BasicRNN
from model_partial_ner.dataset import NERDataset, TrainDataset

from model_word_ada.LM import LM
# from model_word_ada.basic import BasicRNN
from model_word_ada.densenet import DenseRNN
from model_word_ada.ldnet import LDRNN
from model_seq.seqlm import BasicSeqLM 

from torch_scope import wrapper

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
    parser.add_argument('--cp_root', default='./checkpoint')
    parser.add_argument('--checkpoint_name', default='autoner0')
    parser.add_argument('--git_tracking', action='store_true')

    parser.add_argument('--eval_dataset', default='./data/hqner/train_0.pk')
    parser.add_argument('--train_dataset', default='./data/hqner/test.pk')
    parser.add_argument('--batch_token_number', type=int, default=3000)
    parser.add_argument('--label_dim', type=int, default=50)
    parser.add_argument('--hid_dim', type=int, default=300)
    parser.add_argument('--word_dim', type=int, default=100)
    parser.add_argument('--char_dim', type=int, default=30)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--update', choices=['Adam', 'Adagrad', 'Adadelta', 'SGD'], default='Adam')
    parser.add_argument('--rnn_layer', choices=['Basic'], default='Basic')
    parser.add_argument('--rnn_unit', choices=['gru', 'lstm', 'rnn'], default='lstm')
    parser.add_argument('--lr', type=float, default=-1)
    parser.add_argument('--tolerance', type=int, default=5)
    parser.add_argument('--lr_decay', type=float, default=0.05)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--check', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--model', choices=['NER', "ContextNER"], default='NER');
    parser.add_argument('--context_data', default='./lm/lm_maps.pk')
    parser.add_argument('--forward_lm', default='./lm/ld0.th')
    parser.add_argument('--backward_lm', default='./lm/ld_0.th')
    parser.add_argument('--lm_label_dim', type=int, default=-1)
    parser.add_argument('--lm_word_dim', type=int, default=300)
    parser.add_argument('--lm_droprate', type=float, default=0.5)
    parser.add_argument('--lm_rnn_layer', choices=['Basic', 'DenseNet', 'LDNet'], default='LDNet')
    parser.add_argument('--lm_rnn_unit', choices=['gru', 'lstm', 'rnn'], default='lstm')
    parser.add_argument('--lm_hid_dim', type=int, default=300)
    parser.add_argument('--lm_layer_num', type=int, default=10)
 
    args = parser.parse_args()

    pw = wrapper(os.path.join(args.cp_root, args.checkpoint_name), args.checkpoint_name, enable_git_track=args.git_tracking, seed = args.seed)

    gpu_index = pw.auto_device() if 'auto' == args.gpu else int(args.gpu)
    device = torch.device("cuda:" + str(gpu_index) if gpu_index >= 0 else "cpu")
    
    logger.info('loading dataset')
    key_list = ['emb_array', 'w_map', 'c_map', 'tl_map', 'cl_map', 'range', 'test_data', 'dev_data']
    dataset = pickle.load(open(args.eval_dataset, 'rb'))
    emb_array, w_map, c_map, tl_map, cl_map, range_idx, test_data, dev_data = [dataset[tup] for tup in key_list]
    id2label = {v: k for k, v in tl_map.items()}
    assert len(emb_array) == len(w_map)
    ct_dataset = pickle.load(open(args.context_data, 'rb'))
    ct_name_list = ['flm_map', 'blm_map']
    flm_map, blm_map = [ct_dataset[tup] for tup in ct_name_list ]

    train_loader = TrainDataset(args.train_dataset, w_map['<\n>'], c_map['<\n>'], args.batch_token_number, flm_map['\n'], blm_map['\n'], sample_ratio = args.sample_ratio)
    test_loader = NERDataset(test_data, w_map['<\n>'], c_map['<\n>'], args.batch_token_number, flm_map['\n'], blm_map['\n'])
    dev_loader = NERDataset(dev_data, w_map['<\n>'], c_map['<\n>'], args.batch_token_number, flm_map['\n'], blm_map['\n'])

    logger.info('building model')


    rnn_map = {'Basic': BasicRNN}
    rnn_layer = rnn_map[args.rnn_layer](args.layer_num, args.rnn_unit, args.word_dim + args.char_dim, args.hid_dim, args.droprate, args.batch_norm)

    
    
    lm_rnn_map = {'Basic': BasicRNN, 'LDNet': functools.partial(LDRNN, layer_drop = 0)}
    flm_rnn_layer = lm_rnn_map[args.lm_rnn_layer](args.lm_layer_num, args.lm_rnn_unit, args.lm_word_dim, args.lm_hid_dim, args.lm_droprate)
    blm_rnn_layer = lm_rnn_map[args.lm_rnn_layer](args.lm_layer_num, args.lm_rnn_unit, args.lm_word_dim, args.lm_hid_dim, args.lm_droprate)
    flm_model = LM(flm_rnn_layer, None, len(flm_map), args.lm_word_dim, args.lm_droprate, label_dim = args.lm_label_dim)
    blm_model = LM(blm_rnn_layer, None, len(blm_map), args.lm_word_dim, args.lm_droprate, label_dim = args.lm_label_dim)
    flm_file = wrapper.restore_checkpoint(args.forward_lm)['model']
    flm_model.load_state_dict(flm_file, False)
    blm_file = wrapper.restore_checkpoint(args.backward_lm)['model']
    blm_model.load_state_dict(blm_file, False)
    flm_model_seq = BasicSeqLM(flm_model, False, args.lm_droprate, True)
    blm_model_seq = BasicSeqLM(blm_model, True, args.lm_droprate, True)
    

    model_map = {'NER': NER, 'ContextNER': ContextNER}
    ner_model = model_map[args.model](rnn_layer, len(w_map), args.word_dim, len(c_map), args.char_dim, args.label_dim, len(tl_map), args.droprate, flm_model_seq, blm_model_seq)

    ner_model.rand_ini()
    ner_model.load_pretrained_word_embedding(torch.FloatTensor(emb_array))
    ner_config = ner_model.to_params()
    ner_model.to(device)
    
    optim_map = {'Adam' : optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta, 'SGD': functools.partial(optim.SGD, momentum=0.9)}
    if args.lr > 0:
        optimizer=optim_map[args.update](ner_model.parameters(), lr=args.lr)
    else:
        optimizer=optim_map[args.update](ner_model.parameters())

    crit_chunk = nn.BCEWithLogitsLoss()
    crit_type = softCE()

    # writer = SummaryWriter(log_dir='./partial/'+args.log_dir)
    # name_list = ['loss_chunk', 'loss_type', 'scores_chunking', 'scores_typing', 'scores_ner']
    # c_loss, t_loss, chunk_sco, type_sco, ner_sco = [args.log_dir+'/'+tup for tup in name_list]

    logger.info('Saving configues.')

    pw.save_configue(args)
    pw.save_configue({'w_map': w_map, 'c_map': c_map, 'tl_map': tl_map, 'cl_map': cl_map}, 'dict.json')

    logger.info('Setting up training environ.')
                        
    batch_index = 0
    best_eval = float('-inf')
    best_f1, best_pre, best_rec = -1, -1, -1
    best_type2f1, best_type2pre, best_type2rec = {}, {}, {}

    patience = 0
    current_lr = args.lr
    tolerance = args.tolerance

    try:

        for indexs in range(args.epoch):

            logger.info('############')
            logger.info('Epoch: {}'.format(indexs))
            pw.nvidia_memory_map(gpu_index = gpu_index)

            ner_model.train()

            for word_t, char_t, chunk_mask, chunk_label, type_mask, type_label, flm_w, blm_w, blm_ind in train_loader.get_tqdm(device):

                ner_model.zero_grad()
                output = ner_model(flm_w, blm_w, blm_ind, word_t, char_t, chunk_mask)

                chunk_score = ner_model.chunking(output)
                chunk_loss = crit_chunk(chunk_score, chunk_label)

                type_score = ner_model.typing(output, type_mask)
                type_loss = crit_type(type_score, type_label)

                loss = type_loss + chunk_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ner_model.parameters(), args.clip)
                optimizer.step()

                batch_index += 1 

                if 0 == batch_index % args.interval:
                    pw.add_loss_vs_batch({'loss_chunk': utils.to_scalar(chunk_loss), 'loss_type': utils.to_scalar(type_loss)}, batch_index, use_logger = False)
                

                if 0 == batch_index % args.check:

                    # NER evaluation
                    pre_dev, rec_dev, f1_dev, type2pre_dev, type2rec_dev, type2f1_dev = utils.evaluate_ner(dev_loader.get_tqdm(device), ner_model, tl_map['None'], id2label)
                    pw.add_loss_vs_batch({'dev_pre': pre_dev, 'dev_rec': rec_dev}, batch_index, use_logger = False)
                    pw.add_loss_vs_batch({'dev_f1': f1_dev}, batch_index, use_logger = True)

                    pw.save_checkpoint(model = ner_model, is_best = f1_dev > best_eval, s_dict = {'config': ner_config, 'w_map': w_map, 'c_map': c_map, 'tl_map': tl_map, 'cl_map': cl_map})

                    if f1_dev > best_eval:
                        best_eval = f1_dev
                        # best_f1, best_pre, best_rec, best_type2pre, best_type2rec, best_type2f1 = utils.evaluate_ner(test_loader.get_tqdm(device), ner_model, tl_map['None'])
                        best_pre, best_rec, best_f1, best_type2pre, best_type2rec, best_type2f1 = utils.evaluate_ner(test_loader.get_tqdm(device), ner_model, tl_map['None'], id2label)
                        pw.add_loss_vs_batch({'test_pre': best_pre, 'test_rec': best_rec}, batch_index, use_logger = False)
                        pw.add_loss_vs_batch({'test_f1': best_f1}, batch_index, use_logger = True)
                        patience = 0
                        for entity_type in best_type2f1:
                            pw.add_loss_vs_batch({'per_{}_f1'.format(entity_type): best_type2f1[entity_type], 
                                                    'per_{}_pre'.format(entity_type): best_type2pre[entity_type],
                                                    'per_{}_rec'.format(entity_type): best_type2rec[entity_type]}, batch_index, use_logger = False)
                            logger.info('\ttype: %s, f1: %.6f, pre: %.6f, rec: %.6f' % (entity_type, best_type2f1[entity_type], best_type2pre[entity_type], best_type2rec[entity_type]))

                    else:
                        patience += 1
                        if patience >= tolerance:
                            patience = 0
                            current_lr *= 0.9 ** tolerance
                            if args.update == 'SGD':
                                utils.adjust_learning_rate(optimizer, current_lr)
                                logger.info('current_lr = %.10f' % current_lr)

                    ner_model.train()

    except KeyboardInterrupt:

        print('Exiting from training early')

        # NER evaluation
        pre_dev, rec_dev, f1_dev, type2pre_dev, type2rec_dev, type2f1_dev = utils.evaluate_ner(dev_loader.get_tqdm(device), ner_model, tl_map['None'], id2label)
        pw.add_loss_vs_batch({'dev_pre': pre_dev, 'dev_rec': rec_dev}, batch_index, use_logger = False)
        pw.add_loss_vs_batch({'dev_f1': f1_dev}, batch_index, use_logger = True)

        pw.save_checkpoint(model = ner_model, is_best = f1_dev > best_eval, s_dict = {'config': ner_config, 'w_map': w_map, 'c_map': c_map, 'tl_map': tl_map, 'cl_map': cl_map})

        if f1_dev > best_eval:
            best_eval = f1_dev
            best_pre, best_rec, best_f1, best_type2pre, best_type2rec, best_type2f1 = utils.evaluate_ner(test_loader.get_tqdm(device), ner_model, tl_map['None'], id2label)
            pw.add_loss_vs_batch({'test_pre': best_f1, 'test_rec': best_pre}, batch_index, use_logger = False)
            pw.add_loss_vs_batch({'test_f1': best_rec}, batch_index, use_logger = True)
            patience = 0
            for entity_type in best_type2f1:
                pw.add_loss_vs_batch({'per_%s_f1'.format(entity_type): best_type2f1[entity_type], 
                                        'per_%s_pre'.format(entity_type): best_type2pre[entity_type],
                                        'per_%s_rec'.format(entity_type): best_type2rec[entity_type]}, batch_index, use_logger = False)
                logger.info('\ttype: %s, f1: %.6f, pre: %.6f, rec: %.6f' % (entity_type, best_type2f1[entity_type], best_type2pre[entity_type], best_type2rec[entity_type]))

    print ('\nbest dev f1: %.6f, corresponding test f1: %.6f' % (best_eval, best_f1))

    pw.close()
