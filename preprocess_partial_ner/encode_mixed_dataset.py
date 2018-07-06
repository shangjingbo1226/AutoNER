import pickle
import argparse
import os
import random
import numpy as np
from tqdm import tqdm

import itertools
import functools

def filter_words(w_map, emb_array, ck_filenames):
    vocab = set()
    for filename in ck_filenames:
        for line in open(filename, 'r'):
            if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
                line = line.rstrip('\n').split()
                assert len(line) >= 3, 'wrong ck file format'
                word = line[0]
                vocab.add(word)
    new_w_map = {}
    new_emb_array = []
    for (word, idx) in w_map.items():
        if word in vocab:
            new_w_map[word] = len(new_emb_array)
            new_emb_array.append(emb_array[idx])
    for word in ['<unk>', '<s>', '< >', '<\n>']:
        idx = w_map[word]
        new_w_map[word] = len(new_emb_array)
        new_emb_array.append(emb_array[idx])
    print('filtered %d --> %d' % (len(emb_array), len(new_emb_array)))
    return new_w_map, new_emb_array


def read_noisy_corpus(lines):
    features, labels_chunk, labels_chunk_mask, labels_point, labels_typing, is_gold = list(), list(), list(), list(), list(), list()

    tmp_fl, tmp_lpl, tmp_lcml, tmp_lcl, tmp_ltl = list(), list(), list(), list(), list()
    could_be_gold = True
    for line in lines:

        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            
            assert len(line) >= 3 and len(line) <= 4, "the format of noisy corpus"
            # The format should be
            # 0. Token
            # 1. I/O (I means Break, O means Connected)
            # 2. Type (separated by comma)
            # 3. Safe or dangerous?   <-- this is optional
            token = line[0]
            chunk_boundary = line[1]
            entity_types = line[2]

            if len(line) == 3:
                safe = 1
            else:
                safe = int(line[3] == 'S')
                coould_be_gold = False

            tmp_fl.append(token)
            tmp_lcml.append(safe)
            if safe:
                tmp_lcl.append(chunk_boundary)
                if 'I' == chunk_boundary:
                    type_list = entity_types.split(',')
                    tmp_lpl.append(1)
                    tmp_ltl.append(type_list)
                else:
                    tmp_lpl.append(0)
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels_chunk.append(tmp_lcl)
            labels_chunk_mask.append(tmp_lcml)
            labels_point.append(tmp_lpl)
            labels_typing.append(tmp_ltl)
            is_gold.append(coould_be_gold)
            tmp_fl, tmp_lpl, tmp_lcml, tmp_lcl, tmp_ltl = list(), list(), list(), list(), list()
            coould_be_gold = True

    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels_chunk.append(tmp_lcl)
        labels_chunk_mask.append(tmp_lcml)
        labels_point.append(tmp_lpl)
        labels_typing.append(tmp_ltl)
        is_gold.append(coould_be_gold)

    return features, labels_chunk, labels_chunk_mask, labels_point, labels_typing, is_gold

def read_corpus(lines):
    features, labels_chunk, labels_point, labels_typing = list(), list(), list(), list()

    tmp_fl, tmp_lpl, tmp_lcl, tmp_ltl = list(), list(), list(), list()

    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()

            assert len(line) == 3, "the format of corpus"
            # The format should be
            # 0. Token
            # 1. I/O (I means Break, O means Connected)
            # 2. Type (separated by comma)
            token = line[0]
            chunk_boundary = line[1]
            entity_types = line[2]

            tmp_fl.append(token)
            tmp_lcl.append(chunk_boundary)
            if 'I' == chunk_boundary:
                tmp_lpl.append(1)
                tmp_ltl.append(entity_types)
            else:
                tmp_lpl.append(0)
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels_chunk.append(tmp_lcl)
            labels_point.append(tmp_lpl)
            labels_typing.append(tmp_ltl)
            tmp_fl, tmp_lpl, tmp_lcl, tmp_ltl = list(), list(), list(), list()

    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels_chunk.append(tmp_lcl)
        labels_point.append(tmp_lpl)
        labels_typing.append(tmp_ltl)

    return features, labels_chunk, labels_point, labels_typing


def encode_folder(input_folder, output_folder, w_map, c_map, cl_map, tl_map, c_threshold = -1):

    w_st, w_unk, w_con, w_pad = w_map['<s>'], w_map['<unk>'], w_map['< >'], w_map['<\n>']
    c_st, c_unk, c_con, c_pad = c_map['<s>'], c_map['<unk>'], c_map['< >'], c_map['<\n>']

    # list_dirs = os.walk(input_folder)

    range_ind = 0

    # for root, dirs, files in list_dirs:
        # print('loading from ' + ', '.join(files))
        # for file in tqdm(files):
            # with open(os.path.join(root, file), 'r') as fin:
    with open(input_folder, 'r') as fin:
        lines = fin.readlines()

    features, labels_chunk, labels_chunk_mask, labels_point, labels_typing, is_gold = read_noisy_corpus(lines)

    if c_threshold > 0:
        c_count = dict()
        for line in features:
            for tup in line:
                for t_char in tup:
                    c_count[t_char] = c_count.get(t_char, 0) + 1
        c_set = [k for k, v in c_count.items() if v > c_threshold]
        for key in c_set:
            if key not in c_map:
                c_map[key] = len(c_map)

    dataset = list()

    for f_l, l_c, l_c_m, l_m, l_t, i_g in zip(features, labels_chunk, labels_chunk_mask, labels_point, labels_typing, is_gold):
        tmp_w = [w_st, w_con]
        tmp_c = [c_st, c_con]
        tmp_mc = [0, 1]

        for i_f, i_m in zip(f_l[1:-1], l_c_m[1:-1]):
            tmp_w = tmp_w + [w_map.get(i_f, w_map.get(i_f.lower(), w_unk))] * len(i_f) + [w_con]
            tmp_c = tmp_c + [c_map.get(t, c_unk) for t in i_f] + [c_con]
            tmp_mc = tmp_mc + [0] * len(i_f) + [i_m]

        tmp_w.append(w_pad)
        tmp_c.append(c_pad)
        tmp_mc.append(0)


        tmp_lc = [cl_map[tup] for tup in l_c[1:]]
        tmp_mt = l_m[1:]
        tmp_lt = list()
        for tup_list in l_t:
            tmp_mask = [0] * len(tl_map)
            for tup in tup_list:
                tmp_mask[tl_map[tup]] = 1
            tmp_lt.append(tmp_mask)

        dataset.append([tmp_w, tmp_c, tmp_mc, tmp_lc, tmp_mt, tmp_lt, i_g])

    dataset.sort(key=lambda t: len(t[0]), reverse=True)

    with open(output_folder+'train_'+ str(range_ind) + '.pk', 'wb') as f:
        pickle.dump(dataset, f)

    range_ind += 1

    return range_ind


def encode_dataset(input_file, w_map, c_map, cl_map, tl_map):

    print('loading from ' + input_file)

    with open(input_file, 'r') as f:
        lines = f.readlines()

    features, labels_chunk, labels_point, labels_typing = read_corpus(lines)

    w_st, w_unk, w_con, w_pad = w_map['<s>'], w_map['<unk>'], w_map['< >'], w_map['<\n>']
    c_st, c_unk, c_con, c_pad = c_map['<s>'], c_map['<unk>'], c_map['< >'], c_map['<\n>']

    dataset = list()

    for f_l, l_c, l_m, l_t in zip(features, labels_chunk, labels_point, labels_typing):
        tmp_w = [w_st, w_con]
        tmp_c = [c_st, c_con]
        tmp_mc = [0, 1]
        tmp_lc = [cl_map[l_c[1]]]

        for i_f, i_c in zip(f_l[1:-1], l_c[2:]):
            tmp_w = tmp_w + [w_map.get(i_f, w_map.get(i_f.lower(), w_unk))] * len(i_f) + [w_con]
            tmp_c = tmp_c + [c_map.get(t, c_unk) for t in i_f] + [c_con]
            tmp_mc = tmp_mc + [0] * len(i_f) + [1]
            tmp_lc = tmp_lc + [cl_map[i_c]]

        tmp_w.append(w_pad)
        tmp_c.append(c_pad)
        tmp_mc.append(0)

        tmp_mt = l_m[1:]
        tmp_lt = [tl_map[tup] for tup in l_t]

        dataset.append([tmp_w, tmp_c, tmp_mc, tmp_lc, tmp_mt, tmp_lt])

    dataset.sort(key=lambda t: len(t[0]), reverse=True)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train', default="./annotations/debug.ck")
    parser.add_argument('--input_gold_train', default="./data/ner/eng.train.ck")
    parser.add_argument('--input_testa', default="./data/ner/eng.testa.ck")
    parser.add_argument('--input_testb', default="./data/ner/eng.testb.ck")
    parser.add_argument('--pre_word_emb', default="./data/glove.100.pk")
    parser.add_argument('--output_folder', default="./data/hqner/")
    args = parser.parse_args()

    with open(args.pre_word_emb, 'rb') as f:
        w_emb = pickle.load(f)
        w_map = w_emb['w_map']
        emb_array = w_emb['emb_array']

    w_map, emb_array = filter_words(w_map, emb_array, [args.input_train, args.input_testa, args.input_testb])
    
    #four special char/word, <s>, <unk>, < > and <\n>
    c_map = {'<s>': 0, '<unk>': 1, '< >': 2, '<\n>': 3}
    tl_map = {'None': 0,
              'PER': 1, 'ORG': 2, 'LOC': 3,
              'Chemical': 1, 'Disease': 2,
              'AspectTerm': 1}
    cl_map = {'I': 0, 'O': 1}

    range_ind = encode_folder(args.input_train, args.output_folder, w_map, c_map, cl_map, tl_map, 5)
    testa_dataset = encode_dataset(args.input_testa, w_map, c_map, cl_map, tl_map)
    testb_dataset = encode_dataset(args.input_testb, w_map, c_map, cl_map, tl_map)
    gold_train_dataset = encode_dataset(args.input_gold_train, w_map, c_map, cl_map, tl_map)

    with open(args.output_folder+'test.pk', 'wb') as f:
        pickle.dump({'emb_array': emb_array, 'w_map': w_map, 'c_map': c_map, 'tl_map': tl_map, 'cl_map': cl_map, 'range': range_ind, 'test_data':testb_dataset, 'dev_data': testa_dataset}, f)

    print('dumped to the folder: ' + args.output_folder)
    print('done!')
