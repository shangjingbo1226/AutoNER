import pickle
import argparse
import os
import random
import numpy as np
from tqdm import tqdm

import itertools
import functools

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_embedding', default="./embedding/glove.6B.100d.txt")
    parser.add_argument('--output_embedding', default="./data/glove.100.pk")
    parser.add_argument('--unk', default='unk')
    args = parser.parse_args()

    word_dict = dict()
    embedding_array = list()
    for line in open(args.input_embedding, 'r'):
        line = line.split()
        vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
        if line[0] == args.unk:
            if '<unk>' not in word_dict:
                # add <unk>
                word_dict['<unk>'] = len(word_dict)
                if len(embedding_array) > 0:
                    assert len(embedding_array[0]) == len(vector), 'dimension mismatch!'
                embedding_array.append(vector)
        elif line[0] not in word_dict:
            # add a new word
            word_dict[line[0]] = len(word_dict)
            if len(embedding_array) > 0:
                assert len(embedding_array[0]) == len(vector), 'dimension mismatch!'
            embedding_array.append(vector)

    assert len(word_dict) == len(embedding_array)

    bias = 2 * np.sqrt(3.0 / len(embedding_array[0]))

    if '<unk>' not in word_dict:
        print('[Warning] <unk> token not found: ' + args.unk)
        print('Randomly assign values')
        assert len(embedding_array) > 0
        dimension = len(embedding_array[0])
        word_dict['<unk>'] = len(word_dict)
        embedding_array.append([random.random() * bias - bias for tup in embedding_array[0]])

    word_dict['<s>'] = len(word_dict)
    word_dict['< >'] = len(word_dict)
    word_dict['<\n>'] = len(word_dict)
    embedding_array.append([random.random() * bias - bias for tup in embedding_array[0]])
    embedding_array.append([random.random() * bias - bias for tup in embedding_array[0]])
    embedding_array.append([random.random() * bias - bias for tup in embedding_array[0]])

    assert len(word_dict) == len(embedding_array)

    with open(args.output_embedding, 'wb') as f:
        pickle.dump({'w_map': word_dict, 'emb_array': embedding_array}, f)