"""
.. module:: Utils
    :synopsis: Utils
    
.. moduleauthor:: Liyuan Liu, Jingbo Shang
"""
import numpy as np
import torch
import json

import torch
import torch.nn as nn
import torch.nn.init

def adjust_learning_rate(optimizer, lr):
    """
    Shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def to_scalar(var):
    """
    Turn the first element of a tensor to scalar
    """
    return var.view(-1).item()

def evaluate_chunking(iterator, ner_model, none_idx):
    """
    Evaluate the chunking performance.

    Parameters
    ----------
    iterator : ``iterator``, required.
        Dataset loader.
    ner_model : ``torch.nn.Module`` , required.
        Sequence labeling model for evaluation.
    none_idx: ``int``, required.
        The index for the not-target-type entities.
    """

    gold_count = 0
    guess_count = 0
    overlap_count = 0

    ner_model.eval()

    for word_t, char_t, chunk_mask, chunk_label, type_mask, type_label in iterator:
        output = ner_model(word_t, char_t, chunk_mask)
        chunk_score = ner_model.chunking(output)
        pred_chunk = (chunk_score < 0.0)

        if pred_chunk.data.float().sum() <= 1:
            golden_labels = ner_model.to_span(type_mask.cpu(), type_label.cpu(), none_idx)
            gold_count += len(golden_labels)
        else:
            type_score = ner_model.typing(output, pred_chunk)
            max_score, pred_type = type_score.max(dim = 1)

            pred_labels = ner_model.to_span(pred_chunk.long().cpu(), pred_type.long().cpu(), none_idx)

            golden_labels = ner_model.to_span(type_mask.long().cpu(), type_label.long().cpu(), none_idx)

            gold_count += len(golden_labels)
            guess_count += len(pred_labels)
            overlap_count += len(golden_labels & pred_labels)

    pre = overlap_count / (float(guess_count) + 0.000001)
    rec = overlap_count / (float(gold_count) + 0.000001)
    f1 = 2 * pre * rec / (pre + rec + 0.000001)

    return pre, rec, f1

def evaluate_typing(iterator, ner_model, none_idx):
    """
    Evaluate the typing performance.

    Parameters
    ----------
    iterator : ``iterator``, required.
        Dataset loader.
    ner_model : ``torch.nn.Module`` , required.
        Sequence labeling model for evaluation.
    none_idx: ``int``, required.
        The index for the not-target-type entities.
    """

    gold_count = 0
    guess_count = 0
    overlap_count = 0

    ner_model.eval()

    for word_t, char_t, chunk_mask, chunk_label, type_mask, type_label in iterator:
        output = ner_model(word_t, char_t, chunk_mask)
        pred_chunk = (chunk_label <= 0.0)

        if pred_chunk.data.float().sum() <= 1:
            golden_labels = ner_model.to_typed_span(type_mask.cpu(), type_label.cpu(), none_idx)
            gold_count += len(golden_labels)
        else:
            type_score = ner_model.typing(output, pred_chunk)
            max_score, pred_type = type_score.max(dim = 1)

            pred_labels = ner_model.to_typed_span(pred_chunk.long().cpu(), pred_type.long().cpu(), none_idx)

            golden_labels = ner_model.to_typed_span(type_mask.long().cpu(), type_label.long().cpu(), none_idx)

            gold_count += len(golden_labels)
            guess_count += len(pred_labels)
            overlap_count += len(golden_labels & pred_labels)

    pre = overlap_count / (float(guess_count) + 0.000001)
    rec = overlap_count / (float(gold_count) + 0.000001)
    f1 = 2 * pre * rec / (pre + rec + 0.000001)

    return pre, rec, f1

def evaluate_ner(iterator, ner_model, none_idx, id2label):
    """
    Evaluate the NER performance.

    Parameters
    ----------
    iterator : ``iterator``, required.
        Dataset loader.
    ner_model : ``torch.nn.Module`` , required.
        Sequence labeling model for evaluation.
    none_idx: ``int``, required.
        The index for the not-target-type entities.
    """
    gold_count = 0
    guess_count = 0
    overlap_count = 0

    ner_model.eval()

    type2gold, type2guess, type2overlap = {}, {}, {}

    for word_t, char_t, chunk_mask, chunk_label, type_mask, type_label in iterator:
        output = ner_model(word_t, char_t, chunk_mask)
        chunk_score = ner_model.chunking(output)
        pred_chunk = (chunk_score < 0.0)

        if pred_chunk.data.float().sum() <= 1:
            golden_labels = ner_model.to_typed_span(type_mask.cpu(), type_label.cpu(), none_idx, id2label)
            gold_count += len(golden_labels)
        else:
            type_score = ner_model.typing(output, pred_chunk)
            max_score, pred_type = type_score.max(dim = 1)

            pred_labels = ner_model.to_typed_span(pred_chunk.long().cpu(), pred_type.long().cpu(), none_idx, id2label)

            golden_labels = ner_model.to_typed_span(type_mask.long().cpu(), type_label.long().cpu(), none_idx, id2label)

            gold_count += len(golden_labels)
            guess_count += len(pred_labels)
            overlap_count += len(golden_labels & pred_labels)

            for label in golden_labels:
                entity_type = label.split('@')[0]
                type2gold[entity_type] = type2gold.get(entity_type, 0) + 1
            for label in pred_labels:
                entity_type = label.split('@')[0]
                type2guess[entity_type] = type2guess.get(entity_type, 0) + 1
            for label in golden_labels & pred_labels:
                entity_type = label.split('@')[0]
                type2overlap[entity_type] = type2overlap.get(entity_type, 0) + 1

    pre = overlap_count / (float(guess_count) + 0.000001)
    rec = overlap_count / (float(gold_count) + 0.000001)
    f1 = 2 * pre * rec / (pre + rec + 0.000001)

    type2pre, type2rec, type2f1 = {}, {}, {}
    for entity_type in type2gold:
        type2pre[entity_type] = type2overlap.get(entity_type, 0) / float(type2guess.get(entity_type, 0) + 0.000001)
        type2rec[entity_type] = type2overlap.get(entity_type, 0) / float(type2gold.get(entity_type, 0) + 0.000001)
        type2f1[entity_type] = 2 * type2pre[entity_type] * type2rec[entity_type] / (type2pre[entity_type] + type2rec[entity_type] + 0.000001)

    return pre, rec, f1, type2pre, type2rec, type2f1

def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
    
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1