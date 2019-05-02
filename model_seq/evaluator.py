"""
.. module:: evaluator
    :synopsis: evaluator for sequence labeling
 
.. moduleauthor:: Liyuan Liu
"""
import torch
import numpy as np
import itertools

import model_seq.utils as utils
from torch.autograd import Variable

class eval_batch:
    """
    Base class for evaluation, provide method to calculate f1 score and accuracy.

    Parameters
    ----------
    decoder : ``torch.nn.Module``, required.
        the decoder module, which needs to contain the ``to_span()`` method.
    """
    def __init__(self, decoder):
        self.decoder = decoder

    def reset(self):
        """
        reset counters.
        """
        self.correct_labels = 0
        self.total_labels = 0
        self.gold_count = 0
        self.guess_count = 0
        self.overlap_count = 0

    def calc_f1_batch(self, decoded_data, target_data):
        """
        update statics for f1 score.

        Parameters
        ----------
        decoded_data: ``torch.LongTensor``, required.
            the decoded best label index pathes.
        target_data:  ``torch.LongTensor``, required.
            the golden label index pathes.
        """
        batch_decoded = torch.unbind(decoded_data, 1)

        for decoded, target in zip(batch_decoded, target_data):
            length = len(target)
            best_path = decoded[:length]

            correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = self.eval_instance(best_path.numpy(), target)
            self.correct_labels += correct_labels_i
            self.total_labels += total_labels_i
            self.gold_count += gold_count_i
            self.guess_count += guess_count_i
            self.overlap_count += overlap_count_i

    def calc_acc_batch(self, decoded_data, target_data):
        """
        update statics for accuracy score.

        Parameters
        ----------
        decoded_data: ``torch.LongTensor``, required.
            the decoded best label index pathes.
        target_data:  ``torch.LongTensor``, required.
            the golden label index pathes.
        """
        batch_decoded = torch.unbind(decoded_data, 1)

        for decoded, target in zip(batch_decoded, target_data):
            
            # remove padding
            length = len(target)
            best_path = decoded[:length].numpy()

            self.total_labels += length
            self.correct_labels += np.sum(np.equal(best_path, gold))

    def f1_score(self):
        """
        calculate the f1 score based on the inner counter.
        """
        if self.guess_count == 0:
            return 0.0, 0.0, 0.0, 0.0
        precision = self.overlap_count / float(self.guess_count)
        recall = self.overlap_count / float(self.gold_count)
        if precision == 0.0 or recall == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        f = 2 * (precision * recall) / (precision + recall)
        accuracy = float(self.correct_labels) / self.total_labels
        return f, precision, recall, accuracy

    def acc_score(self):
        """
        calculate the accuracy score based on the inner counter.
        """
        if 0 == self.total_labels:
            return 0.0
        accuracy = float(self.correct_labels) / self.total_labels
        return accuracy        

    def eval_instance(self, best_path, gold):
        """
        Calculate statics to update inner counters for one instance.

        Parameters
        ----------
        best_path: required.
            the decoded best label index pathe.
        gold: required.
            the golden label index pathes.
      
        """
        total_labels = len(best_path)
        correct_labels = np.sum(np.equal(best_path, gold))
        gold_chunks = self.decoder.to_spans(gold)
        gold_count = len(gold_chunks)

        guess_chunks = self.decoder.to_spans(best_path)
        guess_count = len(guess_chunks)

        overlap_chunks = gold_chunks & guess_chunks
        overlap_count = len(overlap_chunks)

        return correct_labels, total_labels, gold_count, guess_count, overlap_count

class eval_wc(eval_batch):
    """
    evaluation class for LD-Net

    Parameters
    ----------
    decoder : ``torch.nn.Module``, required.
        the decoder module, which needs to contain the ``to_span()`` and ``decode()`` method.
    score_type : ``str``, required.
        whether the f1 score or the accuracy is needed.
    """
    def __init__(self, decoder, score_type):
        eval_batch.__init__(self, decoder)

        if 'f' in score_type:
            self.eval_b = self.calc_f1_batch
            self.calc_s = self.f1_score
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, seq_model, dataset_loader):
        """
        calculate scores

        Parameters
        ----------
        seq_model: required.
            sequence labeling model.
        dataset_loader: required.
            the dataset loader.

        Returns
        -------
        score: ``float``.
            calculated score.
        """
        seq_model.eval()
        self.reset()

        for f_c, f_p, b_c, b_p, flm_w, blm_w, blm_ind, f_w, _, f_y_m, g_y in dataset_loader:
            scores = seq_model(f_c, f_p, b_c, b_p, flm_w, blm_w, blm_ind, f_w)
            decoded = self.decoder.decode(scores.data, f_y_m)
            self.eval_b(decoded, g_y)

        return self.calc_s()