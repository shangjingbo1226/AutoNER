import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import random

class softCE(nn.Module):
    def __init__(self, if_average = True):
        super(softCE, self).__init__()
        self.logSoftmax = nn.LogSoftmax(dim = 1)
        self.if_average = if_average


    @staticmethod
    def soft_max(vec, mask):
        batch_size = vec.size(0)
        max_score, idx = torch.max(vec, 1, keepdim=True)
        exp_score = torch.exp(vec - max_score.expand_as(vec))
        # exp_score = exp_score.masked_fill_(mask, 0)
        exp_score = exp_score * mask
        exp_score_sum = torch.sum(exp_score, 1).view(batch_size, 1).expand_as(exp_score)
        prob_score = exp_score / exp_score_sum
        return prob_score

    def forward(self, scores, target):
    	#scores: B * L
    	#target: B * L
    	supervision_p = softCE.soft_max(scores, target)
    	scores_logp = self.logSoftmax(scores)
    	CE = (-supervision_p * scores_logp).sum()
    	if self.if_average:
    		CE = CE / scores.size(0)
    	return CE