import torch

import torch.nn.functional as F
from torch import nn

class UncertainModule(object):

    def __init__(self, num_samples, temp):
        self.num_samples = num_samples
        self.temp = temp

    def compute_uncertain(self, inputs, targets, features):
        distribution = self.get_distribution(inputs, features)
        labels = self.smooth_label(targets)
        distribution /= self.temp
        uncertain_score = self.kl_loss(distribution, labels)
        uncertain_score = F.softmax(uncertain_score, dim=0)

        return uncertain_score.sort()

    def get_distribution(self, inputs, features):
        distribution = inputs.mm(features.t())
        return distribution

    def smooth_label(self, targets, lam=0.9):
        value = lam
        labels = torch.ones(targets.shape[0], self.num_samples).cuda()
        labels = labels * ((1. - lam) / (self.num_samples - 1))
        targets = targets.reshape(targets.shape[0], 1)
        labels.scatter_(dim=1, index=targets, value=value)
        return labels

    def kl_loss(self, inputs, targets):
        inputs = F.log_softmax(inputs, dim=1)
        kl = targets * (targets.log() - inputs)
        kl = kl.sum(dim=1)
        return kl


