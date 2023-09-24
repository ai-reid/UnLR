import torch

import torch.nn.functional as F


class UncertainEstimation(object):
    def __int__(self, freedom_degree):
        self.freedom_degree = freedom_degree
