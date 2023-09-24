import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from IPython import embed
from clustercontrast.utils.uncertain_module import UncertainModule


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # targets = targets.max(dim=1)[1]

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Uncertain(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)

        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y, l in zip(inputs, targets, ctx.momentum):
            ctx.features[y] = l * ctx.features[y] + (1. - l) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm_uncertain(inputs, indexes, features, momentum):
    return CM_Uncertain.apply(inputs, indexes, features, momentum)


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets):

        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        return loss


class UncertainClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, um_temp=0.5, sim_temp=0.05):
        super(UncertainClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.sim_temp = sim_temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

        self.uncertain_module = UncertainModule(num_samples, um_temp)

    def forward(self, inputs, targets, uncertain_num=16, lam=0.8):
        bs = inputs.shape[0]
        inputs = F.normalize(inputs, dim=1).cuda()
        if uncertain_num > 0:
            uncertain_score, uncertain_index = self.uncertain_module.compute_uncertain(inputs, targets, self.features)
            uncertain = uncertain_index[uncertain_num * -1:]

            # select hardest
            # uncertain = self.select_hardest(inputs, self.features, targets)

            # random_choose
            # uncertain = torch.randperm(inputs.shape[0])[uncertain_num * -1:]

            # certain = uncertain_index[:uncertain_num * -1]
            # uncertain_score = uncertain_score.detach()

            uncertain_inputs = inputs[uncertain]
            uncertain_targets = targets[uncertain]

            uncertain_outputs = cm(uncertain_inputs, uncertain_targets, self.features, self.momentum)
            # uncertain_labels, lam = self.smooth_label(uncertain_inputs, self.features)
            # uncertain_labels, _ = self.fixed_smooth_label(uncertain_inputs, self.features, lam=lam)
            uncertain_labels, lam = self.hard_label(uncertain_inputs, self.features)
            uncertain_outputs /= self.temp
            uncertain_loss = F.cross_entropy(uncertain_outputs, uncertain_labels)

        else:
            uncertain_loss = torch.Tensor([0]).cuda()
            lam = torch.Tensor([0]).cuda()

        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)

        outputs /= self.temp

        # targets, lam = self.smooth_label(inputs, self.features)
        loss = F.cross_entropy(outputs, targets)

        return loss, uncertain_loss, lam

    @torch.no_grad()
    def smooth_label(self, un_inputs, features):
        c_num = features.shape[0]
        un_num = un_inputs.shape[0]

        lam, max_index = self.sim_softmax(un_inputs, features)

        labels = torch.ones(un_num, c_num).cuda()
        new_labels = []
        for i in range(un_num):
            labels[i] *= (1 - lam[i]) / (c_num - 1)
            labels[i][max_index[i]] = lam[i]
            new_labels.append(labels[i].unsqueeze(0))

        new_labels = torch.cat(new_labels, dim=0)

        return new_labels, lam

    @torch.no_grad()
    def fixed_smooth_label(self, un_inputs, features, lam=0.8):
        c_num = features.shape[0]
        un_num = un_inputs.shape[0]

        _, max_index = self.sim_softmax(un_inputs, features)

        labels = torch.ones(un_num, c_num).cuda()
        new_labels = []
        for i in range(un_num):
            labels[i] *= (1 - lam) / (c_num - 1)
            labels[i][max_index[i]] = lam
            new_labels.append(labels[i].unsqueeze(0))

        new_labels = torch.cat(new_labels, dim=0)

        return new_labels, lam

    @torch.no_grad()
    def hard_label(self, un_inputs, features):
        lam, max_index = self.sim_softmax(un_inputs, features)
        return max_index, lam

    @torch.no_grad()
    def sim_softmax(self, un_inputs, features):
        sim_matrix = un_inputs.mm(features.t())
        sim_matrix /= self.sim_temp
        sim_matrix = F.softmax(sim_matrix, dim=1)

        return sim_matrix.max(dim=1)

    @torch.no_grad()
    def select_hardest(self, inputs, features, targets, num_instance=16):
        similarity = inputs.mm(features.t())
        similarity = similarity[:, targets]
        similarity = similarity[:, 0]

        similarity = similarity.reshape(inputs.shape[0] // num_instance, num_instance)

        _, min_index = similarity.min(dim=0)
        k = 0
        index = []
        while k < len(min_index):
            index.append(k * num_instance + min_index[k])
            k += 1
        index = torch.stack(index)

        return index
