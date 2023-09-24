from __future__ import print_function, absolute_import
import time

import torch

from .utils.meters import AverageMeter


class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            loss = self.memory(f_out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class UncertainTrainer(object):
    def __init__(self, encoder, memory=None):
        super(UncertainTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=20, train_iters=400, progressive=False, un_num=16,
              ratio=0.4, beta=0, lam=0.8):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        un_losses = AverageMeter()

        end = time.time()
        print(train_iters)
        for i in range(train_iters):
        # for i in range(len(data_loader)):
            # load data
            inputs = data_loader.next()
            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)
            bs = inputs.shape[0]

            data_time.update(time.time() - end)

            # forward
            f_out = self._forward(inputs)

            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            if progressive:
                uncertain_num = int(bs * ratio * (train_iters - i) / train_iters)
            else:
                uncertain_num = un_num

            # alpha = 0.625
            # alpha = beta
            alpha = uncertain_num / bs

            if beta != 0:
                alpha = beta
            loss, un_loss, lam = self.memory(f_out, labels, uncertain_num=uncertain_num, lam=lam)
            un_loss *= alpha
            total_loss = loss + un_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            un_losses.update(un_loss.item())
            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'UnLoss {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'un_bs : {}'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              un_losses.val, un_losses.avg,
                              losses.val, losses.avg
                              , uncertain_num
                              ))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
