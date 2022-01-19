# -*- coding: UTF-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # apha如果不定义，就是[1]*num_class
        # self.alpha = alpha
        # if isinstance(alpha, (float, int)):
        #     # binary classifier
        #     self.alpha = torch.Tensor([alpha, 1 - alpha])
        # if isinstance(alpha, list):
        #     self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """
        :param input: B*C (C is num_classes)
        :param target: B
        :return:
        """
        target = target.view(-1, 1)  # B*1

        logp = F.log_softmax(input, dim=1)
        logpt = logp.gather(1, target)
        logpt = logpt.view(-1)  # B
        pt = logpt.exp()

        # if self.alpha is not None:
        #     if self.alpha.type() != input.type():
        #         self.alpha = self.alpha.type_as(input)
        #
        #     at = self.alpha.gather(0, target.view(-1))
        #     logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
