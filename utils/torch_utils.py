import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import os
import numpy as np

from utils.cv_utils import *


writer = SummaryWriter('runs')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1, 1)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, long)):
            self.alpha = torch.tensor([alpha, 1-alpha], dtype=torch.float, device=device)
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float, device=device)
        self.size_average = size_average

    def forward(self, logits, target):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # N,C,H,W => N,C,H*W
            logits = logits.transpose(1, 2)                           # N,C,H*W => N,H*W,C
            logits = logits.contiguous().view(-1, logits.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(logits, 1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        if self.alpha is not None:
            if self.alpha.type() != logits.data.type():
                self.alpha = self.alpha.type_as(logits.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def model_summary(model):
    for idx, m in enumerate(model.modules()):
        print(idx, '->', m)


def save_checkpoint(state, prefix, ckptpath, is_best=False):
    if is_best:
        filename = os.path.join(ckptpath, "%s_best.tar" % (prefix,))
        torch.save(state, filename)
    filename = os.path.join(ckptpath, "%s_latest.tar" % (prefix,))
    torch.save(state, filename)


def load_checkpoint(net, checkpoint):
    loaded_weights = checkpoint['net']
    weight_dic = {}
    net_state_dic = net.state_dict()
    for count, (key, _) in enumerate(net_state_dic.items()):
        weight_dic[key] = loaded_weights.get(key, net_state_dic[key])
    net.load_state_dict(weight_dic)


def adjust_learning_rate(initial, optimizer, epoch, factor=0.1):
    lr = max(initial * (factor ** (epoch // 2)), 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# https://github.com/pytorch/pytorch/issues/2830
def optimizer_cuda(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()


class ScheduledOptimizer(object):

    def __init__(self, opt, initial_lr, num_iterations=1000):
        self._lr = initial_lr
        self.opt = opt
        self.losses = []
        self.window = num_iterations
        self.min_lr = 1e-6
        self.factor = 0.5

    def zero_grad(self):
        self.opt.zero_grad()

    def step_and_update_lr(self, loss):
        self.opt.step()
        losses = self.losses
        while len(losses) > self.window:
            losses.pop(0)
        losses.append(loss)
        if len(losses) < self.window:
            return
        avg_old = np.mean(losses[:self.window//2])
        avg_new = np.mean(losses[self.window//2:])
        if avg_new < avg_old:
            return
        self.lr = max(self.lr * self.factor, self.min_lr)
        self.losses = []     # restart loss count

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, val):
        set_learning_rate(val, self.opt)
        self._lr = val

    def load_state_dict(self, dic):
        self.opt.load_state_dict(dic)

    def state_dict(self):
        return self.opt.state_dict()


def shuffle(data, labels):
    s = np.arange(data.shape[0])
    np.random.shuffle(s)
    return data[s], labels[s]


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def bce_loss(true, logits, pos_weight=None):
    """Computes the weighted binary cross-entropy loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, 1, H, W]. Corresponds to
            the raw output or logits of the model.
        pos_weight: a scalar representing the weight attributed
            to the positive class. This is especially useful for
            an imbalanced dataset.
    Returns:
        bce_loss: the weighted binary cross-entropy loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits.float(),
        true.float(),
        pos_weight=torch.tensor([pos_weight], device=device),
    )
    return bce_loss


def write_true_pred_pairs(modelname, n_iter, ids1, ids2, true, pred):
    for id1, id2, m1, m2 in zip(ids1, ids2, true, pred):
        writer.add_image('%s/%d_iterations/%s-%s_true' % (modelname, n_iter, id1, id2), to_colormap_image(m1), n_iter, dataformats='HWC')
        writer.add_image('%s/%d_iterations/%s-%s_pred' % (modelname, n_iter, id1, id2), to_colormap_image(m2), n_iter, dataformats='HWC')


def write_dist_mats_pairs(modelname, n_iter, ids1, ids2, mtst1, mts2):
    for id1, id2, m1, m2 in zip(ids1, ids2, mtst1, mts2):
        writer.add_image('%s/%d_iterations/%s-%s_m1' % (modelname, n_iter, id1, id2), to_colormap_image(m1), n_iter, dataformats='HWC')
        writer.add_image('%s/%d_iterations/%s-%s_m2' % (modelname, n_iter, id1, id2), to_colormap_image(m2), n_iter, dataformats='HWC')
