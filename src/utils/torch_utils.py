import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from tensorboardX import SummaryWriter
from utils.cv_utils import to_colormap_image


writer = SummaryWriter('runs7')
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_available_devices(d):
    os.environ["CUDA_VISIBLE_DEVICES"] = d


def model_summary(model):
    summary = ''
    for idx, m in enumerate(model.modules()):
        summary += '\n%s->%s' % (idx, m)
    return summary


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


def set_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def step_decay(epoch, drop=0.5, epochs_drop=2.0):
#    initial_lrate = 0.1
#    lrate = initial_lrate * math.pow(drop,
#            math.floor((1+epoch)/epochs_drop))
#    return lrate


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


def upload_images(dmat, dmat_hat, pdb, n_iter, prefix):
    for m1, m2, pdb_id in zip(dmat.data.cpu().numpy(), dmat_hat.data.cpu().numpy(), pdb):
        writer.add_image('%s/%s/cmap_true' % (prefix, pdb_id), to_colormap_image(m1), n_iter, dataformats='HWC')
        writer.add_image('%s/%s/cmap_pred' % (prefix, pdb_id), to_colormap_image(m2), n_iter, dataformats='HWC')
