import os
import math
import numpy as np

import torch
from tensorboardX import SummaryWriter
from src.utils.cv_utils import to_colormap_image

writer = SummaryWriter('runs9')
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


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class ScheduledOptimizer(object):

    def __init__(self, opt, min_lrate=1e-6):
        self.init_lrate = get_learning_rate(opt)
        self.min_lr = min_lrate
        self.opt = opt

    def zero_grad(self):
        self.opt.zero_grad()

    @property
    def lr(self):
        return get_learning_rate(self.opt)

    @lr.setter
    def lr(self, val):
        set_learning_rate(val, self.opt)

    def load_state_dict(self, dic):
        self.opt.load_state_dict(dic)

    def state_dict(self):
        return self.opt.state_dict()


class ScheduledStepOptimizer(ScheduledOptimizer):

    def __init__(self, opt, drop=0.5, iters_drop=1e5):
        super(ScheduledStepOptimizer, self).__init__(opt)
        self.iters_drop = iters_drop
        self.drop = drop

    def step_and_update_lr(self, n_iter):
        self.lr = self.init_lrate * math.pow(self.drop, math.floor(n_iter/self.iters_drop))


class ScheduledMovingAverageOptimizer(ScheduledOptimizer):

    def __init__(self, opt, num_iterations=1000):
        super(ScheduledMovingAverageOptimizer, self).__init__(opt)
        self.losses = []
        self.window = num_iterations
        self.factor = 0.5

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
