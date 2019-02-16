from tqdm import tqdm
from torch import optim
from torch import autograd

from utils.loader import *
from unet import *


THR_DISTANCE = 1.0
UPLOAD_IMAGE_EVERY = 2000
LR = 0.01


def to_onehot(seq_of_idx, num_classes=20):
    batch_sz = seq_of_idx.size(0)
    length = seq_of_idx.size(1)
    onehot = torch.zeros(batch_sz, length, num_classes, dtype=torch.float, device=device)
    onehot.scatter_(2, seq_of_idx.unsqueeze(2), 1)
    return onehot


def hook_func(module, grad_inputs, grad_outputs):
    for inp, out in zip(grad_inputs, grad_outputs):
        if torch.isnan(inp).any() or torch.isnan(out).any():
            raise RuntimeError("Detected NaN in grad")


def mask_distance_matrix(dm, thr=THR_DISTANCE):
    return (dm.abs() >= thr).float() * dm


class OuterProduct(nn.Module):

    def __init__(self):
        super(OuterProduct, self).__init__()

    def forward(self, seq1, seq2):
        assert seq1.size() == seq2.size()
        batch_sz = seq1.size(0)
        n_channels = seq1.size(1)
        length = seq1.size(2)
        op = torch.zeros(batch_sz, n_channels * 2, length, length, dtype=torch.float, device=device)
        for i in range(length):
            a = seq1[:, :, i].unsqueeze(2) - seq2[:, :, :]
            b = seq1[:, :, i].unsqueeze(2) * seq2[:, :, :]
            op[:, :, i, :] = torch.cat([a.abs(), b], 1)
        return op


class M1(nn.Module):
    def __init__(self):
        super(M1, self).__init__()

        self.unet1d_1 = UNet1d(n_channels=61, n_classes=8, inner_channels=32)
        self.unet2d_1 = UNet2d(n_channels=1, n_classes=8, inner_channels=16)
        self.unet2d_2 = UNet2d(n_channels=24, n_classes=1, inner_channels=16)
        self.outer_product = OuterProduct()

    def forward(self, seq1, seq2, prof, dmat, mut_idx):
        batch_sz = dmat.size(0)
        mu = dmat.mean((1, 2)).view(batch_sz, 1, 1)
        sigma = dmat.view(batch_sz, -1).std(1).view(batch_sz, 1, 1)
        dmat.add_(-mu).div_(sigma)
        seq1_onehot = to_onehot(seq1)
        seq2_onehot = to_onehot(seq2)
        msk = torch.zeros(prof.size(0), prof.size(1), 1, dtype=torch.float, device=device)
        msk[:, mut_idx, :] = 1
        si = torch.cat([prof, seq1_onehot, seq2_onehot, msk], 2)
        dm = self.unet2d_1(dmat.unsqueeze(1))
        si = self.unet1d_1(si.transpose(1, 2))
        op = self.outer_product(si, si)
        out = self.unet2d_2(torch.cat([dm, op], 1))
        return out.squeeze(1)


def get_loss(d_hat, d):
    return (d_hat - d).abs().mean((1, 2)).mean()


def predict(model, m1, m2, s1, s2, p1, p2, idx):
    seq1 = torch.cat([s1, s2], 0)
    seq2 = torch.cat([s2, s1], 0)
    prof = torch.cat([p1, p2], 0)
    dm = torch.cat([m1, m2], 0)
    idx = torch.cat([idx, idx], 0)
    ddm_hat = model(seq1, seq2, prof, dm, idx)
    d1 = mask_distance_matrix(m1 - m2)
    d2 = mask_distance_matrix(m2 - m1)
    ddm = torch.cat([d1, d2], 0)
    return ddm_hat, ddm


def train(model, loader, optimizer, n_iter):
    model.train()
    err = 0.0
    i = 0.0
    pbar = tqdm(total=len(loader), desc='pairs loaded')
    for i, (s1, s2, p1, p2, m1, m2, idx, pdb1, pdb2, *_) in enumerate(batch_generator(loader, prepare_torch_batch)):
        optimizer.zero_grad()

        assert s1.shape == s2.shape
        assert m1.shape == m2.shape
        assert p1.shape == p2.shape

        ddm_hat, ddm = predict(model, m1, m2, s1, s2, p1, p2, idx)
        loss = get_loss(ddm_hat, ddm)
        err += loss.item()
        e = err / (i + 1.)

        writer.add_scalars('M1/Loss', {"train": e}, n_iter)

        try:
            with autograd.detect_anomaly():
                loss.backward()
        except RuntimeError:
            print(pdb1)
            print(pdb2)
            continue

        if n_iter % UPLOAD_IMAGE_EVERY == 0:
            delta1 = ddm.unsqueeze(1).data.cpu().numpy()
            delta2 = ddm_hat.data.unsqueeze(1).cpu().numpy()
            for id1, id2, d in zip(pdb1, pdb2, delta1):
                writer.add_image('M1/%d_iterations/%s-%s_true' % (n_iter, id1, id2), d, n_iter)
            for id1, id2, d_hat in zip(pdb1, pdb2, delta2):
                writer.add_image('M1/%d_iterations/%s-%s_pred' % (n_iter, id1, id2), d_hat, n_iter)

        optimizer.step_and_update_lr(loss.item())
        lr = optimizer.lr

        pbar.set_description("Training Loss:%.6f, LR: %.6f (L=%d)" % (e, lr, s1.size(1)))
        pbar.update(len(idx))
        n_iter += 1

    pbar.close()

    return n_iter


def evaluate(model, loader, n_iter):
    model.eval()
    err = 0.0
    i = 0.0
    pbar = tqdm(total=len(loader), desc='pairs loaded')
    for i, (s1, s2, p1, p2, m1, m2, idx, pdb1, pdb2, *_) in enumerate(batch_generator(loader, prepare_torch_batch)):

        assert s1.shape == s2.shape
        assert m1.shape == m2.shape
        assert p1.shape == p2.shape

        ddm_hat, ddm = predict(model, m1, m2, s1, s2, p1, p2, idx)
        loss = get_loss(ddm_hat, ddm)
        err += loss.item()

        pbar.set_description("Validation Loss:%.6f" % (err / (i + 1.),))
        pbar.update(len(idx))

    writer.add_scalars('M1/Loss', {"valid": err / (i + 1.)}, n_iter)

    pbar.close()


def main():

    net = M1()
    net.to(device)
    net.register_backward_hook(hook_func)
    opt = ScheduledOptimizer(optim.Adam(net.parameters(), lr=LR), LR, num_iterations=20000)

    n_iter = 1
    init_epoch = 0
    num_epochs = 20
    train_size = 100000
    test_size = 10000

    trainset = TRAIN_SET
    testset = VALID_SET
    loader_train = Loader(trainset, train_size)
    loader_test = Loader(testset, test_size)

    for epoch in range(init_epoch, num_epochs):
        n_iter = train(net, loader_train, opt, n_iter)
        evaluate(net, loader_test, n_iter)
        loader_train.reset()
        loader_test.reset()


if __name__ == "__main__":
    main()
