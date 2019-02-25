from tqdm import tqdm
from torch import optim
from torch import autograd
from tempfile import gettempdir

from utils.cv_utils import *
from utils.loader import *
from unet import *


THR_DISTANCE = 1.0
UPLOAD_IMAGE_EVERY = 100
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


class M3(nn.Module):
    def __init__(self):
        super(M3, self).__init__()

        self.unet1d_1 = UNet1d(n_channels=40, n_classes=32, inner_channels=32)
        self.unet1d_2 = UNet1d(n_channels=32, n_classes=8, inner_channels=32)
        self.unet2d_1 = UNet2d(n_channels=64, n_classes=1, inner_channels=64)
        self.outer_product = OuterProduct()

    def forward(self, seq1, beta1, prof1):
        seq1_onehot = to_onehot(seq1)
        seq_info = torch.cat([seq1_onehot, prof1], 2).transpose(1, 2)
        seq_info = self.unet1d_1(seq_info)
        dssp = self.unet1d_2(seq_info)
        op1 = self.outer_product(seq_info, seq_info)
        out = self.unet2d_1(op1)
        return torch.sigmoid(out).squeeze(1)


def get_loss(d_hat, d, lam=50.0):
    msk = (d != 0).float() * lam + (d == 0).float() * 1.0
    bce = ((d == 0).float() * torch.log(torch.clamp(1 - d_hat, min=eps)) +
           (d == 1).float() * torch.log(torch.clamp(d_hat, min=eps)))
    bce = (-bce * msk).sum((1, 2)).mean().div_(d.size(1)**2)
    sym = (d_hat.transpose(1, 2) - d_hat).abs().sum((1, 2)).mean()
    return bce + sym


def predict(model, seq, beta, prof, dmat):
    cmap = get_contact_map(dmat)
    cmap_hat = model(seq, beta, prof)
    return cmap_hat, cmap


def train(model, loader, optimizer, n_iter):
    model.train()
    err = 0.0
    i = 0
    pbar = tqdm(total=len(loader), desc='pairs loaded')
    for i, (seq, beta, prof, dmat, pdb, *_) in enumerate(batch_generator(loader, prepare_pdb_batch)):
        optimizer.zero_grad()

        cmap_hat, cmap = predict(model, seq, beta, prof, dmat)
        loss = get_loss(cmap_hat, cmap)
        err += loss.item()
        e = err / (i + 1.)

        writer.add_scalars('M3/Loss', {"train": e}, n_iter)

        try:
            with autograd.detect_anomaly():
                loss.backward()
        except RuntimeError as e:
            raise e

        if n_iter % UPLOAD_IMAGE_EVERY == 0:
            for cm1, cm2, dm, pdb_id in zip(cmap_hat.data.cpu().numpy(), cmap.float().data.cpu().numpy(), dmat.data.cpu().numpy(), pdb):
                writer.add_image('M3/%s/cmap_pred' % pdb_id, to_colormap_image(cm1), n_iter, dataformats='HWC')
                writer.add_image('M3/%s/cmap_true' % pdb_id, to_colormap_image(cm2), n_iter, dataformats='HWC')
                # writer.add_image('M3/%s/dmap_true' % pdb_id, to_colormap_image(dm), n_iter, dataformats='HWC')

        optimizer.step_and_update_lr(loss.item())
        lr = optimizer.lr

        pbar.set_description("Training Loss:%.6f, LR: %.6f (L=%d)" % (e, lr, seq.size(1)))
        pbar.update(seq.size(0))
        n_iter += 1

    pbar.close()
    return n_iter


def evaluate(model, loader, n_iter):
    model.eval()
    err = 0.0
    i = 0
    pbar = tqdm(total=len(loader), desc='pairs loaded')
    for i, (seq, beta, prof, dmat, pdb, *_) in enumerate(batch_generator(loader, prepare_pdb_batch)):

        cmap_hat, cmap = predict(model, seq, beta, prof, dmat)
        loss = get_loss(cmap_hat, cmap)
        err += loss.item()

        pbar.set_description("Validation Loss:%.6f" % (err / (i + 1.),))
        pbar.update(seq.size(0))

    writer.add_scalars('M3/Loss', {"valid": err / (i + 1.)}, n_iter)
    pbar.close()
    return err


def add_arguments(parser):
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-n', "--num_epochs", type=int, default=200,
                        help="How many epochs to train the model?")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    net = M3()
    net.to(device)
    net.register_backward_hook(hook_func)
    opt = ScheduledOptimizer(optim.Adam(net.parameters(), lr=LR), LR, num_iterations=2000)

    n_iter = 1
    init_epoch = 0
    num_epochs = args.num_epochs
    train_size = 10000
    test_size = 1000
    best_yet = np.inf

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '%s'" % args.resume)
            checkpoint = torch.load(args.resume)
            init_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['net'])
            opt.load_state_dict(checkpoint['opt'])
            best_yet = checkpoint['loss']
        else:
            print("=> no checkpoint found at '%s'" % args.resume)

    trainset = TRAIN_SET_CULL_PDB
    testset = VALID_SET_CULL_PDB
    loader_train = PdbLoader(trainset, train_size)
    loader_test = PdbLoader(testset, test_size)
    for epoch in range(init_epoch, num_epochs):
        n_iter = train(net, loader_train, opt, n_iter)
        loss = evaluate(net, loader_test, n_iter)
        loader_train.reset()
        loader_test.reset()

        save_checkpoint({
            'lr': opt.lr,
            'epoch': epoch,
            'net': net.state_dict(),
            'opt': opt.state_dict(),
            'loss': loss,
        }, "m3", args.out_dir, loss < best_yet)

        best_yet = min(best_yet, loss)


if __name__ == "__main__":
    main()
