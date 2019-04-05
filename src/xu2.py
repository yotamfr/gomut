from tqdm import tqdm
from torch import optim
from torch import autograd

import numpy as np
import pandas as pd

from tempfile import gettempdir

from utils.torch_utils import *
from utils.constants import SECONDARY_STRUCTURE
from utils.data import TRAIN_SET_CULL_PDB, VALID_SET_CULL_PDB
from utils.loader import PdbLoader, batch_generator, prepare_pdb_batch_w_dssp
from resnet import ResNet1d, ResNet2d, outconv

THR_DISTANCE = 1.0
UPLOAD_IMAGE_EVERY = 100
LR = 0.0002

# bins = (8.0, 15.0)
bins = (4.5, 8.0, 15.0)
t_bins = torch.tensor((0,) + bins, dtype=torch.float, device=device)
# bins = np.arange(4.5, 15.5, 0.5)
weight_bins = (8.0, 15.0)
weights = (20.5, 5.4, 1.0)


def hook_func(module, grad_inputs, grad_outputs):
    for inp, out in zip(grad_inputs, grad_outputs):
        if torch.isnan(inp).any() or torch.isnan(out).any():
            raise RuntimeError("Detected NaN in grad")


def mask_distance_matrix(dmat, weight_bins=weight_bins):
    """
    Answer: yep, a larger weight is assigned to a pair of residues forming a contact.
    I assigned 20.5, 5.4, 1 to the distance 0-8, 8-15, and >15, respectively, for residue pairs (i, j) where |i-j| >=24.
    These numbers were derived from simple statistics of an old training set.
    However, you don't have to be very accurate here.
    When |i-j| is small, you can reduce 20.5 and 5.4 to smaller values.
    :param dmat: A distance matrix
    :param bins: The quantized distance matrix
    :return: The quantized distance matrix
    """
    b, m, n = dmat.size()
    imj = b * [[[abs(i-j) >= 24 for j in range(n)] for i in range(m)]]
    t_imj = torch.tensor(imj, dtype=torch.float, device=device)
    masks = quantize_distance_matrix(dmat, weight_bins, False)
    return masks, t_imj


def quantize_distance_matrix(dmat, bins=bins, use_ordinal=False):
    b, m, n = dmat.size()
    assert m == n
    qdmat = torch.zeros((b, len(bins) + int(not use_ordinal), m, n), dtype=torch.float, device=device)
    qdmat[:, 0, :, :] = (dmat < bins[0])
    for i in range(1, len(bins)):
        if not use_ordinal:
            qdmat[:, i, :, :] = (dmat >= bins[i - 1]) * (dmat < bins[i])
        else:
            qdmat[:, i, :, :] = (dmat < bins[i])
    if not use_ordinal:
        qdmat[:, len(bins), :, :] = (dmat >= bins[-1])
    return qdmat.float()


def digitize_distance_matrix(dmat, bins=bins):
    ddmat = torch.zeros(dmat.size(), dtype=torch.float, device=device)
    for i in range(1, len(bins)):
        ddmat[(dmat >= bins[i - 1]) * (dmat < bins[i])] = i
    ddmat[(dmat >= bins[-1])] = len(bins)
    return ddmat.float()


class OuterProduct(nn.Module):

    def __init__(self, include_mid=True):
        super(OuterProduct, self).__init__()
        self.include_middle = include_mid

    def forward(self, seq1, seq2):
        m = seq1.size(2)
        v_i = (seq1.transpose(1, 2).unsqueeze(2).repeat(1, 1, m, 1))
        v_j = v_i.transpose(1, 2)
        if self.include_middle:
            v = torch.cat([v_i, v_j, (v_i + v_j).div(2)], 3)
        else:
            v = torch.cat([v_i, v_j], 3)
        return v.permute(0, 3, 1, 2)


class Xu2(nn.Module):
    def __init__(self, input_channels=40, hidden_channels=40, n_blocks=5, bins=bins):
        super(Xu2, self).__init__()
        self.name = 'Xu2'
        oc = hidden_channels + (n_blocks - 1) * 5
        self.resnet1d_1 = ResNet1d(input_size=input_channels, output_size=hidden_channels)
        self.resnet1d_2 = ResNet1d(input_size=hidden_channels, output_size=len(SECONDARY_STRUCTURE))
        self.resnet2d_1 = ResNet2d(ic=hidden_channels * 3, hc=hidden_channels, num_num_blocks=n_blocks)
        self.outer_product = OuterProduct(include_mid=True)
        self.out_conv = outconv(oc, len(bins) + 1)
        self.W = nn.Linear(oc, len(bins) * 2)

    def forward(self, seq1_onehot, beta1, prof1):
        seq_info = torch.cat([seq1_onehot, prof1], 2).transpose(1, 2)
        seq_info = self.resnet1d_1(seq_info)
        dssp_hat = self.resnet1d_2(seq_info)
        op1 = self.outer_product(seq_info, seq_info)
        out = self.resnet2d_1(op1)
        b, c, m, n = out.size()
        assert m == n
        A = self.out_conv(out)
        A = (A + A.transpose(2, 3)) / 2
        return A, dssp_hat


def compute_weighted_loss_w_masks(losses, dmat, weights=weights):
    b, m, n = dmat.size()
    assert m == n
    masks, imj = mask_distance_matrix(dmat)
    masks, imj = masks.contiguous().view(-1, b * m * n), imj.view(b * m * n)
    w_losses = torch.zeros(losses.size(), dtype=torch.float, device=device)
    for i, w in enumerate(weights):
        msk = masks[i]
        w_losses.add_(losses * msk * imj * w)
        w_losses.add_(losses * msk * (1.0 - imj) * max(1.0, w / 2.0))
    return w_losses


def get_cross_entropy_loss(cmap_hat, dmat, dssp_hat, dssp, model, n_iter, loss_type="train"):
    b, c, m, n = cmap_hat.size()
    assert m == n
    indx = dssp.view(-1) != -1
    if indx.any():
        dssp_hat = dssp_hat.view(dssp_hat.size(1), -1)[:, indx]
        dssp = dssp.view(-1)[indx]
        ce_ss = torch.gather(-F.log_softmax(dssp_hat, 0), 0, dssp.unsqueeze(0))
    cmap = quantize_distance_matrix(dmat, use_ordinal=False)
    v_cmap = cmap.view(c, b*m*n).transpose(0, 1).unsqueeze(1)
    v_cmap_hat = cmap_hat.view(c, b*m*n).transpose(0, 1).unsqueeze(2)
    ce = -v_cmap.bmm(F.log_softmax(v_cmap_hat, 1)).view(-1)
    ce = compute_weighted_loss_w_masks(ce, dmat)
    if indx.any():
        ce1, ce2 = ce.mean(), ce_ss.mean()
        writer.add_scalars('%s/CE' % model.name, {loss_type: ce1.item()}, n_iter)
        writer.add_scalars('%s/CE-SS' % model.name, {loss_type: ce2.item()}, n_iter)
        return ce1 + ce2
    return ce.mean()


def predict(model, seq, beta, prof):
    return model(seq, beta, prof)


def ce_cmap_to_dmat(cmap):
    b, _, m, n = cmap.size()
    dmat = torch.gather(t_bins, 0, cmap.argmax(1).view(-1)).view(b, m, n)
    return dmat


def upload_images(dmat, dmat_hat, pdb, n_iter, prefix):
    for m1, m2, pdb_id in zip(dmat.data.cpu().numpy(), dmat_hat.data.cpu().numpy(), pdb):
        writer.add_image('%s/%s/cmap_true' % (prefix, pdb_id), to_colormap_image(m1), n_iter, dataformats='HWC')
        writer.add_image('%s/%s/cmap_pred' % (prefix, pdb_id), to_colormap_image(m2), n_iter, dataformats='HWC')


def train(model, loader, optimizer, n_iter):
    model.train()
    err = 0.0
    i = 0
    pbar = tqdm(total=len(loader), desc='records loaded')
    for i, (seq, beta, prof, dmat, dssp, pdb, *_) in enumerate(batch_generator(loader, prepare_pdb_batch_w_dssp)):
        optimizer.zero_grad()

        cmap_hat, dssp_hat = predict(model, seq, beta, prof)

        if n_iter % UPLOAD_IMAGE_EVERY == 0:
            dmat_hat = cmap_to_dmat(cmap_hat)
            upload_images(dmat, dmat_hat, pdb, n_iter, '%s/%s' % (model.name, 'train'))

        loss = get_loss(cmap_hat, dmat, dssp_hat, dssp, model, n_iter, 'train')
        err += loss.item()
        e = err / (i + 1.)

        writer.add_scalars('%s/Loss' % model.name, {"train": loss.item()}, n_iter)

        try:
            with autograd.detect_anomaly():
                loss.backward()
        except RuntimeError as e:
            raise e

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
    pbar = tqdm(total=len(loader), desc='records loaded')
    for i, (seq, beta, prof, dmat, dssp, pdb, *_) in enumerate(batch_generator(loader, prepare_pdb_batch_w_dssp)):

        cmap_hat, dssp_hat = predict(model, seq, beta, prof)

        if i % UPLOAD_IMAGE_EVERY == 0:
            dmat_hat = cmap_to_dmat(cmap_hat)
            upload_images(dmat, dmat_hat, pdb, i, '%s/%s' % (model.name, 'valid'))

        loss = get_loss(cmap_hat, dmat, dssp_hat, dssp, model, n_iter, 'valid')
        err += loss.item()

        pbar.set_description("Validation Loss:%.6f (L=%d)" % (err / (i + 1.), seq.size(1)))
        pbar.update(seq.size(0))

    writer.add_scalars('%s/Loss' % model.name, {"valid": err / (i + 1.)}, n_iter)
    pbar.close()
    return err / (i + 1.)


def add_arguments(parser):
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-n', "--num_epochs", type=int, default=20,
                        help="How many epochs to train the model?")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")
    # parser.add_argument("-l", "--loss", type=str, choices=["ord", "ce"],
    #                     default="ce", help="Choose what loss function to use.")
    parser.add_argument("-e", '--eval', action='store_true', default=False,
                        help="Run in Eval mode.")


def main():
    global get_loss, cmap_to_dmat
    import argparse
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    get_loss = get_cross_entropy_loss
    cmap_to_dmat = ce_cmap_to_dmat
    net = Xu2(input_channels=40, hidden_channels=20)
    net.to(device)
    net.register_backward_hook(hook_func)
    opt = ScheduledOptimizer(optim.Adam(net.parameters(), lr=LR), LR, num_iterations=20000)

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

    # print(model_summary(net))

    if args.eval:
        evaluate(net, loader_test, n_iter)
        return

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
        }, net.name, args.out_dir, loss < best_yet)

        best_yet = min(best_yet, loss)


if __name__ == "__main__":
    main()
