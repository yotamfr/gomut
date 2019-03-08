from tqdm import tqdm
from torch import optim
from torch import autograd
from tempfile import gettempdir

from utils.data import *
from utils.torch_utils import *
from utils.loader import PdbLoader, batch_generator, prepare_pdb_batch
from resnet.resnet_model import *


THR_DISTANCE = 1.0
UPLOAD_IMAGE_EVERY = 100
LR = 0.0002

bins = (4.5, 8.0, 15.0)
# bins = np.arange(4.5, 8.0, 15.5)
weight_bins = (8.0, 15.0)
weights = (20.5, 5.4, 1.0)


def to_onehot(seq_of_idx, num_classes=20):
    batch_sz, length = seq_of_idx.size(0), seq_of_idx.size(1)
    onehot = torch.zeros(batch_sz, length, num_classes, dtype=torch.float, device=device)
    onehot.scatter_(2, seq_of_idx.unsqueeze(2), 1)
    return onehot


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
    masks = quantize_distance_matrix(dmat, weight_bins)
    return masks, t_imj


def quantize_distance_matrix(dmat, bins=bins):
    qdmat = dmat.unsqueeze(1).repeat(1, len(bins) + 1, 1, 1)
    qdmat[:, 0, :, :] = (dmat < bins[0])
    for i in range(1, len(bins)):
        qdmat[:, i, :, :] = (dmat >= bins[i - 1]) * (dmat < bins[i])
    qdmat[:, len(bins), :, :] = (dmat >= bins[-1])
    return qdmat.float()


class OuterProduct(nn.Module):

    def __init__(self):
        super(OuterProduct, self).__init__()

    def forward(self, seq1, seq2):
        m = seq1.size(2)
        v_i = (seq1.transpose(1, 2).unsqueeze(2).repeat(1, 1, m, 1))
        v_j = v_i.transpose(1, 2)
        v = torch.cat([v_i, v_j, (v_i + v_j).div(2)], 3)
        return v.permute(0, 3, 1, 2)


class Xu(nn.Module):
    def __init__(self, bins=bins):
        super(Xu, self).__init__()
        self.resnet1d_1 = ResNet1d(oc=40)
        self.resnet2d_1 = ResNet2d(oc=len(bins) + 1)
        self.outer_product = OuterProduct()

    def forward(self, seq1, beta1, prof1):
        seq1_onehot = to_onehot(seq1)
        seq_info = torch.cat([seq1_onehot, prof1], 2)
        seq_info = self.resnet1d_1(seq_info.transpose(1, 2))
        op1 = self.outer_product(seq_info, seq_info)
        out = self.resnet2d_1(op1)
        return out


def get_loss(cmap_hat, dmat, weights=weights):
    b, c, m, n = cmap_hat.size()
    cmap = quantize_distance_matrix(dmat)
    masks, imj = mask_distance_matrix(dmat)
    v_cmap = cmap.view(c, b*m*n).transpose(0, 1).unsqueeze(1)
    v_cmap_hat = cmap_hat.view(c, b*m*n).transpose(0, 1).unsqueeze(2)
    ce = -v_cmap.bmm(F.log_softmax(v_cmap_hat, 1)).view(-1)
    losses = torch.zeros(ce.size(), dtype=torch.float, device=device)
    imj = imj.view(b*m*n)
    for i, w in enumerate(weights):
        msk = masks[:, i, :, :].contiguous().view(b*m*n)
        losses.add_(ce * msk * imj * w)
        losses.add_(ce * msk * (1.0 - imj) * max(1.0, w / 2.0))
    return losses.mean()


def predict(model, seq, beta, prof):
    cmap_hat = model(seq, beta, prof)
    return cmap_hat


def train(model, loader, optimizer, n_iter):
    model.train()
    err = 0.0
    i = 0
    pbar = tqdm(total=len(loader), desc='records loaded')
    for i, (seq, beta, prof, dmat, dssp, pdb, *_) in enumerate(batch_generator(loader, prepare_pdb_batch)):
        optimizer.zero_grad()

        cmap_hat = predict(model, seq, beta, prof)

        loss = get_loss(cmap_hat, dmat)
        err += loss.item()
        e = err / (i + 1.)

        writer.add_scalars('Xu/Loss', {"train": loss.item()}, n_iter)

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
    for i, (seq, beta, prof, dmat, dssp, pdb, *_) in enumerate(batch_generator(loader, prepare_pdb_batch)):

        cmap_hat = predict(model, seq, beta, prof)

        loss = get_loss(cmap_hat, dmat)
        err += loss.item()

        writer.add_scalars('M3/Loss', {"train": loss.item()}, n_iter)

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

    net = Xu()
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
        }, "xu", args.out_dir, loss < best_yet)

        best_yet = min(best_yet, loss)


if __name__ == "__main__":
    main()
