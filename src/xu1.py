from tqdm import tqdm
from torch import optim
from torch import autograd
from tempfile import gettempdir

from utils.torch_xu_utils import *
from utils.data import TRAIN_SET_CULL_PDB, VALID_SET_CULL_PDB, YOTAM_TRAIN_SET, YOTAM_VALID_SET
from utils.loader import XuLoader, batch_generator, prepare_xu_batch
from resnet import ResNet1d, ResNet2d, outconv
from unet import UNet1d

THR_DISTANCE = 1.0
UPLOAD_IMAGE_EVERY = 100
LR = 0.0002


def hook_func(module, grad_inputs, grad_outputs):
    for inp, out in zip(grad_inputs, grad_outputs):
        if torch.isnan(inp).any() or torch.isnan(out).any():
            raise RuntimeError("Detected NaN in grad")


class Xu1(nn.Module):
    def __init__(self, ic=40, hc=40, n_blocks=5, bins=bins):
        super(Xu1, self).__init__()
        self.name = 'Xu1'
        oc = hc + (n_blocks - 1) * 5
        self.resnet1d_1 = ResNet1d(output_size=ic)
        self.resnet2d_1 = ResNet2d(ic=ic*3, hc=hc, num_num_blocks=n_blocks)
        self.outer_product = OuterProduct(include_mid=True)
        self.out_conv = outconv(oc, len(bins) + 1)
        self.W = nn.Linear(oc, len(bins) * 2)

    def forward(self, ohot, prof):
        seq_info = torch.cat([ohot, prof], 2)
        seq_info = self.resnet1d_1(seq_info.transpose(1, 2))
        op = self.outer_product(seq_info, seq_info)
        out = self.resnet2d_1(op)
        b, c, m, n = out.size()
        assert m == n
        A = self.out_conv(out)
        A = (A + A.transpose(2, 3)) / 2
        return A


class XuU(nn.Module):
    def __init__(self, input_size=40, ic=40, hc=40, n_blocks=5, bins=bins):
        super(XuU, self).__init__()
        self.name = 'XuU'
        oc = hc + (n_blocks - 1) * 5
        self.unet1d_1 = UNet1d(n_channels=input_size, inner_channels=hc, n_classes=ic)
        self.resnet2d_1 = ResNet2d(ic=ic*3, hc=hc, num_num_blocks=n_blocks)
        self.outer_product = OuterProduct(include_mid=True)
        self.out_conv = outconv(oc, len(bins) + 1)
        self.W = nn.Linear(oc, len(bins) * 2)

    def forward(self, ohot, prof):
        seq_info = torch.cat([ohot, prof], 2)
        seq_info = self.unet1d_1(seq_info.transpose(1, 2))
        op = self.outer_product(seq_info, seq_info)
        out = self.resnet2d_1(op)
        b, c, m, n = out.size()
        assert m == n
        A = self.out_conv(out)
        A = (A + A.transpose(2, 3)) / 2
        return A


class Ord(nn.Module):
    def __init__(self, ic=40, hc=40, n_blocks=5, bins=bins):
        super(Ord, self).__init__()
        self.name = 'Ord'
        oc = hc + (n_blocks - 1) * 5
        self.resnet1d_1 = ResNet1d(output_size=ic)
        self.resnet2d_1 = ResNet2d(ic=ic*3, hc=hc, num_num_blocks=n_blocks)
        self.outer_product = OuterProduct()
        self.W = nn.Linear(oc, len(bins) * 2)

    def forward(self, ohot, prof):
        seq_info = torch.cat([ohot, prof], 2)
        seq_info = self.resnet1d_1(seq_info.transpose(1, 2))
        op = self.outer_product(seq_info, seq_info)
        A = self.resnet2d_1(op)
        b, c, m, n = A.size()
        assert m == n
        A = (A + A.transpose(2, 3)) / 2
        A = self.W(A.view(b, -1, c)).view(b, m, n, -1, 2)
        return A


def predict(model, seq, prof):
    out = model(seq, prof)
    return out


def train(model, loader, optimizer, n_iter):
    model.train()
    err = 0.0
    i = 0
    pbar = tqdm(total=len(loader), desc='records loaded')
    for i, (seq, prof, _, dmat, pdb, *_) in enumerate(batch_generator(loader, prepare_xu_batch)):
        optimizer.zero_grad()

        cmap_hat = predict(model, seq, prof)

        if n_iter % UPLOAD_IMAGE_EVERY == 0:
            dmat_hat = cmap_to_dmat(cmap_hat)
            upload_images(dmat, dmat_hat, pdb, n_iter, '%s/%s' % (model.name, 'train'))

        loss = get_loss(cmap_hat, dmat)
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
    for i, (seq, prof, _, dmat, pdb, *_) in enumerate(batch_generator(loader, prepare_xu_batch)):

        cmap_hat = predict(model, seq, prof)

        if i % UPLOAD_IMAGE_EVERY == 0:
            dmat_hat = cmap_to_dmat(cmap_hat)
            upload_images(dmat, dmat_hat, pdb, i, '%s/%s' % (model.name, 'valid'))

        loss = get_loss(cmap_hat, dmat)
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
    parser.add_argument("-l", "--loss", type=str, choices=["ord", "ce"],
                        default="ce", help="Choose what loss function to use.")
    parser.add_argument("-e", '--eval', action='store_true', default=False,
                        help="Run in Eval mode.")


def main():
    global get_loss, cmap_to_dmat, use_ordinal
    import argparse
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    use_ordinal = args.loss == 'ord'
    get_loss = get_ordinal_log_loss if use_ordinal else get_cross_entropy_loss
    cmap_to_dmat = ordinal_cmap_to_dmat if use_ordinal else ce_cmap_to_dmat
    net = Ord(ic=15, hc=20) if use_ordinal else Xu1(ic=15, hc=20)
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

    trainset = YOTAM_TRAIN_SET
    testset = YOTAM_VALID_SET
    loader_train = XuLoader(trainset)
    loader_test = XuLoader(testset)

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
