from tqdm import tqdm
from torch import optim
from torch import autograd
from tempfile import gettempdir

from utils.torch_xu_utils import *
from utils.data import XU_TRAIN_SET, XU_VALID_SET, YOTAM_TRAIN_SET, YOTAM_VALID_SET
from utils.loader import XuLoader, batch_generator, prepare_xu_batch
from resnet import ResNet1d, ResNet2d, outconv

THR_DISTANCE = 1.0
UPLOAD_IMAGE_EVERY = 100
LR = 0.0002


def hook_func(module, grad_inputs, grad_outputs):
    for inp, out in zip(grad_inputs, grad_outputs):
        if torch.isnan(inp).any() or torch.isnan(out).any():
            raise RuntimeError("Detected NaN in grad")


class Xu0(nn.Module):
    def __init__(self, ic=40, hc=40, n_blocks=5, bins=bins):
        super(Xu0, self).__init__()
        self.name = "Xu0"
        oc = hc + (n_blocks - 1) * 5
        self.resnet1d_1 = ResNet1d(output_size=ic)
        self.resnet2d_1 = ResNet2d(ic=ic*3 + 1, hc=hc, num_num_blocks=n_blocks)
        self.outer_product = OuterProduct(include_mid=True)
        self.out_conv = outconv(oc, len(bins) + 1)
        self.W = nn.Linear(oc, len(bins) * 2)

    def forward(self, ohot, pssm, cmat):
        seq_info = torch.cat([ohot, pssm], 2)
        seq_info = self.resnet1d_1(seq_info.transpose(1, 2))
        op1 = self.outer_product(seq_info, seq_info)
        inp2d = torch.cat([op1, cmat.unsqueeze(1)], 1)
        out = self.resnet2d_1(inp2d)
        b, c, m, n = out.size()
        assert m == n
        A = self.out_conv(out)
        A = (A + A.transpose(2, 3)) / 2
        return A


def predict(model, seq, prof, cmat):
    out = model(seq, prof, cmat)
    return out


def upload_images(dmat, dmat_hat, pdb, n_iter, prefix):
    for m1, m2, pdb_id in zip(dmat.data.cpu().numpy(), dmat_hat.data.cpu().numpy(), pdb):
        writer.add_image('%s/%s/cmap_true' % (prefix, pdb_id), to_colormap_image(m1), n_iter, dataformats='HWC')
        writer.add_image('%s/%s/cmap_pred' % (prefix, pdb_id), to_colormap_image(m2), n_iter, dataformats='HWC')


def train(model, loader, optimizer, n_iter):
    model.train()
    err = 0.0
    i = 0
    pbar = tqdm(total=len(loader), desc='records loaded')
    for i, (seq, prof, cmat, dmat, pdb, *_) in enumerate(batch_generator(loader, prepare_xu_batch)):
        optimizer.zero_grad()

        cmap_hat = predict(model, seq, prof, cmat)

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
    for i, (seq, prof, cmat, dmat, pdb, *_) in enumerate(batch_generator(loader, prepare_xu_batch)):

        cmap_hat = predict(model, seq, prof, cmat)

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
    net = Xu0(ic=20, hc=20)
    net.to(device)
    net.register_backward_hook(hook_func)
    opt = ScheduledOptimizer(optim.Adam(net.parameters(), lr=LR), LR, num_iterations=2000)

    n_iter = 1
    init_epoch = 0
    num_epochs = args.num_epochs
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

    trainset = XU_TRAIN_SET
    testset = XU_VALID_SET
    loader_train = XuLoader(trainset)
    loader_test = XuLoader(testset)

    # print(model_summary(net))

    if args.eval:
        evaluate(net, loader_test, n_iter)
        return

    for epoch in range(init_epoch, num_epochs):
        n_iter = train(net, loader_train, opt, n_iter)
        loss = evaluate(net, loader_test, n_iter)
        loader_train.shuffle()
        loader_test.shuffle()

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
