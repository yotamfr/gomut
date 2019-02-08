from tqdm import tqdm

from utils.loader import *
from unet import *


def train(b_generator, length):
    pbar = tqdm(total=length, desc='pairs loaded')
    for s1, s2, m1, m2, idx in b_generator:
        assert m1.shape == m2.shape
        pbar.update(len(idx))
    pbar.close()


def evaluate():
    pass


def main():

    net = UNet(n_channels=1, n_classes=1)

    init_epoch = 0
    num_epochs = 20
    train_size = 100000
    test_size = 10000

    trainset = TRAIN_SET
    testset = VALID_SET

    for epoch in range(init_epoch, num_epochs):

        train_generator = batch_generator(pairs_loader(trainset, train_size), prepare_torch_batch)
        test_generator = batch_generator(pairs_loader(testset, test_size), prepare_torch_batch)
        train(train_generator, train_size)


if __name__ == "__main__":
    main()
