from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.training
import chainer.training.extensions

from model import VGG16


def main():
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    chainer.set_debug(True)
    # Initialize the model to train
    model = L.Classifier(VGG16())
    model.to_gpu(args.gpu)

    train, test = chainer.datasets.get_cifar100()

    train_iter = chainer.iterators.SerialIterator(train, batch_size=args.batchsize, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(test, batch_size=args.batchsize, shuffle=False, repeat=False)

    optimizer = chainer.optimizers.SGD()
    optimizer.use_cleargrads()
    optimizer.setup(model)

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(chainer.training.extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(chainer.training.extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    main()
