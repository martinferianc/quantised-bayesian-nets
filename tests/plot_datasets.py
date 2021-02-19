
import matplotlib.gridspec as gridspec
import sys
import argparse
import numpy as np
import logging

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../../../../../")

from src.data import *
import src.utils as utils
from src.data import get_test_loader, CIFAR_MEAN, CIFAR_STD, MNIST_MEAN, MNIST_STD
from experiments.presentation.plot_settings import PLT as plt


parser = argparse.ArgumentParser("test_distortions")

parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--data', type=str,
                    default='./../experiments/data', help='experiment name')

parser.add_argument('--label', type=str, default='dataset_sample_plots',
                    help='default experiment category ')
parser.add_argument('--dataset', type=str, default='',
                    help='default dataset ')
parser.add_argument('--batch_size', type=int, default=64,
                    help='default batch size')
parser.add_argument('--num_workers', type=int,
                    default=1, help='default batch size')
parser.add_argument('--valid_portion', type=float,
                    default=0.1, help='portion of training data')
parser.add_argument('--gpu', type=int,
                    default=-1, help='portion of training data')
parser.add_argument('--input_size', nargs='+',
                    default=[1, 3, 32, 32], help='input size')
parser.add_argument('--seed', type=int,
                    default=1, help='input size')
parser.add_argument('--q', type=bool,
                    default=False, help='input size')

INPUT_SIZES = {"mnist": (1, 1, 28, 28), "cifar": (
    1, 3, 32, 32)}


def main():
  args = parser.parse_args()
  logging.info('## Testing datasets ##')
  args, _ = utils.parse_args(args, args.label)
  datasets = ['mnist', 'cifar']

  for i, dataset in enumerate(datasets):
      plt.figure()
      gs = gridspec.GridSpec(5, 4)
      gs.update(wspace=0, hspace=0)
      args.dataset = dataset
      train_loader, valid_loader = get_train_loaders(args)
      test_loader = get_test_loader(args)
      args.input_size = INPUT_SIZES[dataset]
      args.dataset = "random_"+dataset
      random_loader = get_test_loader(args)
      for j, loader in enumerate([train_loader, valid_loader, test_loader, random_loader]):
        input, _ = next(iter(loader))
        input = input[:5]
        for k, image in enumerate(input):
          plt.subplot(gs[k, j])
          if "mnist" in args.dataset:
            plt.imshow(image.squeeze().numpy(), cmap='gray')
          elif "cifar" in args.dataset:
            means = np.array(CIFAR_MEAN).reshape((3, 1, 1))
            stds = np.array(CIFAR_STD).reshape((3, 1, 1))
            image = (image.numpy()*stds)+means
            plt.imshow(np.transpose(image, (1, 2, 0)))
          plt.axis('off')
      plt.tight_layout()
      path = utils.check_path(args.save+'/{}.png'.format(dataset))
      plt.savefig(path)


if __name__ == '__main__':
  main()
