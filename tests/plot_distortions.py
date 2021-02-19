import sys
import argparse
import numpy as np
import logging 

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../../../../../")


from experiments.utils import DISTORTIONS, LEVELS
import src.utils as utils
from src.data import get_test_loader
from experiments.presentation.plot_settings import PLT as plt
import matplotlib.gridspec as gridspec
from src.data import CIFAR_MEAN, CIFAR_STD

parser = argparse.ArgumentParser("test_distortions")

parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--data', type=str, default='./../experiments/data', help='experiment name')

parser.add_argument('--label', type=str, default='test_distortions', help='default experiment category ')
parser.add_argument('--dataset', type=str, default='mnist', help='default dataset ')
parser.add_argument('--batch_size', type=int, default=64, help='default batch size')
parser.add_argument('--num_workers', type=int, default=1, help='default batch size')


parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--debug', action='store_true', help='whether we are currently debugging')

parser.add_argument('--gpu', type=int, default = 0, help='gpu device ids')



def main():
  args = parser.parse_args()
  args, _ = utils.parse_args(args, args.label)
  logging.info('## Testing distortions ##')

  for distortion in DISTORTIONS:
      plt.figure(figsize=(3, 1))
      gs = gridspec.GridSpec(1, 5)
      gs.update(wspace=0, hspace=0)
      for level in range(LEVELS):
          test_loader = get_test_loader(args, distortion=distortion, level=level)
          input, _ = next(iter(test_loader))
          plt.subplot(gs[level])
          if args.dataset == "mnist":
            image = input[0]
            plt.imshow(image.squeeze().numpy(), cmap='gray')
          elif args.dataset == "cifar":
            image = input[2]
            means = np.array(CIFAR_MEAN).reshape((3,1,1))
            stds = np.array(CIFAR_STD).reshape((3,1,1))
            image = (image.numpy()*stds)+means
            plt.imshow(np.transpose(image,(1,2,0)))

          plt.axis('off')
      plt.tight_layout()
      path = utils.check_path(args.save+'/{}.png'.format(distortion))
      plt.savefig(path)

if __name__ == '__main__':
  main()