
import sys
import argparse
import numpy as np
import logging

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../../../../../")
sys.path.append("../../../../../../")

from experiments.presentation.plot_settings import PLT as plt
from matplotlib.ticker import MaxNLocator
import src.utils as utils
from experiments.utils import DISTORTIONS, LEVELS, METRICS, METRICS_UNITS, REGRESSION_DATASETS
from src.utils import BRIGHTNESS_LEVELS, SHIFT_LEVELS, ROTATION_LEVELS

parser = argparse.ArgumentParser("compare_ood_results")

parser.add_argument('--save', type=str, default='EXP', help='experiment name')

parser.add_argument('--pointwise_paths', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--mcd_paths', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--bbb_paths', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--sgld_paths', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--label', type=str, default='',
                    help='default experiment category ')
parser.add_argument('--xaxis', type=str, default='',
                    help='default experiment category ')
parser.add_argument('--weight', action='store_true', default=False, help='random seed')

parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--debug', action='store_true',
                    help='whether we are currently debugging')

parser.add_argument('--gpu', type=int, default=0, help='gpu device ids')
parser.add_argument('--q', action='store_true', default=False,
                    help='whether to do post training quantisation')
parser.add_argument('--at', action='store_true', default=False,
                    help='whether to do training aware quantisation')

def main():
  args = parser.parse_args()
  args, _ = utils.parse_args(args, args.label)
  labels = ['Pointwise', 'MCD', 'BBB', 'SGHMC']
  QUANT = None
  if args.weight:
    QUANT = [32, 8, 7, 6, 5, 4, 3]
  else:
    QUANT = [32, 7, 6, 5, 4, 3]
  if args.xaxis == "distortions":
      # This is for image data
      logging.info('## Loading of result pickles for the experiment ##')
      for yaxis in METRICS:
        fig = plt.figure(figsize=(5,2))
        plt.grid(True)
        bps= []
        for i, path in enumerate([args.pointwise_paths, args.mcd_paths, args.bbb_paths, args.sgld_paths]):
            if len(path)==0:
                continue   
            result = utils.load_pickle(path[0]+"/results.pickle")
            logging.info('### Loading result: {} ###'.format(result))
            data = []
            for level in range(-1,LEVELS):
                level_data = []
                for distortion in DISTORTIONS:
                    if level == -1:
                        level_data.append(result[yaxis]['test'][0])
                    else:
                        level_data.append(
                            result[yaxis][distortion][str(level)][0])

                data.append(level_data)
                
            data = np.array(data)
            positions = np.array([1+k*(5) + i for k in range(LEVELS+1)])
            bp = plt.boxplot(data.T, positions=positions, showfliers=False,
                            patch_artist=True, medianprops=dict(linewidth=2, color='black'), boxprops=dict(facecolor="C"+str(i), hatch='//' if i==0 else ""), widths=1)
            bps.append(bp)

        ax = fig.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=False))
        labels_brightness = [str(x[0])+"," for x in BRIGHTNESS_LEVELS]
        labels_rotation = [str(x[0]) +"°," for x in ROTATION_LEVELS]
        labels_shift = [str(x)  for x in SHIFT_LEVELS]
        ticks = zip(labels_brightness, labels_rotation, labels_shift)
        ticks = ["\n".join(tick) for tick in ticks]
        ticks = ["Test data\nwith no\nAugmentations"] + ticks
        positions = np.array([1+k*(5) + 1.5 for k in range(LEVELS+1)])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(ticks=positions, labels=ticks)
        plt.tick_params(axis="x", which="both", bottom=False)

        plt.xlabel('Distortions')
        plt.ylabel(METRICS_UNITS[yaxis])
        plt.tight_layout()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 - box.height * 0.08,
                         box.width, box.height * 0.92])
        ax.legend([bp["boxes"][0] for bp in bps], labels,
                  loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4)
        path = utils.check_path(
            args.save+'/{}_{}.pdf'.format(args.label, yaxis))
        plt.savefig(path)

  elif args.xaxis == "uci":
      logging.info('## Loading of result pickles for the experiment ##')
      for yaxis in ['error', 'nll']:
        fig = plt.figure(figsize=(5,2))
        plt.grid(True)
        bps= []
        for i, paths in enumerate([args.pointwise_paths, args.mcd_paths, args.bbb_paths, args.sgld_paths]):
            if len(paths)==0:
                continue   
            data = []
            for k, path in enumerate(paths):
                result = utils.load_pickle(path+"/results.pickle")
                logging.info('### Loading result: {} ###'.format(result))
                mean = []
                for dataset, _ in REGRESSION_DATASETS[1:]:
                    val = None
                    if isinstance(result[yaxis]
                                  ['regression_'+dataset]['test'], tuple):
                        val = result[yaxis]['regression_'+dataset]['test'][0]
                    else:
                        val = result[yaxis]['regression_'+dataset]['test']
                    if utils.isoutlier(val):
                        continue
                    if yaxis == "nll":
                        val*=-1
                    mean.append(val)
                mean = np.array(mean)
                data.append(mean)
                
            data = np.array(data, dtype=object)
            positions = np.array(
                [1+k*(5) + i for k in range(0, len(QUANT))])

            bp = plt.boxplot(data.T, positions=positions[:len(data)], showfliers=False,
                            patch_artist=True, medianprops=dict(linewidth=2, color='black'), boxprops=dict(facecolor="C"+str(i), hatch='//' if i==0 else ""), widths=1)
            bps.append(bp)

        ax = fig.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=False))
        if args.weight:
            ticks = ['$Float_{32}$' if j == 32 else '$Q:A_7\\boldsymbol{W_'+str(j)+"}$" for j in QUANT]
        else:
            ticks = ['$Float_{32}$' if j == 32 else '$Q:\\boldsymbol{A_'+str(j)+"}W_8$" for j in QUANT]
        positions = np.array([1+k*(5) + 1.5 for k in range(len(QUANT))])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(ticks=positions, labels=ticks)
        plt.tick_params(axis="x", which="both", bottom=False)
        if yaxis == "error":
            plt.ylim(0,1)
        elif yaxis == "nll":
            pass
        plt.xlabel('Bit-width \& Precision')
        plt.ylabel(METRICS_UNITS[yaxis+"_regression"])
        plt.tight_layout()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 - box.height * 0.08,
                         box.width, box.height * 0.92])
        ax.legend([bp["boxes"][0] for bp in bps], labels,
                  loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4)
        path = utils.check_path(
            args.save+'/{}_{}.pdf'.format(args.label, yaxis))
        plt.savefig(path)

if __name__ == '__main__':
  main()
