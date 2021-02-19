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
from experiments.utils import METRICS_UNITS, RELEVANT_COMBINATIONS, REGRESSION_DATASETS

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
parser.add_argument('--task', type=str,
                    default='classification', help='experiment task')
parser.add_argument('--label', type=str, default='',
                    help='default experiment category ')

parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--debug', action='store_true',
                    help='whether we are currently debugging')
parser.add_argument('--weight', action='store_true',
                    default=False, help='random seed')

parser.add_argument('--gpu', type=int, default=0, help='gpu device ids')
parser.add_argument('--q', action='store_true', default=False,
                    help='whether to do post training quantisation')
parser.add_argument('--at', action='store_true', default=False,
                    help='whether to do training aware quantisation')

def main():
  args = parser.parse_args()
  args, _ = utils.parse_args(args, args.label)
  labels = ['Pointwise', 'MCD', 'BBB', 'SGHMC']
  QUANT= None 
  if args.weight:
    QUANT = [32, 8, 7, 6, 5, 4, 3]
  else:
    QUANT = [32, 7, 6, 5, 4, 3]
  for i, combination in enumerate(RELEVANT_COMBINATIONS[args.task]):
      logging.info('## Loading of result pickles for the experiment ##')
      fig = plt.figure(figsize=(5,1.75))
      plt.grid(True)
      for j, paths in enumerate([args.pointwise_paths, args.mcd_paths, args.bbb_paths, args.sgld_paths]):
        if len(paths)==0:
            continue 
        data = []
        for k, path in enumerate(paths):
            result = utils.load_pickle(path+"/results.pickle")
            logging.info('### Loading result: {} ###'.format(result))
            if args.task=='classification':
                data.append(result[combination[0]][combination[1]])
            elif combination[1]=='uci' and args.task=='regression':
                mean = []
                for dataset, _ in REGRESSION_DATASETS[1:]:
                    val = result[combination[0]
                                 ]['regression_'+dataset]['test'][0]
                    if utils.isoutlier(val):
                      continue
                    mean.append(val)
                  
                data.append([np.mean(mean), np.std(mean)])
                if combination[0] == "nll":
                      data[-1][0]*=-1  
            elif combination[1]=='synthetic' and args.task=='regression':
                d = list(result[combination[0]]["regression_synthetic"]['test'])
                if utils.isoutlier(d[0]):
                      continue
                data.append(d) 
                if combination[0] == "nll":
                  data[-1][0]*=-1  
    
        positions = np.array([x for x in range(len(data))])
        mean = [d[0] for d in data]
        stds = [d[1] for d in data]
        plt.plot(positions, mean,
                 color="C"+str(j), alpha=0.7, label=labels[j])
        plt.errorbar(positions, mean, yerr=stds, fmt='o' if j != 0 else 'v', capsize=10,
                     color="C"+str(j), alpha=0.7)

      ax = fig.gca()
      ax.xaxis.set_major_locator(MaxNLocator(integer=False))
      positions = np.array([k for k in range(len(QUANT))])
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      if args.weight:
        ticks = ['$Float_{32}$' if j == 32 else '$Q:A_7\\boldsymbol{W_'+str(j)+"}$" for j in QUANT]
      else:
        ticks = ['$Float_{32}$' if j == 32 else '$Q:\\boldsymbol{A_'+str(j)+"}W_8$" for j in QUANT]
      plt.tick_params(axis="x", which="both", bottom=False)
      plt.xticks(ticks=positions, labels=ticks)
      if combination[0] == "error" and "mnist" in args.label and not args.weight:
        plt.ylim(0, 10)
      elif combination[0] == "error" and "cifar" in args.label and not args.weight:
        plt.ylim(0, 22.5)

      ax.legend(loc='upper left')
      plt.xlabel('Bit-width \& Precision')
      plt.ylabel(METRICS_UNITS[combination[0] +
                               "_regression" if args.task == "regression" else combination[0]])
      plt.tight_layout()
      path = utils.check_path(
        args.save+'/{}_{}_{}.pdf'.format(combination[0], combination[1], "weight" if args.weight else "activation"))
      box = ax.get_position()
      ax.set_position([box.x0, box.y0 - box.height * 0.08,
                       box.width, box.height * 0.92])
      ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4)
      plt.savefig(path)

if __name__ == '__main__':
  main()

