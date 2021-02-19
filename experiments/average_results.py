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
sys.path.append("../../../../../../../")
sys.path.append("../../../../../../../../")

import src.utils as utils

parser = argparse.ArgumentParser("average_results")

parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--result_paths', nargs='+', default='EXP', help='experiment name')
parser.add_argument('--label', type=str, default='', help='default experiment category ')

parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--debug', action='store_true', help='whether we are currently debugging')

parser.add_argument('--gpu', type=int, default = 0, help='gpu device ids')


def get_dict_path(dictionary, path=[]):
    for key, value in dictionary.items():
        if type(value) is dict:
            return get_dict_path(dictionary[key], path+[key])
        return path+[key]
    return path
                

def get_dict_value(dictionary, path=[], delete =True):
    if len(path)==1:
        val = dictionary[path[0]]
        if delete:
            dictionary.pop(path[0])
        return val
    else:
        return get_dict_value(dictionary[path[0]], path[1:])

def set_dict_value(dictionary, value, path=[]):
    if len(path)==1:
        dictionary[path[0]] = value
    else:
        if not path[0] in dictionary:
            dictionary[path[0]] = {}
        set_dict_value(dictionary[path[0]], value, path[1:])

def main():
  args = parser.parse_args()

  args, _ = utils.parse_args(args,args.label)
  logging.info('# Beginning analysis #')

  final_results = utils.load_pickle(args.save+"/results.pickle")

  logging.info('## Loading of result pickles for the experiment ##')

  results = []
  if len(args.result_paths)==1:
    args.result_paths = args.result_paths[0].split(" ")
  for result_path in args.result_paths:
      result = utils.load_pickle(result_path+"/results.pickle")
      logging.info('### Loading result: {} ###'.format(result))

      results.append(result)

  assert len(results)>1

  final_results['dataset'] = results[0]['dataset']
  final_results['model'] = results[0]['model']

  traversing_result = results[0]
  while len(get_dict_path(traversing_result))!=0:
      path = get_dict_path(traversing_result)
      values = []
      mean = None 
      std = None 
      for result in results:
          val = get_dict_value(result, path)
          if not isinstance(val, dict):
            values.append(val)
        
      if len(values) == 0 or type(values[0]) == str:
          continue

      if type(values[0]) == tuple:
          _values = []
          for i in range(len(values)):
            try:
                count, val = values[i]
                _values.append(val)
            except:
                _values.append(values[i])
          values = _values 

      values = np.array(values)
      mean = np.nanmean(values)
      std = np.nanstd(values)
      set_dict_value(final_results, (mean, std), path)
  
  logging.info('## Results: {} ##'.format(final_results))
  utils.save_pickle(final_results, args.save+"/results.pickle", True)

  

  logging.info('# Finished #')
    

if __name__ == '__main__':
  main()