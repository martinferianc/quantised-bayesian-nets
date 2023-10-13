import os
import numpy as np
import torch
import shutil
import random
import pickle
import torch.nn.functional as F
import sys
import time
import glob
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import re
import shutil
import copy 

UINT_BOUNDS = {8: [0, 255], 7: [0, 127], 6: [0, 63], 5: [0, 31], 4: [0, 15], 3: [0, 7], 2: [0, 3]}
INT_BOUNDS = {8: [-128, 127], 7: [-64, 63], 6: [-32, 31],
                  5: [-16, 15], 4: [-8, 7], 3: [-4, 3], 2: [-2, 1]}
BRIGHTNESS_LEVELS = [(1.5, 1.5), (2., 2.), (2.5, 2.5), (3, 3), (3.5, 3.5)]
ROTATION_LEVELS = [(15,15),(30,30),(45,45),(60,60),(75,75)]
SHIFT_LEVELS = [0.1,0.2,0.3,0.4,0.5]

def clamp_activation(x, args):
    if x.dtype == torch.quint8:
      _min = (UINT_BOUNDS[args.activation_precision][0]-x.q_zero_point())*x.q_scale()
      _max = (UINT_BOUNDS[args.activation_precision][1]-x.q_zero_point())*x.q_scale()
      x = torch.clamp(x, _min, _max)
    return x

def clamp_weight(x, args):
    if x.dtype == torch.qint8:
      _min = (INT_BOUNDS[args.weight_precision][0]-x.q_zero_point())*x.q_scale()
      _max = (INT_BOUNDS[args.weight_precision][1]-x.q_zero_point())*x.q_scale()
      x = torch.clamp(x, _min, _max)
    return x


class Flatten(torch.nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()
    
  def forward(self, x):
    if len(x.shape)==1:
      return x.unsqueeze(dim=0)
    return x.reshape(x.size(0), -1)

class Add(torch.nn.Module):
  def __init__(self):
    super(Add, self).__init__()
    self.add = torch.nn.quantized.FloatFunctional()

  def forward(self, x, y):
    return self.add.add(x,y)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(-?\d+)', text)]

def size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")/1e6
    os.remove('temp.p')
    return size

class AverageMeter(object):
  def __init__(self):
      self.reset()

  def reset(self):
      self.avg = 0.0
      self.sum = 0.0
      self.cnt = 0.0

  def update(self, val, n=1):
      self.sum += val * n
      self.cnt += n
      self.avg = self.sum / self.cnt


def save_model(model, args, special_info=""):
  _model = model
  if args.q and args.at and 'sgld' in args.model:
    from src.quant_utils import convert
    _model_copy = copy.deepcopy(model)
    _model = convert(_model_copy.cpu(), inplace=False)
  torch.save(_model.state_dict(), os.path.join(args.save, 'weights'+special_info+'.pt'))

  with open(os.path.join(args.save, 'args.pt'), 'wb') as handle:
    pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_pickle(data, path, overwrite=False):
  path = check_path(path) if not overwrite else path
  with open(path, 'wb') as fp:
      pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

def isoutlier(val):
    return val == np.inf or val == -np.inf or val<-9e1 or val>9e1 or np.isnan(val)

def load_pickle(path):
    file = open(path, 'rb')
    return pickle.load(file)

def transfer_weights(quantized_model, model):
    state_dict = model.state_dict()
    model = model.to('cpu')
    quantized_model.load_state_dict(state_dict)

def load_model(model, model_path, replace=True):
  state_dict = torch.load(model_path, map_location=torch.device('cpu'))
  model_dict = model.state_dict()
  pretrained_dict = {}
  for k,v in state_dict.items():
    _k = k
    if replace:
      _k = k.replace('module.','').replace('main_net.','')
    pretrained_dict[_k] = v
  pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict.keys()}
  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)
    
def create_exp_dir(path, scripts_to_save=None):
  path = check_path(path)
  os.mkdir(path)

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def check_path(path):
  if os.path.exists(path):
    filename, file_extension = os.path.splitext(path)
    counter = 0
    while os.path.exists(filename+"_"+str(counter)+file_extension):
      counter+=1
    return filename+"_"+str(counter)+file_extension
  return path


def model_to_gpus(model, args):
  if args.gpu!= -1:
    device = torch.device("cuda:"+str(args.gpu))
    model = model.to(device)
  return model

def check_quantized(x):
  return x.dtype == torch.qint8 or x.dtype == torch.quint8

def parse_args(args, label=""):
  if label=="":
    q="not_q"
    if args.q:
      q="q"
    if args.at:
      q+="at"
    label = q
  loading_path = args.save
  dataset = args.dataset if hasattr(args, 'dataset') else ""
  task = args.task if hasattr(args, 'task') else ""
  new_path = '{}-{}-{}-{}'.format(label, dataset, task, time.strftime("%Y%m%d-%H%M%S"))
  
  create_exp_dir(
    new_path, scripts_to_save=glob.glob('*.py') + \
                          glob.glob('../../src/**/*.py', recursive=True) + \
                          glob.glob('../../../src/**/*.py', recursive=True) + \
                          glob.glob('../../../../src/**/*.py', recursive=True) + \
                          glob.glob('../../../../../src/**/*.py', recursive=True) + \
                          glob.glob('../../../experiments/*.py', recursive=True) + \
                          glob.glob('../../../../experiments/*.py', recursive=True) + \
                          glob.glob('../../../../../experiments/*.py', recursive=True))
  args.save = new_path
  if loading_path!="EXP":
    for root, dirs, files in os.walk(loading_path):
        for filename in files:
          if ".pt" in filename:
            shutil.copy(os.path.join(loading_path, filename), os.path.join(new_path, filename))
  
  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  log_path = os.path.join(args.save, 'log.log')
  log_path = check_path(log_path)

  fh = logging.FileHandler(log_path)
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)
  
  print('Experiment dir : {}'.format(args.save))

  writer = SummaryWriter(
      log_dir=args.save+"/",max_queue=5)
  if torch.cuda.is_available() and args.gpu!=-1:
    logging.info('## GPUs available = {} ##'.format(args.gpu))
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
  else:
    logging.info('## No GPUs detected ##')
    
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  logging.info("## Args = %s ##", args)

  path = os.path.join(args.save, 'results.pickle')
  path= check_path(path)
  results = {}
  results["dataset"] = args.dataset if hasattr(args, 'dataset') else ""
  results["model"] = args.model if hasattr(args, 'model') else ""
  results["error"] = {}
  results["nll"] = {}
  results["latency"] = {}
  results["ece"] = {}
  results["entropy"] = {}

  save_pickle(results, path, True)
    
  return args, writer

      

          
  
