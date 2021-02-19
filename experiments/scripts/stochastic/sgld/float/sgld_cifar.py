import sys
import torch
import argparse
from datetime import timedelta
import logging 

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../../../../../")
sys.path.append("../../../../../../")

from experiments.utils import evaluate_cifar_uncertainty
from src.data import *
from src.trainer import Trainer
from src.models import ModelFactory
from src.losses import LOSS_FACTORY
from src.models.stochastic.sgld.utils_sgld import SGLD
import src.utils as utils


parser = argparse.ArgumentParser("cifar_classifier")

parser.add_argument('--task', type=str, default='classification', help='the main task; defines loss')
parser.add_argument('--model', type=str, default='conv_resnet_sgld', help='the model that we want to train')

parser.add_argument('--learning_rate', type=float,
                    default=0.01, help='init learning rate')
parser.add_argument('--loss_scaling', type=str,
                    default='whole', help='smoothing factor')
parser.add_argument('--loss_multiplier', type=float,
                    default=16, help='smoothing factor')
                    

parser.add_argument('--data', type=str, default='./../../../data/',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar',
                    help='dataset')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')

parser.add_argument('--valid_portion', type=float,
                    default=0.1, help='portion of training data')

parser.add_argument('--burnin_epochs', type=int,
                    default=200, help='portion of training data')
parser.add_argument('--resample_momentum_iterations', type=int,
                    default=50, help='portion of training data')
parser.add_argument('--resample_prior_iterations', type=int,
                    default=25, help='portion of training data')

parser.add_argument('--epochs', type=int, default=300,
                    help='num of training epochs')

parser.add_argument('--input_size', nargs='+',
                    default=[1, 3, 32, 32], help='input size')
parser.add_argument('--output_size', type=int,
                    default=10, help='output size')
parser.add_argument('--samples', type=int,
                    default=20, help='output size')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--save_last', action='store_true', default=True,
                    help='whether to just save the last model') 

parser.add_argument('--num_workers', type=int,
                    default=16, help='number of workers')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--debug', action='store_true', help='whether we are currently debugging')

parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default = 0, help='gpu device ids')

parser.add_argument('--q', action='store_true', default=False,
                    help='whether to do post training quantisation')
parser.add_argument('--at', action='store_true', default=False,
                    help='whether to do training aware quantisation')



def main():
  args = parser.parse_args()
  load = False
  if args.save!='EXP':
    load=True
  args, writer = utils.parse_args(args)
  
  logging.info('# Start Re-training #')
  
  criterion = LOSS_FACTORY[args.task](args, args.loss_scaling)

  model_temp = ModelFactory.get_model
  logging.info('## Downloading and preparing data ##')
  train_loader, valid_loader= get_train_loaders(args)

  if not load:
    model= model_temp(args.model, args.input_size, args.output_size, args.at, args, True)

    logging.info('## Model created: ##')
    logging.info(model.__repr__())
    
    logging.info('### Loading model to parallel GPUs ###')
    model = utils.model_to_gpus(model, args)
    
    logging.info('### Preparing schedulers and optimizers ###') 
    optimizer = SGLD(
        model.parameters(),
        args.learning_rate)

    scheduler = None

    logging.info('## Beginning Training ##')

    train = Trainer(model, criterion, optimizer, scheduler, args)

    best_error, train_time, val_time = train.train_loop(
        train_loader, valid_loader, writer)

    logging.info('## Finished training, the best observed validation error: {}, total training time: {}, total validation time: {} ##'.format(
        best_error, timedelta(seconds=train_time), timedelta(seconds=val_time)))

    logging.info('## Beginning Plotting ##')
    del model 

  with torch.no_grad():
    model = model_temp(args.model, args.input_size, args.output_size, args.q, args, False)
    model.load_ensemble(args)

    logging.info('## Model re-created: ##')
    logging.info(model.__repr__())

    model = utils.model_to_gpus(model, args)
    
    model.eval()
    evaluate_cifar_uncertainty(model, args)

    logging.info('# Finished #')
    

if __name__ == '__main__':
  main()
