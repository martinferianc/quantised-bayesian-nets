import sys
import torch
import argparse
from datetime import timedelta
import logging 

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")

from src.data import *
from src.trainer import Trainer
from src.models import ModelFactory
from src.losses import LOSS_FACTORY
import src.utils as utils
from experiments.utils import evaluate_regression_uncertainty, REGRESSION_DATASETS

parser = argparse.ArgumentParser("pointwise_regression")

parser.add_argument('--task', type=str, default='regression', help='the main task; defines loss')
parser.add_argument('--model', type=str, default='linear', help='the model that we want to train')

parser.add_argument('--learning_rate', type=float,
                    default=0.001, help='init learning rate')

parser.add_argument('--loss_scaling', type=str,
                    default='batch', help='smoothing factor')
parser.add_argument('--weight_decay', type=float,
                    default=0.00005, help='weight decay')

parser.add_argument('--data', type=str, default='./../../data/',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='regression',
                    help='dataset')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size')

parser.add_argument('--valid_portion', type=float,
                    default=0.2, help='portion of training data')

parser.add_argument('--epochs', type=int, default=300,
                    help='num of training epochs')

parser.add_argument('--input_size', nargs='+',
                    default=[1], help='input size')
parser.add_argument('--output_size', type=int,
                    default=1, help='output size')
parser.add_argument('--samples', type=int,
                    default=1, help='output size')

parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--save_last', action='store_true', default=True,
                    help='whether to just save the last model') 

parser.add_argument('--num_workers', type=int,
                    default=0, help='number of workers')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--debug', action='store_true', help='whether we are currently debugging')

parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default = 0, help='gpu device ids')

parser.add_argument('--q', action='store_true',default=False,
                    help='whether to do post training quantisation')
parser.add_argument('--at', action='store_true',default=False,
                    help='whether to do training aware quantisation')


def main():
  args = parser.parse_args()
  load = False
  if args.save!='EXP':
    load=True
  
  model_temp = ModelFactory.get_model

  args, writer = utils.parse_args(args)
  logging.info('# Start Re-training #')
  if not load:
    for i, (dataset, n_folds) in enumerate(REGRESSION_DATASETS):
      for j in range(n_folds):
        logging.info('## Dataset: {}, Split: {} ##'.format(dataset, j))

        criterion = LOSS_FACTORY[args.task](args, args.loss_scaling)

        logging.info('## Downloading and preparing data ##')
        args.dataset = "regression_" + dataset

        train_loader, valid_loader= get_train_loaders(args, split=j)
        in_shape = next(iter(train_loader))[0].shape[1]
        args.input_size = [in_shape]

        model= model_temp(args.model, args.input_size, args.output_size, args.at, args)

        logging.info('## Model created: ##')
        logging.info(model.__repr__())
          
        logging.info('### Loading model to parallel GPUs ###')
      
        model = utils.model_to_gpus(model, args)

        logging.info('### Preparing schedulers and optimizers ###')
        optimizer = torch.optim.Adam(
            model.parameters(),
            args.learning_rate,
            weight_decay = args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs)
        
        logging.info('## Beginning Training ##')

        train = Trainer(model, criterion, optimizer, scheduler, args, writer=writer)

        best_error, train_time, val_time = train.train_loop(
            train_loader, valid_loader, special_info="_"+dataset+"_"+str(j))

        logging.info('## Finished training, the best observed validation error: {}, total training time: {}, total validation time: {} ##'.format(
            best_error, timedelta(seconds=train_time), timedelta(seconds=val_time)))


        del model 

  with torch.no_grad():
    logging.info('## Beginning Plotting ##')
    evaluate_regression_uncertainty(model_temp, args)

    logging.info('# Finished #')
    

if __name__ == '__main__':
  main()
