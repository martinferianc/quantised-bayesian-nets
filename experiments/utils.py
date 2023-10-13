import numpy as np
import torch
from tqdm import tqdm
import sys
import logging

sys.path.append('../')

from src.data import *
import src.utils as utils
from experiments.presentation.plot_settings import PLT as plt
import src.quant_utils as quant_utils
from src.metrics import ClassificationMetric, RegressionMetric

DISTORTIONS = ['rotation', 'shift', 'brightness']
METRICS = ['entropy','ece','error','nll']
METRICS_UNITS = {'entropy': "aPE [nats]", 'ece': "ECE [\%]",
                 'error': "Error [\%]", 'nll': "NLL", 'error_regression': "RMSE", 'nll_regression': "NLL"}
LEVELS = 5
N_FOLDS = 10
REGRESSION_DATASETS =[("synthetic",1),("housing",N_FOLDS),("concrete",N_FOLDS),("energy",N_FOLDS),("power",N_FOLDS),("wine",N_FOLDS),("yacht",N_FOLDS)]
RELEVANT_COMBINATIONS = {'classification':[['error', 'test'], ['ece', 'test'], ['entropy', 'test'], ['nll', 'test'], ['ece', 'random'], ['entropy', 'random'], ['nll', 'random']],
                         'regression':[['error', 'synthetic'], ['nll', 'synthetic'], ['error', 'uci'], ['nll', 'uci']]}
def evaluate_mnist_uncertainty(model, args):
    results = utils.load_pickle(args.save+"/results.pickle")
    output, target = _evaluate_and_record(model, results, args)

    f = _plot_ece(output, target, plt)
    path = utils.check_path(args.save+'/ece_test.png')
    plt.savefig(path)
    f = _plot_model_certainty(output, plt)
    path = utils.check_path(args.save+'/certainty_test.png')
    plt.savefig(path)

    args.dataset='random_mnist'
    test_loader = get_test_loader(args)
    error, ece, entropy, nll, output, target = _evaluate_with_loader(test_loader, model, args)
    logging.info("## Random Error: {} ##".format(error))
    logging.info("## Random ECE: {} ##".format(ece))
    logging.info("## Random Entropy: {} ##".format(entropy))
    logging.info("## Random NLL: {} ##".format(nll))

    results["entropy"]["random"] = entropy
    results["ece"]["random"] = ece
    results["error"]["random"] = error
    results["nll"]["random"] = nll
    f = _plot_ece(output, target, plt)
    path = utils.check_path(args.save+'/ece_random.png')
    plt.savefig(path)
    f = _plot_model_certainty(output, plt)
    path = utils.check_path(args.save+'/certainty_random.png')
    plt.savefig(path)

    args.dataset='mnist'
    for distortion in DISTORTIONS:
      for level in range(LEVELS):
        test_loader = get_test_loader(args, distortion, level)
        error, ece, entropy, nll, _, _= _evaluate_with_loader(test_loader, model, args)
        logging.info('## Distortion: {}, Level: {} ##'.format(distortion, level+1))
        logging.info("## Distortion Error: {} ##".format(error))
        logging.info("## Distortion ECE: {} ##".format(ece))
        logging.info("## Distortion Entropy: {} ##".format(entropy))
        logging.info("## Distortion NLL: {} ##".format(nll))

        if distortion not in results["entropy"]:
          results["entropy"][distortion] = {}
          results["ece"][distortion] = {}
          results["error"][distortion] = {}
          results["nll"][distortion] = {}

        results["entropy"][distortion][str(level)] = entropy
        results["ece"][distortion][str(level)] = ece
        results["error"][distortion][str(level)] = error
        results["nll"][distortion][str(level)]= nll

    utils.save_pickle(results, args.save+"/results.pickle", True)
    logging.info("## Results: {} ##".format(results))

def evaluate_cifar_uncertainty(model, args):
    results = utils.load_pickle(args.save+"/results.pickle")
    output, target = _evaluate_and_record(model, results, args)

    f = _plot_ece(output, target, plt)
    path = utils.check_path(args.save+'/ece_test.png')
    plt.savefig(path)
    f = _plot_model_certainty(output, plt)
    path = utils.check_path(args.save+'/certainty_test.png')
    plt.savefig(path)

    args.dataset='random_cifar'
    test_loader = get_test_loader(args)
    error, ece, entropy, nll, output, target = _evaluate_with_loader(test_loader, model, args)
    logging.info("## Random Error: {} ##".format(error))
    logging.info("## Random ECE: {} ##".format(ece))
    logging.info("## Random Entropy: {} ##".format(entropy))
    logging.info("## Random NLL: {} ##".format(nll))

    results["entropy"]["random"] = entropy
    results["ece"]["random"] = ece
    results["error"]["random"] = error
    results["nll"]["random"] = nll
    f = _plot_ece(output, target, plt)
    path = utils.check_path(args.save+'/ece_random.png')
    plt.savefig(path)
    f = _plot_model_certainty(output, plt)
    path = utils.check_path(args.save+'/certainty_random.png')
    plt.savefig(path)

    args.dataset='cifar'
    for distortion in DISTORTIONS:
      for level in range(LEVELS):
        test_loader = get_test_loader(args, distortion, level)
        error, ece, entropy, nll, _, _= _evaluate_with_loader(test_loader, model, args)
        logging.info('## Distortion: {}, Level: {} ##'.format(distortion, level+1))
        logging.info("## Distortion Error: {} ##".format(error))
        logging.info("## Distortion ECE: {} ##".format(ece))
        logging.info("## Distortion Entropy: {} ##".format(entropy))
        logging.info("## Distortion NLL: {} ##".format(nll))

        if distortion not in results["entropy"]:
          results["entropy"][distortion] = {}
          results["ece"][distortion] = {}
          results["error"][distortion] = {}
          results["nll"][distortion] = {}

        results["entropy"][distortion][str(level)] = entropy
        results["ece"][distortion][str(level)] = ece
        results["error"][distortion][str(level)] = error
        results["nll"][distortion][str(level)]= nll

    utils.save_pickle(results, args.save+"/results.pickle", True)
    logging.info("## Results: {} ##".format(results))

def evaluate_regression_uncertainty(model_temp, args):
    results = utils.load_pickle(args.save+"/results.pickle")
    for i, (dataset, n_folds) in enumerate(REGRESSION_DATASETS):
      nlls_train = []
      rmses_train= []
      nlls_valid = []
      rmses_valid= []
      nlls_test = []
      rmses_test= []
      for j in range(n_folds):
        logging.info('## Dataset: {}, Split: {} ##'.format(dataset, j))
        logging.info('## Downloading and preparing data ##')
        args.dataset = "regression_" + dataset

        train_loader, val_loader = get_train_loaders(args, split = j)
        test_loader = get_test_loader(args, split = j)

        in_shape = next(iter(train_loader))[0].shape[1]
        args.input_size = [in_shape]

        model = model_temp(args.model, args.input_size, args.output_size, args.q, args, False)
        if args.q:
          quant_utils.prepare_model(model, args)
          quant_utils.convert(model)
        if not 'sgld' in args.model:
          utils.load_model(model, args.save+"/weights" + "_{}_{}".format(dataset, j)+".pt")
        else:
          model.load_ensemble(args, special_info="{}_{}_".format(dataset, j))
        logging.info('## Model created: ##')
        logging.info(model.__repr__())
        if not args.q:
          model = utils.model_to_gpus(model, args)
        model.eval()
   
        error, _, _, nll, _, _ = _evaluate_with_loader(train_loader, model, args)
        rmses_train.append(error)
        nlls_train.append(nll)
      
        error,_, _, nll, _, _= _evaluate_with_loader(val_loader, model, args)
        rmses_valid.append(error)
        nlls_valid.append(nll)

        error, _, _, nll, _, _ = _evaluate_with_loader(test_loader, model, args)
        rmses_test.append(error)
        nlls_test.append(nll)
        del model
      if n_folds == 1:
        nlls_train = nlls_train[0]
        rmses_train= rmses_train[0]
        nlls_valid = nlls_valid[0]
        rmses_valid= rmses_valid[0]
        nlls_test = nlls_test[0]
        rmses_test= rmses_test[0]
      else:
        nlls_train = np.nanmean(nlls_train)
        rmses_train= np.nanmean(rmses_train)
        nlls_valid = np.nanmean(nlls_valid)
        rmses_valid= np.nanmean(rmses_valid)
        nlls_test = np.nanmean(nlls_test)
        rmses_test= np.nanmean(rmses_test)

      if args.dataset not in results["entropy"]:
        results["entropy"][args.dataset] = {}
        results["ece"][args.dataset] = {}
        results["error"][args.dataset] = {}
        results["nll"][args.dataset] = {}

      logging.info("## Train Error: {} ##".format(rmses_train))
      logging.info("## Train NLL: {} ##".format(nlls_train))

      results["error"][args.dataset]["train"] = rmses_train
      results["nll"][args.dataset]["train"] = nlls_train

      logging.info("## Valid Error: {} ##".format(rmses_valid))
      logging.info("## Valid NLL: {} ##".format(nlls_valid))

      results["error"][args.dataset]["valid"] = rmses_valid
      results["nll"][args.dataset]["valid"] = nlls_valid

      logging.info("## Test Error: {} ##".format(rmses_test))
      logging.info("## Test NLL: {} ##".format(nlls_test))

      results["error"][args.dataset]["test"] = rmses_test
      results["nll"][args.dataset]["test"] = nlls_test

    args.dataset = "regression_synthetic"
    train_loader, _ = get_train_loaders(args, split=-1)
    in_shape = next(iter(train_loader))[0].shape[1]
    args.input_size = [in_shape]
    model= model_temp(args.model, args.input_size, args.output_size, args.q, args, False)
    if args.q:
      quant_utils.prepare_model(model, args)
      quant_utils.convert(model)

    if not 'sgld' in args.model:
      utils.load_model(model, args.save+"/weights_synthetic_0.pt")
    else:
      model.load_ensemble(
          args, special_info="synthetic_0_")

    model.eval()

    fig, ax = plt.subplots(1, 1)
    batch_size = 25
    N_GRID = 1000
    xi_test = np.linspace(-5, 5, N_GRID)
    yi_test = regression_function(xi_test, noise=False)
    xi_test = torch.from_numpy(xi_test).float()
    yi_test = torch.from_numpy(yi_test).float()

    xi_random, yi_random = regression_data_generator(N_points=20)
    mus = []
    epistemic_uncertainties = []
    aleatoric_uncertainties = []
    if 'sgld' not in args.model and args.samples!=1:
      args.samples = 100
    for i in tqdm(range(0, len(xi_test)//batch_size)):
      _mus = []
      _vars = []
      for j in range(args.samples):
        x = xi_test[i*batch_size:(1+i)*batch_size]
        x.unsqueeze_(1)
        if not args.q and next(model.parameters()).is_cuda:
          x = x.cuda()
        mu, var = model(x)
        _mus.append(mu.detach().cpu())
        _vars.append(var.detach().cpu())
      mus.append(torch.stack(_mus, dim=1).mean(dim=1))
      epistemic_uncertainties.append(torch.stack(_mus, dim=1).var(dim=1))
      aleatoric_uncertainties.append(torch.stack(_vars, dim=1).mean(dim=1))

    mean = torch.cat(mus, dim=0).flatten()
    epistemic_uncertainties = torch.cat(
        epistemic_uncertainties, dim=0).flatten()
    aleatoric_uncertainties = torch.cat(
        aleatoric_uncertainties, dim=0).flatten()
    total_uncertainties = (epistemic_uncertainties + aleatoric_uncertainties)**0.5
    ax.plot(xi_test.numpy(), yi_test.numpy(), label='True function', color='k')
    ax.plot(xi_test.numpy(), mean.numpy(), label='Predicted mean', color='r', linestyle='--')
    ax.scatter(xi_random, yi_random,
            label='Random training points', color='b')
    if args.samples>1:
        ax.fill_between(xi_test.numpy(), (mean - total_uncertainties).numpy(),
                        (mean + total_uncertainties).numpy(), color='r', alpha=0.3, label='Total uncertainty')
        ax.fill_between(xi_test.numpy(), (mean - aleatoric_uncertainties**0.5).numpy(), 
                        (mean + aleatoric_uncertainties**0.5).numpy(), color='b', alpha=0.3, label='Aleatoric uncertainty')
        ax.fill_between(xi_test.numpy(), (mean - epistemic_uncertainties**0.5).numpy(),
                        (mean + epistemic_uncertainties**0.5).numpy(), color='g', alpha=0.4, label='Epistemic uncertainty')
    else:
        ax.fill_between(xi_test.numpy(), (mean-aleatoric_uncertainties).numpy(),
                        (mean+aleatoric_uncertainties).numpy(), color='g', alpha=0.4, label='Aleatoric uncertainty')

    ax.legend(loc='upper left')
    plt.tight_layout()
    path = utils.check_path(args.save+'/regression.png')
    plt.savefig(path)
    utils.save_pickle(results, args.save+"/results.pickle", True)
    logging.info("## Results: {} ##".format(results))

def _plot_ece(output, labels, plt, n_bins=10):
    confidences, predictions = output.max(1)
    accuracies = torch.eq(predictions, labels)
    f, rel_ax = plt.subplots(1, 1, figsize=(4, 2.5))

    bins = torch.linspace(0, 1, n_bins + 1)

    bins[-1] = 1.0001
    width = bins[1] - bins[0]
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    bin_corrects = np.nan_to_num(np.array([torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices]))
    bin_scores = np.nan_to_num(np.array([torch.mean(confidences[bin_index]) for bin_index in bin_indices]))
  
    confs = rel_ax.bar(bins[:-1], np.array(bin_corrects), align='edge', width=width, alpha=0.75, edgecolor='b')
    gaps = rel_ax.bar(bins[:-1], bin_scores -bin_corrects, align='edge',
                      bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
    rel_ax.plot([0, 1], [0, 1], '--', color='gray')
    rel_ax.legend([confs, gaps], ['Outputs', 'Gap'], loc='best', fontsize='small')
    rel_ax.set_ylabel('Accuracy')
    rel_ax.set_xlabel('Confidence')
    plt.tight_layout()
    return f

def _plot_model_certainty(output, plt, n_bins=10):
    confidences, _ = output.max(1)
    confidences = np.nan_to_num(confidences)
    f, rel_ax = plt.subplots(1, 1, figsize=(4, 2.5))
    bin_height,bin_boundary = np.histogram(confidences,bins=n_bins)
    width = bin_boundary[1]-bin_boundary[0]
    bin_height = bin_height/float(max(bin_height))
    rel_ax.bar(bin_boundary[:-1],bin_height,width = width, align='center', color='b', label="Normalized counts")
    rel_ax.legend()
    rel_ax.set_xlabel('Confidence')
    f.tight_layout()

    return f

def _evaluate_with_loader(loader, model, args):
    target = []
    output = []
    metric = ClassificationMetric(output_size=args.output_size) if args.task == 'classification' else RegressionMetric(output_size=args.output_size)
    
    for i, (input, _target) in enumerate(loader):
      input = torch.autograd.Variable(input, requires_grad=False)
      _target = torch.autograd.Variable(_target, requires_grad=False)
      if not args.q and next(model.parameters()).is_cuda:
        input = input.cuda()
        _target = _target.cuda()
        
      _output = model(input)
      
      if args.samples>1 and model is not None and model.training is False:
        y = [_output]
        for _ in range(1, args.samples):
          y.append(model(input))
        if "regression" in args.task:
          mu = [_y[0] for _y in y]
          var = [_y[1] for _y in y]
          mean = torch.stack(mu, dim=1).mean(dim=1)
          var = torch.stack(mu, dim=1).var(dim=1) + torch.stack(var, dim=1).mean(dim=1)
          _output = (mean, var)
        else:
          _output = torch.stack(y, dim=1).mean(dim=1)
          
      metric.update(_output, _target)

      target.append(_target)
      output.append(_output)
      if args.debug:
        break
    
    target = torch.cat(target, dim=0).cpu()
    if isinstance(output[0], tuple):
      mu = [_y[0] for _y in output]
      var = [_y[1] for _y in output]
      output = [torch.cat(mu, dim=0).cpu(), torch.cat(var, dim=0).cpu()]
    else:
      output = torch.cat(output, dim=0).cpu()
      
    error = metric.error.compute().item() if args.task == 'classification' else metric.rmse.compute().item()
    ece = metric.ece.compute().item() if args.task == 'classification' else 0.0
    entropy = metric.entropy.compute().item() if args.task == 'classification' else 0.0
    nll = metric.nll.compute().item()
      
    return error, ece, entropy, nll, output, target

def _evaluate_and_record(model, results, args, train=True, valid=True, test=True):
    train_loader, val_loader = get_train_loaders(args)
    test_loader = get_test_loader(args)

    if train:
      error, ece, entropy, nll, _, _ = _evaluate_with_loader(train_loader, model, args)
      logging.info("## Train Error: {} ##".format(error))
      logging.info("## Train ECE: {} ##".format(ece))
      logging.info("## Train Entropy: {} ##".format(entropy))
      logging.info("## Train NLL: {} ##".format(nll))

      results["entropy"]["train"] = entropy
      results["ece"]["train"] = ece
      results["error"]["train"] = error
      results["nll"]["train"] = nll

    if valid:
      error, ece, entropy, nll, _, _= _evaluate_with_loader(val_loader, model, args)
      logging.info("## Valid Error: {} ##".format(error))
      logging.info("## Valid ECE: {} ##".format(ece))
      logging.info("## Valid Entropy: {} ##".format(entropy))
      logging.info("## Valid NLL: {} ##".format(nll))

      results["entropy"]["valid"] = entropy
      results["ece"]["valid"] = ece
      results["error"]["valid"] = error
      results["nll"]["valid"] = nll

    if test:
      error, ece, entropy, nll, output, target = _evaluate_with_loader(test_loader, model, args)
      logging.info("## Test Error: {} ##".format(error))
      logging.info("## Test ECE: {} ##".format(ece))
      logging.info("## Test Entropy: {} ##".format(entropy))
      logging.info("## Test NLL: {} ##".format(nll))

      results["entropy"]["test"] = entropy
      results["ece"]["test"] = ece
      results["error"]["test"] = error
      results["nll"]["test"] = nll
      return output, target
