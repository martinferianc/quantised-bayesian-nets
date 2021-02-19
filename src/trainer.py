
import torch
import src.utils as utils
import time 
import logging 
from src.models.stochastic.sgld.utils_sgld import SGLD
import numpy as np

class Trainer():
  def __init__(self, model, criterion, optimizer, scheduler, args):
    super().__init__()
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.args = args
    
    self.train_step = 0
    self.train_time = 0.0
    self.val_step = 0
    self.val_time = 0.0

    self.grad_buff = []
    self.max_grad = 1e20
    self.grad_std_mul = 30

    self.epoch = 0
    self.iteration = 0

  def _scalar_logging(self, obj, main_obj, nll, kl, error_metric, ece, entropy, info, iteration, writer):
    _error_metric, _main_obj = None, None
    if "classification" in self.args.task:
        _error_metric, _main_obj  = 'error', 'ce'
    elif self.args.task == "regression":
        _error_metric, _main_obj = 'rmse', 'mse'
    writer.add_scalar(info+_error_metric, error_metric, iteration)
    writer.add_scalar(info+'loss', obj, iteration)
    writer.add_scalar(info+_main_obj, main_obj, iteration)
    writer.add_scalar(info+'kl', kl, iteration)
    writer.add_scalar(info+'ece', ece, iteration)
    writer.add_scalar(info+'entropy', entropy, iteration)
    writer.add_scalar(info+'nll', nll, iteration)

    
  def _get_average_meters(self):
    error_metric = utils.AverageMeter()
    obj = utils.AverageMeter()
    main_obj = utils.AverageMeter()
    nll = utils.AverageMeter()
    kl = utils.AverageMeter()
    ece = utils.AverageMeter()
    entropy = utils.AverageMeter()
    return error_metric, obj, main_obj, nll, kl, ece, entropy
    
  def train_loop(self, train_loader, valid_loader, writer=None, special_info=""):
    best_error = float('inf')
    train_error_metric = train_obj = train_main_obj = train_nll = train_ece = train_kl =  train_entropy = None
    val_error_metric = val_obj = val_main_obj = val_nll = val_ece = val_kl =  val_entropy = None

    for epoch in range(self.args.epochs):
      if epoch >= 1 and self.scheduler is not None:
        self.scheduler.step()
      
      if self.scheduler is not None:
        lr = self.scheduler.get_last_lr()[0]
      else:
        lr = self.args.learning_rate

      if writer is not None:
        writer.add_scalar('Train/learning_rate', lr, epoch)

      logging.info(
          '### Epoch: [%d/%d], Learning rate: %e ###', self.args.epochs,
          epoch, lr)
      if hasattr(self.args, 'gamma'):
            logging.info(
            '### Epoch: [%d/%d], Gamma: %e ###', self.args.epochs,
            epoch, self.args.gamma)
            
   
      train_obj, train_main_obj, train_nll, train_kl, train_error_metric, train_ece, train_entropy = self.train(train_loader, self.optimizer, writer)
      
      logging.info('#### Train | Error: %f, Train loss: %f, Train main objective: %f, Train NLL: %f, Train KL: %f, Train ECE %f, Train entropy %f ####',
                     train_error_metric, train_obj, train_main_obj, train_nll, train_kl, train_ece, train_entropy)

      
      if writer is not None:
        self._scalar_logging(train_obj, train_main_obj, train_nll, train_kl, train_error_metric, train_ece, train_entropy, "Train/", epoch, writer)
    
      # validation
      if valid_loader is not None:
        val_obj, val_main_obj, val_nll, val_kl, val_error_metric, val_ece, val_entropy = self.infer(
                                                          valid_loader, writer, "Valid")
        logging.info('#### Valid | Error: %f, Valid loss: %f, Valid main objective: %f, Valid NLL: %f, Valid KL: %f, Valid ECE %f, Valid entropy %f ####',
                      val_error_metric, val_obj, val_main_obj, val_nll, val_kl, val_ece, val_entropy)
        
        if writer is not None:
          self._scalar_logging(val_obj, val_main_obj, val_nll, val_kl, val_error_metric, val_ece, val_entropy, "Valid/", epoch, writer)
      
      if self.args.save_last or val_error_metric <= best_error:
        # Avoid correlation between the samples
        _special_info = None
        if hasattr(self.args, 'burnin_epochs') and epoch>=self.args.burnin_epochs and epoch%2==0 and epoch>=self.args.epochs-self.args.samples*2:
          _special_info= special_info+"_"+str(epoch)
        if _special_info is None:
          _special_info = special_info
        utils.save_model(self.model, self.args, _special_info)
        best_error = val_error_metric
        logging.info(
            '### Epoch: [%d/%d], Saving model! Current best error: %f ###', self.args.epochs,
            epoch, best_error)
      self.epoch+=1
    return best_error, self.train_time, self.val_time
  
  def _step(self, input, target, optimizer, n_batches, n_points, train_timer):
    start = time.time()
    if next(self.model.parameters()).is_cuda:
      input = input.cuda()
      target = target.cuda()
      
    if optimizer is not None:
      optimizer.zero_grad()
    output = self.model(input)
    if hasattr(self.model, 'get_kl_divergence'):
      kl = self.model.get_kl_divergence()
    else:
      kl = torch.tensor([0.0]).view(1).to(input.device)
    obj, main_obj, kl = self.criterion(
        output, target, kl, self.args.gamma if hasattr(self.args, 'gamma') else 0., n_batches, n_points)
    
    if optimizer is not None and obj == obj:
      obj.backward()
      for p in self.model.parameters():
        if p.grad is not None:
          p.grad[p.grad != p.grad] = 0
      if isinstance(optimizer, SGLD):
        if len(self.grad_buff) > 1000:
            self.max_grad = np.mean(self.grad_buff) + \
                self.grad_std_mul * np.std(self.grad_buff)
            self.grad_buff.pop(0)
        # Clipping to prevent explosions
        self.grad_buff.append(torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                                       max_norm=self.max_grad, norm_type=2).item())
        if self.grad_buff[-1] >= self.max_grad:
            self.grad_buff.pop()

        optimizer.step(burn_in=(self.epoch < self.args.burnin_epochs),
                       resample_momentum=(self.iteration % self.args.resample_momentum_iterations == 0),
                       resample_prior=(self.iteration % self.args.resample_prior_iterations == 0))
      else:
        optimizer.step()
      self.iteration+=1
      
    error_metric, ece, entropy, nll, _ = utils.evaluate(output, input, target, self.model, self.args)

    if train_timer:
      self.train_time += time.time() - start
    else:
      self.val_time += time.time() - start
      
    return error_metric, obj.item(), main_obj.item(), nll, kl.item(), ece, entropy


  def train(self, loader, optimizer, writer):
    error_metric, obj, main_obj, nll, kl, ece, entropy = self._get_average_meters()
    self.model.train()

    for step, (input, target) in enumerate(loader):
      n = input.shape[0]
      _error_metric, _obj, _main_obj, _nll, _kl, _ece, _entropy= self._step(input, target, optimizer, len(loader), len(loader.dataset), True)
      
      obj.update(_obj, n)
      main_obj.update(_main_obj, n)
      nll.update(_nll, n)
      kl.update(_kl, n)
      error_metric.update(_error_metric, n)
      ece.update(_ece, n)
      entropy.update(_entropy, n)

      if step % self.args.report_freq == 0:
        logging.info('##### Train step: [%03d/%03d] | Error: %f, Loss: %f, Main objective: %f, NLL: %f, KL: %f, ECE: %f, Entropy: %f #####',
                       len(loader),  step, error_metric.avg, obj.avg, main_obj.avg, nll.avg, kl.avg, ece.avg, entropy.avg)
        self._scalar_logging(obj.avg, main_obj.avg, nll.avg, kl.avg, error_metric.avg, ece.avg, entropy.avg, 'Train/Iteration/', self.train_step, writer)
        self.train_step += 1
      if self.args.debug:
        break

    return obj.avg, main_obj.avg, nll.avg, kl.avg, error_metric.avg, ece.avg, entropy.avg

  def infer(self, loader, writer, dataset="Valid"):
    with torch.no_grad():
      error_metric, obj, main_obj, nll, kl, ece, entropy = self._get_average_meters()
      self.model.eval()

      for step, (input, target) in enumerate(loader):
        n = input.shape[0]
        _error_metric, _obj, _main_obj, _nll, _kl, _ece, _entropy = self._step(
             input, target, None, len(loader), n * len(loader), False)

        obj.update(_obj, n)
        main_obj.update(_main_obj, n)
        nll.update(_nll, n)
        kl.update(_kl, n)
        error_metric.update(_error_metric, n)
        ece.update(_ece, n)
        entropy.update(_entropy, n)
        
        if step % self.args.report_freq == 0:
          logging.info('##### {} step: [{}/{}] | Error: {}, Loss: {}, Main objective: {}, NLL: {}, KL: {}, ECE: {}, Entropy: {} #####'.format(
                       dataset, len(loader), step, error_metric.avg, obj.avg, main_obj.avg, nll.avg, kl.avg, ece.avg, entropy.avg))
          self._scalar_logging(obj.avg, main_obj.avg, nll.avg, kl.avg, error_metric.avg, ece.avg, entropy.avg, '{}/Iteration/'.format(dataset), self.val_step, writer)
          self.val_step += 1

        if self.args.debug:
          break
          

      return obj.avg, main_obj.avg, nll.avg, kl.avg, error_metric.avg, ece.avg, entropy.avg
