
import torch
import src.utils as utils
import time 
import logging 
from src.models.stochastic.sgld.utils_sgld import SGLD
import numpy as np
from src.metrics import ClassificationMetric, RegressionMetric

class Trainer():
  def __init__(self, model, criterion, optimizer, scheduler, args, writer=None):
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
    
    self.train_metrics = ClassificationMetric(output_size=self.args.output_size, writer=writer) if "classification" in self.args.task else RegressionMetric(output_size=self.args.output_size, writer=writer)
    self.valid_metrics = ClassificationMetric(output_size=self.args.output_size, writer=writer) if "classification" in self.args.task else RegressionMetric(output_size=self.args.output_size, writer=writer)
    self.writer = writer

  def train_loop(self, train_loader, valid_loader, special_info=""):
    best_error = float('inf')

    for epoch in range(self.args.epochs):
      if epoch >= 1 and self.scheduler is not None:
        self.scheduler.step()
      
      if self.scheduler is not None:
        lr = self.scheduler.get_last_lr()[0]
      else:
        lr = self.args.learning_rate

      if self.writer is not None:
        self.writer.add_scalar('Train/learning_rate', lr, epoch)

      logging.info(
          '### Epoch: [%d/%d], Learning rate: %e ###', self.args.epochs,
          epoch, lr)
      if hasattr(self.args, 'gamma'):
            logging.info(
            '### Epoch: [%d/%d], Gamma: %e ###', self.args.epochs,
            epoch, self.args.gamma)
            
      self.train_metrics.reset()  
      self.train(train_loader, self.optimizer)
      
      logging.info("#### Train | %s ####", self.train_metrics.get_str())
            
      self.train_metrics.scalar_logging("train", epoch)
    
      # validation
      if valid_loader is not None:
        self.valid_metrics.reset()
        self.infer(valid_loader, "Valid")
        logging.info("#### Valid | %s ####", self.valid_metrics.get_str())
        val_error_metric = self.valid_metrics.get_key_metric()
      
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
      

    if train_timer:
      self.train_metrics.update(output=output, target=target, obj=obj, kl=kl, main_obj=main_obj)
      self.train_time += time.time() - start
    else:
      self.valid_metrics.update(output=output, target=target, obj=obj, kl=kl, main_obj=main_obj)
      self.val_time += time.time() - start
      


  def train(self, loader, optimizer):
    self.model.train()

    for step, (input, target) in enumerate(loader):
      self._step(input, target, optimizer, len(loader), len(loader.dataset), True)


      if step % self.args.report_freq == 0:
        logging.info(
                    "##### Train step: [%03d/%03d] | %s #####",
                    len(loader),
                    step,
                    self.train_metrics.get_str(),
                )
        self.train_step += 1
      if self.args.debug:
        break

  def infer(self, loader, dataset="Valid"):
    with torch.no_grad():
      self.model.eval()

      for step, (input, target) in enumerate(loader):
        n = input.shape[0]
        self._step(
             input, target, None, len(loader), n * len(loader), False)

        if step % self.args.report_freq == 0:
          logging.info(
                      "##### %s step: [%03d/%03d] | %s #####",
                      dataset,
                      len(loader),
                      step,
                      self.valid_metrics.get_str(),
                  )
          self.val_step += 1

        if self.args.debug:
          break