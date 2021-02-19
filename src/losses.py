import torch.nn.functional as F
import torch
import torch.nn as nn

LOSS_FACTORY = {'classification': lambda args, scaling: ClassificationLoss(args, scaling),
                'regression': lambda args, scaling: RegressionLoss(args, scaling)}

class Loss(nn.Module):
    def __init__(self, args, scaling):
        super(Loss, self).__init__()
        self.args = args
        self.scaling = scaling 

class ClassificationLoss(Loss):
  def __init__(self, args, scaling):
      super(ClassificationLoss, self).__init__(args, scaling)
      self.ce = F.nll_loss
  def forward(self, output, target, kl, gamma, n_batches, n_points):
    if self.scaling=='whole':
      ce = n_points*self.ce(torch.log(output+1e-8), target) * self.args.loss_multiplier
      kl = kl / n_batches
    elif self.scaling=='batch':
      ce = self.ce(torch.log(output+1e-8), target)
      kl = kl / (target.shape[0]*n_batches)
    else:
      raise NotImplementedError('Other scaling not implemented!')
    loss = ce + gamma * kl

    return loss, ce, kl

class RegressionLoss(Loss):
  def __init__(self, args, scaling):
    super(RegressionLoss, self).__init__(args, scaling)

  def forward(self, output, target, kl, gamma, n_batches, n_points):
    mean = output[0]
    var = output[1]
    precision = 1/(var+1e-8)
    if self.scaling == 'whole':
      heteroscedastic_loss = n_points * \
          torch.mean(torch.sum(precision * (target - mean)**2 +
                               torch.log(var+1e-8), 1), 0) * self.args.loss_multiplier
      kl = kl / n_batches
    elif self.scaling == 'batch':
      heteroscedastic_loss = torch.mean(
          torch.sum(precision * (target - mean)**2 + torch.log(var+1e-8), 1), 0)
      kl = kl / (target.shape[0]*n_batches)
    else:
      raise NotImplementedError('Other scaling not implemented!')
    loss = heteroscedastic_loss + gamma*kl
    return loss, heteroscedastic_loss, kl

