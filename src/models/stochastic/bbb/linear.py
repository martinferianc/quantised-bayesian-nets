import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import ReLU
from torch.autograd import Variable
from src.models.stochastic.bbb.utils_bbb import kl_divergence

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias, sigma_prior=1.0, args = None):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.std_prior = torch.nn.Parameter(
            torch.ones((1,))*sigma_prior, requires_grad=False)

        self.weight.data.uniform_(-0.01, 0.01)
        self.std = nn.Parameter(torch.zeros_like(self.weight).uniform_(-3, -3))

        self.add_weight = torch.nn.quantized.FloatFunctional()
        self.mul_noise = torch.nn.quantized.FloatFunctional()
        self.args = args

        if self.bias is not None:
          self.bias.data.uniform_(-0.01, 0.01)

    def get_kl_divergence(self):
      return kl_divergence(self.weight, F.softplus(self.std),
                           torch.zeros_like(self.weight).to(
                               self.weight.device),
                           (torch.ones_like(self.std)*self.std_prior).to(self.weight.device))

    def forward(self, x):
      output = None
      if self.training:
          mean = torch.mm(x, self.weight.t())
          std = torch.sqrt(1e-8+torch.mm(torch.pow(x, 2),
                                          torch.pow(F.softplus(self.std).t(), 2)))
          noise = Variable(mean.new(
              mean.size()).normal_())

          bias = self.bias if self.bias is not None else 0.0
          output = mean + std * noise + bias

      else:
          std = F.softplus(self.std)
          noise = Variable(self.weight.data.new(
              self.weight.size()).normal_())
          std = self.mul_noise.mul(noise, std)
          weight_sample = self.add_weight.add(self.weight, std)
          bias = self.bias if self.bias is not None else 0.0

          output = torch.mm(x, weight_sample.t()) + bias

      return output
      
class LinearReLU(torch.nn.Sequential):
    def __init__(self, linear, relu):
      assert type(linear) == Linear and type(relu) == ReLU, \
          'Incorrect types for input modules{}{}'.format(
              type(linear), type(relu))
      super(LinearReLU, self).__init__(linear, relu)
