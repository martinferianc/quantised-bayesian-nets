import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import ReLU
from torch.autograd import Variable
from src.models.stochastic.bbb.utils_bbb import kl_divergence, softplusinv
import copy 

class Conv2d(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
               padding=0, dilation=1, groups=1,
               bias=False, padding_mode='zeros', sigma_prior=-2, args=None):

    super(Conv2d, self).__init__(in_channels, out_channels,kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    self.weight.data.uniform_(-0.01, 0.01)
    self.std = torch.nn.Parameter(
            torch.zeros_like(self.weight).uniform_(-10, -10), requires_grad=True)
    self.std_prior = torch.nn.Parameter(torch.tensor((1,))*sigma_prior, requires_grad=False)
    self.add_weight = torch.nn.quantized.FloatFunctional()
    self.mul_noise = torch.nn.quantized.FloatFunctional()
    self.args = args

  def forward(self, X):
    if self.training:
      Z_mean = F.conv2d(X, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
      Z_std = torch.sqrt(1e-8+F.conv2d(torch.pow(X, 2), torch.pow(F.softplus(self.std), 2),
                          None, self.stride, self.padding, self.dilation, self.groups))
      Z_noise = Variable(Z_mean.new(
          Z_mean.size()).normal_())

      Z = Z_mean + Z_std * Z_noise 
      Z = Z + self.bias if self.bias is not None else Z
    else:
      noise = Variable(self.weight.data.new(
          self.weight.data.size()).normal_())

      std = self.mul_noise.mul(noise, F.softplus(self.std))
      weight = self.add_weight.add(self.weight, std)
      Z = F.conv2d(X, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    return Z


  def get_kl_divergence(self):
    kl = kl_divergence(self.weight, F.softplus(self.std),
                       torch.zeros_like(self.weight).to(self.weight.device), 
                       (torch.ones_like(self.std)*self.std_prior).to(self.weight.device))
    return kl

class ConvBn2d(torch.nn.Sequential):
    def __init__(self, conv, bn):
        assert type(conv) == Conv2d and type(bn) == torch.nn.BatchNorm2d, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(bn))
        super(ConvBn2d, self).__init__(conv, bn)

class ConvReLU2d(torch.nn.Sequential):
    def __init__(self, conv, relu):
        assert type(conv) == Conv2d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(relu))
        super(ConvReLU2d, self).__init__(conv, relu)

class ConvBnReLU2d(torch.nn.Sequential):
    def __init__(self, conv, bn, relu):
        assert type(conv) == Conv2d and type(bn) == torch.nn.BatchNorm2d and \
            type(relu) == ReLU, 'Incorrect types for input modules{}{}{}' \
            .format(type(conv), type(bn), type(relu))
        super(ConvBnReLU2d, self).__init__(conv, bn, relu)

def fuse_conv_bn_weights(conv_w, conv_b, conv_std, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = bn_rm.new_zeros(bn_rm.shape)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    c = (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_w = conv_w * c
    conv_std = softplusinv(F.softplus(conv_std) * c)
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b), torch.nn.Parameter(conv_std)

def fuse_conv_bn_eval(conv, bn):
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias, fused_conv.std= \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias, fused_conv.std,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv

def fuse_conv_bn(conv, bn):
    assert(conv.training == bn.training),\
        "Conv and BN both must be in the same mode (train or eval)."

    if conv.training:
      assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
      assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
      assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
      return ConvBn2d(conv, bn)
    else:
      return fuse_conv_bn_eval(conv, bn)
    
def fuse_conv_bn_relu(conv, bn, relu):
    assert(conv.training == bn.training == relu.training),\
        "Conv and BN both must be in the same mode (train or eval)."

    if conv.training:
      map_to_fused_module_train = {
            Conv2d: ConvBnReLU2d
      }
      assert bn.num_features == conv.out_channels, 'Output channel of Conv must match num_features of BatchNorm'
      assert bn.affine, 'Only support fusing BatchNorm with affine set to True'
      assert bn.track_running_stats, 'Only support fusing BatchNorm with tracking_running_stats set to True'
      fused_module = map_to_fused_module_train.get(type(conv))
      if fused_module is not None:
          return fused_module(conv, bn, relu)
      else:
          raise NotImplementedError("Cannot fuse train modules: {}".format((conv, bn, relu)))

    else:
      map_to_fused_module_eval = {
          Conv2d: ConvReLU2d,
      }
      fused_module = map_to_fused_module_eval[type(conv)]
      if fused_module is not None:
          return fused_module(fuse_conv_bn_eval(conv, bn), relu)
      else:
          raise NotImplementedError("Cannot fuse eval modules: {}".format((conv, bn, relu)))

