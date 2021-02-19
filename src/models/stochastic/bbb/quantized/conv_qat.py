import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from src.models.stochastic.bbb.conv import Conv2d as Conv2dBBB
from src.models.stochastic.bbb.conv import ConvBn2d as ConvBn2dBBB
from src.models.stochastic.bbb.conv import ConvBnReLU2d as ConvBnReLU2dBBB
from src.models.stochastic.bbb.conv import ConvReLU2d as ConvReLU2dBBB


class Conv2d(Conv2dBBB):
    _FLOAT_MODULE = Conv2dBBB

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', qconfig=None, args=None):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                padding, dilation, groups, bias, padding_mode, args=args)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight()  
        self.activation_post_process = qconfig.activation()
        self.std_fake_quant = qconfig.weight()


    def _forward(self, X):
        weight = self.weight_fake_quant(self.weight)
        std = self.std_fake_quant(F.softplus(self.std))
        if self.training: 
            Z_mean = F.conv2d(X, weight, None, self.stride, self.padding, self.dilation, self.groups)
            Z_std = torch.sqrt(1e-8+F.conv2d(torch.pow(X, 2), torch.pow(std, 2),
                                None, self.stride, self.padding, self.dilation, self.groups))
            if X.is_cuda:
                Z_noise = torch.cuda.FloatTensor(Z_mean.shape).normal_(0,1)
            else:
                Z_noise = torch.FloatTensor(Z_mean.shape).normal_(0,1)

            Z = Z_mean + Z_std * Z_noise 
            Z = Z + self.bias if self.bias is not None else Z
        else:
            if X.is_cuda:
                noise = torch.cuda.FloatTensor(self.std.shape).normal_(0,1)
            else:
                noise = torch.FloatTensor(self.std.shape).normal_(0,1)

            std = self.mul_noise.mul(noise, std)
            weight = self.add_weight.add(weight, std)
            Z = F.conv2d(X, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return Z

    def forward(self, input):
        return self.activation_post_process(self._forward(input)) 

    def _get_name(self):
        return 'QATConv2d'

    @classmethod
    def from_float(cls, mod, qconfig=None):
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'
        if isinstance(mod, ConvReLU2dBBB):
            mod = mod[0]

        qconfig = mod.qconfig
        qat_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size, mod.stride,
                mod.padding, mod.dilation, mod.groups, mod.bias is not None, mod.padding_mode, qconfig)
        qat_conv.activation_post_process = mod.activation_post_process
        qat_conv.weight = mod.weight
        qat_conv.std = mod.std
        qat_conv.std_prior = mod.std_prior
        qat_conv.add_weight = mod.add_weight
        qat_conv.add_weight.activation_post_process = qconfig.weight()
        qat_conv.mul_noise = mod.mul_noise
        qat_conv.mul_noise.activation_post_process = qconfig.weight()
        qat_conv.bias = mod.bias
        qat_conv.args = mod.args
        return qat_conv

class ConvBn2d(Conv2d):
    _version = 1
    _FLOAT_MODULE = ConvBn2dBBB

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=False,  padding_mode='zeros', eps=1e-05, momentum=0.1, freeze_bn=False, qconfig=None, args=None):
        super(ConvBn2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                padding, dilation, groups, bias, padding_mode, qconfig, args)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = nn.BatchNorm2d(out_channels, eps, momentum, True, True)
        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        torch.nn.init.uniform_(self.bn.weight)
        torch.nn.init.zeros_(self.bn.bias)
        # note: below is actully for conv, not BN
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super(ConvBn2d, self).reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    def _forward(self, X):
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight = self.weight_fake_quant(self.weight * scale_factor.reshape([-1, 1, 1, 1]))
        std = self.std_fake_quant(F.softplus(self.std) * scale_factor.reshape([-1, 1, 1, 1]))
        if self.training: 
            Z_mean = F.conv2d(X, weight, None, self.stride, self.padding, self.dilation, self.groups)
            Z_std = torch.sqrt(1e-8+F.conv2d(torch.pow(X, 2), torch.pow(std, 2),
                                None, self.stride, self.padding, self.dilation, self.groups))
            if X.is_cuda:
                Z_noise = torch.cuda.FloatTensor(Z_mean.shape).normal_(0,1)
            else:
                Z_noise = torch.FloatTensor(Z_mean.shape).normal_(0,1)

            Z = Z_mean + Z_std * Z_noise 
        else:
            if X.is_cuda:
                noise = torch.cuda.FloatTensor(self.std.shape).normal_(0,1)
            else:
                noise = torch.FloatTensor(self.std.shape).normal_(0,1)

            std = self.mul_noise.mul(noise, std)
            weight = self.add_weight.add(weight, std)
            Z = F.conv2d(X, weight, None, self.stride, self.padding, self.dilation, self.groups)
        Z_orig = Z / scale_factor.reshape([1, -1, 1, 1])
        if self.bias is not None:
            Z_orig = Z_orig + self.bias.reshape([1, -1, 1, 1])
        Z = self.bn(Z_orig)
        return Z

    def forward(self, input):
        return self.activation_post_process(self._forward(input))

    def extra_repr(self):
        return super(ConvBn2d, self).extra_repr()

    def _get_name(self):
        return 'QATConvBn2d'

    @classmethod
    def from_float(cls, mod, qconfig=None):
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'
            qconfig = mod.qconfig
        conv, bn = mod[0], mod[1]
        qat_convbn = cls(conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation,
                         conv.groups, conv.bias is not None,
                         conv.padding_mode,
                         bn.eps, bn.momentum,
                         False,
                         qconfig)
        qat_convbn.activation_post_process = conv.activation_post_process
        qat_convbn.mul_noise = conv.mul_noise
        qat_convbn.mul_noise.activation_post_process = conv.qconfig.weight()
        qat_convbn.add_weight = conv.add_weight
        qat_convbn.add_weight.activation_post_process = conv.qconfig.weight()
        qat_convbn.weight = conv.weight
        qat_convbn.std = conv.std
        qat_convbn.std_prior = conv.std_prior
        qat_convbn.bias = conv.bias
        qat_convbn.bn.weight = bn.weight
        qat_convbn.bn.bias = bn.bias
        qat_convbn.bn.running_mean = bn.running_mean
        qat_convbn.bn.running_var = bn.running_var
        qat_convbn.bn.num_batches_tracked = bn.num_batches_tracked
        qat_convbn.args = conv.args
        return qat_convbn

class ConvBnReLU2d(ConvBn2d):
    _FLOAT_MODULE = ConvBnReLU2dBBB
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 eps=1e-05, momentum=0.1,
                 freeze_bn=False,
                 qconfig=None,
                 args=None):
        super(ConvBnReLU2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias,
                                           padding_mode, eps, momentum,
                                           freeze_bn,
                                           qconfig,args=args)

    def forward(self, input):
        return self.activation_post_process(F.relu(self._forward(input)))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        return super(ConvBnReLU2d, cls).from_float(mod, qconfig)

    def _get_name(self):
        return 'QATConvBnReLU2d'


class ConvReLU2d(Conv2d):
    _FLOAT_MODULE = ConvReLU2dBBB
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False, padding_mode='zeros', qconfig=None, args=None):
        super(ConvReLU2d, self).__init__(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation,
                                         groups, bias, padding_mode,
                                         qconfig, args=args)

    def forward(self, input):
        return self.activation_post_process(F.relu(
            self._forward(input)))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        return super(ConvReLU2d, cls).from_float(mod, qconfig)

    def _get_name(self):
        return 'QATConvReLU2d'

