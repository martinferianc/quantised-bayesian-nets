import torch
import copy 

import torch.nn.functional as F

from torch.nn.quantized.modules.utils import _quantize_weight
from torch.nn.quantized.modules import Conv2d as _Conv2d
from torch.nn.quantized.modules.conv import _ConvNd

from torch.nn.quantized import QFunctional
from torch.nn.modules.utils import _pair

from src.models.stochastic.bbb.conv import Conv2d as Conv2dBBB
from src.models.stochastic.bbb.conv import fuse_conv_bn_weights
from src.models.stochastic.bbb.conv import ConvReLU2d as ConvReLU2dBBB
from src.models.stochastic.bbb.quantized.conv_qat import ConvBnReLU2d as ConvBnReLU2dBBB_QAT
from src.models.stochastic.bbb.quantized.conv_qat import ConvBn2d as ConvBn2dBBB_QAT

from src.models.stochastic.bbb.quantized import NOISE_SCALE, NOISE_ZERO_POINT
from src.utils import clamp_weight

class Conv2d(_Conv2d):
    _version = 1
    _FLOAT_MODULE = Conv2dBBB
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=False,
                padding_mode='zeros',args= None):
        super(_ConvNd, self).__init__()
        if padding_mode != 'zeros':
            raise NotImplementedError(
                "Currently only zero-padding is supported by quantized conv")
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.transposed = False
        self.output_padding = _pair(0)

        self.bias_ = torch.zeros(out_channels, dtype=torch.float) if bias else None

        # Initialize as NCHW. set_weight will internally transpose to NHWC.
        self.weight = torch._empty_affine_quantized(
            [out_channels, in_channels // self.groups] + list(kernel_size),
            scale=1, zero_point=0, dtype=torch.qint8)
        self.std = torch._empty_affine_quantized(
            [out_channels, in_channels // self.groups] + list(kernel_size),
            scale=1, zero_point=0, dtype=torch.qint8)

        self.std_prior = torch.nn.Parameter(torch.ones((1,)), requires_grad=False)
        
        self.scale = 1.0
        self.zero_point = 0

        self.add_weight = torch.nn.quantized.QFunctional()
        self.mul_noise = torch.nn.quantized.QFunctional()
        self.args= args

    def _get_name(self):
        return 'QuantizedConv2d'

    def bias(self):
        return self.bias_

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(_ConvNd, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)
        destination[prefix + 'weight'] = self.weight
        destination[prefix + 'std'] = self.std
        destination[prefix + 'bias_'] = self.bias_

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        self.scale = float(state_dict[prefix + 'scale'])
        state_dict.pop(prefix + 'scale')

        self.zero_point = int(state_dict[prefix + 'zero_point'])
        state_dict.pop(prefix + 'zero_point')

        self.weight = state_dict[prefix + 'weight']
        state_dict.pop(prefix + 'weight')

        self.std = state_dict[prefix + 'std']
        state_dict.pop(prefix + 'std')

        self.bias_ = state_dict[prefix + 'bias_']
        state_dict.pop(prefix + 'bias_')

        super(_ConvNd, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                  missing_keys, unexpected_keys, error_msgs)

    def extra_repr(self):
        return super(_Conv2d, self).extra_repr()

    def __repr__(self):
        return super(_ConvNd, self).__repr__()

    def forward(self, x):
        weight, bias, std = self.weight, self.bias(), self.std
        scale = NOISE_SCALE
        zero_point = NOISE_ZERO_POINT
        qscheme = weight.qscheme()

        noise = torch.FloatTensor(std.shape).normal_(0,1)
        if (qscheme == torch.per_tensor_affine) or (qscheme == torch.per_tensor_symmetric):
            noise = torch.quantize_per_tensor(noise, scale, zero_point, dtype=torch.qint8)
        else:
            raise RuntimeError('Unsupported qscheme specified for quantized Linear noise quantization!')
        weight = self.add_weight.add(weight, self.mul_noise.mul(std, noise))
        weight = clamp_weight(weight, self.args)
        return torch.nn.quantized.functional.conv2d(
            x, weight, bias, stride=self.stride, 
            padding=self.padding, dilation=self.dilation, 
            groups=self.groups, padding_mode=self.padding_mode, 
            scale=self.scale, zero_point=self.zero_point, 
            dtype=torch.quint8)

    @classmethod
    def from_float(cls, mod):
        if hasattr(mod, 'weight_fake_quant'):
            if type(mod) == ConvBn2dBBB_QAT:
                mod.weight, mod.bias, mod.std = fuse_conv_bn_weights(
                    mod.weight, mod.bias, mod.std, mod.bn.running_mean, mod.bn.running_var,
                    mod.bn.eps, mod.bn.weight, mod.bn.bias)
                assert hasattr(mod, 'activation_post_process'), \
                        'Input QAT module must have observer attached'
            weight_post_process = mod.weight_fake_quant
            std_post_process = mod.std_fake_quant
            activation_post_process = mod.activation_post_process
        else:
            assert type(mod) == cls._FLOAT_MODULE, ' nnq.' + cls.__name__ + '.from_float only works for ' + \
                cls._FLOAT_MODULE.__name__
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            if type(mod) == ConvReLU2dBBB:
                activation_post_process = mod[1].activation_post_process
                mod = mod[0]
            else:
                activation_post_process = mod.activation_post_process
            weight_post_process = mod.qconfig.weight()
            std_post_process = copy.deepcopy(mod.qconfig.weight())

        add_weight = mod.add_weight
        mul_noise = mod.mul_noise

        weight_post_process(mod.weight)
        std_post_process(F.softplus(mod.std))

        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()

        assert dtype == torch.qint8, 'Weight observer must have dtype torch.qint8'
        weight = _quantize_weight(mod.weight.float(), weight_post_process)
        std = _quantize_weight(F.softplus(mod.std.float()), std_post_process)

        qconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                    mod.stride, mod.padding, mod.dilation, mod.groups,
                    mod.bias is not None, mod.padding_mode)
        qconv.weight = weight
        qconv.bias_ = mod.bias
        qconv.std = std
        qconv.std_prior = mod.std_prior
        qconv.scale = float(act_scale)
        qconv.zero_point = int(act_zp)
        qconv.add_weight = add_weight
        qconv.mul_noise = mul_noise
        qconv.args = mod.args

        return qconv

class ConvReLU2d(Conv2d):
    _FLOAT_MODULE = ConvReLU2dBBB
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', args=None):
        super(ConvReLU2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, args=args)

    def forward(self, input):
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        
        weight, bias, std = self.weight, self.bias(), self.std
        scale = NOISE_SCALE
        zero_point = NOISE_ZERO_POINT
        qscheme = weight.qscheme()

        noise = torch.FloatTensor(std.shape).normal_(0,1)
        if (qscheme == torch.per_tensor_affine) or (qscheme == torch.per_tensor_symmetric):
            noise = torch.quantize_per_tensor(noise, scale, zero_point, dtype=torch.qint8)
        else:
            raise RuntimeError('Unsupported qscheme specified for quantized Linear noise quantization!')
        
        weight = self.add_weight.add(weight, self.mul_noise.mul(std, noise))
        weight = clamp_weight(weight, self.args)
        _packed_params = torch.ops.quantized.conv2d_prepack(weight, bias, self.stride, self.padding, self.dilation, self.groups)

        return torch.ops.quantized.conv2d_relu(
            input, _packed_params, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedConvReLU2d'

    @classmethod
    def from_float(cls, mod):
        if type(mod) == ConvBnReLU2dBBB_QAT:
            mod.weight, mod.bias, mod.std = fuse_conv_bn_weights(
                mod.weight, mod.bias, mod.std, mod.bn.running_mean, mod.bn.running_var,
                mod.bn.eps, mod.bn.weight, mod.bn.bias)
        mod = super(ConvReLU2d, cls).from_float(mod)
        if not isinstance(mod.add_weight, QFunctional):
            mod.add_weight = QFunctional.from_float(mod.add_weight)
        if not isinstance(mod.mul_noise, QFunctional):
            mod.mul_noise = QFunctional.from_float(mod.mul_noise)
        return mod