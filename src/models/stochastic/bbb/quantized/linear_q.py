import torch
import copy
import torch.nn.functional as F

from torch.nn.quantized.modules.utils import _quantize_weight
from torch.nn.quantized.modules.linear import Linear as _Linear
from torch.nn.quantized import QFunctional

from src.models.stochastic.bbb.linear import Linear as LinearBBB
from src.models.stochastic.bbb.linear import LinearReLU as LinearReLUBBB
from src.models.stochastic.bbb.quantized import NOISE_SCALE, NOISE_ZERO_POINT
from src.utils import clamp_weight
class Linear(_Linear):
    _version = 1
    _FLOAT_MODULE = LinearBBB

    def __init__(self, in_features, out_features, bias_=False, args=None):
        super(_Linear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        if bias_:
            self.bias_ = torch.zeros(out_features)
        else:
            self.bias_ = None

        self.weight = torch._empty_affine_quantized(
            [out_features, in_features], scale=1, zero_point=0, dtype=torch.qint8)
        self.std = torch._empty_affine_quantized(
            [out_features, in_features], scale=1, zero_point=0, dtype=torch.qint8)

        self.scale = 1.0
        self.zero_point = 0

        self.add_weight = torch.nn.quantized.QFunctional()
        self.mul_noise = torch.nn.quantized.QFunctional()
        self.std_prior = torch.nn.Parameter(torch.ones((1,)), requires_grad=False)
        self.args = args

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(_Linear, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = torch.tensor(self.scale).clone().detach()
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point).clone().detach()
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

        super(_Linear, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                  missing_keys, unexpected_keys, error_msgs)

    def _get_name(self):
        return 'QuantizedLinear'

    def extra_repr(self):
        return 'in_features={}, out_features={}, scale={}, zero_point={}, qscheme={}, bias={}'.format(
            self.in_features, self.out_features, self.scale, self.zero_point, self.weight.qscheme(), self.bias() is not None
        )

    def __repr__(self):
        return super(_Linear, self).__repr__()

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
        return torch.nn.quantized.functional.linear(
            x, weight, bias, self.scale, self.zero_point)

    def weight(self):
        return self.weight

    def std(self):
        return self.std

    def bias(self):
        return self.bias_

    @classmethod
    def from_float(cls, mod):
        if hasattr(mod, 'weight_fake_quant'):
            weight_post_process = mod.weight_fake_quant
            std_post_process = mod.std_fake_quant
            activation_post_process = mod.activation_post_process
        else:
            assert type(mod) == cls._FLOAT_MODULE, ' nnq.' + cls.__name__ + '.from_float only works for ' + \
                cls._FLOAT_MODULE.__name__
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            if type(mod) == LinearReLUBBB:
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
        qlinear = cls(mod.in_features, mod.out_features, mod.bias is not None)
        qlinear.weight = weight
        qlinear.std = std
        qlinear.std_prior = mod.std_prior
        qlinear.bias_ = mod.bias
        qlinear.scale = float(act_scale)
        qlinear.zero_point = int(act_zp)
        qlinear.add_weight = add_weight
        qlinear.mul_noise = mul_noise
        qlinear.args = mod.args
        return qlinear

class LinearReLU(Linear):
   
    _FLOAT_MODULE = LinearReLUBBB

    def __init__(self, in_features, out_features, bias=False, args=None):
        super(LinearReLU, self).__init__(in_features, out_features, bias, args)

    def forward(self, input):
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
        _packed_params = torch.ops.quantized.linear_prepack(weight, bias)
        Y_q = torch.ops.quantized.linear_relu(
            input, _packed_params,
            float(self.scale),
            int(self.zero_point))
        return Y_q

    def _get_name(self):
        return 'QuantizedLinearReLU'

    @classmethod
    def from_float(cls, mod):
        mod = super(LinearReLU, cls).from_float(mod)
        if not isinstance(mod.add_weight, QFunctional):
            mod.add_weight = QFunctional.from_float(mod.add_weight)
        if not isinstance(mod.mul_noise, QFunctional):
            mod.mul_noise = QFunctional.from_float(mod.mul_noise)
        return mod
