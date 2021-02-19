import torch
import torch.nn.functional as F
from torch.autograd import Variable

from src.models.stochastic.bbb.linear import Linear as LinearBBB
from src.models.stochastic.bbb.linear import LinearReLU as LinearReLUBBB

class Linear(LinearBBB):
    _FLOAT_MODULE = LinearBBB
    def __init__(self, in_features, out_features, bias=False, qconfig=None, args=None):
        super(Linear, self).__init__(in_features, out_features, bias, args=args)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight()  
        self.activation_post_process = qconfig.activation()
        self.std_fake_quant = qconfig.weight()

    def _forward(self, X):
        output = None
        weight = self.weight_fake_quant(self.weight)
        std = self.std_fake_quant(F.softplus(self.std))
        if self.training: 
            mean = torch.mm(X, weight.t())
            std = torch.sqrt(1e-8+torch.mm(torch.pow(X,2), torch.pow(std.t(), 2))) 
            noise = Variable(mean.new(
                mean.size()).normal_())
            
            bias = self.bias if self.bias is not None else 0.0
            output = mean + std * noise + bias
        else:
            noise = Variable(weight.new(
                weight.size()).normal_())

            std = self.mul_noise.mul(noise, std)
            weight_sample = self.add_weight.add(weight, std)
            bias = self.bias if self.bias is not None else 0.0
            output = torch.mm(X, weight_sample.t()) + bias
        return output

    def forward(self, X):
        return self.activation_post_process(self._forward(X))

    def _get_name(self):
        return 'QATLinear'

    @classmethod
    def from_float(cls, mod, qconfig=None):
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'
        if type(mod) == LinearReLUBBB:
            mod = mod[0]
            
        activation_post_process = mod.activation_post_process

        qconfig = mod.qconfig
        qat_linear = cls(mod.in_features, mod.out_features, mod.bias is not None, qconfig)
        qat_linear.activation_post_process = activation_post_process
        qat_linear.std_prior = mod.std_prior
        qat_linear.weight = mod.weight
        qat_linear.std = mod.std
        qat_linear.add_weight = mod.add_weight
        qat_linear.add_weight.activation_post_process = qconfig.weight()
        qat_linear.mul_noise = mod.mul_noise
        qat_linear.mul_noise.activation_post_process = qconfig.weight()
        qat_linear.bias = mod.bias
        qat_linear.args = mod.args
        return qat_linear

class LinearReLU(Linear):
    _FLOAT_MODULE = LinearReLUBBB
    def __init__(self, in_features, out_features, bias=False,
                 qconfig=None, args=None):
        super(LinearReLU, self).__init__(in_features, out_features, bias = bias, qconfig=qconfig, args=args)

    def forward(self, input):
        return self.activation_post_process(F.relu(self._forward(input)))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        return super(LinearReLU, cls).from_float(mod, qconfig)

    def _get_name(self):
        return 'QATLinearReLU'
