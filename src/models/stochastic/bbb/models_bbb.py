import torch.nn as nn
import torch.nn.functional as F
import torch 

from src.models.stochastic.bbb.linear import Linear, LinearReLU
from src.models.stochastic.bbb.conv import Conv2d, ConvReLU2d, fuse_conv_bn, fuse_conv_bn_relu
from src.utils import Flatten, Add, clamp_activation
from torch.quantization import QuantStub, DeQuantStub

def fuse_bbb_modules(mod_list):
    OP_LIST_TO_FUSER_METHOD = {
        (Linear, torch.nn.ReLU): LinearReLU,
        (Conv2d, torch.nn.ReLU): ConvReLU2d,
        (Conv2d, nn.BatchNorm2d): fuse_conv_bn,
        (Conv2d, nn.BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
    }

    types = tuple(type(m) for m in mod_list)
    fuser_method = OP_LIST_TO_FUSER_METHOD.get(types, None)
    if fuser_method is None:
        raise NotImplementedError("Cannot fuse modules: {}".format(types))
    new_mod = [None] * len(mod_list)
    new_mod[0] = fuser_method(*mod_list)

    for i in range(1, len(mod_list)):
        new_mod[i] = torch.nn.Identity()
        new_mod[i].training = mod_list[0].training

    return new_mod


class LinearNetwork(nn.Module):
  def __init__(self, input_size, output_size, q, args):
    super(LinearNetwork, self).__init__()
    self.args = args
    self.input_size = 1
    layers = [100, 100, 100]

    for i in input_size:
        self.input_size*=int(i) 
    self.output_size = int(output_size)

    self.layers = nn.ModuleList([])
    for i in range(len(layers)):
        if i == 0:
            self.layers.append(Linear(self.input_size, int(layers[0]),sigma_prior=args.sigma_prior,   bias=True, args=args))
        else:
            self.layers.append(Linear(int(layers[i-1]), int(layers[i]), sigma_prior=args.sigma_prior,   bias=True, args=args))
        self.layers.append(nn.ReLU())
    self.mu = Linear(int(layers[len(layers)-1]), 1, sigma_prior=args.sigma_prior,   bias=True, args=args)
    self.log_var = Linear(int(layers[len(layers)-1]), 1,
                     sigma_prior=args.sigma_prior,   bias=True, args=args)

    self.q = q 
    if self.q:
        self.quant = QuantStub()
        self.dequant_mu = DeQuantStub()
        self.dequant_log_var = DeQuantStub()


  def forward(self, x):
    if self.q:
        x = self.quant(x)
        x = clamp_activation(x, self.args)
    
    for layer in self.layers:
       x = layer(x)
       x = clamp_activation(x, self.args)

    mu = self.mu(x)
    mu = clamp_activation(mu, self.args)
    log_var = self.log_var(x)
    log_var = clamp_activation(log_var, self.args)

    if self.q:
        mu = self.dequant_mu(mu)
        log_var = self.dequant_log_var(log_var)
    return (mu, log_var.exp())

  def get_kl_divergence(self):
    kl = 0.0
    for m in self.modules():
        if hasattr(m, 'get_kl_divergence') and not isinstance(m, LinearNetwork):
            kl+=m.get_kl_divergence()
    return kl

  def fuse_model(self):
    fusion = []
    buf = []
    for i,m in enumerate(self.layers):
        if isinstance(m,Linear) or (isinstance(m, nn.ReLU) and isinstance(self.layers[i-1], Linear)):
            buf.append(str(i))
        if len(buf) == 2:
            fusion.append(buf)
            buf = []
    torch.quantization.fuse_modules(self.layers, fusion, fuser_func=fuse_bbb_modules, inplace=True)

class ConvNetwork_LeNet(nn.Module):
    def __init__(self, input_size, output_size, q, args):
        super(ConvNetwork_LeNet, self).__init__()
        self.args = args 

        self.init_channels = input_size[0]

        self.layers = nn.ModuleList([Conv2d(in_channels=self.init_channels, out_channels=20, kernel_size=(5,5), stride=1, padding=2, sigma_prior=args.sigma_prior,  bias=False, args=args),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     Conv2d(in_channels=20, out_channels=50, kernel_size=(5,5), stride=1, padding=2, sigma_prior=args.sigma_prior,  bias=False, args=args),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     Flatten(),
                                     Linear(in_features=50*7*7, out_features=500, sigma_prior=args.sigma_prior,  bias=False, args=args),
                                     nn.ReLU(),
                                     Linear(in_features=500, out_features=output_size, sigma_prior=args.sigma_prior,  bias=False, args=args)])
        
    
        self.q = q 
        if self.q:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        if self.q:
            x = self.quant(x)
            x = clamp_activation(x, self.args)

        for layer in self.layers:
            x = layer(x)
            x = clamp_activation(x, self.args)
            
        if self.q:
            x = self.dequant(x)
        x = F.softmax(x, dim=-1)

        return x

    def get_kl_divergence(self):
        kl = 0.0
        for m in self.modules():
            if hasattr(m, 'get_kl_divergence') and not isinstance(m, ConvNetwork_LeNet):
                kl+=m.get_kl_divergence()
        return kl

    def fuse_model(self):
        torch.quantization.fuse_modules(self.layers, ['5', '6'] ,fuser_func=fuse_bbb_modules, inplace=True)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, q= False, args=None):
        super(BasicBlock, self).__init__()
        self.args = args
        self.q = q 
        self.stem = nn.ModuleList([])

        self.stem.append(Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,sigma_prior=args.sigma_prior, args=args))
        self.stem.append(nn.BatchNorm2d(planes))
        self.stem.append(nn.ReLU())
        self.stem.append(Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, sigma_prior=args.sigma_prior, args=args))
        self.stem.append(nn.BatchNorm2d(planes))

        self.shortcut = nn.ModuleList([])
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut.append(Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, sigma_prior=args.sigma_prior, args=args))
            self.shortcut.append(nn.BatchNorm2d(self.expansion*planes))

        self.add = Add()
        self.end = nn.ReLU()

    def forward(self, x):
        out = x
        for layer in self.stem:
            out = layer(out)
            out = clamp_activation(out, self.args)
        shortcut = x 
        for layer in self.shortcut:
            shortcut = layer(shortcut)
            shortcut = clamp_activation(shortcut, self.args)
        out = self.add(out, shortcut)
        out = clamp_activation(out, self.args)
        out = self.end(out)
        out = clamp_activation(out, self.args)
        return out
        
    def fuse_model(self):
        torch.quantization.fuse_modules(self.stem, [['0', '1','2'], ['3', '4']],fuser_func=fuse_bbb_modules, inplace=True)
        if len(self.shortcut)==2:
            torch.quantization.fuse_modules(self.shortcut, ['0', '1'], fuser_func=fuse_bbb_modules, inplace=True)


class ConvNetwork_ResNet(nn.Module):
    def __init__(self, input_size, output_size, q, args):
        super(ConvNetwork_ResNet, self).__init__()
        self.args = args
        self.q = q
        self.in_planes = 24
        self.init_channels = input_size[1]
        layers = [2, 2, 2, 2]

        self.layers = nn.ModuleList([])
        self.layers.append(Conv2d(self.init_channels, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False,sigma_prior=args.sigma_prior, args=args))
        self.layers.append(nn.BatchNorm2d(self.in_planes))
        self.layers.append(nn.ReLU())
        self.layers.append(self._make_layer(self.in_planes, layers[0], stride=1))
        self.layers.append(self._make_layer(48, layers[1], stride=2))
        self.layers.append(self._make_layer(96, layers[2], stride=2))
        self.layers.append(self._make_layer(192, layers[3], stride=2))
        self.layers.append(nn.AvgPool2d(4))
        self.layers.append(Flatten())
        self.layers.append(Linear(192*BasicBlock.expansion, output_size, sigma_prior=args.sigma_prior, bias=False, args=args))

        self.q = q 
        if self.q:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, self.q, self.args))
            self.in_planes = planes * BasicBlock.expansion
        return nn.ModuleList(layers)

    def forward(self, x):
        if self.q:
            x = self.quant(x)
            x = clamp_activation(x, self.args)

        for layer in self.layers:
            if isinstance(layer, nn.ModuleList):
                for sub_layer in layer:
                    x = sub_layer(x)
                    x = clamp_activation(x, self.args)
            else:
                x = layer(x)
                x = clamp_activation(x, self.args)

        if self.q:
            x = self.dequant(x)

        x = F.softmax(x, dim=-1)
            
        return x


    def fuse_model(self):
        torch.quantization.fuse_modules(self.layers, ['0', '1', '2'],fuser_func=fuse_bbb_modules, inplace=True)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.fuse_model()

    def get_kl_divergence(self):
        kl = 0.0
        for m in self.modules():
            if hasattr(m, 'get_kl_divergence') and not isinstance(m, ConvNetwork_ResNet):
                kl+=m.get_kl_divergence()
        return kl
    

  

  
