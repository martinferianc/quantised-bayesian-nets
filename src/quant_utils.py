import torch
from torch.quantization.observer import MovingAverageMinMaxObserver
from torch.quantization.fake_quantize import FakeQuantize
from torch.quantization.quantization_mappings import *
from torch.quantization.quantize import swap_module
import src.utils as utils 
import copy
import torch.nn.intrinsic as nni

from src.models.stochastic.bbb.conv import Conv2d as Conv2dBBB 
from src.models.stochastic.bbb.conv import ConvReLU2d as ConvReLU2dBBB
from src.models.stochastic.bbb.conv import ConvBn2d as ConvBn2dBBB
from src.models.stochastic.bbb.conv import ConvBnReLU2d as ConvBnReLU2dBBB

from src.models.stochastic.bbb.quantized.conv_q import Conv2d as Conv2dBBB_Q
from src.models.stochastic.bbb.quantized.conv_q import ConvReLU2d as ConvReLU2dBBB_Q

from src.models.stochastic.bbb.quantized.conv_qat import Conv2d as Conv2dBBB_QAT
from src.models.stochastic.bbb.quantized.conv_qat import ConvReLU2d as ConvReLU2dBBB_QAT
from src.models.stochastic.bbb.quantized.conv_qat import ConvBn2d as ConvBn2dBBB_QAT
from src.models.stochastic.bbb.quantized.conv_qat import ConvBnReLU2d as ConvBnReLU2dBBB_QAT

from src.models.stochastic.bbb.linear import Linear as LinearBBB
from src.models.stochastic.bbb.linear import LinearReLU as LinearReLUBBB  
from src.models.stochastic.bbb.quantized.linear_q import Linear as LinearBBB_Q
from src.models.stochastic.bbb.quantized.linear_q import LinearReLU as LinearReLUBBB_Q
from src.models.stochastic.bbb.quantized.linear_qat import Linear as LinearBBB_QAT
from src.models.stochastic.bbb.quantized.linear_qat import LinearReLU as LinearReLUBBB_QAT

DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST = get_qconfig_propagation_list()
DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST.add(LinearBBB)
DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST.add(LinearReLUBBB)
DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST.add(Conv2dBBB)
DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST.add(ConvReLU2dBBB)
DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST.add(ConvBn2dBBB)
DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST.add(ConvBnReLU2dBBB)


QAT_MODULE_MAPPINGS[LinearBBB] = LinearBBB_QAT
QAT_MODULE_MAPPINGS[LinearReLUBBB] = LinearReLUBBB_QAT
QAT_MODULE_MAPPINGS[Conv2dBBB] = Conv2dBBB_QAT
QAT_MODULE_MAPPINGS[ConvReLU2dBBB] = ConvReLU2dBBB_QAT
QAT_MODULE_MAPPINGS[ConvBn2dBBB] = ConvBn2dBBB_QAT
QAT_MODULE_MAPPINGS[ConvBnReLU2dBBB] = ConvBnReLU2dBBB_QAT

STATIC_QUANT_MODULE_MAPPINGS[LinearBBB] = LinearBBB_Q
STATIC_QUANT_MODULE_MAPPINGS[LinearBBB_QAT] = LinearBBB_Q
STATIC_QUANT_MODULE_MAPPINGS[LinearReLUBBB] = LinearReLUBBB_Q
STATIC_QUANT_MODULE_MAPPINGS[LinearReLUBBB_QAT] = LinearReLUBBB_Q
STATIC_QUANT_MODULE_MAPPINGS[Conv2dBBB] = Conv2dBBB_Q
STATIC_QUANT_MODULE_MAPPINGS[Conv2dBBB_QAT] = Conv2dBBB_Q
STATIC_QUANT_MODULE_MAPPINGS[ConvReLU2dBBB] = ConvReLU2dBBB_Q
STATIC_QUANT_MODULE_MAPPINGS[ConvReLU2dBBB_QAT] = ConvReLU2dBBB_Q
STATIC_QUANT_MODULE_MAPPINGS[ConvBn2dBBB_QAT] = Conv2dBBB_Q
STATIC_QUANT_MODULE_MAPPINGS[ConvBnReLU2dBBB_QAT] = ConvReLU2dBBB_Q

STATIC_QUANT_MODULE_MAPPINGS[nni.ConvBn2d] = torch.nn.quantized.Conv2d
STATIC_QUANT_MODULE_MAPPINGS[torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d] = torch.nn.quantized.Conv2d
STATIC_QUANT_MODULE_MAPPINGS[nni.ConvBnReLU2d] = torch.nn.intrinsic.quantized.ConvReLU2d
STATIC_QUANT_MODULE_MAPPINGS[torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d] = torch.nn.intrinsic.quantized.ConvReLU2d

def convert(model, mapping=None, inplace=True):
    def _convert(module, mapping=None, inplace=True):
        if mapping is None:
            mapping = STATIC_QUANT_MODULE_MAPPINGS
        if not inplace:
            module = copy.deepcopy(module)
        reassign = {}
        SWAPPABLE_MODULES = (nni.ConvBn2d,
                            nni.ConvBnReLU2d,
                            torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d,
                            torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d,
                            nni.LinearReLU,
                            nni.BNReLU2d,
                            nni.BNReLU3d,
                            nni.ConvBn1d,
                            nni.ConvReLU1d,
                            nni.ConvBnReLU1d,
                            nni.ConvReLU2d,
                            nni.ConvReLU3d,
                            LinearReLUBBB,
                            ConvReLU2dBBB,
                            ConvBn2dBBB,
                            ConvBnReLU2dBBB)

        for name, mod in module.named_children():
            if type(mod) not in SWAPPABLE_MODULES:
                _convert(mod, mapping, inplace=True)
            swap = swap_module(mod, mapping)
            reassign[name] = swap

        for key, value in reassign.items():
            module._modules[key] = value
        
        return module
    if mapping is None: 
        mapping = STATIC_QUANT_MODULE_MAPPINGS
    model = _convert(model, mapping=mapping, inplace=inplace)
    return model

def postprocess_model(model, args, q=None, at=None, special_info=""):
  if q is None:
      q = args.q
  if at is None: 
      at = args.at
  if q and at and 'sgld' not in args.model:
    model = model.cpu()
    utils.load_model(model, args.save+"/weights{}.pt".format(special_info))
    convert(model)
    utils.save_model(model, args, special_info)

def prepare_model(model, args, q=None, at=None):
    if q is None:
      q = args.q
    if at is None: 
      at =  args.at

    torch.backends.quantized.engine = 'fbgemm'

    assert 2 <= args.activation_precision and args.activation_precision <= 7
    assert 2 <= args.weight_precision and args.weight_precision <= 8

    activation_precision = utils.UINT_BOUNDS[args.activation_precision]
    weight_precision = utils.INT_BOUNDS[args.weight_precision]

    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    
    model.qconfig = torch.quantization.QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                    dtype=torch.quint8,
                                                    quant_min=activation_precision[0],
                                                    quant_max=activation_precision[1],
                                                    qscheme=torch.per_tensor_affine),
                                                weight=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                            quant_min=weight_precision[0],
                                                                            quant_max=weight_precision[1],
                                                                            dtype=torch.qint8,
                                                                            qscheme=torch.per_tensor_affine))
    if not 'bbb' in args.model:
        torch.quantization.prepare_qat(model, inplace=True)
    else:
        torch.quantization.prepare(
            model, allow_list=DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST, inplace=True)
        torch.quantization.prepare(
            model, allow_list=DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST, inplace=True, observer_non_leaf_module_list=[LinearBBB, Conv2dBBB])

        convert(model, mapping=QAT_MODULE_MAPPINGS)

    
