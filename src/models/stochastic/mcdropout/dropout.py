import torch 
import torch.nn as nn
from torch.nn.quantized import QFunctional


class BernoulliDropout(nn.Module):
    def __init__(self, p=0.0):
        super(BernoulliDropout, self).__init__()
        self.p = torch.nn.Parameter(torch.ones((1,))*p, requires_grad=False)
        self.multiplier =  torch.nn.Parameter(torch.ones((1,))/(1.0 - self.p), requires_grad=False)
        
        self.mul_mask = torch.nn.quantized.FloatFunctional()
        self.mul_scalar = torch.nn.quantized.FloatFunctional()
        
    def forward(self, x):
        if self.p<=0.0:
            return x
        mask_ = None
        if len(x.shape)<=2:
            if x.is_cuda:
                mask_ = torch.cuda.FloatTensor(x.shape).bernoulli_(1.-self.p)
            else:
                mask_ = torch.FloatTensor(x.shape).bernoulli_(1.-self.p)
        else:
            if x.is_cuda:
                mask_ = torch.cuda.FloatTensor(x.shape[:2]).bernoulli_(
                    1.-self.p)
            else:
                mask_ = torch.FloatTensor(x.shape[:2]).bernoulli_(
                    1.-self.p)
        if isinstance(self.mul_mask, QFunctional):
            scale = self.mul_mask.scale
            zero_point = self.mul_mask.zero_point
            mask_ = torch.quantize_per_tensor(mask_, scale, zero_point, dtype=torch.quint8)
        if len(x.shape) > 2:
            mask_ = mask_.view(
                mask_.shape[0], mask_.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        x = self.mul_mask.mul(x, mask_)
        x = self.mul_scalar.mul_scalar(x, self.multiplier)
        return x

    def extra_repr(self):
        return 'p={}, quant={}'.format(
            self.p.item(), isinstance(
                self.mul_mask, QFunctional)
        )
