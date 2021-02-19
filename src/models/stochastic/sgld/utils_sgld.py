import torch 
from torch.optim import Optimizer
from numpy.random import gamma

class SGLD(Optimizer):
    def __init__(self, params, lr=1e-2, base_C=0.05, gauss_sig=0.1, alpha0=10, beta0=10):
        self.eps = 1e-6
        self.alpha0 = alpha0
        self.beta0 = beta0

        if gauss_sig == 0:
            self.weight_decay = 0
        else:
            self.weight_decay = 1 / (gauss_sig ** 2)

        if self.weight_decay <= 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(self.weight_decay))
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if base_C < 0:
            raise ValueError("Invalid friction term: {}".format(base_C))

        defaults = dict(
            lr=lr,
            base_C=base_C,
        )
        super(SGLD, self).__init__(params, defaults)

    def step(self, burn_in=False, resample_momentum=False, resample_prior=False):
        loss = None

        # iterate over blocks -> the ones defined in defaults. We dont use groups.
        for group in self.param_groups:
            for p in group["params"]:  # these are weight and bias matrices
                if p.grad is None:
                    continue
                state = self.state[p]  # define dict for each individual param
                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(p)
                    state["g"] = torch.ones_like(p)
                    state["V_hat"] = torch.ones_like(p)
                    state["v_momentum"] = torch.zeros_like(p) 
                    state['weight_decay'] = self.weight_decay

                if resample_prior:
                    alpha = self.alpha0 + p.data.nelement() / 2
                    beta = self.beta0 + (p.data ** 2).sum().item() / 2
                    gamma_sample = gamma(
                        shape=alpha, scale=1 / (beta+self.eps), size=None)
                    state['weight_decay'] = gamma_sample

                base_C, lr = group["base_C"], group["lr"]
                weight_decay = state["weight_decay"]
                tau, g, V_hat = state["tau"], state["g"], state["V_hat"]

                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                if burn_in: 
                    tau.add_(-tau * (g ** 2) / (
                        V_hat + self.eps) + 1)
                    tau_inv = 1. / (tau + self.eps)
                    g.add_(-tau_inv * g + tau_inv * d_p)
                    V_hat.add_(-tau_inv * V_hat + tau_inv * (d_p ** 2))

                V_sqrt = torch.sqrt(V_hat)
                V_inv_sqrt = 1. / (V_sqrt + self.eps) 

                if resample_momentum:
                    state["v_momentum"] = torch.normal(mean=torch.zeros_like(d_p),
                                                       std=torch.sqrt((lr ** 2) * V_inv_sqrt))
                v_momentum = state["v_momentum"]

                noise_var = (2. * (lr ** 2) * V_inv_sqrt * base_C - (lr ** 4))
                noise_std = torch.sqrt(torch.clamp(noise_var, min=1e-16))

                noise_sample = torch.normal(mean=torch.zeros_like(
                    d_p), std=torch.ones_like(d_p) * noise_std)

                v_momentum.add_(- (lr ** 2) * V_inv_sqrt *
                                d_p - base_C * v_momentum + noise_sample)
                
                v_momentum[v_momentum != v_momentum] = 0
                v_momentum[v_momentum == float('inf')] = 0
                v_momentum[v_momentum == -float('inf') ] = 0

                p.data.add_(v_momentum)

        return loss
