import torch 

def kl_divergence(mu, sigma, mu_prior, sigma_prior):
    kl = 0.5 * (2 * torch.log(sigma_prior / sigma) - 1 + (sigma / sigma_prior).pow(2) + ((mu_prior - mu) / sigma_prior).pow(2)).sum() 
    return kl

def softplusinv(x):
    return torch.log(torch.exp(x)-1.)