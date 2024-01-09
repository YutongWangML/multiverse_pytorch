import torch
import torch.nn.functional as F

def fix_grad(G):
    return G + torch.sum(G, dim=1, keepdim=True)

def softmax_cross_entropy_with_integer_labels(discr, labels):
    assert discr.dtype == torch.float32
    assert labels.dtype == torch.int64

    return discr.gather(1, labels.unsqueeze(1)).squeeze() + torch.logsumexp(-discr, dim=1)

def softmax_cross_entropy(discr, labels):
    assert discr.dtype == torch.float32

    discr_min = torch.min(discr, dim=-1)[0].detach()
    discr_min = torch.clamp(discr_min, max=0)

    return torch.sum(discr * labels[..., :-1], dim=-1) - discr_min + torch.log1p(
        torch.expm1(discr_min) + torch.sum(torch.exp(-(discr - discr_min.unsqueeze(-1))), dim=-1)
    )

def fmlc_left(z, Y):
    Ytrim = Y[:, :-1]
    C = torch.sum(Ytrim * z, dim=1, keepdim=True)
    return z - C - Ytrim * C

def exponential(discr, labels):
    assert discr.dtype == torch.float32
    rmarg = fmlc_left(discr, labels)
    return torch.sum(torch.exp(-rmarg), dim=-1)

def sum_hinge(discr, labels):
    assert discr.dtype == torch.float32
    rmarg = fmlc_left(discr, labels)
    return torch.sum(torch.relu(1-rmarg), dim=-1)
