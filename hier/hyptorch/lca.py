import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import hyptorch.pmath as pmath

def isometric_transform(a, x):
    """Reflection (circle inversion of x through orthogonal circle centered at a)."""
    r2 = torch.sum(a ** 2, dim=-1, keepdim=True) - 1.
    u = x - a
    return r2 / torch.sum(u ** 2, dim=-1, keepdim=True) * u + a


def reflection_center(mu):
    """Center of inversion circle."""
    return mu / torch.sum(mu ** 2, dim=-1, keepdim=True)


def euc_reflection(x, a):
    """
    Euclidean reflection (also hyperbolic) of x
    Along the geodesic that goes through a and the origin
    (straight line)
    """
    xTa = torch.sum(x * a, dim=-1, keepdim=True)
    norm_a_sq = torch.sum(a ** 2, dim=-1, keepdim=True).clamp_min(1e-6)
    proj = xTa * a / norm_a_sq
    return 2 * proj - x


def _halve(x):
    """ computes the point on the geodesic segment from o to x at half the distance """
    return x / (1. + torch.sqrt(1 - torch.sum(x ** 2, dim=-1, keepdim=True)))


def hyp_lca(a, b, return_coord=True):
    """
    Computes projection of the origin on the geodesic between a and b, at scale c
    More optimized than hyp_lca1
    """
    r = reflection_center(a)
    b_inv = isometric_transform(r, b)
    o_inv = a
    o_inv_ref = euc_reflection(o_inv, b_inv)
    o_ref = isometric_transform(r, o_inv_ref)
    proj = _halve(o_ref)
    return proj


def hyp_lca_dist(a, b, c=1, clip_r=1, dim=-1):
    """
    Folloinwg Appendix B of https://arxiv.org/pdf/2010.00402.pdf
    """
    sqrt_c = max(c ** 0.5, 1/clip_r)
    a_norm = a.norm(2, dim=dim)
    b_norm = b.norm(2, dim=dim)
    cos_theta = F.cosine_similarity(a,b, dim=dim)
    sin_theta = torch.sqrt((1 - cos_theta.pow(2)).clamp_min(1e-6))
    
    alpha = torch.arctan(1/sin_theta * ((a_norm * (b_norm.pow(2) + 1/c)) / (b_norm * (a_norm.pow(2) + 1/c)) - cos_theta))
    R = torch.sqrt(((a_norm.pow(2) + (1/c)) / (2*a_norm*torch.cos(alpha))).pow(2) - (1/c))
    p_norm = torch.sqrt(R.pow(2) + 1/c) - R
    dist_c = pmath.artanh(sqrt_c * p_norm)
    dist_c = dist_c * 2 / sqrt_c
    return dist_c