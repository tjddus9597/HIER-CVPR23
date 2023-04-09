import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import hyptorch.nn as hypnn
from hyptorch.pmath import dist_matrix
from models.resnet import Resnet50

class CustomSequential(nn.Sequential):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def forward(self, input):
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1)); perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]; inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
            else:
                input = module(input)
        return input

def _build_norm(norm, hidden_dim, **kwargs):
    if norm == 'bn':
        norm = nn.BatchNorm1d(hidden_dim, **kwargs)
    elif norm == 'bn_noaffine':
        norm = nn.BatchNorm1d(hidden_dim, affine=False)
    elif norm == 'syncbn':
        norm = nn.SyncBatchNorm(hidden_dim, **kwargs)
    elif norm == 'csyncbn':
        norm = CSyncBatchNorm(hidden_dim, **kwargs)
    elif norm == 'psyncbn':
        norm =  PSyncBatchNorm(hidden_dim, **kwargs)
    elif norm == 'ln':
        norm = nn.LayerNorm(hidden_dim, **kwargs)
    else:
        assert norm is None, "unknown norm type {}".format(norm)
    return norm

def _build_act(act):
    if act == 'relu':
        act = nn.ReLU(inplace=False)
    elif act == 'gelu':
        act = nn.GELU()
    else:
        assert False, "unknown act type {}".format(act)
    return act
    
def _build_mlp(nlayers, in_dim=384, hidden_dim=2048, out_dim=128, act='gelu', bias=True, norm=None, output_norm=None, c=None):
    """
    build a mlp
    """
    
    norm_func = _build_norm(norm, hidden_dim)
    act_func = _build_act(act)
    layers = []
    for layer in range(nlayers):
        dim1 = in_dim if layer == 0 else hidden_dim
        dim2 = out_dim if layer == nlayers - 1 else hidden_dim

        if c is not None and c > 0:
            layers.append(hypnn.HypLinear(dim1, dim2, bias=bias, c = c))
        else:
            layers.append(nn.Linear(dim1, dim2, bias=bias))

        if layer < nlayers - 1:
            if norm_func is not None:
                layers.append(norm_func)
            layers.append(act_func)
        elif output_norm is not None:
            output_norm_func = _build_norm(output_norm, out_dim)
            layers.append(act_func)
            layers.append(output_norm_func)
            
        mlp = CustomSequential(*layers)
            
    return mlp

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)


class L2norm(nn.Module):
    def __init__(self, p=2.0, dim=1):
        super(L2norm, self).__init__()
        self.p = p
        self.dim = dim
                
    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)
    
class ScalingLayer(nn.Module):
    def __init__(self, s = 1):
        super(ScalingLayer, self).__init__()
        self.s = s
                
    def forward(self, x):
        return x * self.s
        
def init_model(args):
    if args.model.startswith("dino"):
        body = torch.hub.load("facebookresearch/dino:main", args.model)
    elif args.model == 'resnet50':
        body = Resnet50(pretrained=True, bn_freeze=args.bn_freeze)
    else:
        body = timm.create_model(args.model, pretrained=True)
        
    if args.hyp_c > 0:
        last = nn.Sequential(hypnn.ToPoincare(c=args.hyp_c, ball_dim=args.emb, riemannian=True, clip_r=args.clip_r))
    else:
        last = NormLayer()
        
    if args.model == 'resnet50':
        bdim = 2048
    else:
        freeze(body, 0)
        bdim = 384
        
    last_norm = nn.LayerNorm(bdim, elementwise_affine=False).cuda() if args.use_lastnorm else nn.Identity()
    last_layer = nn.Sequential(last_norm, nn.Linear(bdim, args.emb), last)
    last_layer.apply(_init_weights)
    
    if args.model.startswith("vit") or args.model.startswith("deit"):
        body.reset_classifier(0, args.pool)
    else:
        rm_head(body)
    model = HeadSwitch(body, last_layer)
    model.cuda().train()
    return model

class HeadSwitch(nn.Module):
    def __init__(self, body, last_layer):
        super(HeadSwitch, self).__init__()
        self.body = body
        self.last_layer = last_layer
        self.norm = NormLayer()

    def forward(self, x, skip_head=False):
        x = self.body(x)
        if type(x) == tuple:
            x = x[0]
        if not skip_head:
            x = self.last_layer(x)
        else:
            x = self.norm(x)
        return x

class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

def freeze(model, num_block):
    def fr(m):
        for param in m.parameters():
            param.requires_grad = False

    fr(model.patch_embed)
    fr(model.pos_drop)
    for i in range(num_block):
        fr(model.blocks[i])

def rm_head(m):
    names = set(x[0] for x in m.named_children())
    target = {"head", "fc", "head_dist"}
    for x in names & target:
        m.add_module(x, nn.Identity())
