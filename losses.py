# from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/chamfer.html#chamfer_distance

from typing import Union
import torch
import torch.nn.functional as F

from pykeops.torch import LazyTensor


def _chamfer_distance_single_direction(
    x,
    y,
    x_normals,
    y_normals,
    abs_cosine: bool,
    reduction: Union[str, None]
):
    return_normals = x_normals is not None and y_normals is not None

    N, D = x.shape

    x_i = LazyTensor(x[:, None, :])  # (N, 1, D)
    y_j = LazyTensor(y[None, :, :])  # (1, M, D)
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M)
    cham_x, idx = D_ij.min_argmin(dim=1)  # (N,)
    
    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
    
        x_normals_near = y_normals[idx[:,0]]

        cosine_sim = F.cosine_similarity(x_normals, x_normals_near, dim=1, eps=1e-6)
        # If abs_cosine, ignore orientation and take the absolute value of the cosine sim.
        cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)
    
    if reduction is not None:
        # Apply point reduction
        cham_x = cham_x.sqrt().sum()  
        if return_normals:
            cham_norm_x = cham_norm_x.sum()  
        if reduction == "mean":
            cham_x /= N
            if return_normals:
                cham_norm_x /= N

    cham_dist = cham_x
    cham_normals = cham_norm_x if return_normals else None
    return cham_dist, cham_normals

def chamfer_distance(
    x,
    y,
    x_normals=None,
    y_normals=None,
    single_directional: bool = False,
    abs_cosine: bool = True,
    reduction: Union[str, None] = 'mean'
):
 
    cham_x, cham_norm_x = _chamfer_distance_single_direction(
        x,
        y,
        x_normals,
        y_normals,
        abs_cosine,
        reduction
    )
    if single_directional:
        return cham_x, cham_norm_x
    else:
        cham_y, cham_norm_y = _chamfer_distance_single_direction(
        y,
        x,
        y_normals,
        x_normals,
        abs_cosine,
        reduction
    )
        if reduction is not None:
            return (
                cham_x + cham_y,
                (cham_norm_x + cham_norm_y) if cham_norm_x is not None else None,
            )
        return (
            (cham_x, cham_y),
            (cham_norm_x, cham_norm_y) if cham_norm_x is not None else None,
        )
    
def hausdorff_distance(x,y,single_directional=False):

    x_i = LazyTensor(x[:, None, :])  # (N, 1, D)
    y_j = LazyTensor(y[None, :, :])  # (1, M, D)
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M)
    haus_x = D_ij.min(dim=1).max().sqrt()

    if single_directional:
        return haus_x
    else:
        haus_y = D_ij.min(dim=0).max().sqrt()
        return  max(haus_x, haus_y)
    
def bokov_distribution(x, r=1, reduction=None):
    
    N=x.shape[0]
    x_i = LazyTensor(x[:, None, :])  # (N, 1, D)
    y_j = LazyTensor(x[None, :, :])  # (1, M, D)
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M)
    bok = (r**2-D_ij).step().sum(1).view(-1)
    mean_bok=bok.sum()/bok.shape[0]
    bok=bok-mean_bok

    if reduction==None:
        return bok
    elif reduction=='std':
        std_bok=((bok**2).sum()/bok.shape[0]).sqrt()
        return std_bok
    elif reduction=='max':
        max_bok=bok.abs().max()
        return max_bok

