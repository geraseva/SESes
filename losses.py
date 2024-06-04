# from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/chamfer.html#chamfer_distance

from typing import Union
import torch
import torch.nn.functional as F

from pykeops.torch import LazyTensor
from batching import diagonal_ranges

class ChamferDistance:
    def __init__(self, single_directional: bool = False, abs_cosine: bool = True, reduction: Union[str, None] = 'mean'):
        self.single_directional = single_directional
        self.abs_cosine = abs_cosine
        self.reduction = reduction

    def _chamfer_distance_single_direction(self, x, y, x_normals, y_normals, x_batch, y_batch):
        return_normals = x_normals is not None and y_normals is not None

        N, D = x.shape

        x_i = LazyTensor(x[:, None, :].contiguous())  # (N, 1, D)
        y_j = LazyTensor(y[None, :, :].contiguous())  # (1, M, D)
        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M)
        
        if x_batch!=None and y_batch!=None:
            D_ij.ranges = diagonal_ranges(x_batch, y_batch)


        idx = D_ij.argmin(dim=1)  # (N,)
        cham_x=((x - y[idx[:, 0]]) ** 2).sum(-1)

        if return_normals:
            # Gather the normals using the indices and keep only value for k=0
            x_normals_near = y_normals[idx[:, 0]]

            cosine_sim = F.cosine_similarity(x_normals, x_normals_near, dim=1, eps=1e-6)
            # If abs_cosine, ignore orientation and take the absolute value of the cosine sim.
            cham_norm_x = 1 - (torch.abs(cosine_sim) if self.abs_cosine else cosine_sim)

        if self.reduction is not None:
            # Apply point reduction
            cham_x = cham_x.sqrt().sum()
            if return_normals:
                cham_norm_x = cham_norm_x.sum()
            if self.reduction == "mean":
                cham_x /= N
                if return_normals:
                    cham_norm_x /= N

        cham_dist = cham_x
        cham_normals = cham_norm_x if return_normals else None
        return cham_dist, cham_normals

    def __call__(self, x, y, x_normals=None, y_normals=None, x_batch=None, y_batch=None):
        cham_x, cham_norm_x = self._chamfer_distance_single_direction(x, y, x_normals, y_normals, x_batch, y_batch)
        if self.single_directional:
            return cham_x, cham_norm_x
        else:
            cham_y, cham_norm_y = self._chamfer_distance_single_direction(y, x, y_normals, x_normals, y_batch, x_batch)
            if self.reduction is not None:
                return (
                    cham_x + cham_y,
                    (cham_norm_x + cham_norm_y) if cham_norm_x is not None else None,
                )
            return (
                (cham_x, cham_y),
                (cham_norm_x, cham_norm_y) if cham_norm_x is not None else None,
            )

def chamfer_distance(x, y, x_normals=None, y_normals=None, x_batch=None, y_batch=None,
                     single_directional= False, abs_cosine = True, reduction = 'mean'):
    crit=ChamferDistance(single_directional, abs_cosine, reduction)
    return crit(x, y, x_normals, y_normals, x_batch, y_batch)
    
class HausdorffDistance:
    def __init__(self, single_directional=False):

        self.single_directional = single_directional

    def __call__(self, x, y, x_batch=None, y_batch=None):
        x_i = LazyTensor(x[:, None, :].contiguous())  # (N, 1, D)
        y_j = LazyTensor(y[None, :, :].contiguous())  # (1, M, D)
        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M)

        if x_batch!=None and y_batch!=None:
            D_ij.ranges = diagonal_ranges(x_batch, y_batch)
                
        idx = D_ij.argmin(dim=1)
        haus_x=((x - y[idx[:,0]]) ** 2).sum(-1).max().sqrt()

        if self.single_directional:
            return haus_x
        else:
            idx = D_ij.argmin(dim=0)
            haus_y=((y - x[idx[:,0]]) ** 2).sum(-1).max().sqrt()
        return  max(haus_x, haus_y)
    
def hausdorff_distance(x, y, x_normals=None, y_normals=None, x_batch=None, y_batch=None,
                     single_directional=False):
    crit=HausdorffDistance(single_directional)
    return crit(x, y, x_batch, y_batch)

class Distribution:

    def __init__(self, r=1.05, reduction='max'):
        self.r=r
        self.reduction=reduction

    def __call__(self,x, x_batch=None):
        N=x.shape[0]
        x_i = LazyTensor(x[:, None, :].contiguous())  # (N, 1, D)
        y_j = LazyTensor(x[None, :, :].contiguous())  # (1, N, D)
        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M)
        if x_batch!=None:
            D_ij.ranges = diagonal_ranges(x_batch, x_batch)
        b = (self.r**2-D_ij).step().sum(1).view(-1)
        mean_b=b.sum()/b.shape[0]
        b=b-mean_b

        if self.reduction==None:
            return b
        elif self.reduction=='std':
            std_b=((b**2).sum()/b.shape[0]).sqrt()
            return std_b
        elif self.reduction=='max':
            max_b=b.abs().max()
            return max_b

def distribution(x, x_batch=None,
                 r=1.05, reduction=None):
    crit=Distribution(r, reduction)
    return crit(x, x_batch)

class SurfaceCriterion:
    def __init__(self, chamfer_weight=1.0, norm_weight=1.0, hausdorff_weight=1.0, distribution_weight=1.0):
        self.chamfer_weight = chamfer_weight
        self.norm_weight = norm_weight
        self.hausdorff_weight = hausdorff_weight
        self.distribution_weight = distribution_weight

        self.chamfer=ChamferDistance(reduction='mean')
        self.hausdorff=HausdorffDistance()
        self.distribution=Distribution(reduction='max')

    def __call__(self, protein):
        chamfer_loss, norm_loss = self.chamfer(protein['xyz'], protein['target_xyz'],
                                        protein['normals'],protein['target_normals'],
                                        protein['xyz_batch'],protein['target_xyz_batch'])
        hausdorff_loss = self.hausdorff(protein['xyz'], protein['target_xyz'],
                                        protein['xyz_batch'],protein['target_xyz_batch'])
        distribution_loss = self.distribution(protein['xyz'],
                                              protein['xyz_batch'])

        total_loss = (self.chamfer_weight * chamfer_loss +
                      self.norm_weight * norm_loss +
                      self.hausdorff_weight * hausdorff_loss +
                      self.distribution_weight * distribution_loss)
        return total_loss