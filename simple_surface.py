import numpy as np
import torch
from pykeops.torch import LazyTensor
from pykeops.torch.cluster import grid_cluster

import torch.nn as nn
import torch.nn.functional as F

from Bio.PDB import PDBParser

def get_simple_surface(fname):
    
    struct=load_structure_np(fname, center=False)
    points, normals=atoms_to_points_normals(torch.Tensor(struct['xyz']),
                                            atom_rad=torch.Tensor(struct['atom_rad']))
    return points, normals

def load_structure_np(fname, center=False):

    # Load the data
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []
    res=[]
    for atom in atoms:
        coords.append(atom.get_coord())
        types.append(atom.get_name())
        res.append(atom.get_parent().get_resname())

    coords = np.stack(coords)
    types_array = np.array(types)
    res=np.array(res)

    # Normalize the coordinates, as specified by the user:
    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)
        
    return {"xyz": coords, "types": types_array, "resnames": res, 'atom_rad': encode_radii(types_array)/100}


def encode_radii(labels,aa={'H': 110, 'C': 170, 'N': 155, 'O': 152, '-': 180}):

    d=aa.get('-')
    if d==None:
        d=0
    labels_enc=np.array([aa.get(a[0], d) for a in labels])
    return labels_enc

# On-the-fly generation of the surfaces ========================================


def subsample(x, normals=None, scale=1.0):

    labels = grid_cluster(x, scale).long()
    C = labels.max() + 1

    x_1 = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
    D = x_1.shape[1]
    points = torch.zeros_like(x_1[:C])
    points.scatter_add_(0, labels[:, None].repeat(1, D), x_1)
    points= points[:, :-1] / points[:, -1:]
    if normals!=None:
        n_1 = normals
        D = n_1.shape[1]
        norm = torch.zeros_like(n_1[:C])
        norm.scatter_add_(0, labels[:, None].repeat(1, D), n_1)
        norm /= (norm**2).sum(1, keepdim=True).sqrt()

    return points, norm


def atoms_to_points_normals(
    atoms,
    resolution=1.0,
    atom_rad=None,
    probe=1.4,    
    sup_sampling=300,
    reduce_mem=False,
    filter_atoms=False
):
    N, D = atoms.shape
    B=sup_sampling

    print(N, D, B)

    if reduce_mem:
        # draw B random points at r_atom+r_probe spheres and find non-buried

        nb=torch.randn(B, D).type_as(atoms)
        nb /= (nb**2).sum(-1, keepdim=True).sqrt()

        x_i = LazyTensor(nb[:, None,None, :])  
        y_j = LazyTensor(atoms[None, :,None, :])  
        z_k= LazyTensor(atoms[None,None,:, :])
        rad_y=LazyTensor(atom_rad[None,:,None,None]+probe)
        rad_z=LazyTensor(atom_rad[None,None,:, None]+probe)
        points=y_j + rad_y * x_i
        dist = (((points - z_k)/rad_z)**2).sum(dim=3)
        mask=dist.min(dim=2)
        bs, ns, _= (mask>=1 ).nonzero(as_tuple=True)

        normals = nb[bs,:]
        points = atoms[ns,:] + atom_rad[ns,None] * normals

    else:
        if filter_atoms:
            nb=torch.randn(B, D).type_as(atoms)
            nb /= (nb**2).sum(-1, keepdim=True).sqrt()
            x_i = LazyTensor(nb[:, None,None, :])  
            y_j = LazyTensor(atoms[None, :,None, :])  
            z_k= LazyTensor(atoms[None,None,:, :])
            points=y_j + LazyTensor(atom_rad[None,:,None,None]+probe) * x_i
            mask = ((points - z_k) ** 2).sum(3)
            mask=(mask - LazyTensor((atom_rad[None,None,:,None]+probe)**2))  # (B, M,M, 1) masks
            mask=(mask.min(2)>=0).sum(0).to(bool).view(-1)
            N=mask.shape[0]
        else:
            mask=torch.arange(N)


        # draw N*B random points at r_atom+r_probe spheres
        nb=torch.randn(N, B, D).type_as(atoms)
        nb /= (nb**2).sum(-1, keepdim=True).sqrt()
        points = atoms[mask, None, :] + (atom_rad[mask,None,None]+probe) * nb
        normals = nb.view(-1, D)
        points = points.view(-1, D)  # (N*B, D)

        # filter buried points
        x_i = LazyTensor(points[:, None, :])  # (N, 1, 3) points
        y_j = LazyTensor(atoms[None, :, :])  # (1, M, 3) atoms
        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M, 1) distances
        dists, idx=D_ij.min_argmin(1)
        mask=(dists >= (atom_rad[idx]+probe)**2).view(-1)
        points = points[mask]
        normals = normals[mask]

        # move points to r_atom spheres
        points -= probe*normals

    # subsample the point cloud
    points, normals = subsample(points, normals, scale=resolution)

    return points, normals
