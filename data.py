import numpy as np
import torch
import os
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle

from Bio.PDB import PDBParser

from compute_SES import computeMSMS

def encode_labels(labels,aa,onehot=0, to_tensor=True):

    d=aa.get('-')
    if d==None:
        d=onehot
    labels_enc=np.array([aa.get(a, d) for a in labels])
    if not to_tensor:
        return labels_enc
    if onehot>0:
        labels_enc=torch.LongTensor(labels_enc)
        labels_enc=F.one_hot(labels_enc,num_classes=onehot).float()
    else:
        labels_enc=torch.Tensor(labels_enc)
    return labels_enc

def load_structure_np(fname, encoders={
    'atom_types':[{'name': 'atom_rad',
                      'encoder': {'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, '-': 1.80}
                     }]}):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    p={}
    p['atom_xyz']=[]
    p['atom_types']=[]
    p['atom_names']=[]
    p['atom_ids']=[]
    p['atom_resnames']=[]
    p['atom_resids']=[]
    p['atom_chains']=[]

    coords = []
    types = []
    res=[]

    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                p['atom_xyz'].append(atom.get_coord())
                p['atom_types'].append(atom.element)
                p['atom_names'].append(atom.get_name())
                p['atom_ids'].append(atom.get_id())
                p['atom_resnames'].append(residue.get_resname())
                p['atom_resids'].append(residue.get_id()[1])
                p['atom_chains'].append(chain.get_id())
    
    p['atom_xyz'] = np.stack(p['atom_xyz'])
    for key in p:
        p[key]=np.array(p[key])
    list_to_onehot=['atom_types','sequence']
    mask=1 # to mask H atoms, for example
    for key in encoders:
        for aa in encoders[key]:
            if 'mask' in aa['name']:
                mask*=encode_labels(p[key],aa['encoder'],0, to_tensor=False)
    if not isinstance(mask, int):
        mask=(mask>0)
        for key in p:
            p[key]=p[key][mask]
    protein_data={}
    protein_data['atom_xyz']=torch.Tensor(p['atom_xyz'])

    for key in encoders:
        for aa in encoders[key]:
            if 'mask' in aa['name']:
                continue
            o=max(aa['encoder'].values())+1 if aa['name'] in list_to_onehot else 0
            enc=encode_labels(p[key],aa['encoder'],o)
            if aa['name'] in protein_data:
                protein_data[aa['name']]=torch.cat((protein_data[aa['name']],enc), dim=1)
            else:
                protein_data[aa['name']] = enc  
    return protein_data


class AtomSurfaceDataset(Dataset):

    def __init__(self, root_dir='protein_data', list=None, transform=None, pre_transform=None,
                 storage=None, encoders={
    'atom_types':[{'name': 'atom_rad',
                      'encoder': {'H': 1.10, 'C': 1.70, 'N': 1.55, 'O': 1.52, '-': 1.80}
                     }]}):

        self.root_dir = root_dir
        self.list=(os.listdir(root_dir) if list==None else list)
        self.transform = transform
        self.pre_transform=pre_transform
        self.storage=storage
        self.encoders=encoders

        if self.storage==None or not os.path.exists(self.storage):
            self.process()
        else:
            with open(self.storage,'rb') as f:  
                self.data=pickle.load(f)
            self.list=self.data['indexes']
            self.data=self.data['data']

    
    def process(self):

        self.data=[]
        idxs=[]
        for idx in tqdm(self.list):
            prot_name=os.path.join(self.root_dir,idx)
            
            try:
                prot_dict=load_structure_np(prot_name, self.encoders)
                vert, face, norm = computeMSMS(prot_name)
                prot_dict['target_xyz']=torch.FloatTensor(vert)
                prot_dict['target_normals']=torch.FloatTensor(norm)
                self.data.append(prot_dict)
                idxs.append(idx)
                
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                print(f'Failed to load {idx}')
        self.list=idxs
        
        if self.pre_transform!=None:
            idxs=[]
            for i, idx in tqdm(enumerate(self.list), total=len(self.list)):
                try:
                    self.pre_transform(self.data[i])
                    idxs.append(idx)
                
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    print(f'Failed to pre_transform {idx}')
            self.list=idxs
        
        if self.storage!=None:
                with open(self.storage,'wb') as f:
                    pickle.dump({'indexes':self.list, 'data': self.data}, f )

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        
        prot_dict=self.data[idx]

        if self.transform:
            sample = self.transform(prot_dict)

        return prot_dict


class CollateDict:
    """
       data: is a list of dicts 
       which is to be converted 
       to a dict of tensors 
       with new keys corresponding to batches
    """
    def __init__(self, follow_batch=['atom_xyz','target_xyz'], device='cuda'):
        self.follow_batch=follow_batch
        self.device=device

    def __call__(self, data):

        result_dict = {}
        for inc, dict_ in enumerate(data):
            for key, value in dict_.items():   
                if key not in result_dict:
                    result_dict[key] = value.to(self.device)
                else:
                    result_dict[key] = torch.cat((result_dict[key], value.to(self.device)), dim=0)
                if key in self.follow_batch:
                    batch=torch.full((value.shape[0],),inc).to(self.device) 
                    bkey=f'{key}_batch'      
                    if bkey not in result_dict:
                        result_dict[bkey] = batch
                    else:
                        result_dict[bkey] = torch.cat((result_dict[bkey], batch), dim=0)

        return result_dict