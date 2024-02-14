import numpy as np
import torch
import os
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from Bio.PDB import PDBParser

from compute_SES import computeMSMS

def load_structure_np(fname, encoders={
    'atom_encoders':[{'name': 'atom_types',
                      'encoder': {"C": 0, "H": 1, "O": 2, "N": 3, "-": 4}},
                     {'name': 'atom_rad',
                      'encoder': {'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, '-': 1.80}
                     }]}):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []
    res=[]
    for atom in atoms:
        coords.append(atom.get_coord())
        types.append(atom.get_name()[0])
        res.append(atom.get_parent().get_resname())

    coords = np.stack(coords)
    types = np.array(types)
    res=np.array(res)

    protein_data={}
    protein_data['atom_xyz']=torch.from_numpy(coords)

    for aa in encoders['atom_encoders']:
        o=max(aa['encoder'].values())+1
        o=(o if isinstance(o, int) else 0)
        protein_data[aa['name']] = encode_labels(types,aa['encoder'],o)
    
    if encoders.get('residue_encoders') != None:
        for la in encoders['residue_encoders']:
            o=max(la['encoder'].values())+1
            protein_data[la['name']] = encode_labels(res,la['encoder'],o)    

    return protein_data


def encode_labels(labels,aa,onehot=0):

    d=aa.get('-')
    if d==None:
        d=0
    labels_enc=np.array([aa.get(a, d) for a in labels])
    if onehot>0:
        labels_enc=torch.IntTensor(labels_enc)
        labels_enc=F.one_hot(labels_enc,num_classes=onehot).float()
    else:
        labels_enc=torch.FloatTensor(labels_enc)
    return labels_enc


class AtomSurfaceDataset(Dataset):

    def __init__(self, root_dir='protein_data', list=None, transform=None, storage=None, encoders={
    'atom_encoders':[{'name': 'atom_rad',
                      'encoder': {'H': 1.10, 'C': 1.70, 'N': 1.55, 'O': 1.52, '-': 1.80}
                     }]}):

        self.root_dir = root_dir
        self.list=(os.listdir(root_dir) if list==None else list)
        self.transform = transform
        self.storage=storage
        self.encoders=encoders

        if self.storage==None or not os.path.exists(self.storage):
            self.process()
        else:
            self.data=torch.load(self.storage)
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
            except:
                print(f'Failed to load {idx}')
        self.list=idxs
        if self.storage!=None:
                torch.save(self.storage, {'indexes':self.list, 'data': self.data})

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        
        prot_dict=self.data[idx]

        if self.transform:
            sample = self.transform(prot_dict)

        return prot_dict
