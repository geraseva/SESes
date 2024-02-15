import Bio
from Bio.PDB import * 
from Bio.SeqUtils import IUPACData
import numpy as np
from subprocess import Popen, PIPE
from pathlib import Path
import argparse
from tqdm import tqdm
import os

import sys

from xpdb import SloppyStructureBuilder

PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]

protein_dir='protein_data/'
tmp_dir = Path('/tmp')

# Exclude disordered atoms.
class NotDisordered(Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A"  or atom.get_altloc() == "1" 


def find_modified_amino_acids(path):
    """
    Contributed by github user jomimc - find modified amino acids in the PDB (e.g. MSE)
    """
    res_set = set()
    for line in open(path, 'r'):
        if line[:6] == 'SEQRES':
            for res in line.split()[4:]:
                res_set.add(res)
    for res in list(res_set):
        if res in PROTEIN_LETTERS:
            res_set.remove(res)
    return res_set


def extractPDB(
    infilename, outfilename, chain_ids=None 
):
    # extract the chain_ids from infilename and save in outfilename. 
    parser=MMCIFParser(QUIET=True, structure_builder=SloppyStructureBuilder())
    struct = parser.get_structure(infilename, infilename)

    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    modified_amino_acids = find_modified_amino_acids(infilename)

    for chain in struct[0]:
        if (
            chain_ids == None
            or chain.get_id() in chain_ids
        ):
            structBuild.init_chain(chain.get_id()[0])
            for residue in chain:
                het = residue.get_id()
                if het[0] == " ":
                    outputStruct[0][chain.get_id()[0]].add(residue)
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][chain.get_id()[0]].add(residue)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered())

def protonate(in_pdb_file, out_pdb_file):
    # protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
    # in_pdb_file: file to protonate.
    # out_pdb_file: output file where to save the protonated pdb file. 
    
    # Remove protons first, in case the structure is already protonated
    args = ["reduce", "-Trim", in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8').rstrip())
    outfile.close()

    # Now add them again.
    args = ["reduce", "-HIS", "-NOCon", out_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8'))
    outfile.close()

def get_single(pdb_id, chain):
 
    outfile=f"{protein_dir}/{pdb_id}_{chain}.pdb"

    if True: #not os.path.exists(outfile):
        pdbl = PDBList()
        pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=tmp_dir, file_format='mmCif')

        extractPDB(pdb_filename, outfile, [chain])

        protonate(outfile, outfile)


parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument(
    "--pdb", type=str,default='', help="PDB code along with chains to extract, example 1ABC_A_B", required=False
)
parser.add_argument(
    "--pdb_list", type=str,default='', help="Path to a text file that includes a list of PDB codes along with chains, example 1ABC_A_B", required=False
)
parser.add_argument(
    "--save_npys", type=bool ,default=False, help="Whether to save atom coordinates and names as npy files")


if __name__ == '__main__':
    args = parser.parse_args()
    if args.pdb != '':
        pdb_list=[args.pdb]

    elif args.pdb_list != '':
        with open(args.pdb_list) as f:
            pdb_list = f.read().splitlines()
        for pdb_id in tqdm(pdb_list):
            pdb_id = pdb_id.split('_')
            chains = ''.join(pdb_id[1:])
            pdb_id = pdb_id[0]
            get_single(pdb_id, chains)
            #print(f'Failed to extract {pdb_id}_{chains}')
    else:
        raise ValueError('Must specify PDB or PDB list') 