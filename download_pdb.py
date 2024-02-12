import Bio
from Bio.PDB import * 
from subprocess import Popen, PIPE
from pathlib import Path
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument(
    "--pdb", type=str,default='', help="PDB code along with chains to extract, example 1ABC_A_B", required=False
)
parser.add_argument(
    "--pdb_list", type=str,default='', help="Path to a text file that includes a list of PDB codes along with chains, example 1ABC_A_B", required=False
)

pdb_dir=Path('protein_data/')


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
    args = ["reduce", "-HIS", out_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8'))
    outfile.close()



def get_single(pdb_id: str):

    protonated_file = pdb_dir/f"{pdb_id}.pdb"
    if not protonated_file.exists():
        # Download pdb 
        pdbl = PDBList()
        pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=tmp_dir,
                                              file_format='pdb')
        protonate(pdb_filename, protonated_file)

    pdb_filename = protonated_file

    return p


if __name__ == '__main__':
    args = parser.parse_args()
    if args.pdb != '':
        pdb_list=[args.pdb]
    elif args.pdb_list != '':
        with open(args.pdb_list) as f:
            pdb_list = f.read().splitlines()
    else:
        raise ValueError('Must specify PDB or PDB list') 
    
    for pdb_id in tqdm(pdb_list):

        p=get_single(pdb_id)
        