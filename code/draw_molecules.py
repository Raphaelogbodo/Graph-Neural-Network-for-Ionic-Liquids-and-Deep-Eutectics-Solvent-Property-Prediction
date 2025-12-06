import py3Dmol
from openbabel import openbabel
from openbabel import pybel
import os
from rdkit import Chem, rdBase
from rdkit.Chem import Draw
from rdkit.Chem import AllChem


'''A simple function to visualize molecule from smile string'''

def draw_molecule(smiles, molsPerRow=2,show_hydrogens=False, save_path=None):
    '''
    inputs:
        - smiles --> type: list: A list of molecular smile strings eg., ['CCO']
        -show_hydrogens --> type boolean: default (False)
    '''
    if len(smiles) < 2:
        smile = smiles[0]
        mol = Chem.MolFromSmiles(smile)
        if show_hydrogens:
            mol = Chem.AddHs(mol)
        img = Draw.MolToImage(mol)
    else:
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        if show_hydrogens:
            mols = [Chem.AddHs(mol) for mol in mols]
        img = Draw.MolsToGridImage([mol for mol in mols], molsPerRow=molsPerRow, subImgSize=(250,250))
        
    # Save if path provided
    if save_path is not None:
        img.save(save_path)
        print(f"Image saved to: {save_path}")
        
    return img
