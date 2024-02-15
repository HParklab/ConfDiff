import numpy as np
import os
import argparse

from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
from rdkit_type import ATOM_TYPE, BOND_TYPE
from multiprocessing import Pool

RDLogger.DisableLog('rdApp.*')      

def sdf_processing(i, total_num, sdf, input_dir_path):
    mol_id = 0
    atom_num_list = []
    coordinate_data_list = []
    node_data_list = []
    edge_data_list = []
    
    with Chem.SDMolSupplier(str(sdf)) as suppl:
        for mol in suppl:
            if mol==None:
                continue
        
            atom_num = mol.GetNumAtoms()
            coordinate = mol.GetConformer().GetPositions()

            atom_type, num_hs = featurize_node(mol)
            node_data = np.hstack((atom_type, num_hs))
            
            graph_path, order_type = featurize_edge(mol)
            edge_data = np.hstack((graph_path, order_type))
            
            assert atom_num==len(coordinate)
            assert atom_num==len(node_data)
            assert atom_num**2==len(edge_data)
            
            atom_num_list.append(atom_num)
            coordinate_data_list.append(coordinate)
            node_data_list.append(node_data)
            edge_data_list.append(edge_data)
            
            mol_id+=1
    if mol_id==0:
        return mol_id
    
    atom_num = np.vstack(atom_num_list)
    coordinate_data = np.vstack(coordinate_data_list)
    node_data = np.vstack(node_data_list)
    edge_data = np.vstack(edge_data_list)

    atom_num = atom_num.astype(np.int8)
    coordinate_data = coordinate_data.astype(np.float32)
    node_data = node_data.astype(np.int8)
    edge_data = edge_data.astype(np.int8)
    
    np.savez(input_dir_path/sdf.stem, atom_num        = atom_num,
                                      coordinate_data = coordinate_data, 
                                      node_data       = node_data,
                                      edge_data       = edge_data)
    print(f'complete num : {i:4d}, total num : {total_num:4d}')
    return mol_id


def search_index(type_list:list, value:any) -> int:
    if value in type_list:
        return type_list.index(value)
    assert type_list[-1] == 'misc'
    return len(type_list) - 1
    

def featurize_node(mol:Chem.rdchem.Mol):
    atom_type   = np.array([[search_index(ATOM_TYPE['ATOMICNUM'], atom.GetAtomicNum())] for atom in mol.GetAtoms()])
    num_hs      = np.array([[search_index(ATOM_TYPE['NUMHS'], atom.GetTotalNumHs())] for atom in mol.GetAtoms()])
    
    return atom_type, num_hs

def featurize_edge(mol:Chem.rdchem.Mol):
    get_graph_path  = lambda mol,i,j: len(Chem.rdmolops.GetShortestPath(mol,i,j))-1 if i!=j else 0
    get_bond_onehot = lambda mol,i,j: 'None' if mol.GetBondBetweenAtoms(i,j)==None else mol.GetBondBetweenAtoms(i,j).GetBondType().name
  
    graph_path = np.array([
        [search_index(BOND_TYPE['GRAPHPATH'], get_graph_path(mol,i,j))] for j in range(mol.GetNumAtoms()) for i in range(mol.GetNumAtoms())
    ])
    order_type = np.array([
        [search_index(BOND_TYPE['ORDER'], get_bond_onehot(mol,i,j))] for j in range(mol.GetNumAtoms()) for i in range(mol.GetNumAtoms())
    ])
    
    return graph_path, order_type


def sdf_to_input(args, sdf_list):
    input_dir_path = Path(args.outdir)/'input'
    os.makedirs(input_dir_path, exist_ok=True)
    N = len(sdf_list)
    pool = Pool(processes=args.core)
    pool_list = pool.starmap(sdf_processing, [(i, N, sdf, input_dir_path) for i, sdf in enumerate(sdf_list)])
    
    mol_id = 0
    for p in pool_list:
        mol_id += p
    print(f'Number of molecule', mol_id)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--outdir', type=str, default='dataset/')
    parser.add_argument('--core', type=int, default=12)
    args = parser.parse_args()
    
    sdf_list = sorted(Path('dataset/ZINC').rglob('*.sdf'))
    
    # already_list = [path.stem for path in Path('dataset/input').rglob('*.npz')]
    # sdf_list = [sdf for sdf in sdf_list if sdf.stem not in already_list]
    # print(len(sdf_list))
    
    sdf_to_input(args, sdf_list)
