import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm


def load_split_data(cutoff:int, input_path, val_proportion=0.1, test_proportion=0.1):
    
    input_path = Path(input_path)
    coord_data_list = []
    node_data_list = []
    edge_data_list = []
    for npz_path in tqdm(sorted(input_path.rglob('*.npz'))):
        with open(npz_path, 'rb') as f:
            input_data = np.load(f)
            atom_num = input_data['atom_num'].astype(np.int32)
            if len(atom_num)<=1: 
                continue
            coord_data = input_data['coordinate_data']
            node_data = input_data['node_data']
            edge_data = input_data['edge_data']

        assert np.sum(atom_num)==len(coord_data)
        assert np.sum(atom_num)==len(node_data)
        assert np.sum(atom_num**2)==len(edge_data)
        
        split_indices = np.array([np.sum(atom_num[:i+1]) for i in range(0, len(atom_num)-1)])
        coord_data = np.split(coord_data, split_indices)
        node_data = np.split(node_data, split_indices)

        split_indices = np.array([np.sum(atom_num[:i+1]**2) for i in range(0, len(atom_num)-1)])
        edge_data = np.split(edge_data, split_indices)

        for idx, atom in enumerate(atom_num):
            assert atom==len(coord_data[idx])
            assert atom==len(node_data[idx])
            assert atom**2==len(edge_data[idx])
            
        coord_data_list.extend(coord_data)
        node_data_list.extend(node_data)
        edge_data_list.extend(edge_data)
    
    node_limit = [0, 0]
    perm = []
    for idx in sorted(np.random.permutation(list(range(len(node_data_list)))), key=lambda i: len(node_data_list[i])):
        if node_limit[0] != len(node_data_list[idx]):
            node_limit = [len(node_data_list[idx]), 0]
        
        if len(node_data_list[idx]) < 60:
            node_limit[1] += 1
            if node_limit[1] < cutoff:
                perm.append(idx)
                
    perm = np.random.permutation(perm).astype(np.int32)
    coord_data_list = [coord_data_list[i] for i in perm]
    node_data_list = [node_data_list[i] for i in perm]
    edge_data_list = [edge_data_list[i] for i in perm]

    num_mol = len(perm)
    val_index = int(num_mol * val_proportion)
    test_index = val_index + int(num_mol * test_proportion)

    val_coord_data, test_coord_data, train_coord_data = coord_data_list[:val_index], coord_data_list[val_index:test_index], coord_data_list[test_index:] #np.split(coord_data_list, [val_index, test_index])
    val_node_data, test_node_data, train_node_data = node_data_list[:val_index], node_data_list[val_index:test_index], node_data_list[test_index:] #np.split(node_data_list, [val_index, test_index])
    val_edge_data, test_edge_data, train_edge_data = edge_data_list[:val_index], edge_data_list[val_index:test_index], edge_data_list[test_index:] #np.split(edge_data_list, [val_index, test_index])
    
    perm = sorted(list(range(len(train_node_data))), key=lambda idx: len(train_node_data[idx]))
    train_coord_data = [train_coord_data[i] for i in perm]
    train_node_data = [train_node_data[i] for i in perm]
    train_edge_data = [train_edge_data[i] for i in perm]
    return (train_coord_data, train_node_data, train_edge_data), (val_coord_data, val_node_data, val_edge_data), (test_coord_data, test_node_data, test_edge_data)


class LigandDataSet(Dataset):
    
    def __init__(self, coord_data_list, node_data_list, edge_data_list, scale, dtype):
        self.coord_data_list = coord_data_list
        self.node_data_list = node_data_list
        self.edge_data_list = edge_data_list
        
        self.scale = scale
        self.dtype = dtype

    def __len__(self):
        return len(self.node_data_list)
    
    def __getitem__(self, index):
        data = {}
        data['positions'] = torch.from_numpy(self.coord_data_list[index]).to(dtype=self.dtype)
        data['positions'] -= torch.mean(data['positions'], dim=0, keepdim=True)
        data['positions'] /= self.scale
        
        data['atom_mask'] = torch.ones(self.node_data_list[index].shape[0]).to(dtype=self.dtype)
        data['atom_feats'] = torch.from_numpy(self.node_data_list[index]).long()
        data['bond_feats'] = torch.from_numpy(self.edge_data_list[index]).long()
        return data
    
    
class LigandDataLoader(DataLoader):
    
    def __init__(self, data_list, scale, dtype, shuffle:bool=True, BATCH_SIZE:int=4):  

        coord_data_list, node_data_list, edge_data_list = data_list
        dataset = LigandDataSet(coord_data_list ,node_data_list, edge_data_list, scale, dtype)
        generator_params = {
            'shuffle': shuffle,
            'num_workers': 4,
            'pin_memory':  False,
            'collate_fn': collate_fn, 
            'batch_size': BATCH_SIZE,
        }
        super().__init__(dataset, **generator_params)
    

def collate_fn(samples):
    collate_data = {}
    for key in samples[0]:
        collate_data[key] = torch.nn.utils.rnn.pad_sequence(
                                [sample[key] for sample in samples], 
                                batch_first=True, padding_value=0)
    return collate_data
    

if __name__=='__main__':
    
    train_data, valid_data, test_data = load_split_data(Path('../dataset/input'))
    
    _, train_node_data, _ = train_data
    print([len(data) for data in train_node_data])
    # data = next(iter(LigandDataLoader(train_data)))
    # from model import BondEmbedding

    # embedding = BondEmbedding(16)
    # print(embedding(data['bond_feats']).shape)