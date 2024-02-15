import math 
import numpy as np
import torch
import torch.nn as nn

def get_adj_matrix(n_nodes, batch_size, device, edges_dict:dict={}):
    if edges_dict==None:
        edges_dict={}
    if n_nodes in edges_dict:
        edges_dic_b = edges_dict[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx * n_nodes)
                        cols.append(j + batch_idx * n_nodes)
            edges = [torch.LongTensor(rows).to(device),
                        torch.LongTensor(cols).to(device)]
            edges_dic_b[batch_size] = edges
            return edges
    else:
        edges_dict[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device, edges_dict)
    

class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()
    
    
def positional_embedding(N, embed_size):
    idx = torch.arange(N)
    K = torch.arange(embed_size//2)
    pi = 3.141592653589793
    pos_embedding_sin = torch.sin(idx[:,None] * pi / (N**(2*K[None]/embed_size)))
    pos_embedding_cos = torch.cos(idx[:,None] * pi / (N**(2*K[None]/embed_size)))
    pos_embedding = torch.concat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding

def coord2radial(x, norm_constant=1):
    coord_diff = x[:, :, None] - x[:, None]
    radial = torch.sum(coord_diff**2, axis=-1).unsqueeze(-1)
    norm = torch.sqrt(radial).detach() + 1e-8
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff

def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    
    assert not torch.isnan(x).sum()
    N = node_mask.sum(1, keepdims=True)
    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    assert not torch.isnan(N).sum()
    assert not torch.isnan(mean).sum(), (N==0).sum()
    return x


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def get_cosine_betas(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)
    

def kabsch_algorithm(aln_positions:np.ndarray, key_positions:np.ndarray):
    
    aln_CoM = aln_positions.mean(axis=0)
    key_CoM = key_positions.mean(axis=0)

    aln_positions -= aln_CoM
    key_positions -= key_CoM

    h = aln_positions.T @ key_positions
    u, s, vt = np.linalg.svd(h)
    v = vt.T

    d = np.linalg.det(v @ u.T)
    e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    R = v @ e @ u.T
    T = key_CoM - aln_CoM
    
    return R, T


def superimpose_keyatoms(gen_positions:np.ndarray, key_positions:np.ndarray, key_idx_list:list):
    
    out_positions = []
    
    for batch, aln_positions in enumerate(gen_positions[:, key_idx_list]):
        
        R, T = kabsch_algorithm(aln_positions, key_positions)
        Z = gen_positions[batch, ...]
        
        out_positions.append(np.einsum('ij,jk -> ik', Z, R.T) + T)
    
    return np.array(out_positions)


def get_key_positions(mol, pred_path):
    
    with open(pred_path) as f:
        data_list = [line.strip() for line in f.read().split('\n') if len(line)>4]
    
    refix_residue_name = lambda name: name[3]+name[0:3] if len(name)==4 else name.strip()
    name_to_idx = {
        refix_residue_name(atom.GetPDBResidueInfo().GetName().strip()) : atom.GetIdx()
        for atom in mol.GetAtoms()
    }

    try:
        key_idx_list  = [name_to_idx[line[12:16].strip()] for line in data_list]
        key_positions = np.array([
            [float(posi) for posi in line[26:55].split()] 
            for line in data_list]
        ) 
        key_CoM        = key_positions.mean(axis=0)    
        key_positions -= key_CoM
        
        return key_idx_list, key_positions, key_CoM
    
    except:
        raise ValueError(data_list, name_to_idx)