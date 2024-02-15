import torch
import random
import numpy as np
import copy
import os 

from pathlib import PosixPath
from utils import get_cosine_betas, assert_mean_zero_with_mask, superimpose_keyatoms, remove_mean_with_mask

from rdkit.Chem import MolFromSmiles, MolFromPDBFile, AllChem, rdchem, MolToPDBFile, rdDepictor
from rdkit.Geometry import Point3D
from rdkit.Chem.Draw.IPythonConsole import drawMol3D

from sdf_to_dataset import featurize_edge, featurize_node
from utils import get_key_positions

class DiffusionUtils:

    def __init__(self, device, args):

        self.device = device
        self.scale  = args.scale
        

    def draw_mol_from_data(self, data:dict, mol:rdchem.Mol, view:bool=True):
        
        mol_conf = copy.deepcopy(mol)
        conf = mol_conf.GetConformer()
        mol = copy.deepcopy(mol)
        mol.RemoveConformer(0)
        
        for n, position in enumerate(data['positions']):
            
            for i in range(mol_conf.GetNumAtoms()):
                
                x,y,z = position[i] * self.scale
                conf.SetAtomPosition(i, Point3D(x.item(),y.item(),z.item()))
            
            mol.AddConformer(conf, assignId=True)
            if view:
                drawMol3D(mol, confId=n)
        
        return mol


    def save_trajection(self, position_list:list, mol:rdchem.Mol, filename:str='test'):
        
        mol_conf = copy.deepcopy(mol)
        conf = mol_conf.GetConformer()
        mol = copy.deepcopy(mol)
        os.makedirs('sample', exist_ok=True)
        
        for n in range(position_list[0].shape[0]):
            
            rdDepictor.Compute2DCoords(mol)
            mol.RemoveConformer(0)
                
            for _, position in enumerate(position_list):
                
                for i in range(mol.GetNumAtoms()):
                    
                    x,y,z = position[n, i] * self.scale
                    conf.SetAtomPosition(i, Point3D(x.item(),y.item(),z.item()))
                
                mol.AddConformer(conf, assignId=True)
            
            MolToPDBFile(mol, f'sample/{filename}_{n}.pdb')

    
    def get_input_data(self, mol:rdchem.Mol, N:int=1) -> dict :
        
        data = {}
        data['atom_mask'] = torch.ones(mol.GetNumAtoms())
        data['atom_feats'] = torch.from_numpy(np.hstack(featurize_node(mol))).long()
        data['bond_feats'] = torch.from_numpy(np.hstack(featurize_edge(mol))).long()

        data = {
            key : torch.vstack([value.unsqueeze(0) for _ in range(N)]).to(self.device) 
            for key, value in data.items()
        }
        
        data['positions'] = torch.Tensor(np.random.normal(size=[N, mol.GetNumAtoms(), 3])).type(torch.float32)
        data['positions'] = data['positions'].to(self.device) 
        data['positions'] -= torch.mean(data['positions'], dim=1, keepdim=True)
        
        return data
    

    def get_key_positions(self, mol:rdchem.Mol, pred_path:PosixPath):

        key_idx_list, key_positions, key_CoM = get_key_positions(mol, pred_path)
        
        key_CoM       /= self.scale
        key_positions /= self.scale

        key_CoM       = torch.tensor(key_CoM, dtype=torch.float32).to(self.device)
        key_positions = torch.tensor(key_positions, dtype=torch.float32).to(self.device)
        
        return key_idx_list, key_positions, key_CoM
                
                
    def get_key_positions_true(self, ligand_path):
        
        mol = MolFromPDBFile(str(ligand_path))
        key_idx_list = random.sample([i for i in range(mol.GetNumAtoms())], 8)
        key_positions = torch.tensor([
            mol.GetConformer().GetPositions()[idx]
            for idx in key_idx_list], dtype=torch.float32
        ) 
        key_CoM        = key_positions.mean(axis=0)    
        key_positions -= key_CoM
        return key_idx_list, key_positions, key_CoM
        # key_idx_list, key_positions, key_CoM = self._get_key_positions_true(ligand_path)
    
    
class DiffusionProcess:
    
    def __init__(self, diffusion_fn, device, args):
        
        self.diffusion_fn = diffusion_fn
        self.diffusion_fn.eval()

        self.utils  = DiffusionUtils(device, args)
        self.device = device
    
        self.noise_schedule_name = args.noise_schedule
        self.beta_1 = args.beta_1
        self.beta_T = args.beta_T
        self.T      = args.T
    
    
    def _set_noise_schedule(self, noise_schedule_name:str, beta_1:float=1e-4, beta_T:float=0.02, T:int=500):
        
        if noise_schedule_name=='linear':
            self.betas = torch.linspace(start = beta_1, end=beta_T, steps=T).to(self.device)
        elif noise_schedule_name=='cosine':
            self.betas = get_cosine_betas(T).to(self.device)
        else:
            raise NotImplementedError(noise_schedule_name)
        
        self.T               = T
        self.alphas          = 1 - self.betas
        self.alpha_bars      = torch.cumprod(self.alphas, dim = 0).to(device = self.device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device = self.device), self.alpha_bars[:-1]])

        
    def _get_next_coordinate(self, t:int, coordinate:torch.Tensor) -> torch.Tensor:
        
        beta_t   = self.betas[t]
        epsilon  = torch.randn_like(coordinate).to(self.device)
        epsilon  -= torch.mean(epsilon, dim=1, keepdim=True)
        
        x_update = torch.sqrt(1-beta_t)*coordinate + torch.sqrt(beta_t)*epsilon
        x_update -= x_update.mean(dim=0)
        return x_update
        

    # compute p(Zt-1 | Zt)
    def _reverse_process(self, data, t):
        
        predict_epsilon = self.diffusion_fn(data, t)
        
        sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[t]) / (1 - self.alpha_bars[t]) * self.betas[t])
        mu_theta_xt     = torch.sqrt(1 / self.alphas[t]) * (data['positions'] - self.betas[t] / torch.sqrt(1 - self.alpha_bars[t]) * predict_epsilon)
        
        noise = torch.zeros_like(data['positions']) if t == 0 else torch.randn_like(data['positions'])
        noise -= torch.mean(noise, dim=1, keepdim=True)
        
        coord = mu_theta_xt + sqrt_tilde_beta * noise
        coord -= torch.mean(coord, dim=1, keepdim=True)
        coord = coord * data['atom_mask'][...,None]
        assert_mean_zero_with_mask(coord, data['atom_mask'][...,None])
        
        data['positions'] = coord
        return data
    
    
    # compute p(Zt-1 | Zt, Z0)
    def _reverse_process_ddim(self, data, t):
        
        predict_epsilon = self.diffusion_fn(data, t)
        alpha_bar_t     = self.alpha_bars[t]
        alpha_bar_next  = self.alpha_prev_bars[t]
        
        ep = ((1 - alpha_bar_t / alpha_bar_next) * (1 - alpha_bar_next) / (1 - alpha_bar_t)).sqrt()
        predict_x_0 = (data['positions'] - predict_epsilon * (1 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt()
        
        noise = torch.zeros_like(data['positions']) if t == 0 else torch.randn_like(data['positions'])
        noise -= torch.mean(noise, dim=1, keepdim=True)

        coord = alpha_bar_next.sqrt()*predict_x_0 + (1-alpha_bar_next)*predict_epsilon + ep * noise
        coord -= torch.mean(coord, dim=1, keepdim=True)
        coord = coord * data['atom_mask'][...,None]
        assert_mean_zero_with_mask(coord, data['atom_mask'][...,None])
        
        data['positions'] = coord
        return data
    

    # sample q(Zt | Zt-1)
    def _forward_process(self, data:dict, t:torch) -> dict:
        
        used_betas = self.betas[t].view(-1, 1, 1)
        epsilon    = torch.randn_like(data['positions'])
        epsilon    -= torch.mean(epsilon, dim=1, keepdim=True)
        
        data['positions'] = torch.sqrt(1 - used_betas) * data['positions'] + torch.sqrt(used_betas) * epsilon
        data['positions'] = data['positions'] * data['atom_mask'][..., None]
        
        return data
            

    # sample q(Zt | Z0)
    def _forward_process_from_zero(self, data:dict, t:torch) -> dict:
        
        used_alpha_bars   = self.alpha_bars[t].view(-1, 1, 1)
        epsilon           = torch.randn_like(data['positions'])
        epsilon           -= torch.mean(epsilon, dim=1, keepdim=True)
        
        data['positions'] = torch.sqrt(used_alpha_bars) * data['positions'] + torch.sqrt(1 - used_alpha_bars) * epsilon
        data['positions'] = data['positions'] * data['atom_mask'][..., None]
        return data
            
    
    @torch.no_grad()
    def sampling(self, smiles, N:int=1, timesteps:int=0,
                 view:bool=True, save:bool=False):
        
        self._set_noise_schedule(noise_schedule_name=self.noise_schedule_name, beta_1=self.beta_1, beta_T=self.beta_T, T=self.T)
        mol = MolFromSmiles(smiles)
        AllChem.EmbedMolecule(mol)
    
        data = self.utils.get_input_data(mol, N)

        time_seq = range(self.T)
        if timesteps > 0:
            skip = self.T // timesteps
            time_seq  = list(range(0, self.T, skip))

        position_list = []
        for t in reversed(time_seq):

            if timesteps > 0:
                data = self._reverse_process_ddim(data, t)
            else:
                data = self._reverse_process(data, t)

            position_list.append(data['positions'])
        
        if save:
            self.utils.save_trajection(position_list, mol)
    
        if view:
            self.utils.draw_mol_from_data(data, mol)
            return None
        
        mol_out = self.utils.draw_mol_from_data(data, mol, view)
        return mol_out


    @torch.no_grad()
    def conditional_sampling(self, ligand_path:PosixPath, pred_path:PosixPath, N:int=1, 
                             timesteps:int=0, refix_steps:int=100, resampling:int=1, silvr_rate:float=0.01,
                             view:bool=True, save:bool=False, mode:str='fixed'):
        
        self._set_noise_schedule(noise_schedule_name=self.noise_schedule_name, beta_1=self.beta_1, beta_T=self.beta_T, T=self.T)
        mol = MolFromPDBFile(str(ligand_path))

        key_idx_list, key_positions, key_CoM = self.utils.get_key_positions(mol, pred_path)
        key_positions_list = []
        coordinate = copy.deepcopy(key_positions)
        for t in range(self.T):
            coordinate = self._get_next_coordinate(t, coordinate).to(self.device)
            key_positions_list.append(coordinate)
        
        data = self.utils.get_input_data(mol, N)

        time_seq = range(self.T)
        if timesteps > 0:
            skip = self.T * resampling // timesteps
            time_seq  = list(range(0, self.T, skip))

        position_list = []
        for t in reversed(time_seq):

            for r in range(resampling):
                
                if mode=='fixed':
                    data['positions'][:, key_idx_list] = key_positions[None, :, :] + torch.mean(data['positions'][:, key_idx_list], dim=1, keepdim=True)
                
                elif mode=='replacement':
                    data['positions'][:, key_idx_list] = key_positions_list[t][None, :, :] + torch.mean(data['positions'][:, key_idx_list], dim=1, keepdim=True)
                
                elif mode=='silvr':
                    key_mean = torch.mean(data['positions'][:, key_idx_list], dim=1, keepdim=True)
                    data['positions'][:, key_idx_list] += - data['positions'][:, key_idx_list] * self.alphas[t] * silvr_rate + (key_positions_list[t][None, :, :] + key_mean) * silvr_rate
                    # data['positions'] -= torch.mean(data['positions'], dim=1, keepdim=True)
                    
                else:
                    raise ValueError(mode)

                if timesteps > 0:
                    data = self._reverse_process_ddim(data, t)
                else:
                    data = self._reverse_process(data, t)

                if r < resampling - 1:
                    data = self._forward_process(data, t)
            
                position_list.append(data['positions'] + key_CoM)
                
            assert_mean_zero_with_mask(data['positions'], data['atom_mask'][..., None])
            
        self._set_noise_schedule(noise_schedule_name=self.noise_schedule_name, beta_1=self.beta_1, beta_T=self.beta_T, T=self.T * resampling)
        self.diffusion_fn.alpha_bars = self.alpha_bars

        data = self._forward_process_from_zero(data, refix_steps)

        time_seq = range(refix_steps)
        if timesteps > 0:
            skip = self.T // timesteps
            time_seq = list(range(0, refix_steps, skip))

        for t in reversed(time_seq):

            if timesteps > 0:
                data = self._reverse_process_ddim(data, t)
            else:
                data = self._reverse_process(data, t)
            
            position_list.append(data['positions'] + key_CoM)
        
        data['positions'] = key_CoM.detach().cpu().numpy() + superimpose_keyatoms(
                                gen_positions = data['positions'].detach().cpu().numpy(), 
                                key_positions = key_positions.detach().cpu().numpy(),
                                key_idx_list  = key_idx_list
                            )
        
        position_list.append(data['positions'])
        
        if view:
            drawMol3D(mol)
            AllChem.EmbedMolecule(mol)
            drawMol3D(mol)
        
        if save:
            self.utils.save_trajection(position_list, mol, f'{ligand_path.parent.name}_{mode}_resampling={resampling}')
    
        mol_out = self.utils.draw_mol_from_data(data, mol, view)

        return MolFromPDBFile(str(ligand_path), removeHs=True), mol_out