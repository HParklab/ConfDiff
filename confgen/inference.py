import torch
import argparse
import os 

from distutils.util import strtobool
from pathlib import Path, PosixPath
from model import Dynamics
from diffusion import DiffusionProcess


class Inference:
    
    def __init__(self, device, model_path):
        
        with open(model_path, 'rb') as f:
            self.checkpoint = torch.load(f, map_location='cpu')
        self.args = self.checkpoint['args']
        self.T = self.args.T
        self.device = device

        try:
            self.args.scale
        except:
            self.args.scale = 1
        

    def sampling(self, smiles=None, N:int=1, view:bool=True, save:bool=False, timesteps:int=0):
        
        self.dynamics = Dynamics(device=self.device, args=self.args)
        self.dynamics.load_state_dict(self.checkpoint['model'])
        self.diffusion = DiffusionProcess(diffusion_fn=self.dynamics, device=self.device, args=self.args)

        result = self.diffusion.sampling(
            smiles=smiles, N=N, save=save, timesteps=timesteps, view=view
        )

        if result!=None:
            return result 
        
        
    def conditional_sampling(self, pdb_path:str, key_atom_list:list, N:int=1,
                             timesteps:int=0, refix_steps:int=100, resampling:int=1, silvr_rate:float=0.01,
                             view:bool=True, save:bool=False, mode:str='fixed'):
        
        pdb_path = Path(pdb_path)
        assert pdb_path.exists()
        assert key_atom_list!=None
        
        self.args.T = self.T
        assert self.args.T % resampling == 0
        self.args.T = self.args.T // resampling
        
        self.dynamics = Dynamics(device=self.device, args=self.args)
        self.dynamics.load_state_dict(self.checkpoint['model'])
        self.diffusion = DiffusionProcess(diffusion_fn=self.dynamics, device=self.device, args=self.args)

        result = self.diffusion.conditional_sampling(
            pdb_path=pdb_path, key_atom_list=key_atom_list, N=N, 
            timesteps=timesteps, refix_steps=refix_steps, resampling=resampling, silvr_rate=silvr_rate,
            view=view, save=save, mode=mode
        )
         
        if result!=None:
            return result
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--smiles', type=str, default=None, help='Query molecule smiles')
    
    parser.add_argument('--pdb_path', type=str, default=None, help='Query molecule pdb path')
    parser.add_argument('--key_atom_list', type=str, nargs='+', default=None)
    parser.add_argument('--resampling', type=int, default=10, help='Number of resampling for each steps')
    parser.add_argument('--refix', type=int, default=100, help='Number of refix steps')
    parser.add_argument('--mode', choices=['fixed', 'replacement'], default='fixed', help='Conditioning mode')
    
    parser.add_argument('--n_samples', type=int, default=4, help='Number of sampled molecule')
    parser.add_argument('--timesteps', type=int, default=50, help='Compressed time steps for inference speed acceleration (using DDIM)')
    parser.add_argument('--model_path', type=str, help='Model weight path')
    parser.add_argument('--save', type=strtobool, default=True, help='Save sampled molecule with diffusion trajectories')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference = Inference(device, args.model_path)
    
    if args.smiles!=None:
            
        inference.sampling(
            smiles    = args.smiles, 
            N         = args.n_samples,
            save      = args.save,
            timesteps = args.timesteps,
            view      = False
        )
    
    elif args.pdb_path!=None:
        
        inference.conditional_sampling(
            pdb_path = args.pdb_path,
            key_atom_list = args.key_atom_list,
            N = args.n_samples,
            refix_steps = args.refix,
            resampling = args.resampling,
            mode = args.mode,
            save      = args.save,
            timesteps = args.timesteps,
            view      = False
        )
    
    else:
        
        raise AttributeError(f'smiles : {args.smiles} or pdb_path : {args.pdb_path}')