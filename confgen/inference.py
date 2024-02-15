import torch

from pathlib import Path
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
        
        
    def conditional_sampling(self, motifdock_dir_path, N:int=1,
                             timesteps:int=0, refix_steps:int=100, resampling:int=1, silvr_rate:float=0.01,
                             view:bool=True, save:bool=False, mode:str='fixed'):
        
        motifdock_dir_path = Path(motifdock_dir_path)
        pred_path = motifdock_dir_path/f'{motifdock_dir_path.name}_best.ligand.predkey.pdb'
        ligand_path = motifdock_dir_path/'target.ligand.pdb'
        assert pred_path.exists()
        assert ligand_path.exists()
        
        self.args.T = self.T
        assert self.args.T % resampling == 0
        self.args.T = self.args.T // resampling
        
        self.dynamics = Dynamics(device=self.device, args=self.args)
        self.dynamics.load_state_dict(self.checkpoint['model'])
        self.diffusion = DiffusionProcess(diffusion_fn=self.dynamics, device=self.device, args=self.args)

        result = self.diffusion.conditional_sampling(
            ligand_path, pred_path, N=N, 
            timesteps=timesteps, refix_steps=refix_steps, resampling=resampling, silvr_rate=silvr_rate,
            view=view, save=save, mode=mode
        )
         
        if result!=None:
            return result
        

if __name__ == '__main__':
    
    pass