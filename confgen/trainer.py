import os
import torch
import numpy as np
import wandb 
import argparse

from distutils.util import strtobool
from pathlib import Path
from model import Dynamics
from dataloader import LigandDataLoader, load_split_data
from diffusion import DiffusionProcess
from tqdm import tqdm


class Trainer:    
    
    def __init__(self, device, dtype, args,
                 train_dataloader, valid_dataloader):
        
        self.device = device
        self.dtype = dtype
        self.args = args
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        self.beta_1 = args.beta_1
        self.beta_T = args.beta_T
        self.T = args.T
        self.noise_schedule_name = args.noise_schedule
        self.num_epoch = args.num_epochs
        self.start_epoch = 0
        
        self.dynamics = Dynamics(device=device, args=args)
        self.optim = torch.optim.Adam(self.dynamics.parameters(), lr=args.lr, betas=args.betas)
        self.model_path = './model'
        os.makedirs(self.model_path, exist_ok=True)
    
    def train_epoch(self):
        
        with tqdm(self.train_dataloader) as tq:
            
            self.dynamics.train()
            train_loss_list = np.array([])
            
            for data in tq:
            
                input_feats = {
                    key : value.to(self.device) 
                    for key, value in data.items()
                }
                
                self.optim.zero_grad()
                loss = self.dynamics.loss_fn(input_feats)
                loss.backward()
                self.optim.step()
                
                train_loss_list = np.append(train_loss_list, [loss.item()])
                tq.set_postfix(train_loss=loss.item(), mean_loss=np.mean(train_loss_list))
            
        with tqdm(self.valid_dataloader) as tq:
            
            self.dynamics.eval()
            valid_loss_list = np.array([])
            
            with torch.no_grad():
            
                for data in tq:       
            
                    input_feats = {
                        key : value.to(self.device) 
                        for key, value in data.items()
                    }
                
                    loss = self.dynamics.loss_fn(input_feats)
                    valid_loss_list = np.append(valid_loss_list, [loss.item()])
                    tq.set_postfix(valid_loss=loss.item(), mean_loss=np.mean(valid_loss_list))
                    
        return train_loss_list, valid_loss_list
    
    def train(self, smiles=None, model_path=None):
        
        if model_path!=None and Path(model_path).exists():
            self.dynamics.load_state_dict(torch.load(model_path))
        
        min_valid_loss = None
        for epoch in range(self.start_epoch, self.num_epoch):
            
            if epoch % 10 == 0 and smiles!=None:
                DiffusionProcess(self.beta_1, self.beta_T, self.T, self.dynamics, self.device).sampling(smiles)
            
            train_loss_list, valid_loss_list = self.train_epoch()
            if self.args.wandb:
                wandb.log({
                    "train_loss" : np.mean(train_loss_list),
                    "valid_loss" : np.mean(valid_loss_list)
                    }, step=epoch
                )
            else:
                print(f'epoch : {epoch:4d}, train loss : {np.mean(train_loss_list):2.4f}, valid loss : {np.mean(valid_loss_list):2.4f}\n')
            
            if min_valid_loss==None or min_valid_loss>np.mean(valid_loss_list):
                min_valid_loss = np.mean(valid_loss_list)
                self._save_checkpoint(epoch, 'best')
            self._save_checkpoint(epoch, 'last')
    

    def resume(self, model_path):
        with open(model_path, 'rb') as fin:
            checkpoint = torch.load(fin, map_location='cpu')
        self.start_epoch = checkpoint['epoch'] + 1
        self.dynamics.load_state_dict(checkpoint['model'])
        self.optim.load_state_dict(checkpoint['optimizer'])
    

    def _save_checkpoint(self, epoch, model_type):
        state_dict = {
            'model': self.dynamics.state_dict(),
            'optimizer': self.optim.state_dict(),
            'epoch' : epoch,
            'args'  : self.args
        }
        
        if args.wandb:
            name_format = lambda key, value: f'{key}={value}'
            name = f'{model_type}_{"_".join([name_format(key, value) for key, value in wandb.config.items()])}.pt'
        else:
            name = model_type + '.pt'
        filename = os.path.join(self.model_path, name)
        torch.save(state_dict, filename)
        

if __name__=='__main__':
      
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../dataset/input')
    parser.add_argument('--wandb', type=strtobool, default=True)
    parser.add_argument('--T', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--attention', type=strtobool, default=False)
    parser.add_argument('--coords_agg', choices=['sum', 'mean'], default='sum')
    parser.add_argument('--noise_schedule', choices=['linear', 'cosine'], default='linear')
    parser.add_argument('--num_egnn_layers', type=int, default=4)
    parser.add_argument('--num_gcl_layers', type=int, default=1)
    parser.add_argument('--resume', type=strtobool, default=False)
    parser.add_argument('--shuffle', type=strtobool, default=True)
    parser.add_argument('--cutoff', type=int, default=1e+5)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--scale_eps', type=strtobool, default=False)
    parser.add_argument('--num_epochs', type=int, default=100)
    
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
    args = parser.parse_args()

    if args.wandb:
        wandb.init(
            project='confgen',
            config={
                'T'       : args.T,
                'lr'      : args.lr,
                'batch'   : args.batch,
                'cutoff'  : args.cutoff,
                'scale'   : args.scale,
                'scale_eps'       : args.scale_eps,
                'attention'       : args.attention, 
                'coords_agg'      : args.coords_agg,
                'noise_schedule'  : args.noise_schedule,
                'num_egnn_layers' : args.num_egnn_layers,
                'num_gcl_layers'  : args.num_gcl_layers
            }
        )
        name_format = lambda key, value: f'{key}={value}'
        wandb.run.name = "_".join([name_format(key,value) for key,value in wandb.config.items()])
        wandb.run.save()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    train_data, valid_data, test_data = load_split_data(args.cutoff, Path(args.input))

    train_dataloader = LigandDataLoader(train_data, args.scale, dtype, args.shuffle, BATCH_SIZE=args.batch)
    valid_dataloader = LigandDataLoader(valid_data, args.scale, dtype, args.shuffle, BATCH_SIZE=args.batch)

    trainer = Trainer(device, dtype, args, train_dataloader, valid_dataloader)
    if args.wandb and args.resume:
        name_format = lambda key, value: f'{key}={value}'
        name = f'best_{"_".join([name_format(key, value) for key, value in wandb.config.items()])}.pt'
        model_path = Path('model')/name
        assert model_path.exists()
        trainer.resume(model_path=model_path)
    trainer.train()    