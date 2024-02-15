import os 
import copy 
import numpy as np 
import matplotlib.pyplot as plt 
import traceback

from pathlib import Path, PosixPath
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

from inference import Inference
from utils import superimpose_keyatoms, get_key_positions


class MotifDockEvaluation:
    
    def __init__(self, motifdock_path, device:str):
        
        self.motifdock_path = Path(motifdock_path)
        self.device = device
        
        self.output_path = Path('./output')/self.motifdock_path.name
        self.output_path.mkdir(parents=True, exist_ok=True)


    @staticmethod
    def _get_rmsd_list(mol_true, mol_pred):
        
        mol_prb = copy.deepcopy(mol_pred)
        rmsd_total = [] 
        rmsd_conf  = []
        
        for confId in range(mol_prb.GetNumConformers()):    
            rmsd_total.append(cal_total_rmsd(mol_prb, mol_true, confId))
            rmsd_conf.append(AlignMol(mol_prb, mol_true, prbCid=confId))

        return rmsd_conf, rmsd_total
    
    @staticmethod
    def _save_numpy(name:str,
                    conf_rdkit_list:list,
                    conf_fixed_list:list,
                    conf_uncond_list:list,
                    conf_replacement_list:list,
                    conf_resampling_list:list,
                    total_original_list:list,
                    total_rdkit_list:list,
                    total_fixed_list:list,
                    total_uncond_list:list,
                    total_replacement_list:list,
                    total_resampling_list:list,
                    ):
        
        os.makedirs('./result', exist_ok=True)
                
        np.savez(
            './result/'+name,
            conf_rdkit_list = np.array(conf_rdkit_list),
            conf_fixed_list = np.array(conf_fixed_list),
            conf_uncond_list = np.array(conf_uncond_list),
            conf_replacement_list = np.array(conf_replacement_list),
            conf_resampling_list = np.array(conf_resampling_list),
            total_original_list = np.array(total_original_list),
            total_rdkit_list = np.array(total_rdkit_list),
            total_uncond_list = np.array(total_uncond_list),
            total_fixed_list = np.array(total_fixed_list),
            total_replacement_list = np.array(total_replacement_list),
            total_resampling_list = np.array(total_resampling_list)
        )
        
    
    def sampling_original(self, key_path:PosixPath):
        
        mol_true = Chem.MolFromPDBFile(str(key_path.parent/'target.ligand.pdb'))
        key_idx_list, key_positions, key_CoM = get_key_positions(mol_true, str(key_path))
                
        gen_positions = mol_true.GetConformer().GetPositions().reshape([1,-1,3])
        gen_positions -= gen_positions.mean(1)
        
        position = key_CoM + superimpose_keyatoms(
                            gen_positions = gen_positions, 
                            key_positions = key_positions,
                            key_idx_list  = key_idx_list
                        )
            
        mol_pred = copy.deepcopy(mol_true)
        conf = mol_pred.GetConformer()
        mol_pred = copy.deepcopy(mol_true)
        mol_pred.RemoveConformer(0)
        
        for i in range(mol_pred.GetNumAtoms()):
            
            x,y,z = position[0][i]
            conf.SetAtomPosition(i, Point3D(x.item(),y.item(),z.item()))
        
        mol_pred.AddConformer(conf, assignId=True)
        rmsd_conf, rmsd_total = self._get_rmsd_list(mol_true, mol_pred)
        
        with Chem.SDWriter(str(self.output_path/f'{key_path.parent.name}.original.sdf')) as w:
            w.write(mol_pred, confId=0)
        
        return sorted(rmsd_conf), sorted(rmsd_total)


    def sampling_rdkit(self, key_path:PosixPath, N:int):
        
        mol_true = Chem.MolFromPDBFile(str(key_path.parent/'target.ligand.pdb'))
        key_idx_list, key_positions, key_CoM = get_key_positions(mol_true, str(key_path))
                
        gen_positions = mol_true.GetConformer().GetPositions().reshape([1,-1,3])
        gen_positions -= gen_positions.mean(1)
        
        mol_pred = copy.deepcopy(mol_true)
        
        params = Chem.rdDistGeom.srETKDGv3()
        params.randomSeed = 0xf00d
        params.clearConfs = True
        params.numThreads = 0
        params.useExpTorsionAnglePrefs = False
        params.enforceChirality = False
        params.useBasicKnowledge = False
        rdDistGeom.EmbedMultipleConfs(mol_pred, numConfs=N, params=params)    
        
        positions = np.array([conf.GetPositions() for conf in mol_pred.GetConformers()])
        positions -= positions.mean(1).reshape([-1, 1, 3])
        
        positions = key_CoM + superimpose_keyatoms(
                            gen_positions = positions, 
                            key_positions = key_positions,
                            key_idx_list  = key_idx_list
                        )
        
        mol_conf = copy.deepcopy(mol_true)
        conf = mol_conf.GetConformer()
        mol_pred = copy.deepcopy(mol_true)
        mol_pred.RemoveConformer(0)

        for n, position in enumerate(positions):
            
            for i in range(mol_conf.GetNumAtoms()):
                
                x,y,z = position[i]
                conf.SetAtomPosition(i, Point3D(x.item(),y.item(),z.item()))
            
            mol_pred.AddConformer(conf, assignId=True)
        
        rmsd_conf, rmsd_total = self._get_rmsd_list(mol_true, mol_pred)
        save_confId = np.argmin(rmsd_total).tolist()
        
        with Chem.SDWriter(str(self.output_path/f'{key_path.parent.name}.rdkit.sdf')) as w:
            w.write(mol_pred, confId=save_confId)
        
        return sorted(rmsd_conf), sorted(rmsd_total)
    

    def sampling_confgen_unconditional(self, key_path:PosixPath, N:int, 
                                       timesteps:int, label:str):
        
        motifdock_dir_path = key_path.parent
        ligand_path = motifdock_dir_path/'target.ligand.pdb'

        mol_true = Chem.MolFromPDBFile(str(ligand_path))
        smiles = Chem.MolToSmiles(mol_true)

        mol_pred = self.inference.sampling(
                            smiles             = smiles, 
                            N                  = N, 
                            timesteps          = timesteps,
                            view               = False,
                        )
                    
        key_idx_list, key_positions, key_CoM = get_key_positions(mol_true, str(key_path))
        positions = np.array([conf.GetPositions() for conf in mol_pred.GetConformers()])
        positions -= positions.mean(1).reshape([-1, 1, 3])
        
        positions = key_CoM + superimpose_keyatoms(
                            gen_positions = positions, 
                            key_positions = key_positions,
                            key_idx_list  = key_idx_list
                        )
        
        mol_conf = copy.deepcopy(mol_true)
        conf = mol_conf.GetConformer()
        mol_pred = copy.deepcopy(mol_true)
        mol_pred.RemoveConformer(0)

        for n, position in enumerate(positions):
            
            for i in range(mol_conf.GetNumAtoms()):
                
                x,y,z = position[i]
                conf.SetAtomPosition(i, Point3D(x.item(),y.item(),z.item()))
            
            mol_pred.AddConformer(conf, assignId=True)
            
        rmsd_conf, rmsd_total = self._get_rmsd_list(mol_true, mol_pred)
        save_confId = np.argmin(rmsd_total).tolist()
        
        with Chem.SDWriter(str(self.output_path/f'{key_path.parent.name}.confgen.{label}.sdf')) as w:
            w.write(mol_pred, confId=save_confId)
        
        return sorted(rmsd_conf), sorted(rmsd_total)
    

    def sampling_confgen(self, key_path:PosixPath, N:int, mode:str, 
                         resampling:int, timesteps:int, refix_steps:int, label:str, silvr_rate:float=0.01):
        
        mol_true, mol_pred = self.inference.conditional_sampling(
                            motifdock_dir_path = key_path.parent, 
                            mode               = mode, 
                            N                  = N, 
                            refix_steps        = refix_steps,
                            resampling         = resampling,
                            timesteps          = timesteps,
                            view               = False,
                            silvr_rate = silvr_rate
                        )
                    
        rmsd_conf, rmsd_total = self._get_rmsd_list(mol_true, mol_pred)
        save_confId = np.argmin(rmsd_total).tolist()
        
        with Chem.SDWriter(str(self.output_path/f'{key_path.parent.name}.confgen.{label}.sdf')) as w:
            w.write(mol_pred, confId=save_confId)
        
        return sorted(rmsd_conf), sorted(rmsd_total)
    

    def sampling(self, N:int, model_name:str, silvr_rate:float,
                 timesteps:int, refix_steps:int, resampling:int):
        
        self.inference = Inference(self.device, os.path.join('./model', model_name))
        self.model_name = model_name
        
        skip_num = 0 
        sample_num = 0

        conf_rdkit_list = []
        conf_fixed_list = []
        conf_uncond_list = []
        conf_replacement_list = []
        conf_resampling_list = []

        total_original_list = []
        total_rdkit_list = []
        total_fixed_list = []
        total_uncond_list = []
        total_replacement_list = []
        total_resampling_list = []

        for key_path in tqdm(list(self.motifdock_path.rglob('*_best.ligand.predkey.pdb'))):
            
            mol = Chem.MolFromPDBFile(str(next(key_path.parent.rglob('target.ligand.pdb'))))
            if CalcNumRotatableBonds(mol) < 2:
                skip_num += 1
                continue
            
            try:

                _, total_original = self.sampling_original(key_path)
                conf_rdkit, total_rdkit = self.sampling_rdkit(key_path, N=N)
                conf_uncond, total_uncond = self.sampling_confgen_unconditional(key_path=key_path, N=N, timesteps=timesteps, label = 'unconditional')
                conf_fixed, total_fixed = self.sampling_confgen(key_path=key_path, N=N, mode='fixed', resampling=1, timesteps=timesteps, refix_steps=refix_steps, label='fixed')
                conf_replacement, total_replacement = self.sampling_confgen(key_path=key_path, N=N, mode='replacement', resampling=1, timesteps=timesteps, refix_steps=refix_steps, label = 'replacement')
                # conf_resampling, total_resampling = self.sampling_confgen(key_path=key_path, N=N, mode='replacement', resampling=resampling, timesteps=timesteps, refix_steps=refix_steps, label = 'resampling')
                conf_resampling, total_resampling = self.sampling_confgen(key_path=key_path, N=N, mode='silvr', resampling=resampling, timesteps=timesteps, refix_steps=refix_steps, label = 'resampling' ,silvr_rate=silvr_rate)
                
                conf_rdkit_list.append(conf_rdkit) 
                conf_fixed_list.append(conf_fixed)
                conf_uncond_list.append(conf_uncond)
                conf_replacement_list.append(conf_replacement)
                conf_resampling_list.append(conf_resampling)

                total_original_list.append(total_original)
                total_rdkit_list.append(total_rdkit)
                total_fixed_list.append(total_fixed)
                total_uncond_list.append(total_uncond)
                total_replacement_list.append(total_replacement)
                total_resampling_list.append(total_resampling)

                sample_num += 1                
                
                if total_rdkit[0] -2  > total_resampling[0]:
                    print(total_rdkit, total_resampling, key_path)
                    raise ValueError
                
            except Exception as e:

                print(key_path.name, e)
                print(traceback.format_exc())
                continue
        
        print(f'skip : {skip_num}, sample : {sample_num}')

        name = f'dataset={self.motifdock_path.name}_N={N}_timesteps={timesteps}_refix={refix_steps}_resampling={resampling}_{self.model_name}'
        
        self._save_numpy(
            name,
            conf_rdkit_list,
            conf_fixed_list,
            conf_uncond_list,
            conf_replacement_list,
            conf_resampling_list,
            total_original_list,
            total_rdkit_list,
            total_fixed_list,
            total_uncond_list,
            total_replacement_list,
            total_resampling_list,
            )



def cal_total_rmsd(mol_pred, mol_true, confId):
    
    xyz_pred = mol_pred.GetConformer(confId).GetPositions()
    xyz_true = mol_true.GetConformer(0).GetPositions()
    
    rmsd_t = np.sqrt((np.sum((xyz_pred-xyz_true)**2, axis=1)).mean())
    return rmsd_t