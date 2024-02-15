# ConfGen

ConfGen is diffusion model to generate molecule conformation condition with key atom positions.

![](figure/example.gif)

# Installing 

## Conda enviroment

```
$ conda create -n confgen python=3.9
$ conda activate confgen
$ pip install -r requirements.txt
```

## Dataset (Use only for training)

We use custom ZINC20 drug-like dataset to train/valid model, which described `dataset/README.md` for detailed filtering.

```
$ cd datset
$ sh ZINC-downloader-3D-sdf.gz.curl
$ sh additional-ZINC-downloader-3D-sdf.gz.curl

$ mkdir ZINC
$ gzip -f -d *.xaa.sdf.gz
$ mv *.xaa.sdf ZINC/
```

## Model weight

Model weight can be downloaded from [Zenodo](https://zenodo.org/records/10663250)
```
$ mkdir confgen/model
$ cd confgen/model
$ wget https://zenodo.org/records/10663250/files/model.pt
```

# Train

### Preprocessing
Preprocessing takes about an hour.

```
$ python confgen/sdf_to_dataset.py
```

### Training

```
$ python confgen/trainer.py --input dataset/input --T 500 --lr 2e-4 --coords_agg mean --noise_schedule cosine --num_egnn_layers 6 --num_gcl_layers 1 --attention True --batch 64 --scale 5 --scale_eps True
```

Optional flags:

| Flag | Description | 
|--|--|
| --input | Input directory path |
| --wandb | Using Wandb |
| --T | Timesteps for diffusion process |
| --lr | Learning rate |
| --batch | Batch size |
| --Attention | Use attention layer |
| --coords_agg | Aggregation method for coordinate (`sum` or `mean`) |
| --noise_schedule | Noise schedule for diffusion process (`linear` or `cosine`) |
| --num_egnn_layers | Number of egnn layer |
| --num_gcl_layers | Number of sub-egnn layer |
| --resume | Resume with pretrained model |
| --shuffle | Shuffle dataloader |
| --cutoff | Max molecule samples of each heavy atoms number |
| --scale | Reduction factor for coordinate scale |
| --scale_eps | Use epsilon scaling for diffusion process |
| --num_epochs | Num of maxinum epoch |


# Inference

## Option 1. GUI (jupyter notebook)
You can easily inference from `confgen/inference.ipynb` as a user-friendly interface with GUI enviorments.

## Option 2. CLI

### Unconditional Sampling

Generate molecule conformation w/o conditioning. Sample trajectory pdb file was saved in `confgen/sample`

```
$ python confgen/inference.py --smiles <query_smiles> --model_path <model_path> --save True 
```

Optional flags:

| Flag | Description | 
|--|--|
| --smiles | Query molecule smiles | 
| --model_path | Model weight path |
| --n_samples | Number of sampled molecule |
| --save | Save sampled molecule with diffusion trajectories |

### Conditional Sampling