RUN_SERIAL = 1
RANDOM_SEED = 2

print("importing packages...")
import torch
from torch import nn
from torch.optim import Adam, AdamW
import random
import numpy as np
import subprocess

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
random.seed(RANDOM_SEED)                  
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from difs import FullyConditionedUnet, GaussianDiffusionConditional, DiFS
from multiprocessing import cpu_count
from accelerate import Accelerator, DataLoaderConfiguration
from ema_pytorch import EMA
import concurrent.futures
from tqdm import tqdm
import sys
import wandb
from numpy.linalg import norm
import wandb

wandb.login(key="0008e2d125e8e34c1d04105171081d4e8584ad93")

SAVE_PATH = "models/run_" + str(RUN_SERIAL) + ".pt"
TIMESTEPS = 1000    # diffusion sampling timesteps
N = 2
horizon = 32
xdim = 5
px_variance = 500.0
ALPHA = 0.5
MAX_ITERS = 2
TRAIN_NUM_STEPS = 200  # number of diffusion training steps per iteration
TRAIN_BATCH_SIZE = 1
TRAIN_LR = 3e-4
DIM = 64
DIM_MULTS = (2, 4, 8, 16)
USE_CFG = False

print("Generating initial disturbances...")
data = px_variance * torch.randn(N,xdim,horizon)

def sim_risk(disturbances):
    # disturbances of shape (5, 32)
    # site1[0] , site1[1] , ..., site1[31]
    # site2[0] , site2[1] , ..., site2[31]
    # site3[0] , site3[1] , ..., site3[31]
    # site4[0] , site4[1] , ..., site4[31]
    # reward[0], reward[1], ..., reward[31]

    disturbances = disturbances[:, :30]
    matrix_str = '\n'.join(' '.join(f'{x:.8f}' for x in row) for row in disturbances)
    
    result = subprocess.run(
    ['julia', 'simulate_runner.jl'],
    input=matrix_str.encode('utf-8'),
    capture_output=True,
    # check=True
    )

    return float(result.stdout.decode('utf-8'))

print("Making model...")
model = FullyConditionedUnet(
        dim = DIM,
        # dim_mults = (1, 2, 4),
        dim_mults = DIM_MULTS,
        channels = xdim,
        cond_dim = horizon,
).to('cuda')

diffusion = GaussianDiffusionConditional(
    model,
    seq_length=horizon,
    classifier_free_guidance=USE_CFG,
    timesteps=TIMESTEPS
).to('cuda')

sampler = DiFS(
    RANDOM_SEED,
    diffusion,
    evaluate_fn=sim_risk,
    init_disturbances=data,
    run_serial=RUN_SERIAL,
    save_path=SAVE_PATH,
    alpha=ALPHA,
    N=N,
    max_iters=MAX_ITERS,
    train_num_steps=TRAIN_NUM_STEPS,
    train_batch_size=TRAIN_BATCH_SIZE,
    train_lr=TRAIN_LR,
    use_wandb=True,
    save_intermediate=False
)

print("Logging hyperparameter choice to wandb...")
config = {
    "alpha": ALPHA,
    "dim": DIM,
    "dim_mults": DIM_MULTS,
    "timesteps": TIMESTEPS,
    "N": N,
    "train_num_steps": TRAIN_NUM_STEPS,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "train_lr": TRAIN_LR,
    "classifier_free_guidance": USE_CFG,
    "random_seed": RANDOM_SEED
}

wandb.init(entity="distillation_difs", 
           project="GAN",
           config=config,
           )

print("TRAINING STARTS!")

sampler.train()
