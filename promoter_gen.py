import torch
import ml_collections
import time
import os
import json
from datetime import datetime
import pickle
from torch import nn, Tensor
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import numpy as np
from models import MLP1, TransformerDenoiser
from promoter_model import WrappedModel
from model import Transformer
from dna_model import CNNModel
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.promoter_dataset import PromoterDataset 
from selene_sdk.utils import NonStrandSpecific
from promoter_model import PromoterModel
from utils.sei import Sei
from utils.esm import upgrade_state_dict
import pandas as pd

# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    if torch.cuda.is_available():
        device='cuda:0'
        print("Using gpu")
    else:
        device='cpu'
        print("using cpu")

    torch.manual_seed(42)
    def get_sei_profile(seq_one_hot,seifeatures,sei,device):
        seq_one_hot = seq_one_hot[:, :, :4] 
        B, L, K = seq_one_hot.shape
        sei_inp = torch.cat([torch.ones((B, 4, 1536), device=device) * 0.25,
                             seq_one_hot.transpose(1, 2),
                             torch.ones((B, 4, 1536), device=device) * 0.25], 2) # batchsize x 4 x 4,096
        with torch.no_grad():
            sei_out = sei(sei_inp).cpu().detach().numpy() # batchsize x 21,907
        sei_out = sei_out[:,seifeatures[1].str.strip().values == 'H3K4me3'] # batchsize x 2,350
        predh3k4me3 = sei_out.mean(axis=1) # batchsize
        return predh3k4me3

    scheduler = PolynomialConvexScheduler(n=2)
    path = MixtureDiscreteProbPath(scheduler=scheduler)
    seifeatures = pd.read_csv('data/promoter_design/target.sei.names', sep='|', header=None)
    #class WrappedModel(ModelWrapper):
    #    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
    #        return torch.softmax(self.model(x, t), dim=-1)

    if cfg.source_distribution == "uniform":
        added_token = 0
    elif cfg.source_distribution == "mask":
        mask_token = cfg.model.vocab_size  # tokens starting from zero
        added_token = 1
    else:
        raise NotImplementedError
        
    # additional mask token
    cfg.model.vocab_size += added_token
    seq_length=1024
        
    checkpoint_path = "/home/hmuhammad/flow/checkpoints/promoter_masked_epoch_500.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device),weights_only=False)
    probability_denoiser = PromoterModel(vocab_size=cfg.model.vocab_size).to(device)
    probability_denoiser.load_state_dict(checkpoint['model_state_dict'])
    probability_denoiser.eval()  # Set to evaluation mode
    wrapped_probability_denoiser = WrappedModel(probability_denoiser)
    solver = MixtureDiscreteEulerSolver(model=wrapped_probability_denoiser, path=path, vocabulary_size=cfg.model.vocab_size)

    sei = NonStrandSpecific(Sei(4096, 21907))
    sei.load_state_dict(upgrade_state_dict(
        torch.load('data/promoter_design/best.sei.model.pth.tar', map_location=device, weights_only=False)['state_dict'],
        prefixes=['module.']))
    sei.to(device)
    generator = np.random.default_rng(seed=137)

    nfe = 100
    step_size = 1 / nfe

    safe_sampling = True
    batch_size = 256
    dim = seq_length
    epsilon=1e-3
    n_plots = 9
    linspace_to_plot = torch.linspace(0,  1 - epsilon, n_plots)

    test_ds = PromoterDataset(split="test", n_tsses=100000, rand_offset=0) #pytorch dataset object, 7497 sequences for test
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4, pin_memory=True,persistent_workers=True,prefetch_factor=4)
    num_batches = len(test_loader)
    total_mse=0
    for batch in test_loader:
        seq_one_hot = batch[:,:,:4].to(device) #(B,1024,4)
        seq = torch.argmax(seq_one_hot, dim=-1) # (B,1024)
        signal = batch[:, :, 4:5].to(device)
        sei_profile=get_sei_profile(seq_one_hot,seifeatures=seifeatures,sei=sei,device=device)
        if cfg.source_distribution == "uniform":
            x_init = torch.randint(size=(batch.shape[0], dim), high=cfg.model.vocab_size, device=device)
        elif cfg.source_distribution == "mask":
            x_init = (torch.zeros(size=(batch.shape[0], dim),device=device) + mask_token).long() 
        sol = solver.sample(x_init=x_init,
                    step_size=step_size,
                    verbose=True,
                    return_intermediates=True,
                    time_grid=linspace_to_plot,
                    label=signal)
        seq_pred=sol[-1] # [n_samples, seq_length]
        seq_pred_one_hot = torch.nn.functional.one_hot(seq_pred, num_classes=cfg.model.vocab_size).float()
        sei_profile_pred = get_sei_profile(seq_pred_one_hot,seifeatures=seifeatures,sei=sei,device=device)
        batch_mse = ((sei_profile - sei_profile_pred) ** 2).mean()
        print(batch_mse)
        total_mse+=batch_mse.item()

    mean_mse = total_mse / num_batches
    print(mean_mse)

        

if __name__ == "__main__":
    main()
