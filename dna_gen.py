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
from models import MLP1, TransformerDenoiser,WrappedModel
from model import Transformer
from dna_model import CNNModel
import hydra
from omegaconf import DictConfig, OmegaConf

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

    scheduler = PolynomialConvexScheduler(n=2)
    path = MixtureDiscreteProbPath(scheduler=scheduler)
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
    seq_length=cfg.dataset.seq_length 
        
    checkpoint_path = "/home/hmuhammad/flow/checkpoints/cnn_masked_epoch_680.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device),weights_only=False)

    # 2. Initialize your model architecture first (must match original)
    # Assuming you have a ProbabilityDenoiser class defined elsewhere
    #probability_denoiser = MLP1(input_dim=vocab_size, time_dim=1, hidden_dim=hidden_dim, length=seq_length).to(device)
    #probability_denoiser = TransformerDenoiser(vocab_size=vocab_size,seq_length=seq_length,d_model=256,nhead=8, num_layers=8).to(device)

    if cfg.model.name == "CNN":
        probability_denoiser = CNNModel(vocab_size=cfg.model.vocab_size,hidden_dim=cfg.model.hidden_dim,num_cnn_stacks=cfg.model.num_cnn_stacks,p_dropout=0,num_classes=cfg.dataset.num_classes).to(device)
    if cfg.model.name == "Transformer":
        probability_denoiser = Transformer(vocab_size=cfg.model.vocab_size,masked=cfg.model.masked,hidden_size=cfg.model.hidden_dim,dropout=cfg.model.p_dropout,n_blocks=cfg.model.n_blocks,cond_dim=cfg.model.cond_dim,n_heads=cfg.model.n_heads).to(device)
    #probability_denoiser = Transformer(vocab_size=4,masked=False,hidden_size=128,dropout=0.1,n_blocks=8,cond_dim=128,n_heads=8).to(device)
    # 3. Load the state dict
    #probability_denoiser = nn.DataParallel(probability_denoiser)
    probability_denoiser.load_state_dict(checkpoint['model_state_dict'])
    probability_denoiser.eval()  # Set to evaluation mode

    wrapped_probability_denoiser = WrappedModel(probability_denoiser)
    solver = MixtureDiscreteEulerSolver(model=wrapped_probability_denoiser, path=path, vocabulary_size=cfg.model.vocab_size)
    nfe = 512
    step_size = 1 / nfe

    safe_sampling = True
    n_samples = 10000
    dim = seq_length
    epsilon=1e-3

    if cfg.source_distribution == "uniform":
        x_init = torch.randint(size=(n_samples, dim), high=cfg.model.vocab_size, device=device)
    elif cfg.source_distribution == "mask":
        x_init = (torch.zeros(size=(n_samples, dim),device=device) + mask_token).long()  


    y = torch.ones(n_samples,dtype=int,device=device)
    n_plots = 9
    linspace_to_plot = torch.linspace(0,  1 - epsilon, n_plots)

    sol = solver.sample(x_init=x_init,
                        step_size=step_size,
                        verbose=True,
                        return_intermediates=True,
                        time_grid=linspace_to_plot,
                        label=y,
                        cfg_scale=3)
    print(sol.shape) # [sampling_steps, n_samples, seq_length]
    final_seqs=sol[-1].cpu().numpy()
    np.save("output_samples/fb_class1_masked.npy",final_seqs)
    # Assuming sol is your tensor with shape [9, 2, 500]
    last_generation = sol[-1, 2]  # Gets last timestep (-1), first sample (0)
    last_generation = last_generation.cpu().numpy()
    decoded_text = []
    # Load tokenizer and decode
    order = {"A": 0, "C": 1, "G": 2, "T": 3}
    decode = {0: 'A', 1: "C", 2: "G", 3: "T"}
    for i in last_generation:
        decoded_text.append(decode[i])
    print("Last generated sequence (first sample):")
    print(decoded_text)

if __name__ == "__main__":
    main()
