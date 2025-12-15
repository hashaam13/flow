import torch
import torch.nn.functional as F
import random
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
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # When creating test_loader, set a generator
    g = torch.Generator()
    g.manual_seed(42)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)



    scheduler = PolynomialConvexScheduler(n=2)
    path = MixtureDiscreteProbPath(scheduler=scheduler)
    generalized_kl_fn = MixturePathGeneralizedKL(
    path=path,
    reduction="none"
    )

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

    nfe = 100
    step_size = 1 / nfe

    safe_sampling = True
    batch_size = 256
    dim = seq_length

    test_ds = PromoterDataset(split="test", n_tsses=100000, rand_offset=0) #pytorch dataset object, 7497 sequences for test
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4, pin_memory=True,persistent_workers=True,prefetch_factor=4,generator=g,worker_init_fn=seed_worker)
    num_batches = len(test_loader)
    print(num_batches)

    num_batches_run=2

    n_discretization = 1024
    n_samples = 4          # usually 1–4 is enough
    epsilon = 1e-3
    total_elbo = 0.0
    total_loss=0.0
    total_tokens = 0
    num_batches_run = 2

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            #if i >= num_batches_run:
            #    break

            seq_one_hot = batch[:, :, :4].to(device)  # (B, L, 4)
            x_1 = torch.argmax(seq_one_hot, dim=-1)   # (B, L)
            signal = batch[:, :, 4:5].to(device)
            B, L = x_1.shape

            elbo = torch.zeros(B, device=device)

            for _ in range(n_samples):
                # sample time t ~ Uniform(0,1)
                t = torch.rand(B, device=device) * (1 - epsilon)

                # sample x_0
                if cfg.source_distribution == "uniform":
                    x_0 = torch.randint(
                        high=cfg.model.vocab_size,
                        size=x_1.shape,
                        device=device
                    )
                elif cfg.source_distribution == "mask":
                    x_0 = torch.full_like(x_1, mask_token)

                # forward path: x_t ~ p_t(. | x_1)
                path_sample=path.sample(t=t, x_0=x_0, x_1=x_1)

                # model prediction
                logits = probability_denoiser(x=path_sample.x_t, t=path_sample.t, signal=signal)

                # accumulate ELBO
                #elbo -= generalized_kl_fn(logits=logits,x_1=x_1, x_t=path_sample.x_t,t=path_sample.t).sum(dim=1)
                loss = F.cross_entropy(
                logits.transpose(1, 2),  # (B, vocab, L)
                x_1,                     # target
                reduction='none'
                )                             # (B, L)
                print(loss.sum().item())

                total_loss += loss.sum().item()
                total_tokens += B * L

            #elbo /= n_samples

            #total_elbo += elbo.sum().item()
            #total_tokens += B * L

            #print(f"Batch {i}: mean ELBO/token = {elbo.mean().item() / L:.4f}")
    #mean_elbo_per_token = total_elbo / total_tokens
    #perplexity = math.exp(-mean_elbo_per_token)
    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)

    print(f"Denoising perplexity: {perplexity:.4f}")

    #print(f"Test perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main() 