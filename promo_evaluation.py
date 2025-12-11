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
device = 'cuda:0'

def get_sei_profile(seq_one_hot,seifeatures,sei,device):
    seq_one_hot = seq_one_hot[:, :, :4] 
    B, L, K = seq_one_hot.shape
    sei_inp = torch.cat([torch.ones((B, 4, 1536), device=device) * 0.25,
                             seq_one_hot.transpose(1, 2),
                             torch.ones((B, 4, 1536), device=device) * 0.25], 2) # batchsize x 4 x 4,096
    with torch.no_grad():
        sei_out = sei(sei_inp).cpu().detach().numpy() # batchsize x 21,907
    return sei_out

seifeatures = pd.read_csv('data/promoter_design/target.sei.names', sep='|', header=None)

# Load SEI model
sei = NonStrandSpecific(Sei(4096, 21907))
sei.load_state_dict(torch.load('data/promoter_design/best.sei.model.pth.tar', map_location=device)['state_dict'])
sei.to(device)
sei.eval()

# Load your generated samples
test_ds = PromoterDataset(split="test", n_tsses=100000, rand_offset=0) #pytorch dataset object, 7497 sequences for test
test_loader = DataLoader(test_ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True,persistent_workers=True,prefetch_factor=4)
allsamples = np.load('allsamples.npy')  # shape: (5, 7497, 1024)
num_samples, num_seqs, seq_len = allsamples.shape

batch_size = 128  # process SEI in batches to avoid memory issues
allsamples_pred = np.zeros((5, num_seqs, 21907))
testsamples_pred = np.zeros(( num_seqs, 21907))

with torch.no_grad():
    for j in range(num_samples):  # 5 independent generations
        for i in range(0, num_seqs, batch_size):
            batch = allsamples[j, i:i+batch_size] # [Batch_size,seq_length]
            batch_tensor = torch.FloatTensor(batch).to(device)
            seq_pred_one_hot = torch.nn.functional.one_hot(batch, num_classes=4).float()
            sei_out = get_sei_profile(seq_pred_one_hot,seifeatures=seifeatures,sei=sei,device=device)
            allsamples_pred[j, i:i+batch_size] = sei_out

    for batch in test_loader:
        seq_one_hot = batch[:,:,:4].to(device) #(B,1024,4)
        sei_profile=get_sei_profile(seq_one_hot,seifeatures=seifeatures,sei=sei,device=device)
        testsamples_pred=testsamples_pred+sei_profile

# Compute H3K4me3 mean
h3k4_idx = seifeatures[1].str.strip().values == 'H3K4me3'
allsamples_pred_h3k4me3 = allsamples_pred[:, :, h3k4_idx].mean(axis=2)  # (5, 7497)

testseqs_pred_h3k4me3 = testsamples_pred[:, h3k4_idx].mean(axis=1)  # (7497)

# Compute MSE per generation
acc = []
for i in range(5):
    acc.append(((allsamples_pred_h3k4me3[i] - testseqs_pred_h3k4me3)**2).mean())
mean_mse = np.mean(acc)
stderr_mse = np.std(acc)/np.sqrt(4)

print("Mean MSE:", mean_mse, "StdErr:", stderr_mse)
