import argparse
import copy
import numpy as np
import torch
from pathlib import Path
import os
import pickle
import json
import time
from dna_model import CNNModel

if torch.cuda.is_available():
    device='cuda:0'
    print("Using gpu")
else:
    device='cpu'
    print("using cpu")
seqs = np.load("output_samples/fb_uncond_masked_1060_256.npy")
gen_seqs=torch.from_numpy(copy.deepcopy(seqs)).to(device)

mask_token = 4   # assuming A,C,G,T = 0,1,2,3 and MASK=4
vocab_size = 4   # classifier expects only 4 bases
mask_positions = (gen_seqs == mask_token)
gen_seqs[mask_positions] = torch.randint(low=0, high=vocab_size, size=(mask_positions.sum().item(),)).to(device)
    # 2 enhancer datasets, DeepFlyBrain_data.pkl and DeepMEL2_data.pkl 
with open("/home/hmuhammad/flow/data/DeepFlyBrain_data.pkl", "rb") as f:            
    data = pickle.load(f)                                #dict with keys:['train_data','y_train','valid_data','y_valid','test_data', 'y_test']
    train_data = data['train_data']                          #numpy array (83726, 500, 4) for DeepFlyBrain data, (70892, 500, 4) for DeepMEL2 data
    y_train = data['y_train']                                #numpy array (83726, 81) for DeepFlyBrain data, (70892, 47) for DeepMEL2 data,

seqs1 = torch.argmax(torch.from_numpy(copy.deepcopy(train_data)), dim=-1) #numpy array (83726, 500)
clss = torch.argmax(torch.from_numpy(copy.deepcopy(y_train)), dim=-1 ) #numpy array (83726)
total_sequences = len(seqs1)
class0_indices = torch.where(clss == 0)[0]
random_indices = torch.randperm(total_sequences)[:10000]
random_seqs = seqs1[random_indices].to(device)
# Randomly choose 1000 of them
#selected_indices = class0_indices[torch.randperm(len(class0_indices))[:1000]]
selected_indices = class0_indices[:]

# Select the sequences
real_seqs = seqs1[selected_indices].to(device)
#real_seqs = seqs1.to(device)
#selected_indices = class0_indices[1000:2000]
#gen_seqs = seqs1[selected_indices]
                  
cls_clean_model = CNNModel(vocab_size = 4, hidden_dim=128, num_cnn_stacks=1,p_dropout=0,num_classes=81,classifier=True,clean_data=True).to(device)
#print(cls_clean_model)
ckpt_path = "/home/hmuhammad/flow/checkpoints/classifiers//clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt"
cls_ckpt = torch.load(ckpt_path,map_location=torch.device(device),weights_only=False)
state_dict = dict()
for k, v in cls_ckpt["state_dict"].items():
    new_k = k[6:]
    state_dict[new_k] = v

cls_clean_model.load_state_dict(state_dict)
#logits=cls_clean_model(real_seqs)
with torch.no_grad():
    real_embs=cls_clean_model(random_seqs,return_embedding=True)[1]
    gen_embs=cls_clean_model(gen_seqs,return_embedding=True)[1]
    #print("real embeddings",real_embs.shape)
    #print("gen embeddings",gen_embs.shape)
def sqrtm_torch(mat, eps=1e-6):
    """Matrix square root using eigen decomposition (Torch only)."""
    # Ensure symmetric
    mat = (mat + mat.T) / 2
    
    eigvals, eigvecs = torch.linalg.eigh(mat)  # symmetric eigendecomposition
    eigvals = torch.clamp(eigvals, min=eps)    # avoid negative/zero
    sqrt_eigvals = torch.sqrt(eigvals)
    sqrtm = (eigvecs * sqrt_eigvals.unsqueeze(0)) @ eigvecs.T
    return sqrtm

def calculate_frechet_distance_torch(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Torch implementation of Frechet Distance."""
    diff = mu1 - mu2
    covmean = sqrtm_torch(sigma1 @ sigma2, eps=eps)

    fid = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean)
    return fid

def compute_fid_torch(real_embs, gen_embs, eps=1e-6):
    """
    real_embs, gen_embs: torch.Tensor of shape [N, d]
    Returns: scalar FID (torch.Tensor)
    """
    mu_real = real_embs.mean(dim=0)
    mu_gen = gen_embs.mean(dim=0)

    # unbiased covariance
    sigma_real = torch.cov(real_embs.T)
    sigma_gen = torch.cov(gen_embs.T)

    return calculate_frechet_distance_torch(mu_real, sigma_real, mu_gen, sigma_gen, eps=eps)

# Example:
fid_score = compute_fid_torch(real_embs, gen_embs)
print("FID:", fid_score.item())
#print("embedding shape",real_embs.shape)

#pred_class=torch.argmax(logits,dim=1)
#ones= (pred_class==0).sum().item()
#print(ones)