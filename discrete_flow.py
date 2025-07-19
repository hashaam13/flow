import torch
import time
import os
import pickle
from torch import nn, Tensor
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from models import MLP1

# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL

if torch.cuda.is_available():
    device='cuda:0'
    print("Using gpu")
else:
    device='cpu'
    print("using cpu")

torch.manual_seed(42)

from tokenizers import Tokenizer, models, trainers
save_dir = "/home/hmuhammad/flow/data"
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Load dataset and tokenizer
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize all samples and save
def tokenize_and_save(split, max_length=512):
    encoded_data = tokenizer(
        dataset[split]["text"],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    
    # Save token IDs, attention masks, and original text
    torch.save({
        "input_ids": encoded_data["input_ids"],
        "attention_mask": encoded_data["attention_mask"],
        "texts": dataset[split]["text"]  # Optional: save original text
    }, os.path.join(save_dir, f"imdb_{split}_tokenized.pt"))

# Save train/test splits
# tokenize_and_save("train")
# tokenize_and_save("test")

train_data = torch.load(f"{save_dir}/imdb_train_tokenized.pt")  # dictionary {input_ids,attention_mask,texts}
test_data = torch.load(f"{save_dir}/imdb_test_tokenized.pt")    

# Load vocabulary (with full path)
with open(f"{save_dir}/imdb_vocab.pkl", "rb") as f:            # Updated path
    vocab = pickle.load(f)
    id_to_token = vocab["id_to_token"]

# Example usage (unchanged)
batch_input_ids = train_data["input_ids"]    # [samples,sequences_length=512]

# print("Sample Token IDs:", batch_input_ids[0])
#  print("Decoded Text:", " ".join([id_to_token[id.item()] for id in batch_input_ids[0] if id != tokenizer.pad_token_id]))
batch_size = 32
vocab_size = tokenizer.vocab_size
epsilon = 1e-3
hidden_dim=64
seq_length=512
lr=0.001

# instantiate a convex path object
scheduler = PolynomialConvexScheduler(n=2)
path = MixtureDiscreteProbPath(scheduler=scheduler)

#loss function
loss_fn = MixturePathGeneralizedKL(path=path)

probability_denoiser = MLP1(input_dim=vocab_size, time_dim=1, hidden_dim=hidden_dim, length=seq_length).to(device)
optim=optim.Adam(probability_denoiser.parameters(),lr=lr)


dataloader = DataLoader(
    dataset=batch_input_ids,
    batch_size=batch_size,
    shuffle=True
    )

for i,data in enumerate(dataloader):
    x_1=data.to(device) # (batch_size,seq_length=512)
    x_0=torch.randint_like(x_1,high=vocab_size,device=device)
    t = torch.rand(x_1.shape[0]).to(device)*(1-epsilon)
    # sample probability path
    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

    logits = probability_denoiser(x=path_sample.x_t, t=path_sample.t)
    # discrete lfow matching generalizedKL loss
    loss = loss_fn(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t)
    loss.backward()
    optim.step()
    break
