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
tokenize_and_save("train")
tokenize_and_save("test")

# Save vocabulary for decoding later
with open(os.path.join(save_dir, "imdb_vocab.pkl"), "wb") as f:
    pickle.dump({
        "vocab": tokenizer.get_vocab(),
        "id_to_token": {v: k for k, v in tokenizer.get_vocab().items()}
    }, f)


train_data = torch.load(f"{save_dir}/imdb_train_tokenized.pt")  # Updated path
test_data = torch.load(f"{save_dir}/imdb_test_tokenized.pt")    # Updated path

# Load vocabulary (with full path)
with open(f"{save_dir}/imdb_vocab.pkl", "rb") as f:            # Updated path
    vocab = pickle.load(f)
    id_to_token = vocab["id_to_token"]

# Example usage (unchanged)
batch_input_ids = train_data["input_ids"][:32]  # First 32 samples
batch_attention_mask = train_data["attention_mask"][:32]

print("Sample Token IDs:", batch_input_ids[0])
print("Decoded Text:", 
      " ".join([id_to_token[id.item()] for id in batch_input_ids[0] if id != tokenizer.pad_token_id]))