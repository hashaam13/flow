import torch
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
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from models import MLP1,TransformerDenoiser

# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from torch.nn.parallel import DataParallel

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
batch_size = 64
vocab_size = tokenizer.vocab_size
epsilon = 1e-3
hidden_dim=64
seq_length=512
lr=0.0001
epochs=40

# instantiate a convex path object
scheduler = PolynomialConvexScheduler(n=2)
path = MixtureDiscreteProbPath(scheduler=scheduler)

#loss function
loss_fn = MixturePathGeneralizedKL(path=path)

#probability_denoiser = MLP1(input_dim=vocab_size, time_dim=1, hidden_dim=hidden_dim, length=seq_length).to(device)
probability_denoiser = TransformerDenoiser(vocab_size=vocab_size,seq_length=seq_length,d_model=256,nhead=8, num_layers=8).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    probability_denoiser = DataParallel(probability_denoiser, device_ids=[0,1])
optim=optim.Adam(probability_denoiser.parameters(),lr=lr)

dataloader = DataLoader(
    dataset=batch_input_ids,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1
    )
# 1. Setup plot directory
plot_dir = "/home/hmuhammad/flow/plots"
os.makedirs(plot_dir, exist_ok=True)  # Create if doesn't exist
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 2. Initialize tracking
train_losses = []
best_loss = float('inf')
start_time=time.time()
for epoch in range(epochs):
    epoch_loss = 0.0
    probability_denoiser.train()
    
    for i, data in enumerate(dataloader):
        optim.zero_grad()
        
        x_1 = data.to(device)
        x_0 = torch.randint_like(x_1, high=vocab_size, device=device)
        t = torch.rand(x_1.shape[0]).to(device) * (1 - epsilon)
        
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        logits = probability_denoiser(x=path_sample.x_t, t=path_sample.t)
        loss = loss_fn(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t)
        
        loss.backward()
        optim.step()
        epoch_loss += loss.item()

        # Print batch progress
        if i % 100 == 0:
            print(f"Epoch {epoch} | Batch {i} | Avg Loss: {epoch_loss/(i+1):.4f}")

    # Store epoch loss
    avg_epoch_loss = epoch_loss / len(dataloader)
    train_losses.append(avg_epoch_loss)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Time: {time.time()-start_time:.2f}s')
    
    if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': probability_denoiser.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')
    
    # Save live updating plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.title(f"Training Loss (Epoch {epoch+1})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    # Save with timestamp and epoch number
    plot_path = f"{plot_dir}/loss_{timestamp}_epoch{epoch}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {plot_path}")
    print(f"Epoch {epoch} Complete | Loss: {avg_epoch_loss:.4f}")

# Save final loss curve
final_plot_path = f"{plot_dir}/loss_{timestamp}_final.png"
plt.figure(figsize=(10, 6))
plt.plot(train_losses, 'b-', linewidth=2)
plt.title("Final Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(final_plot_path, dpi=300)
plt.close()

print(f"\nTraining complete! All plots saved to {plot_dir}")
print(f"Final loss curve: {final_plot_path}")