import torch
import copy
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
from torch.utils.data import DataLoader,Dataset
import math
import numpy as np
from models import MLP1,TransformerDenoiser
from dna_model import CNNModel
from model import Transformer
import hydra
from omegaconf import DictConfig, OmegaConf



# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from torch.nn.parallel import DataParallel

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg, DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if torch.cuda.is_available():
        device='cuda:0'
        print("Using gpu")
    else:
        device='cpu'
        print("using cpu")

    torch.manual_seed(42)
    np.random.seed(42)
    save_dir = "/home/hmuhammad/flow/data"


    # Load vocabulary (with full path)
    # 2 enhancer datasets, DeepFlyBrain_data.pkl and DeepMEL2_data.pkl 
    with open(f"{save_dir}/DeepFlyBrain_data.pkl", "rb") as f:            
        data = pickle.load(f)                                #dict with keys:['train_data','y_train','valid_data','y_valid','test_data', 'y_test']
    train_data = data['train_data']                          #numpy array (83726, 500, 4) for DeepFlyBrain data, (70892, 500, 4) for DeepMEL2 data
    y_train = data['y_train']                                #numpy array (83726, 81) for DeepFlyBrain data, (70892, 47) for DeepMEL2 data,

    seqs = torch.argmax(torch.from_numpy(copy.deepcopy(train_data)), dim=-1) #numpy array (83726, 500)
    clss = torch.argmax(torch.from_numpy(copy.deepcopy(y_train)), dim=-1 ) + 1 #numpy array (83726)

    class SequenceDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = sequences
            self.labels = labels
            
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx]
        
    def get_masked_classes(clss, cfg_drop_prob=0.1):
        mask = torch.rand_like(clss.float()) > cfg_drop_prob  # 1=keep, 0=drop
        return clss * mask.long()  # Keeps original class or sets to 0 (unconditional)

    dataset = SequenceDataset(seqs, clss) # create dataset
    batch_size = 2048
    vocab_size = 4
    epsilon = 1e-3
    hidden_dim=64
    seq_length=500
    lr=0.0005 # 5e-4
    epochs=220 # 918 for flybrain,1084 for deeplMEL2
    n_iters=300000
    warmup=500
    clip_grad = True

    # instantiate a convex path object
    scheduler = PolynomialConvexScheduler(n=2)
    path = MixtureDiscreteProbPath(scheduler=scheduler)

    #loss function
    loss_fn = MixturePathGeneralizedKL(path=path)

    #probability_denoiser = MLP1(input_dim=vocab_size, time_dim=1, hidden_dim=hidden_dim, length=seq_length).to(device)
    #probability_denoiser = TransformerDenoiser(vocab_size=vocab_size,seq_length=seq_length,d_model=256,nhead=8, num_layers=8).to(device)
    probability_denoiser = CNNModel(vocab_size = vocab_size, hidden_dim=128, num_cnn_stacks=4,p_dropout=0.1,num_classes=81).to(device)
    #probability_denoiser = Transformer(vocab_size=4,masked=False,hidden_size=128,dropout=0.1,n_blocks=8,cond_dim=128,n_heads=8).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        probability_denoiser = DataParallel(probability_denoiser, device_ids=[0,1])
    optimizer=optim.Adam(probability_denoiser.parameters(),lr=lr)

    dataloader = DataLoader(
        dataset=dataset,
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
    n_iter = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        probability_denoiser.train()
        
        for i, (data,y) in enumerate(dataloader):
            n_iter +=1
            optimizer.zero_grad()
            
            x_1 = data.to(device)
            y = get_masked_classes(clss=y).to(device)
            x_0 = torch.randint_like(x_1, high=vocab_size, device=device)
            t = torch.rand(x_1.shape[0]).to(device) * (1 - epsilon)
            
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            logits = probability_denoiser(x=path_sample.x_t, t=path_sample.t, cls=y)
            loss = loss_fn(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t)
            
            # if clip_grad:
            #     torch.nn.utils.clip_grad_norm_(probability_denoiser.parameters(), 1.0)
            # if warmup > 0:
            #     for g in optimizer.param_groups:
            #         g["lr"] = lr * np.minimum(n_iter / warmup, 1.0)


            loss.backward()
            optimizer.step()
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
                'optimizer_state_dict': optimizer.state_dict(),
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
