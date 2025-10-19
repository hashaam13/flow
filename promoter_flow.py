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
from promoter_model import PromoterModel
import hydra
from omegaconf import DictConfig, OmegaConf


from utils.promoter_dataset import PromoterDataset 
# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from torch.nn.parallel import DataParallel

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if torch.cuda.is_available():
        device='cuda:0'
        print("Using gpu")
    else:
        device='cpu'
        print("using cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    save_dir = cfg.train.save_dir


    train_ds = PromoterDataset(split="train", n_tsses=100000, rand_offset=10) #pytorch dataset object 88570 sequences
    #val_ds = PromoterDataset(split="test", n_tsses=100000, rand_offset=0)
    
    print("Len train_ds: ", len(train_ds))
    #print("Len val_ds: ", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4, pin_memory=True,persistent_workers=True,prefetch_factor=4)
    #val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=1)

    epsilon = cfg.epsilon
    seq_length=1024
    lr=cfg.train.lr # 5e-4
    epochs=cfg.train.epochs # 918 for flybrain,1084 for deeplMEL2
    warmup=cfg.train.warmup
    clip_grad = cfg.train.clip_grad

    # instantiate a convex path object
    scheduler = PolynomialConvexScheduler(n=2)
    path = MixtureDiscreteProbPath(scheduler=scheduler)

    #loss function
    loss_fn = MixturePathGeneralizedKL(path=path)


    if cfg.source_distribution == "uniform":
        added_token = 0
    elif cfg.source_distribution == "mask":
        mask_token = cfg.model.vocab_size  # tokens starting from zero
        added_token = 1
        print("masked source distribution")
    else:
        raise NotImplementedError
        
    # additional mask token
    cfg.model.vocab_size += added_token
    probability_denoiser = PromoterModel(vocab_size=cfg.model.vocab_size).to(device)
    optimizer=optim.AdamW(probability_denoiser.parameters(),lr=lr)

    # 1. Setup plot directory
    plot_dir = cfg.train.plot_dir
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
        
        for i, batch in enumerate(train_loader): #(B,1024,6)
            n_iter +=1
            optimizer.zero_grad()
            x_1 = batch[:,:,:4].to(device) #(B,1024,4)
            x_1 = torch.argmax(x_1,-1).to(device) #(B,1024)
            signal = batch[:,:,4:5].to(device) #(B,1024,1)
            if cfg.source_distribution == "uniform":
                x_0 = torch.randint_like(x_1, high=cfg.model.vocab_size, device=device)
            elif cfg.source_distribution == "mask":
                x_0 = torch.zeros_like(x_1) + mask_token              
            t = torch.rand(x_1.shape[0]).to(device) * (1 - epsilon)            
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            logits = probability_denoiser(x=path_sample.x_t, t=path_sample.t, signal=signal)
            loss = loss_fn(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t)
            
            if warmup > 0:
                 for g in optimizer.param_groups:
                     g["lr"] = lr * np.minimum(n_iter / warmup, 1.0)

            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(probability_denoiser.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

            # Print batch progress
            if i % 100 == 0:
                print(f"Epoch {epoch} | Batch {i} | Avg Loss: {epoch_loss/(i+1):.4f}")

        # Store epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Time: {time.time()-start_time:.2f}s')
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            checkpoint_path = f'checkpoints/promoter_masked_epoch_{epoch+1}.pth'
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

if __name__ == "__main__":
    main()
