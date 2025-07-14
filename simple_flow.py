import torch
import time
import os
from torch import nn, Tensor
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
import matplotlib.pyplot as plt
from matplotlib import cm
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from models import MLP, FourierEncoder,MNISTUNet


if torch.cuda.is_available():
    device='cuda:0'
    print("Using gpu")
else:
    device='cpu'
    print("using cpu")

torch.manual_seed(42)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Training data
trainset = torchvision.datasets.CIFAR10(
    root='/home/hmuhammad/flow/data',
    train=True,
    download=True,
    transform=transform_train
)

# Test data
testset = torchvision.datasets.CIFAR10(
    root='/home/hmuhammad/flow/data',
    train=False,
    download=True,
    transform=transform_test
)
    
batch_size = 128  # You can adjust this based on your GPU memory
lr=0.001
epochs=1
print_every=2000
t_embed_dim = 10
trainloader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2  # For parallel data loading
)

testloader = DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2)

#vf=MLP().to(device)
vf = MNISTUNet(
    channels = [32, 64, 128],
    num_residual_layers = 2,
    t_embed_dim = 40,
    y_embed_dim = 40,
).to(device)
path = AffineProbPath(scheduler=CondOTScheduler())
optim = torch.optim.Adam(vf.parameters(),lr=lr)
criterion = nn.MSELoss()

train_loss_history = []
start_time=time.time()
for epoch in range(epochs):
    running_loss = 0
    for i, data in enumerate(trainloader):
        optim.zero_grad()
        x_1, y = data
        x_0 = torch.randn_like(x_1).to(device)
        t = torch.rand(x_1.shape[0]).to(device)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        x_t=path_sample.x_t
        ts=path_sample.t
        u_pred = vf(x_t,t=t,y=y)
        u_target = path_sample.dx_t
        # print(u_pred.shape,u_target.shape)
        loss = criterion(u_pred,u_target)
        loss.backward()
        optim.step()
        running_loss += loss.item()
    
    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(trainloader)
    train_loss_history.append(epoch_loss)

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Time: {time.time()-start_time:.2f}s')
    
    # Save checkpoint every N epochs or at the end
    if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': vf.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')
    
    # Plot and save loss curve
    plt.figure()
    plt.plot(train_loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('plots/training_loss.png')
    plt.close()

# Save final model
final_model_path = 'checkpoints/final_model.pth'
torch.save(vf.state_dict(), final_model_path)
print(f'Saved final model to {final_model_path}')

# Plot final loss curve
plt.figure(figsize=(10, 6))
plt.plot(train_loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.legend()
plt.savefig('plots/final_training_loss.png')
plt.show()

