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
from models import MLP, FourierEncoder,MNISTUNet, WrappedModel
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


if torch.cuda.is_available():
    device='cuda:0'
    print("Using gpu")
else:
    device='cpu'
    print("using cpu")

torch.manual_seed(42)

transform_train = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Training data
trainset = torchvision.datasets.MNIST(
    root='/home/hmuhammad/flow/data',
    train=True,
    download=True,
    transform=transform_train
)
    
batch_size = 500  # You can adjust this based on your GPU memory
lr=0.0001
epochs=100
print_every=2000
t_embed_dim = 40
y_embed_dim = 40
trainloader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1  # For parallel data loading
)

#vf=MLP().to(device)
vf = MNISTUNet(
    channels = [32, 64, 128],
    num_residual_layers = 2,
    t_embed_dim = t_embed_dim,
    y_embed_dim = y_embed_dim,
    image_channels=1
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
        x_1 = x_1.to(device)
        y = y.to(device)
        x_0 = torch.randn_like(x_1).to(device)
        t = torch.rand(x_1.shape[0]).to(device)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        x_t=path_sample.x_t.to(device)
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

wrapped_vf = WrappedModel(vf)


step_size = 0.05

norm = cm.colors.Normalize(vmax=50, vmin=0)

batch_size = 11  # batch size
eps_time = 1e-2
T = torch.linspace(0,1,10).to(device)  # sample times
Y = torch.linspace(0,10,11,dtype=torch.int).to(device)
if len(Y) != batch_size:
    print("number of labels should match the batch size")

x_init = torch.randn((batch_size, 1, 32, 32), dtype=torch.float32, device=device)

solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
sol = solver.sample( time_grid=T,x_init=x_init, method='midpoint', label=Y,step_size=step_size,return_intermediates=True)  # sample from the model
print(sol.shape) # (sample_times, batch_size, channels, height,width)

# Denormalize the images (reverse the Normalize transform)
def denormalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    """Denormalize a tensor image with mean and std."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Multiply by std and add mean
    return tensor.clamp_(0, 1)  # Clamp to [0,1] range

# Add this after your visualization code
output_dir = "/home/hmuhammad/flow/output_samples/"

# Save final images
final_images = sol[-1].detach().cpu()
for i in range(batch_size):
    img = final_images[i]
    img_denorm = denormalize(img)  # Use your denormalize function
    plt.imshow(img_denorm.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(f"{output_dir}final_image_{i}.png", bbox_inches='tight', pad_inches=0)
    plt.close()

print(f"Saved {batch_size} images to {output_dir}")