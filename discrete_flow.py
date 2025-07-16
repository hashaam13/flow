import torch
import time
import os
from torch import nn, Tensor
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
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


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

trainset = AG_NEWS(split='train')
    
batch_size = 128  # You can adjust this based on your GPU memory
lr=0.0001
epochs=1
print_every=2000
t_embed_dim = 40
y_embed_dim = 40
for label,line in trainset:
    print(line)
