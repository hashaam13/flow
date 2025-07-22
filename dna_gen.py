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
save_dir = "/home/hmuhammad/flow/data"


# Load vocabulary (with full path)
with open(f"{save_dir}/DeepFlyBrain_data.pkl", "rb") as f:            
    data = pickle.load(f)#dict with keys:['train_data','y_train','valid_data','y_valid','test_data', 'y_test']
train_data = data['train_data'] #numpy array (83726, 500, 4)
y_train = data['y_train'] #numpy array (83726, 81)
#print(train_data[0][:10])
#print(y_train[0])

seqs = torch.argmax(torch.from_numpy(copy.deepcopy(train_data)), dim=-1) #numpy array (83726, 500)
clss = torch.argmax(torch.from_numpy(copy.deepcopy(y_train)), dim=-1 ) #numpy array (83726)
vocab = 4


