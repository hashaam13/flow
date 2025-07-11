import torch
import time
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
import matplotlib.pyplot as plt
from matplotlib import cm
import torchvision


if torch.cuda.is_available():
    device='cuda:0'
    print("Using gpu")
else:
    device='cpu'
    print("using cpu")

torch.manual_seed(42)

cifar=torchvision.datasets.CIFAR10(root="/home/hmuhammad/flowgit /data",train=True,download=True)


