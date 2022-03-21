""" Make training and any ops involving random reproducible """

import random

import numpy as np
import torch as Torch


def set_seed(seed=0, cudnn=False, torch=True):
    if cudnn and Torch.cuda.is_available():
        Torch.backends.cudnn.deterministic = True
        Torch.backends.cudnn.benchmark = False
    if torch:
        Torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
