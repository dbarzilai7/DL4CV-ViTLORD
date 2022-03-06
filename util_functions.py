import random
import sys
import numpy as np
import torch
import yaml

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGENET_NORMALIZATION_VALS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_conf(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed):
    seed = seed
    if seed == -1:
        seed = np.random.randint(2 ** 32 - 1, dtype=np.int64)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f'running with seed: {seed}.')