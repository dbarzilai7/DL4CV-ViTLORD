import torch
import torchvision

class MNISTIndexed(torchvision.datasets.MNIST):
    def __getitem__(self, idx):
        data, target = super().__getitem__(idx)
        return data, target, idx
