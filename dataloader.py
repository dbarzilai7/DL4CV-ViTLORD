import torch
import torchvision
from torchvision import transforms, datasets
import numpy as np

MNIST_PATH = "./datasets/MNIST/"
LFW_PATH = "./datasets/LFW/"


class MNISTIndexed(torchvision.datasets.MNIST):
    def __getitem__(self, idx):
        data, target = super().__getitem__(idx)
        return data, target, idx


class LFWIndexed(torchvision.datasets.LFWPeople):
    def __getitem__(self, idx):
        data, target = super().__getitem__(idx)
        return data, target, idx


def load_mnist():
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the data
    return MNISTIndexed(MNIST_PATH, download=True, train=True, transform=transform)


def load_LFW():
    transform = transforms.Compose([transforms.CenterCrop(160),
                                    transforms.PILToTensor(),
                                    transforms.ConvertImageDtype(torch.float)])
    #,transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    dataset = LFWIndexed(LFW_PATH, download=True, split='train', transform=transform)

    labels_counts = np.unique(dataset.targets, return_counts=True)
    most_prominent = labels_counts[1].argsort()[-8:-1]
    labels = labels_counts[0][most_prominent]
    good_indices = np.isin(dataset.targets, labels)
    dataset.data = list(np.array(dataset.data)[good_indices])
    dataset.targets = list(np.array(dataset.targets)[good_indices])

    return dataset


def load_datasets(name, max_images_to_use, batch_size):
    if name == "MNIST":
        dataset = load_mnist()
    elif name == "LFW":
        dataset = load_LFW()
    else:
        print("Dataset not supported")

    len_dataset = len(dataset.data)
    max_images_to_use = min(max_images_to_use, len_dataset)
    used_indices = np.random.randint(0, max_images_to_use, max_images_to_use)
    subsample = torch.utils.data.Subset(dataset, used_indices)
    return torch.utils.data.DataLoader(subsample, batch_size=batch_size, shuffle=True)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        normalizedTensor = torch.zeros_like(tensor)
        normalizedTensor[0] = (tensor[0] - self.mean[0]) / (self.std[0])
        normalizedTensor[1] = (tensor[1] - self.mean[1]) / (self.std[1])
        normalizedTensor[2] = (tensor[2] - self.mean[2]) / (self.std[2])

        return normalizedTensor
