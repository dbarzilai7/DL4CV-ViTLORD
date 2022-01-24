import torch
import torchvision
from torchvision import transforms, datasets
import numpy as np

MNIST_PATH = "./datasets/MNIST/"

class MNISTIndexed(torchvision.datasets.MNIST):
    def __getitem__(self, idx):
        data, target = super().__getitem__(idx)
        return data, target, idx

def load_mnist():
  # Define a transform to normalize the data
  transform = transforms.Compose([transforms.ToTensor()])

  # Download and load the data
  return MNISTIndexed(MNIST_PATH, download=True, train=True, transform=transform)

def load_datasets(name, max_images_to_use, batch_size):
  if name == "MNIST":
    dataset = load_mnist()
  else:
    print("Dataset not supported")

  len_dataset = len(dataset.data)
  max_images_to_use = min(max_images_to_use, len_dataset)
  used_indices = np.random.randint(0, len_dataset, max_images_to_use)
  subsample = torch.utils.data.Subset(dataset, used_indices)
  return torch.utils.data.DataLoader(subsample, batch_size=batch_size, shuffle=True)
