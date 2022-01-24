import torchvision
from torchvision import transforms, datasets
import models
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from dataloader import MNISTIndexed
from losses import LossG, NaiveLoss
import yaml
import random
import pandas as pd

from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_SIZE = 60000
IMAGES_TO_USE = 1600
CONTENT_CODE_LEN = 28
BATCH_SIZE = 64

with open("./conf.yaml", "r") as f:
    cfg = yaml.safe_load(f)
seed = cfg['seed']
if seed == -1:
    seed = np.random.randint(2 ** 32 - 1, dtype=np.int64)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
print(f'running with seed: {seed}.')


data_path = "./datasets/MNIST/"

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              ])

# Download and load the data
mnist_data = MNISTIndexed(data_path, download=True, train=True, transform=transform)
used_indices = np.random.randint(0, DATASET_SIZE, IMAGES_TO_USE)
mnist_subsample = torch.utils.data.Subset(mnist_data, used_indices)
mnist_dataloader = torch.utils.data.DataLoader(mnist_subsample, batch_size=BATCH_SIZE, shuffle=True)

criterion = LossG(cfg)

model = models.GeneratorBasic(CONTENT_CODE_LEN, 4, 10, (BATCH_SIZE, 1, 28, 28))

def train_model(model, tboard_name, loss_func, train_loader, device, epochs=50, lr=1e-3, noise_std=0.3, reg_factor=1e-6):
    writer = SummaryWriter(log_dir='logs/' + tboard_name)
    optimizer = optim.Adam(model.parameters(), lr=lr)
 
    # prepare the data
    # TODO play with initizalization and refactor?
    class_codes = torch.normal(0.5, noise_std, (10, 10)).to(device)
    content_codes = torch.normal(0.5, noise_std, (DATASET_SIZE, CONTENT_CODE_LEN)).to(device)
    
    # set up some variables for the visualizations
    display_contents = used_indices[:4]
    display_classes = [0, 1, 2, 3]

    # prepare model
    model = model.to(device)
    
    # sets up some stuff for visualization
    sample_content_images = [train_loader.dataset.dataset[i][0] for i in display_contents]
    labels = list(mnist_dataloader.dataset.dataset.targets)
    sample_class_indices = [labels.index(i) for i in display_classes]
    samples_classes_ims = [train_loader.dataset.dataset[i][0] for i in sample_class_indices]
    tboard_classses = torch.cat([torch.zeros(1, 28, 28)] + samples_classes_ims).unsqueeze(1).to(device)
    tboard_contents = torch.cat([train_loader.dataset.dataset[i][0] for i in display_contents]).unsqueeze(1).to(device)
    
    tboard_batch = torch.zeros(((len(display_classes) + 1) * (len(display_contents) + 1), 1, 28, 28), device=device)
    non_first_col = np.arange(tboard_batch.shape[0])
    non_first_col = non_first_col[non_first_col % (len(display_contents) + 1) != 0]
    tboard_batch[:tboard_batch.shape[0]:len(display_classes)+1, ...] = tboard_classses
    
    # start of train
    for epoch in range(epochs):
        model.train()

        all_losses = []
        for data_row in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            images, labels, indices = data_row
            images = images.to(device)

            # create input for network
            cur_content, cur_class = content_codes[indices], class_codes[labels]
            cur_content.requires_grad_(True)
            cur_class.requires_grad_(True)
            noise = torch.randn(CONTENT_CODE_LEN, device=device) * noise_std
            inputs = torch.cat((cur_class, cur_content + noise), 1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            
            losses = loss_func(torch.cat([outputs, outputs, outputs], dim=1), torch.cat([images, images, images], dim=1), cur_content)
            losses['loss'].backward()
            optimizer.step()

            # statistics
            all_losses.append(losses)
            if len(all_losses) > 1:
              break
        
        model.eval()
        inputs = []
        for disp_classes in display_classes:
            for disp_contents in display_contents:
                inputs.append(torch.cat((class_codes[disp_classes], content_codes[disp_contents])).unsqueeze(0))
        outputs = model(torch.cat(inputs, 0))
        tboard_batch[non_first_col, ...] = torch.cat((tboard_contents, outputs))
        
        img_grid = torchvision.utils.make_grid(tboard_batch,nrow=len(display_classes) + 1)
        writer.add_image('images',img_grid, global_step=epoch)

        losses_means = pd.DataFrame(all_losses).mean(axis=0)
        print(losses_means)
        for index, value in losses_means.items():
          writer.add_scalar(index, value, global_step=epoch)
        writer.flush()
        print("Epoch: {}, loss: {}\n".format(epoch, losses_means.loc['loss']))

    writer.close()

train_model(model, "MNIST_nL_bG_1", criterion, mnist_dataloader, device, epochs=50, lr=1e-3, noise_std=0.3, reg_factor=1e-6)

