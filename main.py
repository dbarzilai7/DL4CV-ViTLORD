import torchvision
from torchvision import transforms, datasets
import models
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from losses import *
import yaml
import random
import pandas as pd
from dataloader import *
from datetime import datetime

from matplotlib import pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_PATH = "./conf.yaml"
IMAGES_TO_USE = 1600
CONTENT_CODE_LEN = 28

def load_conf():
  with open(CONF_PATH, "r") as f:
      cfg = yaml.safe_load(f)
  seed = cfg['seed']
  if seed == -1:
      seed = np.random.randint(2 ** 32 - 1, dtype=np.int64)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  print(f'running with seed: {seed}.')
  return cfg

def train_model(model, tboard_name, loss_func, train_loader, device, cfg):
    writer = SummaryWriter(log_dir='logs/' + tboard_name)
    writer.add_text('TrainConfig', str(cfg))

    epochs, lr, noise_std = cfg['epochs'], cfg['lr'], cfg['noise_std']
    optimizer = optim.Adam(model.parameters(), lr=lr)
 
    # prepare the data
    dataset_size = len(train_loader.dataset.dataset.data)
    num_classes = len(train_loader.dataset.dataset.classes)
    class_codes = torch.normal(0.5, noise_std, (num_classes, num_classes)).to(device)
    content_codes = torch.normal(0.5, noise_std, (dataset_size, CONTENT_CODE_LEN)).to(device)
    
    # set up some variables for the visualizations
    display_contents = train_loader.dataset.indices[:4]
    display_classes = [0, 1, 2, 3]

    # prepare model
    model = model.to(device)
    
    # sets up some stuff for visualization
    sample_content_images = [train_loader.dataset.dataset[i][0] for i in display_contents]
    c, h, w = sample_content_images[0].shape
    labels = list(train_loader.dataset.dataset.targets)
    sample_class_indices = [labels.index(i) for i in display_classes]
    samples_classes_ims = [train_loader.dataset.dataset[i][0] for i in sample_class_indices]
    tboard_classses = torch.cat([torch.zeros(1, h, w)] + samples_classes_ims).unsqueeze(1).to(device)
    tboard_contents = torch.cat([train_loader.dataset.dataset[i][0] for i in display_contents]).unsqueeze(1).to(device)
    
    tboard_batch = torch.zeros(((len(display_classes) + 1) * (len(display_contents) + 1), 1, h, w), device=device)
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
        for index, value in losses_means.items():
          writer.add_scalar(index, value, global_step=epoch)
        writer.flush()
        print("Epoch: {}, loss: {}\n".format(epoch, losses_means.loc['loss']))
    
    writer.add_hparams(cfg, {'hparam/' + index: value for index, value in losses_means.items()})

    writer.close()

if __name__ == "__main__":
  cfg = load_conf()
  dataset_name, criterion_name, batch_size = cfg['dataset'], cfg['criterion'], cfg['batch_size']

  dataloader = load_datasets(dataset_name, IMAGES_TO_USE, batch_size)
  c, h, w = dataloader.dataset.dataset[0][0].shape
  num_classes = len(dataloader.dataset.dataset.classes)

  criterion = get_criterion(criterion_name, cfg)

  model = models.GeneratorBasic(CONTENT_CODE_LEN, 4, num_classes, (batch_size, c, h, w ))

  log_name = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
  train_model(model, log_name, criterion, dataloader, DEVICE, cfg)

