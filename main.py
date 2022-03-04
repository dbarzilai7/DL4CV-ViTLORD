import itertools

import torchvision
from torchvision import transforms, datasets
import models
from torch import optim
import torch
from numpy import copy
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from config import base_config
import modules
from evaluation import Evaluation
from losses import *
import yaml
import random
import pandas as pd
from dataloader import *
from datetime import datetime
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR

from matplotlib import pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_PATH = sys.argv[1]  # "./conf.yaml"
IMAGES_TO_USE = 500
CONTENT_CODE_LEN = 128
CLASS_CODE_LEN = 256


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


def train_model(model, tboard_name, loss_func, train_loader, device, cfg, config):
    writer = SummaryWriter(log_dir='logs/' + tboard_name)
    writer.add_text('TrainConfig', str(cfg))

    epochs = cfg['epochs']

    targets = np.unique(train_loader.dataset.dataset.targets)
    class_mapping = np.arange(np.amax(targets) + 1)
    class_mapping[targets] = np.arange(targets.size)

    evaluator = Evaluation(train_loader, device, writer, class_mapping)

    # prepare the data
    dataset_size = train_loader.dataset.indices.size
    num_classes = len(np.unique(train_loader.dataset.dataset.targets))
    # class_codes = torch.normal(0.5, noise_std, (num_classes, CLASS_CODE_LEN)).to(device)
    # class_codes = torch.FloatTensor(num_classes, CLASS_CODE_LEN).uniform_(-0.05, 0.05).to(device)
    # content_codes = torch.FloatTensor(dataset_size, CONTENT_CODE_LEN).uniform_(-0.05, 0.05).to(device)

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.Adam([
        {
            'params': itertools.chain(model.modulation.parameters(),
                                      model.generator.parameters()),
            'lr': config['train']['learning_rate']['generator']
        },
        {
            'params': itertools.chain(model.content_embedding.parameters(),
                                      model.class_embedding.parameters()),
            'lr': config['train']['learning_rate']['latent']
        }
    ], betas=(0.5, 0.999))
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_loader),
        eta_min=config['train']['learning_rate']['min']
    )
    # prepare model
    model = model.to(device)

    # sets up some stuff for visualization
    #c, h, w = train_loader.dataset.dataset[0][0].shape

    # start of train
    for epoch in range(epochs):
        # if cfg['swap_gen'] and epoch == cfg['warm_up_epochs']:
        #     model = models.GeneratorDone(CONTENT_CODE_LEN, 4, num_classes, (cfg['batch_size'], c, h, w))
        #     model = model.to(device)
        model.train()

        all_losses = []
        for data_row in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            images, labels, indices = data_row
            images = images.to(device)

            # create input for network
            # cur_content, cur_class = content_codes[indices.to(torch.long)], \
            #                          class_codes[class_mapping[labels.to(torch.long)]]
            # cur_content.requires_grad_(True)
            # cur_class.requires_grad_(True)
            # noise = torch.randn(CONTENT_CODE_LEN, device=device) * noise_std

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model((indices.to(torch.long)).to(device), (torch.from_numpy(class_mapping[labels])).to(device))

            losses = loss_func(outputs['img'], images, outputs['content_code'], epoch)

            losses['loss'].backward()
            optimizer.step()
            scheduler.step()
            # statistics
            all_losses.append({key: float(value) for key, value in losses.items()})

        evaluator.eval(model, epoch, all_losses)

    writer.close()


if __name__ == "__main__":
    print("STARTING")
    cfg = load_conf()
    dataset_name, criterion_name, batch_size = cfg['dataset'], cfg['criterion'], cfg['batch_size']

    dataloader = load_datasets(dataset_name, IMAGES_TO_USE, batch_size)
    c, h, w = dataloader.dataset.dataset[0][0].shape
    num_classes = len(np.unique(dataloader.dataset.dataset.targets))

    criterion = get_criterion(criterion_name, cfg).to(DEVICE)

    # model = models.GeneratorBasic(CONTENT_CODE_LEN, 4, num_classes, (batch_size, c, h, w))
    # model = modules.Generator(CONTENT_CODE_LEN, 4, num_classes, (batch_size, c, h, w))
    config = dict(
        img_shape=(h, w, c),
        n_imgs=dataloader.dataset.indices.size,
        n_classes=num_classes,
    )

    config.update(base_config)
    model = modules.LatentModel(config)
    model.init()
    log_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    train_model(model, log_name, criterion, dataloader, DEVICE, cfg, config)
