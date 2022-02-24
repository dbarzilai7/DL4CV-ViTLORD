import torchvision
from torchvision import transforms, datasets
import models
from torch import optim
import torch
from numpy import copy
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from losses import *
import yaml
import random
import pandas as pd
from dataloader import *
from datetime import datetime
import sys
from matplotlib import pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_PATH = sys.argv[1]  # "./conf.yaml"
IMAGES_TO_USE = 1600
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


def train_model(model, tboard_name, loss_func, train_loader, device, cfg):
    writer = SummaryWriter(log_dir='logs/' + tboard_name)
    writer.add_text('TrainConfig', str(cfg))

    epochs, lr, noise_std = cfg['epochs'], cfg['lr'], cfg['noise_std']
    optimizer = optim.Adam(model.parameters(), lr=lr)
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # prepare the data
    dataset_size = train_loader.dataset.indices.size
    num_classes = len(np.unique(train_loader.dataset.dataset.targets))
    # class_codes = torch.normal(0.5, noise_std, (num_classes, CLASS_CODE_LEN)).to(device)
    class_codes = torch.FloatTensor(num_classes, CLASS_CODE_LEN).uniform_(-0.05, 0.05).to(device)
    targets = np.unique(train_loader.dataset.dataset.targets)
    class_mapping = np.arange(np.amax(targets) + 1)
    for i, t in enumerate(targets):
        class_mapping[t] = i

    # content_codes = torch.normal(0.5, noise_std, (dataset_size, CONTENT_CODE_LEN)).to(device)
    content_codes = torch.normal(0, noise_std, (dataset_size, CONTENT_CODE_LEN)).to(device)

    # set up some variables for the visualizations
    display_contents = train_loader.dataset.indices[:4]
    labels_counts = np.unique(train_loader.dataset.dataset.targets, return_counts=True)
    display_classes = labels_counts[0][labels_counts[1].argsort()[-4:]]
    # prepare model
    model = model.to(device)

    # sets up some stuff for visualization
    sample_content_images = [(train_loader.dataset.dataset[i][0]) for i in display_contents]
    c, h, w = sample_content_images[0].shape
    labels = list(train_loader.dataset.dataset.targets)
    sample_class_indices = [labels.index(i) for i in display_classes]
    samples_classes_ims = [unorm(train_loader.dataset.dataset[i][0]).unsqueeze(0) for i in sample_class_indices]
    tboard_classes = torch.cat([torch.zeros(1, c, h, w)] + samples_classes_ims).to(device)
    tboard_contents = torch.cat([unorm(train_loader.dataset.dataset[i][0]).unsqueeze(0) for i in display_contents]).to(device)

    tboard_batch = torch.zeros(((len(display_classes) + 1) * (len(display_contents) + 1), c, h, w), device=device)
    non_first_col = np.arange(tboard_batch.shape[0])
    non_first_col = non_first_col[non_first_col % (len(display_contents) + 1) != 0]
    tboard_batch[:tboard_batch.shape[0]:len(display_classes) + 1, ...] = tboard_classes

    # start of train
    for epoch in range(epochs):
        model.train()

        all_losses = []
        for data_row in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            images, labels, indices = data_row
            images = images.to(device)

            # create input for network
            cur_content, cur_class = content_codes[indices.to(torch.long)], class_codes[
                class_mapping[labels.to(torch.long)]]
            cur_content.requires_grad_(True)
            cur_class.requires_grad_(True)
            noise = torch.randn(CONTENT_CODE_LEN, device=device) * noise_std
            inputs = torch.cat((cur_class, cur_content + noise), 1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if isinstance(model, models.GeneratorDone):
                outputs = model(cur_content + noise, cur_class)
                losses = loss_func(outputs, images, cur_content)
            else:
                outputs = model(inputs)
                losses = loss_func(torch.cat([outputs, outputs, outputs], dim=1),
                                   torch.cat([images, images, images], dim=1), cur_content)

            losses['loss'].backward()
            optimizer.step()

            # statistics
            all_losses.append({key: float(value) for key, value in losses.items()})

        model.eval()
        inputs = []
        input_classes = []
        input_contents = []
        with torch.no_grad():
            if isinstance(model, models.GeneratorDone):
                display_classes_norm = [(labels_counts[0].tolist()).index(cls) for cls in display_classes]
                for disp_classes in display_classes_norm:
                    for disp_contents in display_contents:
                        input_contents.append(content_codes[disp_contents].unsqueeze(0))
                        input_classes.append((class_codes[disp_classes].unsqueeze(0)))
                outputs = model(torch.cat(input_contents, 0), torch.cat(input_classes))
                tboard_batch[non_first_col, ...] = torch.cat((tboard_contents, outputs))

            else:
                for disp_classes in display_classes:
                    for disp_contents in display_contents:
                        inputs.append(torch.cat((class_codes[disp_classes], content_codes[disp_contents])).unsqueeze(0))
                outputs = model(torch.cat(inputs, 0))
                tboard_batch[non_first_col, ...] = torch.cat((tboard_contents, outputs))

            img_grid = torchvision.utils.make_grid(tboard_batch, nrow=len(display_classes) + 1)
            writer.add_image('images', img_grid, global_step=epoch)

            losses_means = pd.DataFrame(all_losses).mean(axis=0)
            for index, value in losses_means.items():
                writer.add_scalar(index, value, global_step=epoch)
            writer.flush()
            print("Epoch: {}, loss: {}\n".format(epoch, losses_means.loc['loss']))

    writer.add_hparams(cfg, {'hparam/' + index: value for index, value in losses_means.items()})

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
    model = models.GeneratorDone(CONTENT_CODE_LEN, 4, num_classes, (batch_size, c, h, w))

    log_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    train_model(model, log_name, criterion, dataloader, DEVICE, cfg)
