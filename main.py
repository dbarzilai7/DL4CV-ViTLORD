import itertools
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import modules
from evaluator import Evaluator
from losses import *
from dataloader import *
from datetime import datetime
from util_functions import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from matplotlib import pyplot as plt


def train_model(model, tboard_name, loss_func, train_loader, device, cfg):
    writer = SummaryWriter(log_dir='logs/' + tboard_name)
    writer.add_text('TrainConfig', str(cfg))

    epochs = cfg['epochs']

    targets = np.unique(train_loader.dataset.dataset.targets)
    class_mapping = np.arange(np.amax(targets) + 1)
    class_mapping[targets] = np.arange(targets.size)

    # set up class for evaluations and visualizations
    evaluator = Evaluator(train_loader, device, writer, class_mapping)

    # set optimizer and scheduler
    optimizer = optim.Adam([
        {
            'params': itertools.chain(model.modulation.parameters(),
                                      model.generator.parameters()),
            'lr': cfg['lr_generator']
        },
        {
            'params': itertools.chain(model.content_embedding.parameters(),
                                      model.class_embedding.parameters()),
            'lr': cfg['lr_latent_codes']
        }
    ], betas=(0.5, 0.999))
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_loader),
        eta_min=cfg['min_lr']
    )

    # prepare model
    model = model.to(device)

    # start of train
    for epoch in range(epochs):
        if cfg['swap_gen'] and epoch == cfg['warm_up_epochs']:
            model.reset_generator()

        model.train()

        all_losses = []
        for data_row in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            images, labels, indices = data_row
            images = images.to(device)

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

    dataloader = load_datasets(dataset_name, cfg['max_images_to_use'], batch_size)
    c, h, w = dataloader.dataset.dataset[0][0].shape
    num_classes = len(np.unique(dataloader.dataset.dataset.targets))

    criterion = get_criterion(criterion_name, cfg).to(DEVICE)

    model = modules.LatentModel(cfg, dataloader.dataset.indices.size, num_classes, h, w, c)
    model.init()

    log_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    train_model(model, log_name, criterion, dataloader, DEVICE, cfg)
