import itertools
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import modules
import random
from evaluator import Evaluator
from losses import *
from dataloader import *
from datetime import datetime
from util_functions import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from matplotlib import pyplot as plt

CONF_PATH = sys.argv[1]  # "./conf.yaml"


def train_model(model, optimizer, tboard_name, loss_func, train_loader, device, cfg, embedding_criterion=None):
    writer = SummaryWriter(log_dir='logs/' + tboard_name)
    writer.add_text('TrainConfig', str(cfg))

    epochs = cfg['epochs']

    targets = np.unique(train_loader.dataset.dataset.targets)
    class_mapping = np.arange(np.amax(targets) + 1)
    class_mapping[targets] = np.arange(targets.size)

    # set up class for evaluations and visualizations
    evaluator = Evaluator(train_loader, device, writer, class_mapping)

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
            model.reset_generator(device)

        model.train()

        all_losses = []
        prev_images, prev_labels, prev_indices = None, None, None
        for data_row in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            images, labels, indices = data_row
            images = images.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images, (indices.to(torch.long)).to(device),
                            images, (torch.from_numpy(class_mapping[labels])).to(device))

            losses = loss_func(outputs['img'], images, outputs['content_code'], outputs['class_code'], outputs, epoch)

            if prev_images is not None and embedding_criterion is not None and cfg['mix_codes_training'] \
                    and images.shape[0] == prev_images.shape[0]:
                outputs_mixed = model(images, (indices.to(torch.long)).to(device),
                                      prev_images, (torch.from_numpy(class_mapping[prev_labels])).to(device))

                losses['loss'] += embedding_criterion(outputs_mixed['img'], images, outputs_mixed['content_code'],
                                                      outputs_mixed['class_code'], outputs_mixed, epoch)['loss']

            prev_images, prev_labels, prev_indices = images, labels, indices

            losses['loss'].backward()
            optimizer.step()
            scheduler.step()

            # statistics
            all_losses.append({key: float(value) for key, value in losses.items()})

        evaluator.eval(model, epoch, all_losses)

    writer.close()


def get_model_and_optimizer(cfg):
    if cfg['model'].lower() == "decoderencoder":
        model = modules.DecoderEncoder(cfg, dataloader.dataset.indices.size, num_classes, h, w, c)
        optimizer = optim.Adam([
            {
                'params': itertools.chain(model.decoder.modulation.parameters(),
                                          model.decoder.generator.parameters()),
                'lr': cfg['lr_generator']
            },
            {
                'params': itertools.chain(model.decoder.embeddings.parameters()),
                'lr': cfg['lr_latent_codes']
            },
            {
                'params': itertools.chain(model.encoder.parameters()),
                'lr': cfg['lr_encoder']
            },
        ], betas=(0.5, 0.999))
    elif cfg['model'].lower() == "latent":
        model = modules.LatentModel(cfg, dataloader.dataset.indices.size, num_classes, h, w, c)
        optimizer = optim.Adam([
        {
            'params': itertools.chain(model.modulation.parameters(),
                                      model.generator.parameters()),
            'lr': cfg['lr_generator']
        },
        {
            'params': itertools.chain(model.embeddings.parameters()),
            'lr': cfg['lr_latent_codes']
        }
        ], betas=(0.5, 0.999))
    else:
        print("Invalid model name")
        exit(0)
    return model,optimizer


if __name__ == "__main__":
    print("STARTING")
    cfg = load_conf(CONF_PATH)
    set_seed(cfg['seed'])
    dataset_name, criterion_name, batch_size = cfg['dataset'], cfg['criterion'], cfg['batch_size']

    dataloader = load_datasets(dataset_name, cfg['max_images_to_use'], batch_size, cfg['classes_to_use'])
    c, h, w = dataloader.dataset.dataset[0][0].shape
    num_classes = len(np.unique(dataloader.dataset.dataset.targets))

    criterion, embedding_criterion = get_criterion(criterion_name, cfg)

    model, optimizer = get_model_and_optimizer(cfg)

    model.init()
    rand_num = random.randint(0, 1000)
    log_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + "-" + str(rand_num)
    train_model(model, optimizer, log_name, criterion, dataloader, DEVICE, cfg, embedding_criterion)
