from sklearn.neighbors import NearestNeighbors
import numpy as np
from matplotlib import pyplot as plt

import modules
from dataloader import load_datasets
from evaluator import Evaluator
from util_functions import *

LOG_NAME = sys.argv[1]


def get_model(cfg):
    if cfg['model'].lower() == "decoderencoder":
        model = modules.DecoderEncoder(cfg, dataloader.dataset.indices.size, num_classes, h, w, c)

    elif cfg['model'].lower() == "latent":
        model = modules.LatentModel(cfg, dataloader.dataset.indices.size, num_classes, h, w, c)
    else:
        print("Invalid model name")
        exit(0)
    return model

if __name__ == "__main__":
    print("STARTING")
    cfg = load_conf('./confs/{}'.format(LOG_NAME)+'.yaml')
    set_seed(cfg['seed'])

    dataset_name, criterion_name, batch_size = cfg['dataset'], cfg['criterion'], cfg['batch_size']
    dataloader = load_datasets(dataset_name, cfg['max_images_to_use'], batch_size, cfg['classes_to_use'])
    c, h, w = dataloader.dataset.dataset[0][0].shape
    num_classes = len(np.unique(dataloader.dataset.dataset.targets))

    model = get_model(cfg)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load('./models/{}'.format(LOG_NAME)))
    else:
        model.load_state_dict(torch.load('./models/{}'.format(LOG_NAME), map_location=torch.device('cpu')))
    model.eval()

    dataset = dataloader.dataset.dataset
    s = len(dataloader.dataset.dataset)

    images = np.array([dataloader.dataset.dataset[i][0].permute(1, 2, 0).numpy() for i in range(s)])

    targets = np.unique(dataloader.dataset.dataset.targets)
    class_mapping = np.arange(np.amax(targets) + 1)
    class_mapping[targets] = np.arange(targets.size)

    # set up class for evaluations and visualizations
    evaluator = Evaluator(dataloader, DEVICE, None, class_mapping)

    nbrs = NearestNeighbors(n_neighbors=1,metric='euclidean').fit(images.reshape(s, -1))

    outputs = model(evaluator.content_imgs_for_eval, evaluator.contents_ids_for_eval,
                    evaluator.class_imgs_for_eval, evaluator.classes_ids_for_eval)['img'].detach().numpy()

    distances, indices = nbrs.kneighbors(outputs.reshape(outputs.shape[0], -1))

    #fig, axs = plt.subplots(4,4)
    rows = 4
    cols = 4
    fig, axs = plt.subplots(rows, cols)
    for i in range(rows):
        for j in range(cols):
            axs[i][j].imshow(np.moveaxis(outputs[i*4+j],0,2))

    fig2, axs2 = plt.subplots(rows, cols)
    for i in range(rows):
        for j in range(cols):
            axs2[i][j].imshow(images[indices[i*4+j]].squeeze())

    plt.show()
    print(indices)
