import io

from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

from dataloader import UnNormalize


class Evaluation:
    def __init__(self, train_loader, device, writer, class_mapping):
        self.unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        # set up some variables for the visualizations
        self.display_contents = train_loader.dataset.indices[:4]
        sample_content_images = [(train_loader.dataset.dataset[i][0]) for i in self.display_contents]
        c, h, w = sample_content_images[0].shape
        self.labels_counts = np.unique(train_loader.dataset.dataset.targets, return_counts=True)
        self.display_classes = self.labels_counts[0][self.labels_counts[1].argsort()[-4:]]

        labels = list(train_loader.dataset.dataset.targets)
        sample_class_indices = [labels.index(i) for i in self.display_classes]
        samples_classes_ims = [train_loader.dataset.dataset[i][0].unsqueeze(0) for i in sample_class_indices]
        tboard_classes = torch.cat([torch.zeros(1, c, h, w)] + samples_classes_ims).to(device)
        self.tboard_contents = torch.cat(
            [train_loader.dataset.dataset[i][0].unsqueeze(0) for i in self.display_contents]).to(
            device)

        self.tboard_batch = torch.zeros(((len(self.display_classes) + 1) * (len(self.display_contents) + 1), c, h, w),
                                        device=device)
        self.tboard_batch[:self.tboard_batch.shape[0]:len(self.display_classes) + 1, ...] = tboard_classes

        self.non_first_col = np.arange(self.tboard_batch.shape[0])
        self.non_first_col = self.non_first_col[self.non_first_col % (len(self.display_contents) + 1) != 0]

        self.classes_for_eval, self.contents_for_eval = [], []
        for disp_class in self.display_classes:
            for disp_contents in self.display_contents:
                self.classes_for_eval.append(disp_class)
                self.contents_for_eval.append(disp_contents)
        self.classes_for_eval, self.contents_for_eval = np.array(self.classes_for_eval), np.array(self.contents_for_eval)

        self.writer = writer
        self.device = device

        self.class_mapping = class_mapping

    def eval(self, model, epoch, all_losses):
        non_first_col = np.arange(self.tboard_batch.shape[0])
        non_first_col = non_first_col[non_first_col % (len(self.display_contents) + 1) != 0]

        model.eval()
        input_classes = []
        input_contents = []
        with torch.no_grad():
            if epoch % 5 == 0:
                #display_classes_norm = [(self.labels_counts[0].tolist()).index(cls) for cls in self.display_classes]
                # for disp_classes in display_classes_norm:
                #     for disp_contents in self.display_contents:
                #         input_contents.append(content_codes[disp_contents].unsqueeze(0))
                #         input_classes.append((class_codes[disp_classes].unsqueeze(0)))
                outputs = model(torch.from_numpy(self.contents_for_eval).to(self.device),
                                (torch.from_numpy(self.class_mapping[self.classes_for_eval])).to(self.device))['img']
                #outputs = self.unorm(outputs)
                self.tboard_batch[non_first_col, ...] = torch.cat((self.tboard_contents, outputs))
                #self.tboard_batch[0, ...] = model(content_codes[1549].unsqueeze(0), class_codes[18].unsqueeze(0))

                img_grid = torchvision.utils.make_grid(self.tboard_batch, nrow=len(self.display_classes) + 1)
                self.writer.add_image('images', img_grid, global_step=epoch)

                #tsne = TSNE(n_components=2, init='random')
                # for points, name in zip([content_codes, class_codes], ['content_codes', 'class_codes']):
                #     points_2d = tsne.fit_transform(points.cpu())
                #     plt.figure()
                #     plt.scatter(points_2d[..., 1], points_2d[..., 0])
                #     self.writer.add_image(name, self.plt_to_tensor(), global_step=epoch)
                #
                #     plt.figure()
                #     plt.hist(np.linalg.norm(points.cpu(), axis=1))
                #     self.writer.add_image(name + " hist", self.plt_to_tensor(), global_step=epoch)


            losses_means = pd.DataFrame(all_losses).mean(axis=0)
            for index, value in losses_means.items():
                self.writer.add_scalar(index, value, global_step=epoch)
            self.writer.flush()
            print("Epoch: {}, loss: {}\n".format(epoch, losses_means.loc['loss']))

    def plt_to_tensor(self):
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)

        image = Image.open(buf)
        return ToTensor()(image)
