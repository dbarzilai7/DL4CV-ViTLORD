"""
Much of the code is taken from https://github.com/avivga/lord-pytorch
"""

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn
from torchvision import models
from torchvision.transforms import transforms, Resize

from extractor import VitExtractor
from util_functions import *


class LatentModel(nn.Module):

    def __init__(self, cfg, n_imgs, n_classes, image_height, image_width, channels):
        super().__init__()

        self.image_shape = (image_height, image_width, channels)
        self.cfg = cfg

        self.embeddings = LatentModel.build_embedding(cfg, n_imgs, n_classes)

        self.modulation = Modulation(cfg['class_dim'], cfg['n_adain_layers'], cfg['adain_dim'])

        self.generator = Generator(cfg['content_dim'], cfg['n_adain_layers'], cfg['adain_dim'], self.image_shape)

    def forward(self, content_imgs, content_id, class_imgs, class_id):
        content_code, class_code = self.embeddings(content_imgs, content_id, class_imgs, class_id)
        class_adain_params = self.modulation(class_code)
        generated_img = self.generator(content_code, class_adain_params)

        return {
            'img': generated_img,
            'content_code': content_code,
            'class_code': class_code
        }

    def init(self):
        self.apply(self.weights_init)

    def reset_generator(self, device=None):
        self.generator = Generator(self.cfg['content_dim'], self.cfg['n_adain_layers'], self.cfg['adain_dim'],
                                   self.image_shape)
        LatentModel.weights_init(self.generator)
        if device is not None:
            self.generator = self.generator.to(device)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, a=-0.05, b=0.05)

    @staticmethod
    def build_embedding(cfg, n_imgs, n_classes):
        if cfg['embedding_type'].lower() == 'basic':
            return LatentEmbeddings(cfg, n_imgs, n_classes)
        elif cfg['embedding_type'].lower() == 'dino':
            return DinoEmbedding(cfg)
        else:
            print("Invalid value in embedding_type")
            exit(0)


class AmortizedModel(nn.Module):

    def __init__(self, cfg, image_height, image_width, channels):
        super().__init__()

        self.image_shape = (image_height, image_width, channels)
        self.cfg = cfg

        self.content_encoder = Encoder(self.image_shape, cfg['content_dim'])
        self.class_encoder = Encoder(self.image_shape, cfg['class_dim'])
        self.modulation = Modulation(cfg['class_dim'], cfg['n_adain_layers'], cfg['adain_dim'])
        self.generator = Generator(cfg['content_dim'], cfg['n_adain_layers'], cfg['adain_dim'], self.image_shape)

    def forward(self, img):
        return self.convert(img, img)

    def convert(self, content_img, class_img):
        content_code = self.content_encoder(content_img)
        class_code = self.class_encoder(class_img)
        class_adain_params = self.modulation(class_code)
        generated_img = self.generator(content_code, class_adain_params)

        return {
            'img': generated_img,
            'content_code': content_code,
            'class_code': class_code
        }


class RegularizedEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, stddev):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.stddev = stddev

    def forward(self, x):
        x = self.embedding(x)

        if self.training and self.stddev != 0:
            noise = torch.zeros_like(x)
            noise.normal_(mean=0, std=self.stddev)

            x = x + noise

        return x


class Modulation(nn.Module):

    def __init__(self, code_dim, n_adain_layers, adain_dim):
        super().__init__()

        self.__n_adain_layers = n_adain_layers
        self.__adain_dim = adain_dim

        self.adain_per_layer = nn.ModuleList([
            nn.Linear(in_features=code_dim, out_features=adain_dim * 2)
            for _ in range(n_adain_layers)
        ])

    def forward(self, x):
        adain_all = torch.cat([f(x) for f in self.adain_per_layer], dim=-1)
        adain_params = adain_all.reshape(-1, self.__n_adain_layers, self.__adain_dim, 2)

        return adain_params


class Generator(nn.Module):

    def __init__(self, content_dim, n_adain_layers, adain_dim, img_shape):
        super().__init__()

        self.__initial_height = img_shape[0] // (2 ** n_adain_layers)
        self.__initial_width = img_shape[1] // (2 ** n_adain_layers)
        self.__adain_dim = adain_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(
                in_features=content_dim,
                out_features=self.__initial_height * self.__initial_width * (adain_dim // 8)
            ),

            nn.LeakyReLU(),

            nn.Linear(
                in_features=self.__initial_height * self.__initial_width * (adain_dim // 8),
                out_features=self.__initial_height * self.__initial_width * (adain_dim // 4)
            ),

            nn.LeakyReLU(),

            nn.Linear(
                in_features=self.__initial_height * self.__initial_width * (adain_dim // 4),
                out_features=self.__initial_height * self.__initial_width * adain_dim
            ),

            nn.LeakyReLU()
        )

        self.adain_conv_layers = nn.ModuleList()
        for i in range(n_adain_layers):
            self.adain_conv_layers += [
                nn.Upsample(scale_factor=(2, 2)),
                nn.Conv2d(in_channels=adain_dim, out_channels=adain_dim, padding=1, kernel_size=3),
                nn.LeakyReLU(),
                AdaptiveInstanceNorm2d(adain_layer_idx=i)
            ]

        self.adain_conv_layers = nn.Sequential(*self.adain_conv_layers)

        self.last_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=adain_dim, out_channels=64, padding=2, kernel_size=5),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=img_shape[2], padding=3, kernel_size=7),
            nn.Sigmoid()
        )

    def assign_adain_params(self, adain_params):
        for m in self.adain_conv_layers.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                m.bias = adain_params[:, m.adain_layer_idx, :, 0]
                m.weight = adain_params[:, m.adain_layer_idx, :, 1]

    def forward(self, content_code, class_adain_params):
        self.assign_adain_params(class_adain_params)

        x = self.fc_layers(content_code)
        x = x.reshape(-1, self.__adain_dim, self.__initial_height, self.__initial_width)
        x = self.adain_conv_layers(x)
        x = self.last_conv_layers(x)

        return x


class Encoder(nn.Module):

    def __init__(self, img_shape, code_dim):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=img_shape[-1], out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=4096, out_features=256),
            nn.LeakyReLU(),

            nn.Linear(in_features=256, out_features=256),
            nn.LeakyReLU(),

            nn.Linear(256, code_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv_layers(x)
        x = x.view((batch_size, -1))

        x = self.fc_layers(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, adain_layer_idx):
        super().__init__()
        self.weight = None
        self.bias = None
        self.adain_layer_idx = adain_layer_idx

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]

        x_reshaped = x.contiguous().view(1, b * c, *x.shape[2:])
        weight = self.weight.contiguous().view(-1)
        bias = self.bias.contiguous().view(-1)

        out = F.batch_norm(
            x_reshaped, running_mean=None, running_var=None,
            weight=weight, bias=bias, training=True
        )

        out = out.view(b, c, *x.shape[2:])
        return out


class SimpleLinearEncoder(nn.Module):

    def __init__(self, out_dim):
        super().__init__()

        self.fc_layers = nn.Sequential(
            nn.LazyLinear(out_features=out_dim),
            # nn.LeakyReLU(),
            #
            # nn.Linear(in_features=out_dim * 2, out_features=out_dim),
            # nn.LeakyReLU(),
            #
            # nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.fc_layers(x)


class LatentEmbeddings(nn.Module):
    def __init__(self, cfg, n_imgs, n_classes):
        super().__init__()

        self.content_embedding = RegularizedEmbedding(n_imgs, cfg['content_dim'], cfg['noise_std'])
        self.class_embedding = nn.Embedding(n_classes, cfg['class_dim'])

    def forward(self, content_imgs, content_id, class_imgs, class_id):
        content_code = self.content_embedding(content_id)
        class_code = self.class_embedding(class_id)

        return content_code, class_code


class DinoEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.extractor = VitExtractor(cfg['dino_model_name'], DEVICE, cfg)
        self.cfg = cfg
        imagenet_norm = transforms.Normalize(*IMAGENET_NORMALIZATION_VALS)
        self.global_resize_transform = Resize(cfg['dino_global_patch_size'], max_size=480)

        self.global_transform = transforms.Compose([self.global_resize_transform, imagenet_norm])

        self.content_encoder = SimpleLinearEncoder(cfg['content_dim'])
        self.class_encoder = SimpleLinearEncoder(cfg['class_dim'])

        self.pca = PCA(n_components=5)

    def forward(self, content_imgs, content_id, class_imgs, class_id):
        content_codes, class_codes = [], []
        for content_im, class_im in zip(content_imgs, class_imgs):  # avoid memory limitations
            content_im = self.global_transform(content_im).unsqueeze(0).to(DEVICE)
            class_im = self.global_transform(class_im).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                if self.cfg['use_id_for_content']:
                    structure = self.extractor.get_keys_from_input(class_im, 11)
                else:
                    structure = self.extractor.get_keys_self_sim_from_input(class_im, layer_num=11)
                cls_token = self.extractor.get_feature_from_input(content_im)[-1][0, 0, :]

                # self_sim = torch.Tensor(self.pca.fit_transform(self_sim.squeeze().cpu())).to(DEVICE)

            content_codes.append(structure)
            class_codes.append(cls_token)

        A = torch.stack(content_codes, dim=0).squeeze()
        if(self.cfg['pca_ssim']):
            _, _, V = torch.pca_lowrank(A)
            A = torch.bmm(A, V[:, :, :self.cfg['pca_ssim']])
        content_codes_ret = self.content_encoder(A.reshape(A.shape[0], -1))
        class_codes_tensor = torch.stack(class_codes, dim=0)
        class_codes_ret = self.class_encoder(class_codes_tensor)

        return content_codes_ret, class_codes_ret


class DecoderEncoder(nn.Module):
    def __init__(self, cfg, n_imgs, n_classes, image_height, image_width, channels):
        super().__init__()
        self.decoder = LatentModel(cfg, n_imgs, n_classes, image_height, image_width, channels)
        self.encoder = DinoEmbedding(cfg)

    def forward(self, content_imgs, content_id, class_imgs, class_id):
        out = self.decoder(content_imgs, content_id, class_imgs, class_id)
        out['out_content_codes'], out['out_class_codes'] = self.encoder(out['img'], content_id, out['img'], class_id)
        return out

    def init(self):
        self.decoder.init()

    def reset_generator(self, device=None):
        self.decoder.reset_generator(device)
