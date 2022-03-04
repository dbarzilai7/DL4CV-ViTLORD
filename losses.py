from torchvision.transforms import Resize
from torchvision import transforms
import torch.nn.functional as F
import torchvision
from extractor import VitExtractor
from util_functions import *


class LossG(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.extractor = VitExtractor(model_name=cfg['dino_model_name'], device=DEVICE)

        imagenet_norm = transforms.Normalize(*IMAGENET_NORMALIZATION_VALS)
        self.global_resize_transform = Resize(cfg['dino_global_patch_size'], max_size=480)

        self.global_transform = transforms.Compose([self.global_resize_transform, imagenet_norm])

        self.lambdas = dict(
            content_embedding_reg=cfg['content_reg_dino'],
            lambda_l1=cfg['lambda_l1'],
            lambda_l2=cfg['lambda_l2'],
            lambda_cls=cfg['lambda_global_cls'],
            lambda_ssim=cfg['lambda_global_ssim'],
            lambda_identity=cfg['lambda_global_identity']
        )

    def forward(self, outputs, inputs, content_embedding, epoch=None):
        losses = {}
        loss_G = 0

        if self.lambdas['lambda_ssim'] > 0:
            losses['lambda_ssim'] = self.calculate_global_ssim_loss(outputs, inputs)
            loss_G += losses['lambda_ssim'] * self.lambdas['lambda_ssim']

        if self.lambdas['lambda_cls'] > 0:
            losses['lambda_cls'] = self.calculate_crop_cls_loss(outputs, inputs)
            loss_G += losses['lambda_cls'] * self.lambdas['lambda_cls']

        if self.lambdas['lambda_identity'] > 0:
            losses['lambda_identity'] = self.calculate_global_id_loss(outputs, inputs)
            loss_G += losses['lambda_identity'] * self.lambdas['lambda_identity']

        if self.lambdas['lambda_l1'] > 0:
            losses['lambda_l1'] = torch.nn.functional.l1_loss(inputs, outputs)
            loss_G += losses['lambda_l1'] * self.lambdas['lambda_l1']

        if self.lambdas['lambda_l2'] > 0:
            losses['lambda_l2'] = torch.nn.functional.mse_loss(inputs, outputs)
            loss_G += losses['lambda_l2'] * self.lambdas['lambda_l2']

        if self.lambdas['content_embedding_reg'] > 0:
            loss_G += self.lambdas['content_embedding_reg'] * (torch.sum(content_embedding ** 2, dim=1).mean())

        losses['loss'] = loss_G
        return losses

    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
                keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss/len(outputs)

    def calculate_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):  # avoid memory limitations
            a = self.global_transform(a).unsqueeze(0).to(DEVICE)
            b = self.global_transform(b).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
                target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss/len(outputs)

    def calculate_global_id_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                keys_a = self.extractor.get_keys_from_input(a.unsqueeze(0), 11)
                keys_b = self.extractor.get_keys_from_input(b.unsqueeze(0), 11)
            loss += F.mse_loss(keys_a, keys_b)
        return loss/len(outputs)


class NaiveLoss(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, outputs, inputs, content_embedding, epoch=None):
        l1_loss = torch.nn.L1Loss(reduction='mean')
        l2_loss = torch.nn.MSELoss()
        reg_factor = 1e-3
        return {'loss': l1_loss(inputs, outputs) + l2_loss(inputs, outputs) + reg_factor * torch.norm(
            content_embedding) ** 2}


class NetVGGFeatures(torch.nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vggnet = torchvision.models.vgg16(pretrained=True)
        self.layer_ids = layer_ids

    def forward(self, x):
        output = []
        for i in range(self.layer_ids[-1] + 1):
            x = self.vggnet.features[i](x)

            if i in self.layer_ids:
                output.append(x)

        return output


class VGGDistance(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.layer_ids = [2, 7, 12, 21, 30]
        self.vgg = NetVGGFeatures(self.layer_ids)
        self.cfg = cfg

        self.imagenet_norm = transforms.Normalize(*IMAGENET_NORMALIZATION_VALS)

    def forward(self, outputs, images, content_embedding, epoch=None):
        I1 = outputs
        I2 = images

        b_sz = I1.size(0)
        f1 = self.vgg(I1)
        f2 = self.vgg(I2)

        loss = torch.abs(I1 - I2).view(b_sz, -1).mean(1)

        for i in range(len(self.layer_ids)):
            layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
            loss = loss + layer_loss

        content_penalty = torch.sum(content_embedding ** 2, dim=1).mean()
        return {'loss': self.cfg['lambda_VGG'] * loss.mean() + self.cfg['content_reg_vgg'] * content_penalty}


class ViTVGG(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vgg = VGGDistance(cfg)
        self.vit = LossG(cfg)
        self.cfg = cfg
        self.warmup_epochs = cfg['warm_up_epochs']

    def forward(self, outputs, inputs, content_embedding, epoch=None):
        if epoch < self.warmup_epochs:
            return self.vgg.forward(outputs, inputs, content_embedding, epoch)
        return self.vit.forward(outputs, inputs, content_embedding, epoch)


def get_criterion(name, cfg):
    if name == "Naive":
        return NaiveLoss(cfg)
    elif name == "ViT":
        return LossG(cfg)
    elif name == "VGG":
        return VGGDistance(cfg)
    elif name == "ViTVGG":
        return ViTVGG(cfg)
    else:
        print("Loss not found")
        raise NotImplementedError
