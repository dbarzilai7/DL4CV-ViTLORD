import torch
import torch.nn as nn
import torch.nn.functional as F


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class GeneratorBasic(nn.Module):
    """
    very simple generator
    code_dim is the number of units in code (it's a 1-dimensional vector)
    taken from here    https://github.com/tneumann/minimal_glo/blob/master/glo.py
    """

    def __init__(self, content_dim, adain_layers, adain_dim, img_shape):
        super(GeneratorBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(content_dim + adain_dim, 84), nn.ReLU(True),
            nn.Linear(84, 120), nn.ReLU(True),
            nn.Linear(120, 16 * 5 * 5), nn.ReLU(True),
            View(shape=(-1, 16, 5, 5)),
            torch.nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, 16, 5),
            nn.BatchNorm2d(16), nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, img_shape[1], 5, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, code):
        return self.net(code)

    def test(self):
        code_dim = 50
        batch_size = 32
        random_tensor = torch.rand(batch_size, code_dim)
        print(f'the shape of the code is {random_tensor.shape}')
        result = self.forward(random_tensor)
        print(f'the shape of the result is {result.shape}')


class GeneratorVitLord(nn.Module):
    """
    Our optimal generator
    """

    def __init__(self, content_dim, adain_layers, adain_dim, img_shape):
        super(GeneratorVitLord, self).__init__()
        batches, c, h, w = img_shape
        self.net = nn.Sequential(
            nn.Linear(content_dim + adain_dim, h * w * (adain_dim // 8)),
            # TODO aplpha=0.3 ?
            nn.LeakyReLU(),

            nn.Linear(h * w * (adain_dim // 8), h * w * (adain_dim // 4)),
            nn.LeakyReLU(),

            nn.Linear(h * w * (adain_dim // 4), h * w * adain_dim),
            nn.LeakyReLU(),

            View(shape=(batches, adain_dim, h, w)),

            # TODO next four lines in for loops for i in range(adain_layers)
            #             torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(adain_dim, adain_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # TODO ADAIN

            torch.nn.Conv2d(adain_dim, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            torch.nn.Conv2d(64, c, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, code):
        return self.net(code)


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


class GeneratorDone(nn.Module):

    def __init__(self, content_dim, n_adain_layers, adain_dim, img_shape):
        super().__init__()
        class_dim = 256
        self.modulation = Modulation(class_dim, n_adain_layers, adain_dim)

        self.__initial_height = img_shape[2] // (2 ** n_adain_layers)
        self.__initial_width = img_shape[3] // (2 ** n_adain_layers)
        self.__adain_dim = adain_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(
                in_features=content_dim,
                out_features=self.__initial_height * self.__initial_width * (adain_dim // 4)
            ),

            nn.LeakyReLU(),

            # nn.Linear(
            #     in_features=self.__initial_height * self.__initial_width * (adain_dim // 8),
            #     out_features=self.__initial_height * self.__initial_width * (adain_dim // 4)
            # ),
            #
            # nn.LeakyReLU(),

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

            nn.Conv2d(in_channels=64, out_channels=img_shape[1], padding=3, kernel_size=7),
            nn.Sigmoid()
        )

    def assign_adain_params(self, adain_params):
        for m in self.adain_conv_layers.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                m.bias = adain_params[:, m.adain_layer_idx, :, 0]
                m.weight = adain_params[:, m.adain_layer_idx, :, 1]

    def forward(self, content_code, class_code):
        class_adain_params = self.modulation(class_code)
        self.assign_adain_params(class_adain_params)

        x = self.fc_layers(content_code)
        x = x.reshape(-1, self.__adain_dim, self.__initial_height, self.__initial_width)
        x = self.adain_conv_layers(x)
        x = self.last_conv_layers(x)

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
