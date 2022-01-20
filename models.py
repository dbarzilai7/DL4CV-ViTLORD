import torch
import torch.nn as nn


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
    def __init__(self,     content_dim, adain_layers, adain_dim, img_shape):
        super(GeneratorBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(content_dim + adain_dim, 84), nn.ReLU(True),
            nn.Linear(84, 120), nn.ReLU(True),
            nn.Linear(120, 16*5*5), nn.ReLU(True),
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

