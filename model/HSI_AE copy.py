import torch
import torch.nn as nn
import torch.nn.functional as F

class HSI_AE(nn.Module):
    def __init__(self, n_wavelength, n_latent):
        super(HSI_AE, self).__init__()

        #encoder
        self.en1 = nn.Conv2d(n_wavelength, 300, kernel_size=3, stride=1, padding=2)
        self.en2 = nn.Conv2d(300, 200, kernel_size=3, stride=2, padding=2)
        self.en3 = nn.Conv2d(200, 100, kernel_size=3, stride=2, padding=2)
        self.en4 = nn.Conv2d(100, n_latent, kernel_size=3, stride=2, padding=2)
        #decoder
        self.de4 = nn.ConvTranspose2d(n_latent, 100, kernel_size=3, stride=2, padding=2)
        self.de3 = nn.ConvTranspose2d(100, 200, kernel_size=3, stride=2, padding=2)
        self.de2 = nn.ConvTranspose2d(200, 300, kernel_size=3, stride=2, padding=2)
        self.de1 = nn.ConvTranspose2d(300, n_wavelength, kernel_size=3, stride=1, padding=2)

    def forward(self, x):
        x = F.relu(self.en1(x))
        x = F.relu(self.en2(x))
        x = F.relu(self.en3(x))
        x = F.relu(self.en4(x))

        features = x

        x = F.relu(self.de4(x))
        x = F.relu(self.de3(x))
        x = F.relu(self.de2(x))
        x = F.relu(self.de1(x))

        return features, x

def test():
    b, c, h, w = 10, 360, 241, 201
    images = torch.randn(b, c, h, w)
    net = HSI_AE(c, 50)
    features, reconstruct = net(images)
    print(features.shape)
    print(reconstruct.shape)

if __name__ == "__main__":
    test()
