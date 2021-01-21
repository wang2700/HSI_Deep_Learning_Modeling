import torch
import torch.nn as nn
import torch.nn.functional as F

class Spec_AE(nn.Module):
    def __init__(self, n_wavelength, n_latent, n_layers):
        super(Spec_AE, self).__init__()

        steps = (n_latent - n_wavelength) / (n_layers)
        channels = [int(n_wavelength + steps * i) for i in range(n_layers)]
        channels.append(n_latent)
        self.encoderList = nn.ModuleList()
        self.decoderList = nn.ModuleList()

        #encoder
        prev_channel = channels[0]
        for i in range(1, n_layers+1):
            curr_channel = channels[i]
            self.encoderList.append(nn.Linear(prev_channel, curr_channel))
            prev_channel = curr_channel

        #decoder
        for i in range(n_layers-1, -1, -1):
            curr_channel = channels[i]
            self.decoderList.append(nn.Linear(prev_channel, curr_channel))
            prev_channel = curr_channel

        self.batchnorm = nn.BatchNorm1d(n_wavelength)
        
    def forward(self, x):
        # x = self.batchnorm(x)
        for i, layer in enumerate(self.encoderList):
            x = F.relu(layer(x))

        features = x

        for i, layer in enumerate(self.decoderList):
            x = F.relu(layer(x))

        return features, x

if __name__ == "__main__":
    b, spec = 48000, 360
    specs = torch.randn(b,spec)
    net = Spec_AE(360, 5, 3)
    print(net)
    print(sum(p.numel() for p in net.parameters()))
    features, reconstruct = net(specs)
    print(features.shape)
    print(reconstruct.shape)