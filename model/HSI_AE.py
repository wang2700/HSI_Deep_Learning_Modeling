import torch
import torch.nn as nn
import torch.nn.functional as F

class HSI_AE(nn.Module):
    def __init__(self, layer_channel, n_layers, maxpool_after, stride_before):
        super(HSI_AE, self).__init__()
        
        self.encoderList = nn.ModuleList()
        self.decoderList = nn.ModuleList()
        self.maxpool_after = maxpool_after

        #encoder
        prev_channel = layer_channel[0]
        for i in range(1, n_layers):
            n_channel = layer_channel[i]
            stride = 1
            if i <= stride_before:
                stride = 2
            self.encoderList.append(nn.Conv2d(prev_channel, n_channel, kernel_size=3, stride=stride))
            prev_channel = n_channel

        #decoder
        for i in range(n_layers-2, -1, -1):
            n_channel = layer_channel[i]
            stride = 1
            if i < stride_before:
                stride = 2
            self.decoderList.append(nn.ConvTranspose2d(prev_channel, n_channel, kernel_size=3, stride=stride))
            prev_channel = n_channel

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2, return_indices=True)
        # self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.batchnorm = nn.BatchNorm2d(num_features=layer_channel[0])

    def forward(self, x):
        x = self.batchnorm(x)
        indices_list = []
        output_size = []
        for i, layer in enumerate(self.encoderList):
            x = F.relu(layer(x))
            # if i > self.maxpool_after:
            #     output_size.append(x.size())
            #     x, indices = self.pool(x)
            #     indices_list.append(indices)

        features = x
        unpool_before = len(self.decoderList) - self.maxpool_after - 1
        for i, layer in enumerate(self.decoderList):
            # if i < unpool_before:
            #     indices = indices_list.pop()
            #     x = self.unpool(x, indices, output_size=output_size.pop()) 
            x = F.relu(layer(x))

        return features, x

def test():
    b, c, h, w = 1, 360, 241, 201
    images = torch.randn(b, c, h, w)
    layer_channel = list(range(360, 0, -10))
    net = HSI_AE(layer_channel, len(layer_channel), 6, 1)
    print(net)
    print(sum(p.numel() for p in net.parameters()))
    features, reconstruct = net(x=images)
    print(images.shape)
    print(features.shape)
    print(reconstruct.shape)

if __name__ == "__main__":
    test()
