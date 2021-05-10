import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


class HSI_Regre(nn.Module):
    def __init__(self, input_channel, prediction_channel):
        super(HSI_Regre, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, 3, kernel_size=1)
        self.model = models.resnet34(pretrained=True)
        self.fc = nn.Linear(1000, prediction_channel)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.normalize(x)
        x = self.model(x)
        x = F.relu(self.fc(x))
        return x


if __name__ == "__main__":
    b, c, w, h = 3, 5, 240, 200
    image = torch.rand(b, c, w, h)
    model = HSI_Regre(c, 2)
    print(model)
    output = model(image)
    print(output.shape)
