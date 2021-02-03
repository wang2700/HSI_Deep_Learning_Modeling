import torch
import torch.nn as nn
import torch.nn.functional as F

class Treatment_Classifier(nn.Module):
    def __init__(self, n_classes, input_ch):
        super(Treatment_Classifier, self).__init__()

        self.conv1 = nn.Conv2d(input_ch, 50, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(50, 10, kernel_size=3, stride=2)
        self.fc = nn.Linear(10*30*26, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.fc(x.view(-1, 10*30*26))
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    b, c, h, w = 10, 150, 124, 109
    n_classes = 1
    input_ch = 150
    features = torch.randn(b, c, h, w)
    net = Treatment_Classifier(n_classes, input_ch)
    classfication = net(features)
    print(classfication)