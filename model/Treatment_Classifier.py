import torch
import torch.nn as nn
import torch.nn.functional as F

class Treatment_Classifier(nn.Module):
    def __init__(self, n_classes, n_input, n_stage):
        super(Treatment_Classifier, self).__init__()

        self.n_stage = n_stage
        feature_steps = [n_input]
        for i in range(n_stage-1):
            feature_steps.append(int(feature_steps[-1] - (n_input-n_classes)/n_stage))
        feature_steps.append(n_classes)
        print(feature_steps)
        #linear layer for classification
        self.linear_list = []
        for i in range(n_stage):
          self.linear_list.append(nn.Linear(feature_steps[i], feature_steps[i+1]))

    def forward(self, x):
        for i in range(self.n_stage-1):
          x = F.relu(self.linear_list[i](x))
        x = torch.sigmoid(self.linear_list[-1](x))
        return x

if __name__ == "__main__":
    n_classes = 2
    n_input = 100
    n_stage = 6
    b = 5
    features = torch.randn(b, n_input)
    net = Treatment_Classifier(n_classes, n_input, n_stage)
    classfication = net(features)
    print(classfication)