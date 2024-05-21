

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.fc1 = nn.Linear(512, 120)
        self.fc2 = nn.Linear(120, 2)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out


##class Network(nn.Module):
##    def __init__(self):
##        super(Network, self).__init__()
##
##        self.layer1 = nn.Sequential(
##            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
##            nn.ReLU(),
##            nn.MaxPool2d(kernel_size=3, stride=2))
##
##        self.layer2 = nn.Sequential(
##            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
##            nn.ReLU(),
##            nn.MaxPool2d(kernel_size=3, stride=2))
##
##        self.layer3 = nn.Sequential(
##            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
##            nn.ReLU(),
##            nn.MaxPool2d(kernel_size=3, stride=2))
##
##        self.pool = nn.AdaptiveAvgPool2d(30)
##        self.drop_out = nn.Dropout()
##        self.fc1 = nn.Linear(30 * 30 * 3, 2)
##        self.fc2 = nn.Linear(1000, 2)
##
##    def forward(self, x):
##        out = x
##        out = self.layer1(x)
##        out = self.layer2(out)
##        out = self.layer3(out)
##        #out = self.layer4(out)
##        out = self.pool(out)
##        #print(out)
##        out = torch.flatten(out)
##        out = self.drop_out(out)
##        out = self.fc1(out)
##        #out = self.fc2(out)
##        return out
##
