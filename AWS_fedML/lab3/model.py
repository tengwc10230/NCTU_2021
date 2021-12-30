import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Net_femnist(nn.Module):
    def __init__(self):
        super(Net_femnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 62)
        self.same_padding = nn.ReflectionPad2d(2)

    def forward(self, x):
        x = self.same_padding(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.same_padding(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 3136)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
