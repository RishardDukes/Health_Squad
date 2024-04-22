# MyModels.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the model architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1).double()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1).double()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1).double()
        self.conv4 = nn.Conv2d(128, 1, kernel_size=3, padding=1).double()

    def forward(self, x):
        # Forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))  # Sigmoid activation for binary classification
        return x
