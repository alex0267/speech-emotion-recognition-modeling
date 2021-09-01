import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LeakyReLU

from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DnnModel(BaseModel):
    def __init__(self, seed, num_classes, dropout=0.25):
        super(DnnModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1)
        )
        self.conv1_drop = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1)
        )
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)
        )
        self.conv3_drop = nn.Dropout2d(p=dropout)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)
        )
        self.conv4_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(64 * 4, num_classes)

    def forward(self, x):
        x = F.elu(F.max_pool2d(self.conv1(x), (2, 1)))
        x = self.conv1_drop(x)
        x = F.elu(F.max_pool2d(self.conv2(x), (2, 2)))
        x = self.conv2_drop(x)
        x = F.elu(F.max_pool2d(self.conv3(x), (2, 2)))
        x = self.conv3_drop(x)
        x = F.elu(F.max_pool2d(self.conv4(x), (2, 2)))
        x = self.conv4_drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class DnnModelLr(BaseModel):
    def __init__(self, seed, num_classes, dropout=0.25):
        super(DnnModelLr, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1)
        )
        self.conv1_drop = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1)
        )
        self.conv2_drop = nn.Dropout2d(p=dropout)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)
        )
        self.conv3_drop = nn.Dropout2d(p=dropout)

        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)
        )
        self.conv4_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(64 * 4, num_classes)

    def forward(self, x):
        x = LeakyReLU()(F.max_pool2d(self.conv1(x), (2, 1)))
        x = self.conv1_drop(x)
        x = LeakyReLU()(F.max_pool2d(self.conv2(x), (2, 2)))
        x = self.conv2_drop(x)
        x = LeakyReLU()(F.max_pool2d(self.conv3(x), (2, 2)))
        x = self.conv3_drop(x)
        x = LeakyReLU()(F.max_pool2d(self.conv4(x), (2, 2)))
        x = self.conv4_drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
