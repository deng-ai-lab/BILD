import torch
import torch.nn as nn
import torch.nn.functional as F


class Data_prob(nn.Module):
    def __init__(self, bs):
        super(Data_prob, self).__init__()
        self.down_conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.block1_1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.block1_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))

        self.down_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.block2_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.block2_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))

        self.down_conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.block3_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.block3_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(nn.Linear(128, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, 1))
        self.batchsize = bs

    def forward(self, x1):
        x = F.leaky_relu(self.down_conv1(x1), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(x + self.block1_1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(x + self.block1_2(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.down_conv2(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(x + self.block2_1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(x + self.block2_2(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.down_conv3(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(x + self.block3_1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(x + self.block3_2(x), negative_slope=0.2, inplace=True)
        x = self.pool(x).reshape(self.batchsize, 128)
        x = self.linear(x).reshape(self.batchsize)
        x = (x - x.mean())/torch.std(x)
        x = F.softmax(x, dim=0)
        # x = F.sigmoid(x)
        return x
