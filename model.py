#####################################
# METU - CENG483 HW3                #
# Author:                           #
#  Necip Fazil Yildiran             #
#####################################

import torch
import torch.nn as nn
import torch.nn.functional as F

# network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, padding=2) # 5x5, in: L, 16 filters
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 24, 3, padding=1) # 3x3, in: 16 from conv1, 24 filters
        self.batchnorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 2, 3, padding=1) # 3x3, in: 16 from conv1, 2 filters: a,b output

    def forward(self, x):
        # normalization
        x = (x / 50.0) - 1.0
        # Notice that relu and maxpool operations are applied directly from torch.nn.functional
        # .. since these layers do not have any parameters to be learnt. 
        x = self.conv1(x)      # conv1
        x = F.max_pool2d(x, 2) # maxpool: 2x2 kernel to decrease height&width to half
        x = self.batchnorm1(x)      # batch normalization
        x = F.relu(x)          # ReLU
        x = self.conv2(x)      # conv2
        x = F.max_pool2d(x, 2) # maxpool: 2x2 kernel to decrease height&width to half
        x = self.batchnorm2(x)      # batch normalization
        x = F.relu(x)          # ReLU
        x = self.conv3(x)      # conv3

        return x
