import torch
import copy
from torch.nn import functional as F

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.filter_size = 3

        self.strided_conv = torch.nn.Conv2d(self.in_channels, self.out_channels, self.filter_size, self.stride, padding=(1,1))
        self.conv = torch.nn.Conv2d(self.out_channels, self.out_channels, self.filter_size, padding=(1,1))
        self.batchnorm1 = torch.nn.BatchNorm2d(self.out_channels)
        self.batchnorm2 = torch.nn.BatchNorm2d(self.out_channels)
        self.batchnorm_input = torch.nn.BatchNorm2d(self.out_channels)
        self.relu = torch.nn.ReLU()
        self.conv1x1 = torch.nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride)

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.batchnorm_input(y)

        x = self.strided_conv(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.batchnorm2(x)

        assert y.size() == x.size(), f"shape error: input shape: {y.size()}, x shape: {x.size()}"
        return self.relu(torch.add(x, y))


class ResNet(torch.nn.Module):
    '''
    Conv2D(3, 64, 7, 2)
    BatchNorm()
    ReLU()
    MaxPool(3, 2)
    ResBlock(64, 64, 1)
    ResBlock(64, 128, 2)
    ResBlock(128, 256, 2)
    ResBlock(256, 512, 2)
    GlobalAvgPool()
    Flatten()
    FC(512, 4)
    Softmax()
    '''
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = torch.nn.Conv2d(3, 64, 7, 2)
        self.batch_norm = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(3,2)
        self.resblocks1 = ResBlock(64,64,1)
        self.resblocks2 = ResBlock(64,128,2)
        self.resblocks3 = ResBlock(128,256,2)
        self.resblocks4 = ResBlock(256,512,2)
        self.globalavgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(512,4)
        self.drop_out = torch.nn.Dropout(.2)
        self.softmax = torch.nn.Softmax(dim=1)
#         self.initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.resblocks1(x)
        x = self.resblocks2(x)
        x = self.resblocks3(x)
        x = self.resblocks4(x)
        x = self.globalavgpool(x)
        # x = F.adaptive_avg_pool2d(x,(1,1))
        x = self.flatten(x)
        x = self.fc(x)
        # x = self.drop_out(x)
        return self.softmax(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                # torch.nn.init.constant_(m.weight,1)
                # torch.nn.init.constant_(m.bias,0)
                pass