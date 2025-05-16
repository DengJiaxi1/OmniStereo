import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                             padding=dilation if dilation > 1 else pad, dilation=dilation, bias=True),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                             padding=dilation if dilation > 1 else pad, dilation=dilation, bias=True),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class Fusion(nn.Module):
    def __init__(self, channels):
        super(Fusion, self).__init__()

        self.depth_layer1 = nn.Sequential(BasicBlock(1, channels, 1, 1, 1),
                                          BasicBlock(channels, channels, 1, 1, 1))

        self.depth_layer2 = nn.Sequential(BasicBlock(channels, channels * 2, 2, 1, 1))

        self.depth_layer4 = nn.Sequential(BasicBlock(channels * 2, channels * 4, 2, 1, 1),
                                          nn.ConvTranspose2d(channels * 4, channels * 2, 2, 2),
                                          nn.ReLU(inplace=True))

        self.depth_layer6 = nn.Sequential(BasicBlock(channels * 4, channels * 2, 1, 1, 1),
                                          nn.ConvTranspose2d(channels * 2, channels, 2, 2),
                                          nn.ReLU(inplace=True))

        self.depth_layer7 = nn.Sequential(BasicBlock(channels * 2, channels, 1, 1, 1),
                                          BasicBlock(channels, channels, 1, 1, 1),
                                          nn.Conv2d(channels, 1, kernel_size=1, padding=0, stride=1, bias=True))


    def forward(self, depth, conf):

        mask = depth == 0
        conf_out = conf.masked_fill(mask, -torch.finfo(conf.dtype).max)
        conf_softmax = F.softmax(conf_out, dim=1)
        depth_fusion = torch.sum(depth * conf_softmax, dim=1, keepdim=True)

        # depth encoder
        depth_fusion1 = self.depth_layer1(depth_fusion)
        depth_fusion2 = self.depth_layer2(depth_fusion1)
        depth_fusion4 = self.depth_layer4(depth_fusion2)

        # decoder
        depth_fusion6 = self.depth_layer6(torch.cat((depth_fusion2, depth_fusion4), 1))
        depth_fusion7 = self.depth_layer7(torch.cat((depth_fusion1, depth_fusion6), 1))

        return depth_fusion7
