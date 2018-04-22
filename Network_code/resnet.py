import torch.nn as nn
import match
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3 (in_planes,out_planes,stride = 1):
    "3 x 3 convolution with padding"
    return nn.Conv2d(in_planes,out_channels,kernel_size = 3,
                    stride= stride,padding= 1 ,bias= False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,inplanes,planes,stride =1 ,downsample = None):
        super(BasicBlock,self)>__init__()
        self.conv1 = conv3x3(inplanes,planes,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        slef.relu = nn.ReLU(inplace=  True)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def foeward(self,x):
        residual = x
        























