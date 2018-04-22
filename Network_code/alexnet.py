import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet','alexnet']

model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    """
    2012年ImageNet上大放异彩的卷积神经网路
    """

    def __init__(self,num_classes = 1000):
        super(AlexNet,self).__init__()
        self.features = nn.Sequential(
            
        )



