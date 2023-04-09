import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class Resnet50(nn.Module):
    def __init__(self, pretrained=True, bn_freeze = True):
        super(Resnet50, self).__init__()

        self.model = resnet50(pretrained)
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)
        self.model.fc = nn.Identity()
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])
                    
        if bn_freeze:
            for m in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                m.eval()
                m.train = lambda _: None

    def forward(self, x):        
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        x = x.view(x.size(0), -1)
        
        return x