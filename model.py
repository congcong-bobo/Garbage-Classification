import torch
import torch.nn as nn  
from torchvision.models import resnet50

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)