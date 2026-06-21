import torch.nn as nn
from torchvision import models

class AmazonResnet(nn.Module):
    """
    Image classification model based on ResNet50.
    To adapt the last fully connected layer (fc) to the desired number of classes
    and to allow freezing the backbone feature extractor weights.
    """
    def __init__(self, num_classes=17, freeze_backbone=True):
        super().__init__()
        # To load ResNet50 with pre-trained weights from ImageNet
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # To freeze backbone layers for transfer learning
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # To obtain the number of input features of the original final layer
        in_features = self.backbone.fc.in_features
        # To replace the original fc layer with a linear layer adapted to the target classes
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
