
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

class RETFoundRegression(nn.Module):
    def __init__(self, backbone: VisionTransformer):
        super(RETFoundRegression, self).__init__()
        self.backbone = backbone
        self.backbone.reset_classifier(0)  # Remove classification head
        self.head = nn.Linear(self.backbone.embed_dim, 1)  # Regression output

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output
