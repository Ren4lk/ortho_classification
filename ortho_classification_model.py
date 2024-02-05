from torch import nn
from torchvision import models


class OCModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model_name = 'resnet18'
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = OCModel()
    print(model)
