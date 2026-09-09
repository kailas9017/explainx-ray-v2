import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """DenseNet121 with its classifier head replaced for `num_classes` outputs.

    Same architecture as the earlier project (transfer-learned from ImageNet),
    which reached 0.82 validation AUROC on the 14-class set. Kept unchanged here —
    the previous project's problem was never the architecture, it was the class
    list going out of sync with the saved weights.
    """
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model
