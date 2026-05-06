from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from config import DEFAULT_LABELS, DEFAULT_MODEL_NAME


def build_model(
    model_name: str = DEFAULT_MODEL_NAME,
    num_classes: int = len(DEFAULT_LABELS),
    pretrained: bool = False,
) -> nn.Module:
    if model_name != "resnet18":
        raise ValueError(f"Unsupported model_name: {model_name}")

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    labels = checkpoint.get("labels", list(DEFAULT_LABELS))
    model = build_model(
        model_name=checkpoint.get("model_name", DEFAULT_MODEL_NAME),
        num_classes=len(labels),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint

