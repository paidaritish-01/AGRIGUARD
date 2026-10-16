"""
Unit tests for preprocessing and image transformations.
"""

import pytest
import torch
from PIL import Image
import numpy as np

from agriscan.data.transforms import (
    get_training_transforms,
    get_eval_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def test_eval_transforms_shape_and_type():
    """Verify eval transforms output tensor of exact shape [3, 224, 224] without NaNs."""
    img = Image.new("RGB", (640, 480), color=(100, 150, 80))
    transform = get_eval_transforms(image_size=224)
    tensor = transform(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


def test_training_transforms_augmentation():
    """Verify training transforms produce valid tensors across random augmentations."""
    img = Image.new("RGB", (300, 300), color=(80, 180, 60))
    train_transform = get_training_transforms(image_size=224)

    for _ in range(5):
        tensor = train_transform(img)
        assert tensor.shape == (3, 224, 224)
        assert not torch.isnan(tensor).any()


def test_custom_image_size():
    """Verify custom image sizes are respected."""
    img = Image.new("RGB", (512, 512), color=(50, 120, 50))
    eval_transform = get_eval_transforms(image_size=256)
    tensor = eval_transform(img)
    assert tensor.shape == (3, 256, 256)
