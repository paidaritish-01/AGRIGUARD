"""
Image transformation pipelines for training and evaluation.

Preserves crop-specific pathology cues by keeping color jitter subtle
(avoiding chromatic alterations that would mask true chlorosis or necrosis).
"""

from typing import Tuple
from torchvision import transforms


# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_training_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Transforms for training with appropriate agricultural augmentations.
    Flips, subtle rotations, and mild lighting variations to simulate varying outdoor sunlight.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # Gentle color jitter to simulate cloudy vs sunny field conditions
        # without distorting disease pigmentation (mild hue alteration max 0.02)
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Deterministic transformations for validation, testing, and production inference.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
