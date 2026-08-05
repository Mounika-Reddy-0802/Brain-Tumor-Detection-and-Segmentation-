from __future__ import annotations

import albumentations as A


def get_train_transforms() -> A.Compose:
    """Joint image/mask augmentations for segmentation training.

    GaussNoise takes std_range in normalised [0, 1] units. The old var_limit=(10, 50)
    spelling was removed in albumentations 2.x, where unknown arguments are warned
    about and then ignored, so the noise silently fell back to library defaults.
    sqrt(10)/255 and sqrt(50)/255 reproduce the intended strength.

    "mask" is a built-in target, so it must not be declared in additional_targets.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(std_range=(0.012, 0.028), p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.GridDistortion(p=0.2),
        ]
    )


def get_eval_transforms() -> None:
    return None
