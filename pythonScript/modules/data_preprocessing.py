import os
from enum import Enum
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


class ImageType(Enum):
    CANNY = "canny"
    MORPHOLOGY = "morphology"
    NORMAL = "normal"
    ORIGINAL = "original"


class Label(Enum):
    WITHOUT_SIGN = 0
    WITH_SIGN = 1


# This function does to much - should be one for loading image and one for applying cmbw
def load_images_from_folder(
    folder: Path,
    label: Label,
    target_size: Tuple[int],
    img_type: ImageType = ImageType.NORMAL,
) -> Tuple[np.ndarray]:
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

        if img_type == ImageType.CANNY:
            img = apply_canny(img, target_size)
        elif img_type == ImageType.MORPHOLOGY:
            img = apply_morphology(img, target_size)
        elif img_type == ImageType.NORMAL:
            img = black_and_white(img, target_size)

        images.append(img)
        labels.append(label.value)
    return np.array(images), np.array(labels)


def apply_canny(image, image_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    tight = cv2.Canny(blurred, 140, 160)
    tight = cv2.resize(tight, image_size)
    tight = np.expand_dims(tight, axis=-1)
    return tight


def apply_morphology(image, target_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 190, 210, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    image = cv2.resize(eroded, target_size)
    image = np.expand_dims(image, axis=-1)
    return image


def black_and_white(image, target_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=-1)
    return image
