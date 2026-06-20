from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.services.image_processor import read_image_safe


@dataclass(frozen=True)
class ImageVariant:
    name: str
    path: Path


DEFAULT_VARIANTS = (
    "original",
    "grayscale_soft",
    "contrast",
    "denoise",
    "deskew",
    "line_removed",
)


def create_image_variants(
    image_path: Path,
    output_dir: Path,
    variant_names: list[str] | tuple[str, ...] = DEFAULT_VARIANTS,
) -> list[ImageVariant]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = []

    for variant_name in variant_names:
        variant_path = output_dir / f"{image_path.stem}__{variant_name}.png"
        create_image_variant(image_path, variant_path, variant_name)
        variants.append(ImageVariant(name=variant_name, path=variant_path))

    return variants


def create_image_variant(image_path: Path, output_path: Path, variant_name: str) -> Path:
    img = read_image_safe(str(image_path))
    if img is None:
        raise ValueError(f"Cannot open image: {image_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if variant_name == "original":
        shutil.copyfile(image_path, output_path)
        return output_path

    if variant_name == "grayscale_soft":
        result = grayscale_soft(img)
    elif variant_name == "contrast":
        result = enhance_contrast(img)
    elif variant_name == "denoise":
        result = denoise(img)
    elif variant_name == "deskew":
        result = deskew(img)
    elif variant_name == "line_removed":
        result = remove_notebook_lines(img)
    elif variant_name == "otsu_binary":
        result = otsu_binary(img)
    else:
        raise ValueError(f"Unknown image variant: {variant_name}")

    cv2.imwrite(str(output_path), result)
    return output_path


def grayscale_soft(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)


def deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(inverted > 0))

    if len(coords) == 0:
        return img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.2 or abs(angle) > 15:
        return img

    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def remove_notebook_lines(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    line_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    result = cv2.inpaint(img, line_mask, 3, cv2.INPAINT_TELEA)
    return result


def otsu_binary(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
