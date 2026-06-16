import os
import hashlib
import logging
from uuid import uuid4

import cv2
import numpy as np

from app.services.ocr import run_ocr

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {
    'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'
}

def allowed_file(filename: str) -> bool:
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def hash_image(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def save_uploaded_file(file_storage, upload_folder: str) -> str:

    if not allowed_file(file_storage.filename):
        raise ValueError("Недопустимый формат файла")

    os.makedirs(upload_folder, exist_ok=True)

    ext = os.path.splitext(file_storage.filename)[1].lower()
    if not ext:
        raise ValueError("Файл без расширения")

    fname = f"{uuid4().hex}{ext}"
    path = os.path.join(upload_folder, fname)

    file_storage.save(path)
    return path


def read_image_safe(path: str):

    img = cv2.imread(path)
    if img is not None:
        return img

    try:
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def preprocess_image(path: str) -> str:
    img = read_image_safe(path)

    if img is None:
        raise ValueError(f"Не удалось открыть изображение: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    max_width = 1200

    if width > max_width:
        scale = max_width / width
        new_size = (int(width * scale), int(height * scale))
        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cv2.imwrite(path, thresh)
    return path


def ocr_extract_text(path: str, config: dict) -> str:
    try:
        return run_ocr(path, config).text
    except Exception:
        logger.exception("OCR error")
        return ''
