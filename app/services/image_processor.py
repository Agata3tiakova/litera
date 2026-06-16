import os
import base64
import hashlib
import logging
import requests
from uuid import uuid4

import cv2
import numpy as np

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
    vision_key = config.get('VISION_API_KEY') or os.getenv('VISION_API_KEY')
    vision_url = config.get('VISION_API_URL') or os.getenv('VISION_API_URL')

    if not vision_key or not vision_url:
        raise RuntimeError("Yandex Vision API key or URL not set")

    with open(path, 'rb') as f:
        image_bytes = f.read()

    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    payload = {
        "analyze_specs": [{
            "content": image_b64,
            "features": [{
                "type": "TEXT_DETECTION",
                "text_detection_config": {
                    "language_codes": ["ru"]
                }
            }]
        }]
    }

    headers = {
        "Authorization": f"Api-Key {vision_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            vision_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        pages = (
            data
            .get('results', [{}])[0]
            .get('results', [{}])[0]
            .get('textDetection', {})
            .get('pages', [])
        )

        all_text = []

        for page in pages:
            for block in page.get('blocks', []):
                for line in block.get('lines', []):
                    words = [
                        w.get('text', '')
                        for w in line.get('words', [])
                    ]
                    all_text.append(' '.join(words))

        text = '\n'.join(all_text).strip()

        return text

    except Exception:
        logger.exception("Yandex Vision error")
        return ''
