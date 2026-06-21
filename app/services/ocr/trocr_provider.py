import os
import time

from PIL import Image

from app.services.ocr.base import OCRResult

_models = {}


class TrOCRProvider:
    name = "trocr"
    model_config_key = "TROCR_MODEL"
    default_model = "microsoft/trocr-base-handwritten"
    device_config_key = "TROCR_DEVICE"
    default_device = "auto"
    max_new_tokens_config_key = "TROCR_MAX_NEW_TOKENS"
    default_max_new_tokens = 128
    segment_lines_config_key = "TROCR_SEGMENT_LINES"
    default_segment_lines = False

    def recognize(self, image_path: str, config: dict) -> OCRResult:
        started_at = time.perf_counter()
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are not installed. Install optional OCR dependencies from requirements-ocr.txt."
            ) from exc

        model_name = (
            config.get(self.model_config_key)
            or os.getenv(self.model_config_key)
            or self.default_model
        )
        device = _resolve_device(
            torch,
            config.get(self.device_config_key)
            or os.getenv(self.device_config_key)
            or self.default_device,
        )
        max_new_tokens = int(
            config.get(self.max_new_tokens_config_key)
            or os.getenv(self.max_new_tokens_config_key)
            or self.default_max_new_tokens
        )
        segment_lines = _as_bool(
            config.get(self.segment_lines_config_key)
            or os.getenv(self.segment_lines_config_key),
            self.default_segment_lines,
        )
        processor, model = _get_model(model_name)
        model.to(device)
        model.eval()

        images = _line_images(image_path) if segment_lines else [Image.open(image_path).convert("RGB")]
        lines = []
        with torch.no_grad():
            for image in images:
                pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                )
                line = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                if line:
                    lines.append(line)

        return OCRResult(
            provider=self.name,
            text="\n".join(lines).strip(),
            elapsed_seconds=time.perf_counter() - started_at,
            metadata={
                "model": model_name,
                "device": device,
                "max_new_tokens": max_new_tokens,
                "segment_lines": segment_lines,
                "line_count": len(images),
            },
        )


class CyrillicTrOCRProvider(TrOCRProvider):
    name = "cyrillic_trocr"
    model_config_key = "CYRILLIC_TROCR_MODEL"
    default_model = "cyrillic-trocr/trocr-handwritten-cyrillic"
    device_config_key = "CYRILLIC_TROCR_DEVICE"
    max_new_tokens_config_key = "CYRILLIC_TROCR_MAX_NEW_TOKENS"
    default_max_new_tokens = 96
    segment_lines_config_key = "CYRILLIC_TROCR_SEGMENT_LINES"
    default_segment_lines = True


def _get_model(model_name: str):
    if model_name not in _models:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        _models[model_name] = (
            TrOCRProcessor.from_pretrained(model_name),
            VisionEncoderDecoderModel.from_pretrained(model_name),
        )
    return _models[model_name]


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_device(torch, configured_device: str) -> str:
    device = (configured_device or "auto").strip().lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "TROCR_DEVICE/CYRILLIC_TROCR_DEVICE is set to cuda, but this PyTorch build cannot access CUDA."
        )
    return device


def _line_images(image_path: str) -> list[Image.Image]:
    import cv2
    import numpy as np

    image = cv2.imread(image_path)
    if image is None:
        return [Image.open(image_path).convert("RGB")]

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        11,
    )

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, width // 18), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, height // 18)))
    lines = cv2.bitwise_or(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel),
    )
    text_mask = cv2.subtract(binary, lines)

    line_bands = _find_line_bands(text_mask, width, height)
    boxes = []
    for y1, y2 in line_bands:
        band = text_mask[y1:y2, :]
        columns = np.where(np.sum(band > 0, axis=0) > 0)[0]
        if len(columns) == 0:
            continue
        x1 = int(columns[0])
        x2 = int(columns[-1])
        if x2 - x1 < 20 or y2 - y1 < 8:
            continue
        boxes.append((x1, y1, x2, y2))
    crops = []
    for x1, y1, x2, y2 in boxes:
        pad_x = max(8, int((x2 - x1) * 0.03))
        pad_y = max(6, int((y2 - y1) * 0.25))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)
        crop = image[y1:y2, x1:x2]
        crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

    return crops or [Image.open(image_path).convert("RGB")]


def _find_line_bands(mask, width: int, height: int) -> list[tuple[int, int]]:
    import cv2
    import numpy as np

    row_density = np.sum(mask > 0, axis=1).astype("float32") / max(1, width)
    window = max(7, height // 180)
    kernel = np.ones(window, dtype="float32") / window
    smoothed = np.convolve(row_density, kernel, mode="same")
    threshold = max(0.006, float(np.percentile(smoothed, 72)) * 0.45)
    active = smoothed > threshold

    bands = []
    start = None
    for y, is_active in enumerate(active):
        if is_active and start is None:
            start = y
        elif not is_active and start is not None:
            if y - start >= 8:
                bands.append((start, y))
            start = None
    if start is not None and height - start >= 8:
        bands.append((start, height))

    bands = _merge_close_bands(bands, max_gap=max(5, height // 140))
    refined = []
    for y1, y2 in bands:
        band = mask[y1:y2, :]
        contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        ys = []
        for contour in contours:
            _, cy, _, ch = cv2.boundingRect(contour)
            if ch >= 3:
                ys.extend((y1 + cy, y1 + cy + ch))
        if ys:
            refined.append((max(0, min(ys)), min(height, max(ys))))

    return _merge_close_bands(refined, max_gap=max(5, height // 160))


def _merge_close_bands(bands: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for y1, y2 in bands:
        if not merged or y1 - merged[-1][1] > max_gap:
            merged.append((y1, y2))
        else:
            py1, py2 = merged[-1]
            merged[-1] = (py1, max(py2, y2))
    return merged
