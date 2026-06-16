import os
import time

from app.services.ocr.base import OCRResult

_readers = {}


class EasyOCRProvider:
    name = "easyocr"

    def recognize(self, image_path: str, config: dict) -> OCRResult:
        started_at = time.perf_counter()
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "easyocr is not installed. Install optional OCR dependencies from requirements-ocr.txt."
            ) from exc

        languages = _split_languages(
            config.get("EASYOCR_LANGS") or os.getenv("EASYOCR_LANGS", "ru,en")
        )
        gpu = _as_bool(config.get("EASYOCR_GPU") or os.getenv("EASYOCR_GPU"), default=False)
        reader_key = (tuple(languages), gpu)

        if reader_key not in _readers:
            _readers[reader_key] = easyocr.Reader(languages, gpu=gpu)

        result = _readers[reader_key].readtext(image_path, detail=1, paragraph=True)
        lines = [item[1] for item in result]
        confidences = [item[2] for item in result if len(item) > 2]

        return OCRResult(
            provider=self.name,
            text="\n".join(lines).strip(),
            elapsed_seconds=time.perf_counter() - started_at,
            metadata={
                "languages": languages,
                "gpu": gpu,
                "blocks": len(result),
                "avg_confidence": _avg(confidences),
            },
        )


def _split_languages(value: str) -> list[str]:
    return [lang.strip() for lang in value.split(",") if lang.strip()]


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
