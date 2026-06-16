import os
import time

from app.services.ocr.base import OCRResult

_instances = {}


class PaddleOCRProvider:
    name = "paddleocr"

    def recognize(self, image_path: str, config: dict) -> OCRResult:
        started_at = time.perf_counter()
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "paddleocr is not installed. Install optional OCR dependencies from requirements-ocr.txt."
            ) from exc

        lang = config.get("PADDLEOCR_LANG") or os.getenv("PADDLEOCR_LANG", "cyrillic")
        use_angle_cls = _as_bool(
            config.get("PADDLEOCR_USE_ANGLE_CLS") or os.getenv("PADDLEOCR_USE_ANGLE_CLS"),
            default=True,
        )
        key = (lang, use_angle_cls)

        if key not in _instances:
            _instances[key] = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

        result = _instances[key].ocr(image_path, cls=use_angle_cls)
        lines = []
        confidences = []

        for page in result or []:
            for item in page or []:
                if len(item) >= 2 and len(item[1]) >= 2:
                    lines.append(item[1][0])
                    confidences.append(float(item[1][1]))

        return OCRResult(
            provider=self.name,
            text="\n".join(lines).strip(),
            elapsed_seconds=time.perf_counter() - started_at,
            metadata={
                "lang": lang,
                "use_angle_cls": use_angle_cls,
                "blocks": len(lines),
                "avg_confidence": _avg(confidences),
            },
        )


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
