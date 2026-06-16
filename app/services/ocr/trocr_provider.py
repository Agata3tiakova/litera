import os
import time

from PIL import Image

from app.services.ocr.base import OCRResult

_models = {}


class TrOCRProvider:
    name = "trocr"

    def recognize(self, image_path: str, config: dict) -> OCRResult:
        started_at = time.perf_counter()
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed. Install optional OCR dependencies from requirements-ocr.txt."
            ) from exc

        model_name = config.get("TROCR_MODEL") or os.getenv(
            "TROCR_MODEL",
            "microsoft/trocr-base-handwritten",
        )
        processor, model = _get_model(model_name)

        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return OCRResult(
            provider=self.name,
            text=text.strip(),
            elapsed_seconds=time.perf_counter() - started_at,
            metadata={"model": model_name},
        )


def _get_model(model_name: str):
    if model_name not in _models:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        _models[model_name] = (
            TrOCRProcessor.from_pretrained(model_name),
            VisionEncoderDecoderModel.from_pretrained(model_name),
        )
    return _models[model_name]
