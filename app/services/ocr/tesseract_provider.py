import os
import time

from app.services.ocr.base import OCRResult


class TesseractProvider:
    name = "tesseract"

    def recognize(self, image_path: str, config: dict) -> OCRResult:
        started_at = time.perf_counter()
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "pytesseract is not installed. Install optional OCR dependencies "
                "from requirements-ocr.txt and the Tesseract binary with Russian language data."
            ) from exc

        command = config.get("TESSERACT_CMD") or os.getenv("TESSERACT_CMD")
        if command:
            pytesseract.pytesseract.tesseract_cmd = command

        lang = config.get("TESSERACT_LANG") or os.getenv("TESSERACT_LANG", "rus+eng")
        psm = config.get("TESSERACT_PSM") or os.getenv("TESSERACT_PSM", "6")
        text = pytesseract.image_to_string(
            image_path,
            lang=lang,
            config=f"--oem 1 --psm {psm}",
        )

        return OCRResult(
            provider=self.name,
            text=text.strip(),
            elapsed_seconds=time.perf_counter() - started_at,
            metadata={"lang": lang, "psm": psm},
        )
