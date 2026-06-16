import base64
import os
import time

import requests

from app.services.ocr.base import OCRResult


class GoogleVisionProvider:
    name = "google_vision"

    def recognize(self, image_path: str, config: dict) -> OCRResult:
        started_at = time.perf_counter()
        api_key = config.get("GOOGLE_VISION_API_KEY") or os.getenv("GOOGLE_VISION_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_VISION_API_KEY is not set")

        with open(image_path, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "requests": [{
                "image": {"content": content},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": ["ru"]},
            }],
        }
        response = requests.post(
            "https://vision.googleapis.com/v1/images:annotate",
            params={"key": api_key},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        annotation = data.get("responses", [{}])[0].get("fullTextAnnotation", {})

        return OCRResult(
            provider=self.name,
            text=annotation.get("text", "").strip(),
            elapsed_seconds=time.perf_counter() - started_at,
            metadata={"pages": len(annotation.get("pages", []))},
        )
