import base64
import os
import time

import requests

from app.services.ocr.base import OCRResult


class YandexVisionProvider:
    name = "yandex"

    def recognize(self, image_path: str, config: dict) -> OCRResult:
        started_at = time.perf_counter()
        vision_key = config.get("VISION_API_KEY") or os.getenv("VISION_API_KEY")
        vision_url = config.get("VISION_API_URL") or os.getenv("VISION_API_URL")

        if not vision_key or not vision_url:
            raise RuntimeError("Yandex Vision API key or URL not set")

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "analyze_specs": [{
                "content": image_b64,
                "features": [{
                    "type": "TEXT_DETECTION",
                    "text_detection_config": {
                        "language_codes": ["ru"],
                    },
                }],
            }],
        }
        headers = {
            "Authorization": f"Api-Key {vision_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            vision_url,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        pages = (
            data
            .get("results", [{}])[0]
            .get("results", [{}])[0]
            .get("textDetection", {})
            .get("pages", [])
        )

        lines = []
        for page in pages:
            for block in page.get("blocks", []):
                for line in block.get("lines", []):
                    words = [w.get("text", "") for w in line.get("words", [])]
                    lines.append(" ".join(words))

        return OCRResult(
            provider=self.name,
            text="\n".join(lines).strip(),
            elapsed_seconds=time.perf_counter() - started_at,
            metadata={"raw_pages": len(pages)},
        )
